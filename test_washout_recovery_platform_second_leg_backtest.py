"""
延伸test_washout_recovery_with_support_backtest.py：那支只驗證過「夾起單
+ ma120_second_leg（半年線MA120版兩次腳未破）」，這支改用廣義版
platform_second_leg_level（不限MA120、不限日線的兩次腳未破），看看
擴大支撐定義後，組合版的優勢還在不在。

逐日模擬「當天視角只看得到當天為止的資料」：
    fast_slope_up = ma_slope(window, FAST_MA["日"]) == "上揚"
    platform_bias, support_levels = _step6_left_side_platform(window, "日", latest, fast_slope_up, [], [])
    platform_second_leg_level = _detect_platform_second_leg(window, "日", latest, support_levels)

簡化假設：w_bottom_neckline/m_top_neckline用預設值None，沒有另外跑
_detect_w_bottom()/_detect_m_top()先算頸線——這兩個只是_step6_left_
side_platform()用來輔助校正支撐/壓力候選的額外資訊，不是必要輸入，
省略後支撐偵測仍會運作，只是候選價位可能少考慮兩個特殊頸線候選，這次
只是要看「廣義版兩次腳未破+夾起單」這個組合的大方向效果，不要求跟
正式pipeline的支撐候選清單逐一對齊。

用法（在專案資料夾內執行）：
    python3 test_washout_recovery_platform_second_leg_backtest.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import FAST_MA, _detect_platform_second_leg, _step6_left_side_platform
from stock_core import STOCK_NAME_MAP, get_stock_data, ma_slope, run_all_indicators, unique_watchlist
from test_washout_recovery_buy import _detect_mtm_turnup, _detect_washout_recovery

BACKTEST_MONTHS_DEFAULT = 24
SLOPE_LOOKBACK = 3


def _platform_second_leg_series(df_day: pd.DataFrame) -> pd.Series:
    fast_col = FAST_MA["日"]
    flags = []
    for i in range(len(df_day)):
        window = df_day.iloc[:i + 1].reset_index(drop=True)
        if len(window) < 20:  # left_side_platform/platform_second_leg都要求min_bars，資料太短必為None
            flags.append(False)
            continue
        latest = window.iloc[-1]
        fast_slope_up = ma_slope(window, fast_col, lookback=SLOPE_LOOKBACK) == "上揚"
        _bias, support_levels = _step6_left_side_platform(window, "日", latest, fast_slope_up, [], [])
        level = _detect_platform_second_leg(window, "日", latest, support_levels)
        flags.append(level is not None)
    return pd.Series(flags, index=df_day.index)


def analyze_one(name: str, code: str, market: str, api_token: str, backtest_months: int) -> dict:
    today = date.today()
    backtest_start = today - timedelta(days=backtest_months * 30)
    fetch_start = (backtest_start - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")

    try:
        df_day = get_stock_data(code, market, fetch_start, str(today), api_token)
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "code": code, "error": str(exc)}
    if df_day.empty:
        return {"name": name, "code": code, "error": "日線資料為空"}
    df_day = run_all_indicators(df_day, "日")
    if "MA35" not in df_day.columns or "MTM" not in df_day.columns:
        return {"name": name, "code": code, "error": "缺少MA35或MTM欄位"}
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    washout = _detect_washout_recovery(df_day)
    platform_second_leg = _platform_second_leg_series(df_day)
    washout_plus_platform = washout & platform_second_leg

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    recent_mask = df_day["date"] >= backtest_start_str

    def _collect(mask):
        out = []
        for idx in df_day.index[recent_mask & mask]:
            d = df_day.loc[idx, "date"]
            fwd = _forward_returns(df_day, d)
            out.append({f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS})
        return out

    return {
        "name": name, "code": code, "error": None,
        "platform_second_leg_only_events": _collect(platform_second_leg),
        "washout_plus_platform_events": _collect(washout_plus_platform),
        "baseline_events": _collect(pd.Series(True, index=df_day.index)),
    }


def run(api_token: str, backtest_months: int = BACKTEST_MONTHS_DEFAULT) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(analyze_one, name, code, market, api_token, backtest_months)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _print_stats(label: str, events: list) -> None:
    df = pd.DataFrame(events)
    print(f"--- {label}（共{len(events)}筆）---")
    for n in FORWARD_WINDOWS:
        valid = df[f"fwd_ret_{n}d"].dropna() if not df.empty else pd.Series(dtype=float)
        if len(valid):
            print(f"{n}日：平均報酬 {valid.mean()*100:.2f}%　正報酬比例 {(valid>0).mean()*100:.1f}%　(n={len(valid)})")
        else:
            print(f"{n}日：無資料")


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    buckets = {"platform_second_leg_only_events": [], "washout_plus_platform_events": [], "baseline_events": []}
    for r in results:
        if r.get("error"):
            continue
        for key in buckets:
            buckets[key].extend(r[key])

    print(f"\n=== 回測{months_arg}個月，13檔加總（廣義版platform_second_leg_level）===\n")
    _print_stats("廣義版兩次腳未破單獨（對照用）", buckets["platform_second_leg_only_events"])
    print()
    _print_stats("夾起單 + 廣義版兩次腳未破同時成立", buckets["washout_plus_platform_events"])
    print()
    _print_stats("無條件基準（隨便哪天進場）", buckets["baseline_events"])
