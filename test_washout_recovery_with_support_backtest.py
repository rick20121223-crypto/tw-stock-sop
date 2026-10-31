"""
重新測試「夾起單」訊號，但這次搭配「左側平台/半年線兩次腳未破」的支撐
前提一起算——原本test_washout_recovery_buy.py測的是「單純夾起單」（假
跌破MA35後迅速站回），完全沒有搭配支撐context，結果比無條件基準還差。
但盲點3討論時引用的原文是把「左側歷史platform強支撐未破」跟「夾起單」
綁在一起講的，不是獨立訊號，所以這裡給它一個更貼近原文的公平測試：
夾起單 AND 兩次腳未破（重用sop_decision._detect_ma120_second_leg，逐日
模擬「當天視角只看得到當天為止的資料」，不能用未來資訊）。

範圍限縮說明：只用_detect_ma120_second_leg（半年線MA120版「兩次腳未
破」），沒有另外接_detect_platform_second_leg（廣義版，不限MA120/日線）
——後者需要先跑過_step6_left_side_platform()一整條依賴鏈（fast_slope_
up/w_bottom_neckline/m_top_neckline），逐日重算的複雜度明顯更高，這次
先用比較單純的MA120版本驗證概念，如果結果有效再考慮要不要擴大到廣義版。

用法（在專案資料夾內執行）：
    python3 test_washout_recovery_with_support_backtest.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import _detect_ma120_second_leg
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist
from test_washout_recovery_buy import _detect_mtm_turnup, _detect_washout_recovery

BACKTEST_MONTHS_DEFAULT = 24


def _second_leg_series(df_day: pd.DataFrame) -> pd.Series:
    """逐日呼叫_detect_ma120_second_leg()，模擬「當天只看得到當天為止的
    歷史資料」，避免用到未來資訊造成回測偷看答案。"""
    flags = [_detect_ma120_second_leg(df_day.iloc[:i + 1], "日") for i in range(len(df_day))]
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
    if "MA35" not in df_day.columns or "MTM" not in df_day.columns or "MA120" not in df_day.columns:
        return {"name": name, "code": code, "error": "缺少MA35/MTM/MA120欄位"}
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    washout = _detect_washout_recovery(df_day)
    mtm_turnup = _detect_mtm_turnup(df_day)
    second_leg = _second_leg_series(df_day)

    washout_plus_support = washout & second_leg
    washout_plus_support_plus_mtm = washout & second_leg & mtm_turnup

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
        "washout_only_events": _collect(washout),
        "second_leg_only_events": _collect(second_leg),
        "washout_plus_support_events": _collect(washout_plus_support),
        "washout_plus_support_plus_mtm_events": _collect(washout_plus_support_plus_mtm),
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

    buckets = {
        "washout_only_events": [], "second_leg_only_events": [],
        "washout_plus_support_events": [], "washout_plus_support_plus_mtm_events": [],
        "baseline_events": [],
    }
    for r in results:
        if r.get("error"):
            continue
        for key in buckets:
            buckets[key].extend(r[key])

    print(f"\n=== 回測{months_arg}個月，13檔加總 ===\n")
    _print_stats("夾起單單獨（原本的版本，對照用）", buckets["washout_only_events"])
    print()
    _print_stats("兩次腳未破單獨（MA120版support，對照用）", buckets["second_leg_only_events"])
    print()
    _print_stats("夾起單 + 兩次腳未破同時成立（這次要驗證的組合版）", buckets["washout_plus_support_events"])
    print()
    _print_stats("夾起單 + 兩次腳未破 + MTM勾起（三者同時成立）", buckets["washout_plus_support_plus_mtm_events"])
    print()
    _print_stats("無條件基準（隨便哪天進場）", buckets["baseline_events"])
