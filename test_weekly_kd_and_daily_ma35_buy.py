"""
一次測兩個筆記裡提到、資料抓得到的買點條件：

A) 日35MA支撐＋站穩5/10日線：日線生死線(MA35)上揚且站上、同時站上
   10MA(第二關快均線)。這個現有sop_decision.py的_step2_ma()已經在算
   （"中長線結構偏多"跟"短線止跌表態更明確"這兩句reasons文字分別對應
   MA35跟MA10的確認），這裡用文字關鍵字比對重新驗證它事後報酬好不好，
   不是全新機制，是重新檢驗現有機制。

B) 週KD金叉＋站穩週5MA/週35MA(生死線向上)：完全新的條件，程式裡原本
   沒有KD這個指標，這裡用台股慣用的KD公式（RSV 9週期、K/D各用2/3舊值+
   1/3新值平滑，初始值50）現算，套在_build_weekly()重建出來的週線上。

兩個都只看過去資料，可以在完整歷史上一次算好整段序列，不用逐日切片
重建sop_decision.py的判讀。

用法（在專案資料夾內執行）：
    python3 test_weekly_kd_and_daily_ma35_buy.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from diagnose_sell_lag import _build_weekly
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24

MARKER_MA35_BULLISH = "中長線結構偏多"   # 只有key_slope=="上揚" 且 站上 時才會出現
MARKER_MA10_GATE = "短線止跌表態更明確"  # 只有同時站上MA10時才會出現


def _calc_kd(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    """台股慣用KD：RSV取n週期，K/D各用「2/3舊值+1/3新值」平滑，初始值50。"""
    low_n = df["min"].rolling(window=n, min_periods=n).min()
    high_n = df["max"].rolling(window=n, min_periods=n).max()
    denom = (high_n - low_n).replace(0, pd.NA)
    rsv = ((df["close"] - low_n) / denom * 100).fillna(50.0)

    k_vals, d_vals = [], []
    prev_k, prev_d = 50.0, 50.0
    for r in rsv:
        k = prev_k * 2 / 3 + r / 3
        d = prev_d * 2 / 3 + k / 3
        k_vals.append(k)
        d_vals.append(d)
        prev_k, prev_d = k, d
    df = df.copy()
    df["K"] = k_vals
    df["D"] = d_vals
    return df


def _detect_daily_ma35_signal(df_day: pd.DataFrame) -> pd.Series:
    """逐日重新評估_step2_ma()的reasons文字，看MA35+MA10雙重確認那天有
    沒有同時出現。因為evaluate_timeframe()本身只看傳入df的「最後一列」，
    這裡可以直接對每個截到第i天的子序列呼叫一次，但為了效能改用
    「整欄已經算好指標，逐列檢查用得到的欄位本身」的向量化寫法，不用
    重新呼叫evaluate_timeframe()。
    """
    fast_up = (df_day["close"] >= df_day["MA5"]) & (df_day["MA5"] > df_day["MA5"].shift(3))
    second_gate = df_day["close"] >= df_day["MA10"]
    key_up = (df_day["close"] >= df_day["MA35"]) & (df_day["MA35"] > df_day["MA35"].shift(3))
    return (fast_up & second_gate & key_up).fillna(False)


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
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    recent_mask_day = df_day["date"] >= backtest_start_str

    # --- A) 日35MA支撐+站穩5/10日線 ---
    daily_signal = _detect_daily_ma35_signal(df_day)
    daily_events = []
    for idx in df_day.index[recent_mask_day & daily_signal]:
        d = df_day.loc[idx, "date"]
        fwd = _forward_returns(df_day, d)
        daily_events.append({f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS})

    # --- B) 週KD金叉+站穩週5MA/週35MA(向上) ---
    week_full = _build_weekly(df_day)
    week_full = run_all_indicators(week_full, "週")
    week_full = _calc_kd(week_full)
    week_full["date"] = week_full["date"].astype(str)

    kd_golden_cross = (week_full["K"] > week_full["D"]) & (week_full["K"].shift(1) <= week_full["D"].shift(1))
    above_week5 = week_full["close"] >= week_full["MA5"]
    week35_up = week_full["MA35"] > week_full["MA35"].shift(1)
    above_week35 = week_full["close"] >= week_full["MA35"]
    weekly_signal = (kd_golden_cross & above_week5 & above_week35 & week35_up).fillna(False)

    recent_mask_week = week_full["date"] >= backtest_start_str
    weekly_events = []
    for idx in week_full.index[recent_mask_week & weekly_signal]:
        d = week_full.loc[idx, "date"]
        fwd = _forward_returns(df_day, d)  # 用日線價格算forward return，週訊號對應到的那個交易日
        weekly_events.append({f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS})

    baseline_events = []
    for idx in df_day.index[recent_mask_day]:
        d = df_day.loc[idx, "date"]
        fwd = _forward_returns(df_day, d)
        baseline_events.append({f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS})

    return {"name": name, "code": code, "error": None,
            "daily_events": daily_events, "weekly_events": weekly_events,
            "baseline_events": baseline_events}


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

    all_daily, all_weekly, all_baseline = [], [], []
    for r in results:
        if r.get("error"):
            continue
        all_daily.extend(r["daily_events"])
        all_weekly.extend(r["weekly_events"])
        all_baseline.extend(r["baseline_events"])

    print(f"\n=== 回測{months_arg}個月，13檔加總 ===\n")
    _print_stats("A) 日35MA支撐+站穩5/10日線（現有機制重新驗證）", all_daily)
    print()
    _print_stats("B) 週KD金叉+站穩週5MA/週35MA向上（全新假設）", all_weekly)
    print()
    _print_stats("無條件基準（隨便哪天進場）", all_baseline)
