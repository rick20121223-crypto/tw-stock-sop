"""
v1（test_weekly_ma35_washout_recovery_buy.py）回測結果：單獨「跌破週35MA
後站回」、或加週MTM勾起，都沒有打贏無條件基準，樣本數變大後優勢直接消失。
但丁老師原話是「配合技術面操作」才是加碼點，不是單一條件就成立——這支
在base訊號(週夾起單)上疊加幾種「技術面轉強」確認，看哪個組合站得住：

    A) base + 日線站回5MA、10MA（使用者自己對SLS講的轉強條件之一）
    B) base + 日MACD轉強（DIF由跌轉漲的弧度勾起，不要求零軸之上，因為
       像SLS這種案例日MACD本來就在零軸下，不能用「零軸之上」卡門檻）
    C) base + A + B 同時成立（日線面全套確認）
    D) base + 週MACD轉強（DIF由跌轉漲，更長週期的動能確認）

日線的MA5/MA10/DIF取「這週最後一個交易日」的值（用merge_asof往回找
週線標記日期(週日)之前最近一個有交易的日線日期），週線訊號本身的日期
也用同一天當作進場基準，跟v1一致。

用法（在專案資料夾內執行）：
    python3 test_weekly_ma35_washout_recovery_buy_v2.py [FinMind_Token] [回測月數=48]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from diagnose_sell_lag import _build_weekly
from historical_backtest import WARMUP_CALENDAR_DAYS
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 48
LOOKBACK_WEEKS = 3


def _detect_weekly_washout_recovery(week_df: pd.DataFrame, lookback: int = LOOKBACK_WEEKS) -> pd.Series:
    was_below_recently = (week_df["close"] < week_df["MA35"]).rolling(window=lookback).sum().shift(1) >= 1
    now_above = week_df["close"] >= week_df["MA35"]
    return (was_below_recently & now_above).fillna(False)


def _detect_macd_turnup(df: pd.DataFrame) -> pd.Series:
    dif = df["DIF"]
    rising_now = dif > dif.shift(1)
    was_falling = dif.shift(1) <= dif.shift(2)
    return (rising_now & was_falling).fillna(False)


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

    # 日線面的轉強條件，先在日線序列上算好（向量化，避免逐筆查表）
    df_day["_reclaim_5_10"] = (df_day["close"] >= df_day["MA5"]) & (df_day["close"] >= df_day["MA10"])
    df_day["_macd_turnup"] = _detect_macd_turnup(df_day)

    week_full = _build_weekly(df_day)
    week_full = run_all_indicators(week_full, "週")
    if "MA35" not in week_full.columns or "DIF" not in week_full.columns:
        return {"name": name, "code": code, "error": "缺少週MA35或週DIF欄位"}
    week_full["date"] = week_full["date"].astype(str)

    base = _detect_weekly_washout_recovery(week_full)
    week_macd_turnup = _detect_macd_turnup(week_full)

    # 把「這週最後一個交易日」的日線狀態，對齊回週線那一列
    day_for_asof = df_day[["date", "_reclaim_5_10", "_macd_turnup"]].copy()
    day_for_asof["date"] = pd.to_datetime(day_for_asof["date"])
    week_for_asof = week_full[["date"]].copy()
    week_for_asof["date"] = pd.to_datetime(week_for_asof["date"])
    aligned = pd.merge_asof(
        week_for_asof.sort_values("date"), day_for_asof.sort_values("date"),
        on="date", direction="backward",
    )
    daily_reclaim = aligned["_reclaim_5_10"].fillna(False).reset_index(drop=True)
    daily_macd_turnup = aligned["_macd_turnup"].fillna(False).reset_index(drop=True)

    signals = {
        "A_base+日站回5_10MA": base & daily_reclaim,
        "B_base+日MACD轉強": base & daily_macd_turnup,
        "C_base+日站回+日MACD轉強": base & daily_reclaim & daily_macd_turnup,
        "D_base+週MACD轉強": base & week_macd_turnup,
    }

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    recent_mask_week = week_full["date"] >= backtest_start_str
    recent_mask_day = df_day["date"] >= backtest_start_str

    def _collect(mask):
        out = []
        for idx in week_full.index[recent_mask_week & mask]:
            d = week_full.loc[idx, "date"]
            fwd = _forward_returns(df_day, d)
            out.append({f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS})
        return out

    baseline_events = []
    for idx in df_day.index[recent_mask_day]:
        d = df_day.loc[idx, "date"]
        fwd = _forward_returns(df_day, d)
        baseline_events.append({f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS})

    result = {"name": name, "code": code, "error": None, "baseline_events": baseline_events}
    for label, mask in signals.items():
        result[label] = _collect(mask)
    return result


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

    labels = ["A_base+日站回5_10MA", "B_base+日MACD轉強", "C_base+日站回+日MACD轉強", "D_base+週MACD轉強"]
    all_events = {label: [] for label in labels}
    all_baseline = []
    for r in results:
        if r.get("error"):
            continue
        all_baseline.extend(r["baseline_events"])
        for label in labels:
            all_events[label].extend(r[label])

    print(f"\n=== 回測{months_arg}個月，13檔加總（在週夾起單base上疊加確認條件）===\n")
    for label in labels:
        _print_stats(label, all_events[label])
        print()
    _print_stats("無條件基準（隨便哪天進場）", all_baseline)
