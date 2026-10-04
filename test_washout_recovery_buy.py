"""
測試「洗盤結束/夾起單」買點：假跌破關鍵均線(日35MA/生死線)後，迅速拉回
站上，判定為主力洗盤結束、相對安全買點。筆記裡這個條件還可以加上
「MTM勾起」一起確認，這裡測兩個版本方便比較：單獨用夾起單 vs 夾起單+
MTM勾起。

偵測定義：
    夾起單：過去lookback天內，至少一天收盤跌破MA35，但今天收盤重新站回
            MA35之上（短暫假跌破後迅速拉回，不是真的跌破）
    夾起單+MTM：上面的條件成立，且當天MTM由跌轉漲（跟test_mtm_turnup_
                buy.py同一個「弧度勾起」定義）

兩個指標都只看過去資料，可以在完整歷史上一次算好整段序列。

用法（在專案資料夾內執行）：
    python3 test_washout_recovery_buy.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24
LOOKBACK_BARS = 3  # 「過去幾天內」曾經跌破才算「剛剛假跌破」，太久以前的跌破不算


def _detect_washout_recovery(df: pd.DataFrame, lookback: int = LOOKBACK_BARS) -> pd.Series:
    was_below_recently = (df["close"] < df["MA35"]).rolling(window=lookback).sum().shift(1) >= 1
    now_above = df["close"] >= df["MA35"]
    return (was_below_recently & now_above).fillna(False)


def _detect_mtm_turnup(df: pd.DataFrame) -> pd.Series:
    mtm = df["MTM"]
    rising_now = mtm > mtm.shift(1)
    was_falling = mtm.shift(1) <= mtm.shift(2)
    return (rising_now & was_falling & (mtm < 0)).fillna(False)


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
    mtm_turnup = _detect_mtm_turnup(df_day)
    washout_plus_mtm = washout & mtm_turnup

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
        "washout_events": _collect(washout),
        "combo_events": _collect(washout_plus_mtm),
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

    all_washout, all_combo, all_baseline = [], [], []
    for r in results:
        if r.get("error"):
            continue
        all_washout.extend(r["washout_events"])
        all_combo.extend(r["combo_events"])
        all_baseline.extend(r["baseline_events"])

    print(f"\n=== 回測{months_arg}個月，13檔加總 ===\n")
    _print_stats("夾起單單獨使用（假跌破MA35後迅速拉回站上）", all_washout)
    print()
    _print_stats("夾起單+MTM勾起（兩個同時成立）", all_combo)
    print()
    _print_stats("無條件基準（隨便哪天進場）", all_baseline)
