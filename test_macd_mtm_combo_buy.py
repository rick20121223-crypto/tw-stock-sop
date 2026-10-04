"""
測試「順勢黃金加碼點」完整版：MACD零軸之上（長多結構未變）＋ MTM弧度
勾起來，兩個條件同時成立才算——筆記裡寫這是「系統中勝率最高的順勢
加碼點」。

上一輪只測了MTM勾起單獨使用，結果輸給無條件基準（見
test_mtm_turnup_buy.py），原因很可能是MTM自己上下擺動太頻繁，大多是
雜訊。但丁老師筆記裡從來沒說MTM可以單獨用，一定要搭配「MACD在零軸
之上」這個條件過濾掉雜訊，才是完整版本。這支腳本補上這個過濾條件，
跟上一輪的MTM-only結果一起比較，才能公平驗證筆記裡的完整邏輯。

偵測定義：
    MTM由跌轉漲（跟test_mtm_turnup_buy.py同一個定義，局部轉折+MTM<0）
    且 DIF（MACD快慢線）> 0（零軸之上，代表長多結構未變，只是短中線
    拉回修正，不是空頭趨勢中的反彈）

MACD(DIF)跟MTM都只看過去資料，不會用到未來，可以在完整歷史上一次
算好整段序列，不用逐日切片重建。

用法（在專案資料夾內執行）：
    python3 test_macd_mtm_combo_buy.py [FinMind_Token] [回測月數=24]
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


def _detect_combo_signal(df: pd.DataFrame) -> pd.Series:
    """回傳布林Series：MTM弧度勾起(低檔) 且 MACD(DIF)在零軸之上。"""
    mtm = df["MTM"]
    rising_now = mtm > mtm.shift(1)
    was_falling = mtm.shift(1) <= mtm.shift(2)
    mtm_turnup = rising_now & was_falling & (mtm < 0)
    macd_above_zero = df["DIF"] > 0
    return (mtm_turnup & macd_above_zero).fillna(False)


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
    if "MTM" not in df_day.columns or "DIF" not in df_day.columns:
        return {"name": name, "code": code, "error": "沒有算出MTM或DIF欄位"}
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    combo = _detect_combo_signal(df_day)

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    recent_mask = df_day["date"] >= backtest_start_str

    events = []
    for idx in df_day.index[recent_mask & combo]:
        d = df_day.loc[idx, "date"]
        close = float(df_day.loc[idx, "close"])
        fwd = _forward_returns(df_day, d)
        events.append({"name": name, "code": code, "date": d, "close": close,
                        **{f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS}})

    recent = df_day[recent_mask].reset_index(drop=True)
    crash_info = {"peak_date": None, "trough_date": None, "crash_pct": None,
                  "first_flag_date": None, "lag_trading_days": None, "price_vs_trough_pct": None}
    if not recent.empty:
        running_peak = recent["close"].cummax()
        drawdown = (recent["close"] - running_peak) / running_peak
        trough_pos = drawdown.idxmin()
        trough_date = recent.loc[trough_pos, "date"]
        trough_price = float(recent.loc[trough_pos, "close"])
        peak_price = running_peak.loc[trough_pos]
        peak_pos = recent[(recent["date"] <= trough_date) & (recent["close"] == peak_price)].index[-1]
        peak_date = recent.loc[peak_pos, "date"]
        crash_info["peak_date"] = peak_date
        crash_info["trough_date"] = trough_date
        crash_info["crash_pct"] = round((trough_price / peak_price - 1) * 100, 1) if peak_price else None

        all_dates = recent["date"].tolist()
        trough_idx = all_dates.index(trough_date)
        check_dates = set(all_dates[max(0, trough_idx - 20):trough_idx + 15])
        candidates = [e for e in events if e["date"] in check_dates]
        if candidates:
            candidates.sort(key=lambda e: e["date"])
            first = candidates[0]
            crash_info["first_flag_date"] = first["date"]
            crash_info["lag_trading_days"] = all_dates.index(trough_date) - all_dates.index(first["date"])
            crash_info["price_vs_trough_pct"] = (
                round((first["close"] / trough_price - 1) * 100, 1) if trough_price else None
            )

    baseline_events = []
    for idx in df_day.index[recent_mask]:
        d = df_day.loc[idx, "date"]
        fwd = _forward_returns(df_day, d)
        baseline_events.append({f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS})

    return {"name": name, "code": code, "error": None, "events": events,
            "baseline_events": baseline_events, **crash_info}


def run(api_token: str, backtest_months: int = BACKTEST_MONTHS_DEFAULT) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(analyze_one, name, code, market, api_token, backtest_months)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)

    print(f"=== 崩盤低點附近，MACD零軸之上+MTM勾起有沒有準時亮起（回測{months_arg}個月）===")
    all_events, all_baseline_events = [], []
    hit, miss = 0, 0
    for r in sorted(results, key=lambda x: x.get("name", "")):
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")
            continue
        all_events.extend(r["events"])
        all_baseline_events.extend(r["baseline_events"])
        if r["first_flag_date"] is None:
            print(f"{r['name']:8s} 高點{r['peak_date']} 低點{r['trough_date']}({r['crash_pct']}%) "
                  f"| 完全沒有亮起")
            miss += 1
        else:
            lag = r["lag_trading_days"]
            lag_desc = f"領先{lag}個交易日" if lag > 0 else (f"晚了{-lag}個交易日" if lag < 0 else "剛好當天")
            pct = r["price_vs_trough_pct"]
            pct_desc = f"比低點價{pct:+.1f}%, " if pct is not None else ""
            print(f"{r['name']:8s} 高點{r['peak_date']} 低點{r['trough_date']}({r['crash_pct']}%) "
                  f"| 首次亮起:{r['first_flag_date']}({pct_desc}{lag_desc})")
            hit += 1
    print(f"\n統計：13檔中有{hit}檔在低點附近亮起過，{miss}檔全程沒亮")

    print(f"\n=== 全部事件的事後報酬統計（共{len(all_events)}筆，全部13檔全窗口）===")
    df = pd.DataFrame(all_events)
    for n in FORWARD_WINDOWS:
        valid = df[f"fwd_ret_{n}d"].dropna() if not df.empty else pd.Series(dtype=float)
        if len(valid):
            print(f"{n}日：平均報酬 {valid.mean()*100:.2f}%　正報酬比例 {(valid>0).mean()*100:.1f}%　(n={len(valid)})")

    print(f"\n=== 無條件基準（同一個{months_arg}個月窗口，隨便哪天進場）===")
    base_df = pd.DataFrame(all_baseline_events)
    for n in FORWARD_WINDOWS:
        valid = base_df[f"fwd_ret_{n}d"].dropna() if not base_df.empty else pd.Series(dtype=float)
        if len(valid):
            print(f"{n}日：平均報酬 {valid.mean()*100:.2f}%　正報酬比例 {(valid>0).mean()*100:.1f}%　(n={len(valid)})")

    print("\n對照：上一輪測過的「MTM勾起單獨使用」是 5日+1.10%/48.8%　10日+1.69%/49.4%　20日+4.29%/50.2%（689筆）")
