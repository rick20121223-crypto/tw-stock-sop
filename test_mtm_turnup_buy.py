"""
測試「MTM弧度勾起來」當買進/加碼進場訊號——但丁老師筆記裡「第一線試單」
「順勢黃金加碼點」兩個最常用的買點，核心都是這個，但現在sop_decision.py
的_step5_mtm()只拿MTM用在賣出示警（60分線出場警示），從來沒被當成
買進訊號驗證過。

偵測定義（「低檔...弧度勾起來」白話翻譯成可計算條件）：
    MTM由跌轉漲（MTM[t] > MTM[t-1] 且 MTM[t-1] <= MTM[t-2]，局部轉折點）
    且 MTM[t] < 0（還在負值/低檔區，不是漲多之後的高檔鈍化訊號）

MTM本身(stock_core.calc_mtm)只看close.shift(period)跟自己的rolling
mean，不會用到未來資料，所以可以直接在完整歷史上一次算好整段序列，
不用像sop_decision.py那樣逐日切片重建，速度快很多。

兩種驗證方式：
1. 統計驗證：對13檔24個月窗口裡所有「MTM勾起」事件，看之後5/10/20個
   交易日的報酬/正報酬比例，跟無條件基準比較，看這個訊號本身夠不夠格
   當進場依據。
2. 崩盤低點驗證：在兩段最大崩盤的低點附近，「MTM勾起」有沒有準時亮起
   （跟之前測半年線第二隻腳/左側平台支撐用同一個方法，方便直接比較）。

用法（在專案資料夾內執行）：
    python3 test_mtm_turnup_buy.py [FinMind_Token] [回測月數=24]
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


def _detect_mtm_turnup(df: pd.DataFrame) -> pd.Series:
    """回傳布林Series：MTM弧度勾起來（由跌轉漲、且仍在MTM<0的低檔區）。"""
    mtm = df["MTM"]
    rising_now = mtm > mtm.shift(1)
    was_falling = mtm.shift(1) <= mtm.shift(2)
    in_low_zone = mtm < 0
    return (rising_now & was_falling & in_low_zone).fillna(False)


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
    if "MTM" not in df_day.columns:
        return {"name": name, "code": code, "error": "沒有算出MTM欄位"}
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    turnup = _detect_mtm_turnup(df_day)

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    recent_mask = df_day["date"] >= backtest_start_str

    events = []
    for idx in df_day.index[recent_mask & turnup]:
        d = df_day.loc[idx, "date"]
        close = float(df_day.loc[idx, "close"])
        fwd = _forward_returns(df_day, d)
        events.append({"name": name, "code": code, "date": d, "close": close,
                        **{f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS}})

    # 無條件基準：同一個窗口、同一批股票，每一天都算一次forward return，
    # 不管那天是不是MTM勾起，拿來跟MTM勾起的事件比較才是公平的對照組
    # （用同一個24個月窗口重算，不是沿用之前6個月窗口算出來的舊數字）。
    baseline_events = []
    for idx in df_day.index[recent_mask]:
        d = df_day.loc[idx, "date"]
        fwd = _forward_returns(df_day, d)
        baseline_events.append({f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS})

    # 崩盤低點驗證：找回測窗口內最大的一次崩盤(高點->低點)，看MTM勾起在
    # 低點前後20個交易日內有沒有亮起、亮起時機跟低點差多少天/差多少價。
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

    print(f"=== 崩盤低點附近，MTM弧度勾起有沒有準時亮起（回測{months_arg}個月）===")
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
                  f"| MTM勾起全程沒有亮起")
            miss += 1
        else:
            lag = r["lag_trading_days"]
            lag_desc = f"領先{lag}個交易日" if lag > 0 else (f"晚了{-lag}個交易日" if lag < 0 else "剛好當天")
            pct = r["price_vs_trough_pct"]
            pct_desc = f"比低點價{pct:+.1f}%, " if pct is not None else ""
            print(f"{r['name']:8s} 高點{r['peak_date']} 低點{r['trough_date']}({r['crash_pct']}%) "
                  f"| 首次亮起:{r['first_flag_date']}({pct_desc}{lag_desc})")
            hit += 1
    print(f"\n統計：13檔中有{hit}檔在低點附近亮起過MTM勾起，{miss}檔全程沒亮")

    print(f"\n=== 全部MTM勾起事件的事後報酬統計（共{len(all_events)}筆，全部13檔全窗口）===")
    df = pd.DataFrame(all_events)
    for n in FORWARD_WINDOWS:
        valid = df[f"fwd_ret_{n}d"].dropna() if not df.empty else pd.Series(dtype=float)
        if len(valid):
            print(f"{n}日：平均報酬 {valid.mean()*100:.2f}%　正報酬比例 {(valid>0).mean()*100:.1f}%　(n={len(valid)})")

    print(f"\n=== 無條件基準（同一個{months_arg}個月窗口，不管是不是MTM勾起，隨便哪天進場）===")
    base_df = pd.DataFrame(all_baseline_events)
    for n in FORWARD_WINDOWS:
        valid = base_df[f"fwd_ret_{n}d"].dropna() if not base_df.empty else pd.Series(dtype=float)
        if len(valid):
            print(f"{n}日：平均報酬 {valid.mean()*100:.2f}%　正報酬比例 {(valid>0).mean()*100:.1f}%　(n={len(valid)})")
