"""
檢查「左側平台支撐」（sop_decision.py Step 6）這個真正會影響買進分數的抄底
機制，在核心持股近24個月窗口內最大的一次崩盤低點附近，有沒有準確地
亮起（觸發support_bonus、在判讀理由裡留下「相對安全買點」字樣）。

跟 diagnose_sell_lag.py 系列用同一個「日線指標算一次、逐日切片模擬」
手法，只讀 evaluate_timeframe() 回傳的 reasons 文字做關鍵字比對（找
「相對安全買點」這個只有Step 6真的加分時才會寫入reasons的字串），不碰
私有函式、不複製一份判斷邏輯。

用法（在專案資料夾內執行）：
    python3 diagnose_platform_buy.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24
PLATFORM_BUY_MARKER = "相對安全買點"


def diagnose_one(name: str, code: str, market: str, api_token: str, backtest_months: int) -> dict:
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
    recent = df_day[df_day["date"] >= backtest_start_str].reset_index(drop=True)
    if recent.empty:
        return {"name": name, "code": code, "error": "回測窗口內沒有資料"}

    running_peak = recent["close"].cummax()
    drawdown = (recent["close"] - running_peak) / running_peak
    trough_pos = drawdown.idxmin()
    trough_date = recent.loc[trough_pos, "date"]
    trough_price = float(recent.loc[trough_pos, "close"])
    peak_price = running_peak.loc[trough_pos]
    peak_pos = recent[(recent["date"] <= trough_date) & (recent["close"] == peak_price)].index[-1]
    peak_date = recent.loc[peak_pos, "date"]
    crash_pct = round((trough_price / peak_price - 1) * 100, 1)

    # 檢查低點前後20個交易日內，「相對安全買點」有沒有亮起過，第一次亮起是哪天
    all_dates = recent["date"].tolist()
    trough_idx = all_dates.index(trough_date)
    check_dates = all_dates[max(0, trough_idx - 20):trough_idx + 15]

    first_flag_date, first_flag_price = None, None
    for d in check_dates:
        day_slice = df_day[df_day["date"] <= d]
        verdict = evaluate_timeframe(day_slice, "日")
        if any(PLATFORM_BUY_MARKER in r for r in verdict.reasons):
            first_flag_date = d
            first_flag_price = float(df_day.loc[df_day["date"] == d, "close"].iloc[-1])
            break

    result = {
        "name": name, "code": code, "error": None,
        "peak_date": peak_date, "trough_date": trough_date, "trough_price": trough_price,
        "crash_pct": crash_pct,
        "first_flag_date": first_flag_date, "first_flag_price": first_flag_price,
    }
    if first_flag_date is not None:
        result["lag_trading_days"] = check_dates.index(trough_date) - check_dates.index(first_flag_date)
        # trough_price有時會是0（除權息/股票分割等事件造成的資料斷點，不是
        # 真的跌到0元），這種情況跳過百分比計算避免除以零，不影響其他欄位。
        result["price_vs_trough_pct"] = (
            round((first_flag_price / trough_price - 1) * 100, 1) if trough_price else None
        )
    return result


def run(api_token: str, backtest_months: int = BACKTEST_MONTHS_DEFAULT) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(diagnose_one, name, code, market, api_token, backtest_months)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    hit, miss = 0, 0
    for r in sorted(results, key=lambda x: x.get("name", "")):
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")
            continue
        if r["first_flag_date"] is None:
            print(f"{r['name']:8s} 高點{r['peak_date']} 低點{r['trough_date']}({r['crash_pct']}%) "
                  f"| 左側平台支撐全程沒有亮起")
            miss += 1
        else:
            lag = r["lag_trading_days"]
            lag_desc = f"領先{lag}個交易日" if lag > 0 else (f"晚了{-lag}個交易日" if lag < 0 else "剛好當天")
            pct = r["price_vs_trough_pct"]
            pct_desc = f"比低點價{pct:+.1f}%, " if pct is not None else ""
            print(f"{r['name']:8s} 高點{r['peak_date']} 低點{r['trough_date']}({r['crash_pct']}%) "
                  f"| 首次亮起:{r['first_flag_date']}(價{r['first_flag_price']:.2f}, "
                  f"{pct_desc}{lag_desc})")
            hit += 1

    print(f"\n=== 統計：13檔中有{hit}檔在低點附近亮起過「左側平台支撐」買點，{miss}檔全程沒亮 ===")
