"""
測試但丁老師的建議：「一條線日線35MA(生死線)就解決問題了」——把賣出減碼
的觸發條件，從現行「生死線下彎／四關價跌破／MACD死亡交叉／OBV量能背離
任一觸發即賣」，改成「只信生死線（MA35）：生死線沒有同時下彎+跌破，
其他三個條件再怎麼吵都不算數」。

跟test_confirmed_sell_rule.py測過的版本不一樣：那支測的是「四關價跌破
需要搭配任一個其他條件才算數」（四個條件裡只拔掉四關價的獨立資格）；
這支測的是更激進的版本，直接把另外三個條件(四關價/MACD死叉/OBV背離)
全部拔掉獨立觸發資格，只剩生死線(MA35)能真正觸發賣出減碼。

比較三組規則：
    現行規則：四個條件任一觸發即賣
    規則A（上次測過）：四關價跌破需搭配其他條件才算
    規則B（這次，但丁老師建議）：只有生死線下彎+跌破才算，其他三個
        條件完全不能單獨觸發賣出減碼

一樣沿用diagnose_sell_lag.py系列的逐日切片重建手法，只讀公開函式
evaluate_timeframe()的reasons文字做關鍵字比對，不碰私有函式。

用法（在專案資料夾內執行）：
    python3 test_key_ma_rule.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from diagnose_sell_lag import _build_weekly, _matched_markers
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import Verdict, classify_final, combine_timeframes, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24
KEY_MA_MARKER = "生死線下彎(均線)"


def _keep_only_key_ma(verdict: Verdict, markers: set) -> Verdict:
    """規則B：賣出減碼只有在markers包含生死線下彎時才放行，否則降級觀望
    （不管是四關價/MACD死叉/OBV背離單獨觸發，還是分數門檻路徑，一律
    不算數）。"""
    if verdict.conclusion == "賣出減碼" and KEY_MA_MARKER not in markers:
        return Verdict(
            conclusion="觀望", confidence=verdict.confidence, score=verdict.score,
            reasons=verdict.reasons,
            caveats=list(verdict.caveats) + ["[規則B模擬] 生死線未下彎跌破，其他訊號不算數，降級觀望"],
        )
    return verdict


def _evaluate_rule_b(df_day_full: pd.DataFrame, as_of_date: str):
    day_slice = df_day_full[df_day_full["date"] <= as_of_date]
    if day_slice.empty:
        return None
    week_slice = _build_weekly(day_slice)
    if week_slice.empty:
        return None
    week_slice = run_all_indicators(week_slice, "週")

    verdict_day = evaluate_timeframe(day_slice, "日")
    verdict_week = evaluate_timeframe(week_slice, "週")
    day_markers = _matched_markers(verdict_day.reasons)
    week_markers = _matched_markers(verdict_week.reasons)

    original_signal = classify_final(
        combine_timeframes({"日": verdict_day, "週": verdict_week})["最終建議"])

    ruleB_day = _keep_only_key_ma(verdict_day, day_markers)
    ruleB_week = _keep_only_key_ma(verdict_week, week_markers)
    ruleB_signal = classify_final(
        combine_timeframes({"日": ruleB_day, "週": ruleB_week})["最終建議"])

    return original_signal, ruleB_signal


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
    as_of_dates = [d for d in df_day["date"] if d >= backtest_start_str]

    rows, original_events, ruleB_events = [], [], []
    prev_original, prev_ruleB = None, None
    for d in as_of_dates:
        result = _evaluate_rule_b(df_day, d)
        if result is None:
            continue
        original_signal, ruleB_signal = result
        close = float(df_day.loc[df_day["date"] == d, "close"].iloc[-1])
        fwd = _forward_returns(df_day, d)
        row_fwd = {f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS}

        if original_signal == "賣出減碼" and prev_original not in (None, "賣出減碼"):
            original_events.append({"name": name, "date": d, "close": close, **row_fwd})
        if ruleB_signal == "賣出減碼" and prev_ruleB not in (None, "賣出減碼"):
            ruleB_events.append({"name": name, "date": d, "close": close, **row_fwd})

        rows.append({"date": d, "close": close,
                      "original_signal": original_signal, "ruleB_signal": ruleB_signal})
        prev_original, prev_ruleB = original_signal, ruleB_signal

    if not rows:
        return {"name": name, "code": code, "error": "回測窗口內沒有有效資料"}

    daily = pd.DataFrame(rows)
    running_peak = daily["close"].cummax()
    drawdown = (daily["close"] - running_peak) / running_peak
    trough_pos = drawdown.idxmin()
    trough_date = daily.loc[trough_pos, "date"]
    peak_price = running_peak.loc[trough_pos]
    peak_pos = daily[(daily["date"] <= trough_date) & (daily["close"] == peak_price)].index[-1]
    peak_date = daily.loc[peak_pos, "date"]

    crash_window = daily[(daily["date"] > peak_date) & (daily["date"] <= trough_date)]
    orig_first = crash_window.loc[crash_window["original_signal"] == "賣出減碼", "date"]
    ruleB_first = crash_window.loc[crash_window["ruleB_signal"] == "賣出減碼", "date"]

    return {
        "name": name, "code": code, "error": None,
        "peak_date": peak_date, "trough_date": trough_date,
        "crash_pct": round((daily.loc[trough_pos, "close"] / peak_price - 1) * 100, 1)
        if peak_price else None,
        "orig_sell_date": orig_first.iloc[0] if not orig_first.empty else None,
        "ruleB_sell_date": ruleB_first.iloc[0] if not ruleB_first.empty else None,
        "original_events": original_events, "ruleB_events": ruleB_events,
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


def _summary_table(events: list) -> dict:
    df = pd.DataFrame(events)
    row = {"事件數": len(df)}
    for n in FORWARD_WINDOWS:
        valid = df[f"fwd_ret_{n}d"].dropna() if not df.empty else pd.Series(dtype=float)
        row[f"{n}日_平均報酬%"] = round(valid.mean() * 100, 2) if len(valid) else None
        row[f"{n}日_續跌比例%"] = round((valid < 0).mean() * 100, 1) if len(valid) else None
    return row


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)

    print(f"=== 窗口內最大崩盤：現行規則 vs 規則B(只信生死線) 的賣出時機比較（回測{months_arg}個月）===")
    all_original_events, all_ruleB_events = [], []
    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")
            continue
        orig = r["orig_sell_date"] or "(未觸發)"
        ruleb = r["ruleB_sell_date"] or "(未觸發)"
        delay = ""
        if r["orig_sell_date"] and r["ruleB_sell_date"]:
            d1 = pd.Timestamp(r["orig_sell_date"])
            d2 = pd.Timestamp(r["ruleB_sell_date"])
            delay = f"（規則B晚了{(d2-d1).days}天）" if d2 > d1 else ("（一樣快）" if d2 == d1 else "（規則B反而更快）")
        elif r["orig_sell_date"] and not r["ruleB_sell_date"]:
            delay = "（規則B這次崩盤完全沒躲！）"
        print(f"{r['name']:8s} 高點{r['peak_date']} 低點{r['trough_date']}({r['crash_pct']}%)　"
              f"現行:{orig}　規則B:{ruleb} {delay}")
        all_original_events.extend(r["original_events"])
        all_ruleB_events.extend(r["ruleB_events"])

    print(f"\n=== 整體賣出事件統計（全部股票、全窗口加總）===")
    print("現行規則：", _summary_table(all_original_events))
    print("規則B(只信生死線)：", _summary_table(all_ruleB_events))
    print("\n讀法：事件數大減代表大幅減少誤殺；續跌比例大升代表賣飛機率大降；")
    print("崩盤時機如果「規則B晚了」甚至「完全沒躲」，代表只信生死線的代價是躲崩盤變慢/變弱，要自己權衡。")
