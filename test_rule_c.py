"""
測試折中規則C：「生死線(MA35)下彎+跌破 OR 另外三個條件(四關價/MACD死叉/
OBV背離)裡至少2個同時成立，才算真的賣出減碼」。

動機：規則B(只信生死線)賣飛大幅減少，但反應明顯變慢（平均晚約1個月），
環球晶那種急跌甚至完全沒觸發過——因為急跌時生死線(均線)本來就來不及
轉彎。規則C想同時接住兩種情境：慢但準的生死線轉弱、跟快但通常是真的
（多個獨立條件同時爆，不是單一條件自己吵）的急跌。

跟現行規則(任一條件觸發即賣)、規則B(只信生死線)一起三組比較：
    現行規則：四個條件任一觸發即賣
    規則B：只有生死線下彎+跌破才算
    規則C（這次）：生死線下彎+跌破，或者其他三個條件裡至少2個同時成立

用法（在專案資料夾內執行）：
    python3 test_rule_c.py [FinMind_Token] [回測月數=24]
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
FAST_MARKERS = {"四關價跌破", "MACD死亡交叉", "OBV量能背離"}


def _apply_rule_c(verdict: Verdict, markers: set) -> Verdict:
    if verdict.conclusion != "賣出減碼":
        return verdict
    confirmed = (KEY_MA_MARKER in markers) or (len(markers & FAST_MARKERS) >= 2)
    if confirmed:
        return verdict
    return Verdict(
        conclusion="觀望", confidence=verdict.confidence, score=verdict.score,
        reasons=verdict.reasons,
        caveats=list(verdict.caveats) + ["[規則C模擬] 生死線未破、快速條件也不足2個，降級觀望"],
    )


def _evaluate_rule_c(df_day_full: pd.DataFrame, as_of_date: str):
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

    ruleC_day = _apply_rule_c(verdict_day, day_markers)
    ruleC_week = _apply_rule_c(verdict_week, week_markers)
    ruleC_signal = classify_final(
        combine_timeframes({"日": ruleC_day, "週": ruleC_week})["最終建議"])

    return original_signal, ruleC_signal


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

    rows, original_events, ruleC_events = [], [], []
    prev_original, prev_ruleC = None, None
    for d in as_of_dates:
        result = _evaluate_rule_c(df_day, d)
        if result is None:
            continue
        original_signal, ruleC_signal = result
        close = float(df_day.loc[df_day["date"] == d, "close"].iloc[-1])
        fwd = _forward_returns(df_day, d)
        row_fwd = {f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS}

        if original_signal == "賣出減碼" and prev_original not in (None, "賣出減碼"):
            original_events.append({"name": name, "date": d, "close": close, **row_fwd})
        if ruleC_signal == "賣出減碼" and prev_ruleC not in (None, "賣出減碼"):
            ruleC_events.append({"name": name, "date": d, "close": close, **row_fwd})

        rows.append({"date": d, "close": close,
                      "original_signal": original_signal, "ruleC_signal": ruleC_signal})
        prev_original, prev_ruleC = original_signal, ruleC_signal

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
    ruleC_first = crash_window.loc[crash_window["ruleC_signal"] == "賣出減碼", "date"]

    return {
        "name": name, "code": code, "error": None,
        "peak_date": peak_date, "trough_date": trough_date,
        "crash_pct": round((daily.loc[trough_pos, "close"] / peak_price - 1) * 100, 1)
        if peak_price else None,
        "orig_sell_date": orig_first.iloc[0] if not orig_first.empty else None,
        "ruleC_sell_date": ruleC_first.iloc[0] if not ruleC_first.empty else None,
        "original_events": original_events, "ruleC_events": ruleC_events,
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

    print(f"=== 窗口內最大崩盤：現行規則 vs 規則C(折中) 的賣出時機比較（回測{months_arg}個月）===")
    all_original_events, all_ruleC_events = [], []
    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")
            continue
        orig = r["orig_sell_date"] or "(未觸發)"
        rulec = r["ruleC_sell_date"] or "(未觸發)"
        delay = ""
        if r["orig_sell_date"] and r["ruleC_sell_date"]:
            d1 = pd.Timestamp(r["orig_sell_date"])
            d2 = pd.Timestamp(r["ruleC_sell_date"])
            delay = f"（規則C晚了{(d2-d1).days}天）" if d2 > d1 else ("（一樣快）" if d2 == d1 else "（規則C反而更快）")
        elif r["orig_sell_date"] and not r["ruleC_sell_date"]:
            delay = "（規則C這次崩盤完全沒躲！）"
        print(f"{r['name']:8s} 高點{r['peak_date']} 低點{r['trough_date']}({r['crash_pct']}%)　"
              f"現行:{orig}　規則C:{rulec} {delay}")
        all_original_events.extend(r["original_events"])
        all_ruleC_events.extend(r["ruleC_events"])

    print(f"\n=== 整體賣出事件統計（全部股票、全窗口加總）===")
    print("現行規則：", _summary_table(all_original_events))
    print("規則C(折中)：", _summary_table(all_ruleC_events))
