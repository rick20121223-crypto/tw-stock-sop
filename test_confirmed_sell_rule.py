"""
測試「四關價跌破需要搭配第二個確認條件才真的賣出」這個修改規則，跟現行
規則（四個硬賣條件：生死線下彎／四關價跌破／MACD死亡交叉／OBV量能背離，
任一觸發即賣）比較：
1. 賣出訊號次數有沒有減少（減少代表少掉很多「四關價單獨觸發」的誤殺）
2. 剩下的賣出訊號，賣飛機率（事後續跌比例）有沒有改善
3. 窗口內最大的一次崩盤，修改後的規則還能不能及時躲過、慢了幾天

做法：沿用diagnose_sell_lag.py/diagnose_sell_whipsaw.py已經驗證過的逐日
重建判讀邏輯，額外做一個「修改版」：如果某個timeframe(日/週)的結論是
賣出減碼、且唯一比對到的關鍵字只有「四關價跌破」（沒有生死線/MACD死叉/
OBV背離任何一個同時出現），就把那個timeframe的Verdict結論downgrade回
觀望，再丟回正式的combine_timeframes()/classify_final()重新算一次最終
結論——其餘邏輯完全沿用正式SOP，不是另外發明一套獨立判斷，確保長線續抱
短線出場、強制降級規則等其他決策路徑行為跟正式系統一致。

⚠️ 這是文字關鍵字比對的簡化版模擬，不是真的重寫evaluate_timeframe()內部
的hard_sell布林邏輯——如果「四關價跌破」單獨存在，但其他因素剛好讓綜合
分數本來就會跌破賣出門檻(score<=門檻那條後備路徑)，這個模擬會跟真正改寫
內部邏輯的結果有微小落差。足夠拿來篩選「這個方向值不值得正式投入」，
但正式要改sop_decision.py/sop_rules.yaml之前，應該用這裡篩選出來、
看起來有效的方向，再去正式改規則後完整重新驗證。

用法（在專案資料夾內執行）：
    python3 test_confirmed_sell_rule.py [FinMind_Token] [回測月數=24]
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


def _downgrade_lone_four_key(verdict: Verdict, markers: set) -> Verdict:
    """修改規則的核心：單獨「四關價跌破」(沒有搭配其他三個硬賣條件任一個)
    不算數，降級回觀望；其他情況（沒賣出減碼、或有其他條件一起觸發）原樣
    放行。"""
    if verdict.conclusion == "賣出減碼" and markers == {"四關價跌破"}:
        return Verdict(
            conclusion="觀望", confidence=verdict.confidence, score=verdict.score,
            reasons=verdict.reasons,
            caveats=list(verdict.caveats) + ["[模擬規則] 四關價跌破缺乏第二條件確認，降級觀望"],
        )
    return verdict


def _evaluate_both(df_day_full: pd.DataFrame, as_of_date: str):
    """回傳當天「現行規則」跟「修改規則」各自算出的最終長期訊號。"""
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

    modified_day = _downgrade_lone_four_key(verdict_day, day_markers)
    modified_week = _downgrade_lone_four_key(verdict_week, week_markers)
    modified_signal = classify_final(
        combine_timeframes({"日": modified_day, "週": modified_week})["最終建議"])

    return original_signal, modified_signal


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

    rows, original_events, modified_events = [], [], []
    prev_original, prev_modified = None, None
    for d in as_of_dates:
        result = _evaluate_both(df_day, d)
        if result is None:
            continue
        original_signal, modified_signal = result
        close = float(df_day.loc[df_day["date"] == d, "close"].iloc[-1])
        fwd = _forward_returns(df_day, d)
        row_fwd = {f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS}

        if original_signal == "賣出減碼" and prev_original not in (None, "賣出減碼"):
            original_events.append({"name": name, "date": d, "close": close, **row_fwd})
        if modified_signal == "賣出減碼" and prev_modified not in (None, "賣出減碼"):
            modified_events.append({"name": name, "date": d, "close": close, **row_fwd})

        rows.append({"date": d, "close": close,
                      "original_signal": original_signal, "modified_signal": modified_signal})
        prev_original, prev_modified = original_signal, modified_signal

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
    mod_first = crash_window.loc[crash_window["modified_signal"] == "賣出減碼", "date"]

    return {
        "name": name, "code": code, "error": None,
        "peak_date": peak_date, "trough_date": trough_date,
        "crash_pct": round((daily.loc[trough_pos, "close"] / peak_price - 1) * 100, 1),
        "orig_sell_date": orig_first.iloc[0] if not orig_first.empty else None,
        "mod_sell_date": mod_first.iloc[0] if not mod_first.empty else None,
        "original_events": original_events, "modified_events": modified_events,
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


def _whipsaw_table(events: list) -> dict:
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

    print(f"=== 窗口內最大崩盤：現行規則 vs 修改規則 的賣出時機比較（回測{months_arg}個月）===")
    all_original_events, all_modified_events = [], []
    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")
            continue
        orig = r["orig_sell_date"] or "(未觸發)"
        mod = r["mod_sell_date"] or "(未觸發)"
        delay = ""
        if r["orig_sell_date"] and r["mod_sell_date"]:
            d1 = pd.Timestamp(r["orig_sell_date"])
            d2 = pd.Timestamp(r["mod_sell_date"])
            delay = f"（修改規則晚了{(d2-d1).days}天）" if d2 > d1 else ("（一樣快）" if d2 == d1 else "（修改規則反而更快）")
        print(f"{r['name']:8s} 高點{r['peak_date']} 低點{r['trough_date']}({r['crash_pct']}%)　"
              f"現行:{orig}　修改:{mod} {delay}")
        all_original_events.extend(r["original_events"])
        all_modified_events.extend(r["modified_events"])

    print(f"\n=== 整體賣出事件統計（全部股票、全窗口加總）===")
    print("現行規則：", _whipsaw_table(all_original_events))
    print("修改規則：", _whipsaw_table(all_modified_events))
    print("\n讀法：事件數減少代表少掉誤殺；續跌比例上升代表賣飛機率下降；")
    print("崩盤時機如果「修改規則晚了」，代表少誤殺是用「躲崩盤變慢」換來的，要自己權衡。")
