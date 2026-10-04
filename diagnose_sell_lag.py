"""
診斷「賣出減碼」訊號為什麼慢：對核心持股近6個月窗口裡的崩盤區間（高點→
低點），逐日比對日線/週線各自的判讀理由(reasons)，找出「訊號翻轉成賣出
減碼」那一天，新增/首次出現了哪個SOP分層條件，藉此判斷卡住賣出訊號的
瓶頸主要是哪一層：生死線(均線)下彎、四關價跌破、MACD死亡交叉、還是OBV
量能背離。

純粹讀 evaluate_timeframe()/combine_timeframes()/classify_final() 這些
公開函式回傳的判讀理由文字(reasons)做關鍵字比對，不碰sop_decision.py
內部的私有函式、不複製一份判斷邏輯，只是離線分析用途，不影響正式判讀。

跟historical_backtest.py一樣，日線指標只算一次、用切片模擬逐日判讀；
週線則每天重新resample+算指標（理由同historical_backtest.py docstring）。

用法：
    python3 diagnose_sell_lag.py [FinMind_Token] [回測月數=6]
"""
import os
import sys
from datetime import date, timedelta

import pandas as pd

from historical_backtest import BACKTEST_MONTHS_DEFAULT, WARMUP_CALENDAR_DAYS, _build_weekly
from sop_decision import classify_final, combine_timeframes, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

# 判讀理由裡的關鍵字片段 -> 人看得懂的分層名稱。順序無所謂，比對用「文字
# 是否包含這個片段」，不是精確比對整句（sop_decision.py的文字會帶變數，
# 例如均線欄位名稱），所以這裡只取不會變動的核心片段。
# ⚠️ 「生死線下彎(均線)」原本比對"生死線"這個字串，但sop_decision.py裡
# 「生死線上揚且站上，中長線結構偏多」(多方情境)跟「生死線已下彎，任何
# 買訊都要降級」(空方情境，slope下彎但價格還沒跌破、尚未構成硬賣否決)
# 都含有"生死線"兩個字，會把多方、以及「下彎但未跌破」的情況也誤判成
# 真正觸發賣出的key_ma_down_veto。"且股價已跌破"這句話只有在
# key_ma_down_veto真的變成True（slope下彎 且 價格也跌破）時才會被加進
# reasons，是唯一不會跟其他情境混淆的片段，改用這個才對。
MARKERS = {
    "生死線下彎(均線)": "且股價已跌破",
    "四關價跌破": "四關價：今開跌破昨低",
    "MACD死亡交叉": "MACD 出現死亡交叉",
    "OBV量能背離": "量價背離",
}


def _matched_markers(reasons: list) -> set:
    text = " ".join(reasons)
    return {label for label, pattern in MARKERS.items() if pattern in text}


def _evaluate_detail(df_day_full: pd.DataFrame, as_of_date: str):
    """回傳 (day_verdict, week_verdict, 長期合併後的分類桶)，資料不足回傳
    (None, None, None)。"""
    day_slice = df_day_full[df_day_full["date"] <= as_of_date]
    if day_slice.empty:
        return None, None, None
    week_slice = _build_weekly(day_slice)
    if week_slice.empty:
        return None, None, None
    week_slice = run_all_indicators(week_slice, "週")

    verdict_day = evaluate_timeframe(day_slice, "日")
    verdict_week = evaluate_timeframe(week_slice, "週")
    combined = combine_timeframes({"日": verdict_day, "週": verdict_week})
    signal = classify_final(combined["最終建議"])
    return verdict_day, verdict_week, signal


def diagnose_one(name: str, code: str, market: str, api_token: str,
                  backtest_months: int) -> dict:
    today = date.today()
    backtest_start = (today - timedelta(days=backtest_months * 30))
    fetch_start = (backtest_start - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")

    df_day = get_stock_data(code, market, fetch_start, str(today), api_token)
    if df_day.empty:
        return {"name": name, "code": code, "error": "日線資料為空"}
    df_day = run_all_indicators(df_day, "日")
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    recent = df_day[df_day["date"] >= backtest_start_str].reset_index(drop=True)
    if recent.empty:
        return {"name": name, "code": code, "error": "回測窗口內沒有資料"}

    # 在回測窗口內找「崩盤高點 -> 低點」：用running peak算drawdown，取全
    # 期最大drawdown對應的高點日/低點日，不是單純的全期最低價（避免抓到
    # 窗口邊界或不相干的次要低點）。
    running_peak = recent["close"].cummax()
    drawdown = (recent["close"] - running_peak) / running_peak
    trough_pos = drawdown.idxmin()
    trough_date = recent.loc[trough_pos, "date"]
    peak_price = running_peak.loc[trough_pos]
    peak_pos = recent[(recent["date"] <= trough_date) & (recent["close"] == peak_price)].index[-1]
    peak_date = recent.loc[peak_pos, "date"]

    window_dates = recent[(recent["date"] > peak_date) & (recent["date"] <= trough_date)]["date"].tolist()

    history = []
    first_sell_date, first_sell_source = None, None
    for d in window_dates:
        verdict_day, verdict_week, signal = _evaluate_detail(df_day, d)
        if verdict_day is None:
            continue
        history.append({
            "date": d,
            "day_markers": _matched_markers(verdict_day.reasons),
            "week_markers": _matched_markers(verdict_week.reasons),
            "day_concl": verdict_day.conclusion,
            "week_concl": verdict_week.conclusion,
        })
        if signal == "賣出減碼":
            first_sell_date = d
            first_sell_source = "日" if verdict_day.conclusion == "賣出減碼" else "週"
            break

    result = {
        "name": name, "code": code, "error": None,
        "peak_date": peak_date, "peak_price": float(peak_price),
        "trough_date": trough_date, "trough_price": float(recent.loc[trough_pos, "close"]),
        "first_sell_date": first_sell_date, "first_sell_source": first_sell_source,
    }
    if first_sell_date is None:
        result["note"] = "整段崩盤期間合併訊號都沒有轉為賣出減碼"
        return result

    key = "day_markers" if first_sell_source == "日" else "week_markers"
    markers_on_flip_day = history[-1][key]
    markers_day_before = history[-2][key] if len(history) >= 2 else set()
    newly_triggered = markers_on_flip_day - markers_day_before

    result["markers_on_flip_day"] = sorted(markers_on_flip_day)
    result["newly_triggered_markers"] = sorted(newly_triggered)
    if not markers_on_flip_day:
        result["note"] = "沒有比對到任何已知關鍵字，可能是「綜合分數跌破門檻」路徑觸發，不是四個硬賣訊任一條件"
    elif not newly_triggered:
        result["note"] = "翻轉當天的條件前一天就已經存在了（可能是另一個週期/來源同時到齊才翻轉，不是單一條件新觸發）"
    return result


def run_diagnosis(api_token: str, backtest_months: int = BACKTEST_MONTHS_DEFAULT) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    for name, code, market in watchlist:
        try:
            results.append(diagnose_one(name, code, market, api_token, backtest_months))
        except Exception as exc:  # noqa: BLE001
            results.append({"name": name, "code": code, "error": str(exc)})
    return results


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run_diagnosis(api_token, months_arg)

    print(f"{'股票':8s} {'高點日':12s} {'轉賣日':12s} {'來源':4s} {'新觸發的關鍵條件':30s} 備註")
    bottleneck_tally = {}
    for r in results:
        if r.get("error"):
            print(f"{r['name']:8s} ⚠️ {r['error']}")
            continue
        if r["first_sell_date"] is None:
            print(f"{r['name']:8s} {r['peak_date']:12s} {'(無)':12s} {'':4s} {'':30s} {r.get('note','')}")
            continue
        newly = "、".join(r["newly_triggered_markers"]) or "(無匹配關鍵字)"
        print(f"{r['name']:8s} {r['peak_date']:12s} {r['first_sell_date']:12s} "
              f"{r['first_sell_source']:4s} {newly:30s} {r.get('note','')}")
        for m in (r["newly_triggered_markers"] or ["(無匹配關鍵字/綜合分數路徑)"]):
            bottleneck_tally[m] = bottleneck_tally.get(m, 0) + 1

    print("\n=== 各條件「是觸發賣出訊號的最後一塊拼圖」次數統計 ===")
    for marker, count in sorted(bottleneck_tally.items(), key=lambda kv: -kv[1]):
        print(f"{marker}：{count}檔")
