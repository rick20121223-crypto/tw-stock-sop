"""
驗證W底突破時，KD(36,5,5)「剛黃金交叉」是不是比「已經是多頭一陣子」更
能預示真正的翻身——對應課程「KD(365)與MACD同步轉強」講的是同步轉折，
不是單純已經處於多頭狀態。

這是第二版操作化方式。第一版（見2026-10-13 sop_rules.yaml changelog）
測「均線+MACD兩者同步時，KD365目前狀態是不是多頭(K365>D365)」，結果
不支持假設（KD365目前狀態幾乎93%機率已經同步，篩選力很弱，而且
「不同步」那組報酬反而更高）。這次改測「KD365是不是最近剛黃金交叉」
（fresh_window=3個交易日內），更貼近課程講的「同步」轉折時刻，而不是
「已經維持一段時間」的狀態。

比較兩組「訊號後N日報酬／上漲比例」，互斥：
    A) W底突破+均線斜率+MACD同步轉多，但KD365不是最近剛黃金交叉（可能
       已經多頭一陣子，或還沒轉多）
    B) W底突破+均線斜率+MACD同步轉多，且KD365在過去3個交易日內剛黃金
       交叉——真正的「同步轉強」時刻

注意：這次不依賴_detect_w_bottom()內部的synced判斷（那個判斷用的是
「目前狀態」版KD365，是Phase1已經上線的定義）——A/B兩組都要求W底突破+
均線+MACD同步這個基礎條件成立，只是對KD365的操作化方式不同，用來比較
兩種候選定義何者更有預測力。

沿用第一版學到的教訓，排除「訊號中途已經反轉」的事件（價格跌破頸線或
KD365死叉）。

這支腳本只跑日線。

用法（在專案資料夾內執行，需要FinMind token）：
    python3 test_kd365_w_bottom_buy.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import RULES, _detect_w_bottom
from stock_core import (FAST_MA, STOCK_NAME_MAP, get_stock_data, ma_slope,
                         run_all_indicators, unique_watchlist)

BACKTEST_MONTHS_DEFAULT = 24
FRESH_CROSS_WINDOW = 3  # 幾個交易日內的黃金交叉算「剛交叉」，比照left_side_platform/cci既有confirm_bars=3的慣例
MAX_REVERSAL_LOOKAHEAD = max(FORWARD_WINDOWS)


def _macd_bias_sign(latest: pd.Series) -> float:
    """重建_step3_macd()裡macd_bias>0的那一半判斷：DIF在零軸之上給
    正值，不需要完整重算bias分數，_detect_w_bottom()只在意macd_bias>0
    這個布林結果。"""
    if pd.isna(latest.get("DIF")):
        return 0.0
    return 1.0 if latest["DIF"] > 0 else -1.0


def _kd365_bullish_state(latest: pd.Series) -> bool:
    return (pd.notna(latest.get("K365")) and pd.notna(latest.get("D365"))
            and latest["K365"] > latest["D365"])


def _kd365_crossed_up_recently(df: pd.DataFrame, idx: int, window: int = FRESH_CROSS_WINDOW) -> bool:
    """檢查K365是否在[idx-window+1, idx]這段範圍內的某一天黃金交叉
    （前一天K<=D，當天K>D）。idx本身也算在檢查範圍內。"""
    start = max(1, idx - window + 1)
    for i in range(start, idx + 1):
        prev, cur = df.iloc[i - 1], df.iloc[i]
        pk, pd_ = prev.get("K365"), prev.get("D365")
        ck, cd = cur.get("K365"), cur.get("D365")
        if pd.notna(pk) and pd.notna(pd_) and pd.notna(ck) and pd.notna(cd):
            if pk <= pd_ and ck > cd:
                return True
    return False


def _find_reversal_day(df: pd.DataFrame, event_idx: int, neckline: float,
                        max_days: int = MAX_REVERSAL_LOOKAHEAD):
    """從event_idx後一天開始找，股價收盤跌破頸線、或KD365死叉（K365穿破
    D365向下），兩者較早發生的那天偏移量(1~max_days)。找不到就回傳None，
    代表訊號在max_days交易日內都還維持。

    跌破頸線的門檻刻意用neckline*(1-tolerance)，不是neckline本身——
    _detect_w_bottom()自己判斷「回測有沒有破」用的就是這個容忍帶（見
    sop_rules.yaml pattern_recognition.w_bottom.neckline_tolerance_pct），
    如果這裡用更嚴格的門檻，會把型態本身還算「沒破」的正常价格波動也
    誤判成反轉，系統性低估訊號的實際維持天數。
    """
    cfg = RULES.get("pattern_recognition", {}).get("w_bottom", {})
    tolerance = cfg.get("neckline_tolerance_pct", 0.0)
    break_threshold = neckline * (1 - tolerance)
    for offset in range(1, max_days + 1):
        idx = event_idx + offset
        if idx >= len(df):
            return None
        row = df.iloc[idx]
        if pd.notna(row.get("close")) and row["close"] < break_threshold:
            return offset
        if idx >= 1:
            prev = df.iloc[idx - 1]
            k, d = row.get("K365"), row.get("D365")
            pk, pd_ = prev.get("K365"), prev.get("D365")
            if pd.notna(k) and pd.notna(d) and pd.notna(pk) and pd.notna(pd_):
                if pk >= pd_ and k < d:
                    return offset
    return None


def _collect_events(df: pd.DataFrame) -> dict:
    """逐日重建W底偵測+均線+MACD基礎確認，再依KD365是不是「最近剛黃金
    交叉」分成A/B兩組，互斥。不使用_detect_w_bottom()內部的synced文字
    （那是Phase1已上線的「目前狀態」版定義），這裡用自己算的fresh-cross
    版本做分組。
    """
    cfg = RULES.get("pattern_recognition", {}).get("w_bottom", {})
    lookback = cfg.get("lookback_bars") or 60
    kd365_cfg = RULES["indicators"]["kd365"]
    kd365_warmup = kd365_cfg["n"] + kd365_cfg["m1"] + kd365_cfg["m2"]
    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
    start = max(lookback, kd365_warmup, slope_lookback + 1)

    not_fresh_dates, fresh_cross_dates = [], []

    for i in range(start, len(df)):
        window = df.iloc[:i + 1]
        latest = window.iloc[-1]

        fast_slope_up = ma_slope(window, FAST_MA["日"], lookback=slope_lookback) == "上揚"
        macd_bias = _macd_bias_sign(latest)
        if not (fast_slope_up and macd_bias > 0):
            continue

        kd365_bullish = _kd365_bullish_state(latest)
        reasons: list = []
        neckline = _detect_w_bottom(window, fast_slope_up, macd_bias, kd365_bullish, reasons)
        if neckline is None:
            continue

        signal_date = df.iloc[i]["date"]
        if _kd365_crossed_up_recently(df, i):
            fresh_cross_dates.append((signal_date, neckline))
        else:
            not_fresh_dates.append((signal_date, neckline))

    return {"not_fresh": not_fresh_dates, "fresh_cross": fresh_cross_dates}


def _events_to_forward_returns(df: pd.DataFrame, events: list) -> list:
    results = []
    for signal_date, neckline in events:
        idx_matches = df.index[df["date"] == signal_date]
        if len(idx_matches) == 0:
            continue
        event_idx = idx_matches[0]
        raw = _forward_returns(df, signal_date)
        reversal = _find_reversal_day(df, event_idx, neckline)
        if reversal is None:
            results.append(raw)
        else:
            results.append({w: (v if reversal > w else None) for w, v in raw.items()})
    return results


def run(api_token: str, months: int) -> list:
    start = (date.today() - timedelta(days=months * 31 + WARMUP_CALENDAR_DAYS)).isoformat()
    results = []

    def _one(name, code, market):
        try:
            df = get_stock_data(code, market, start, str(date.today()), api_token)
            if df.empty:
                return {"name": name, "code": code, "error": "無資料"}
            df = run_all_indicators(df, "日")
            events = _collect_events(df)
            return {
                "name": name, "code": code,
                "not_fresh_events": _events_to_forward_returns(df, events["not_fresh"]),
                "fresh_cross_events": _events_to_forward_returns(df, events["fresh_cross"]),
            }
        except Exception as exc:  # noqa: BLE001
            return {"name": name, "code": code, "error": str(exc)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_one, name, code, market)
                   for name, code, market in unique_watchlist(STOCK_NAME_MAP)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _print_stats(label: str, events: list) -> None:
    if not events:
        print(f"{label}：無事件")
        return
    n = len(events)
    print(f"{label}（{n}筆事件）：")
    for window in FORWARD_WINDOWS:
        rets = [e[window] for e in events if e.get(window) is not None]
        excluded = n - len(rets)
        if not rets:
            continue
        avg = sum(rets) / len(rets)
        up_ratio = sum(1 for r in rets if r > 0) / len(rets)
        excl_note = f"，排除{excluded}筆訊號已反轉" if excluded else ""
        print(f"  {window}日後：平均報酬{avg:+.2%}，上漲比例{up_ratio:.1%}"
              f"（{len(rets)}筆{excl_note}）")


def main() -> None:
    api_token = sys.argv[1] if len(sys.argv) > 1 else ""
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    all_not_fresh, all_fresh_cross = [], []
    for r in results:
        if r.get("error"):
            continue
        all_not_fresh.extend(r["not_fresh_events"])
        all_fresh_cross.extend(r["fresh_cross_events"])

    print(f"\n=== 回測{months_arg}個月，{len(unique_watchlist(STOCK_NAME_MAP))}檔加總 ===")
    print("（各天期後面的「排除N筆訊號已反轉」：價格跌破頸線或KD365死叉，\n"
          "不算進那個天期的統計）\n")
    _print_stats(f"A) W底+均線+MACD確認，KD365非剛交叉（{FRESH_CROSS_WINDOW}日內無交叉）", all_not_fresh)
    print()
    _print_stats(f"B) W底+均線+MACD確認，KD365剛黃金交叉（{FRESH_CROSS_WINDOW}日內）", all_fresh_cross)


if __name__ == "__main__":
    main()
