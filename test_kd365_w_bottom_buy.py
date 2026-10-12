"""
驗證W底偵測加KD(36,5,5)確認後，「三者同步確認」的買點是不是真的比
「只有均線+MACD兩者同步、KD365不同步」更可靠。

對應課程錄音《中山北路二段65巷6號.m4a》「抓起漲與打底進場點」：低檔
打底、KD(365)與MACD同步轉強才是明確進場買點。2026-10-12 Phase 1已經
把KD365接進_detect_w_bottom()的synced判斷，這支腳本是Phase 2的backtest
驗證。

比較兩組「訊號後N日報酬／上漲比例」，互斥：
    A) 均線斜率+MACD同步轉多，但KD365不同步（K365<=D365）——舊版兩者
       確認的買點
    B) 均線斜率+MACD+KD365三者同步轉多——新版三者確認的買點（現在
       _detect_w_bottom()標記「轉折訊號確認」的那組）

用意：如果B比A更有保護力（後續上漲比例更高、平均報酬更高），代表KD365
確實能篩掉只有兩者確認、其實還不夠可靠的假突破；如果差不多甚至更差，
代表加KD365沒有實質幫助。

比照賣出側test_kd365_double_check.py學到的教訓，排除「訊號中途已經
反轉」的事件——如果W底突破後，股價又跌破頸線，或KD365死叉翻空，代表
這個「打底成功」的訊號已經失效，不該把反轉之後的漲跌也算進這個訊號的
功過。reversal_day取「價格跌破頸線」或「KD365死叉」兩者較早發生的那個。

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
CONFIRMED_MARKER = "轉折訊號確認"
MAX_REVERSAL_LOOKAHEAD = max(FORWARD_WINDOWS)


def _macd_bias_sign(latest: pd.Series) -> float:
    """重建_step3_macd()裡macd_bias>0的那一半判斷：DIF在零軸之上給
    正值，不需要完整重算bias分數，_detect_w_bottom()只在意macd_bias>0
    這個布林結果。"""
    if pd.isna(latest.get("DIF")):
        return 0.0
    return 1.0 if latest["DIF"] > 0 else -1.0


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
    """逐日重建_detect_w_bottom()的偵測，標出A組(兩者確認但KD365不同步)
    /B組(三者同步確認)的事件，互斥。"""
    cfg = RULES.get("pattern_recognition", {}).get("w_bottom", {})
    lookback = cfg.get("lookback_bars") or 60
    kd365_cfg = RULES["indicators"]["kd365"]
    kd365_warmup = kd365_cfg["n"] + kd365_cfg["m1"] + kd365_cfg["m2"]
    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
    start = max(lookback, kd365_warmup, slope_lookback + 1)

    two_way_dates, three_way_dates = [], []

    for i in range(start, len(df)):
        window = df.iloc[:i + 1]
        latest = window.iloc[-1]

        fast_slope_up = ma_slope(window, FAST_MA["日"], lookback=slope_lookback) == "上揚"
        macd_bias = _macd_bias_sign(latest)
        kd365_bullish = (
            pd.notna(latest.get("K365")) and pd.notna(latest.get("D365"))
            and latest["K365"] > latest["D365"]
        )

        reasons: list = []
        neckline = _detect_w_bottom(window, fast_slope_up, macd_bias, kd365_bullish, reasons)
        if neckline is None:
            continue

        signal_date = df.iloc[i]["date"]
        confirmed_three_way = any(CONFIRMED_MARKER in r for r in reasons)

        if confirmed_three_way:
            three_way_dates.append((signal_date, neckline))
        elif fast_slope_up and macd_bias > 0:
            # 均線+MACD兩者同步，但KD365不同步（kd365_bullish=False才會
            # 走到這裡，因為synced的另外兩個條件已經成立）
            two_way_dates.append((signal_date, neckline))

    return {"two_way": two_way_dates, "three_way": three_way_dates}


def _events_to_forward_returns(df: pd.DataFrame, events: list) -> list:
    results = []
    for i, (signal_date, neckline) in enumerate(events):
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
                "two_way_events": _events_to_forward_returns(df, events["two_way"]),
                "three_way_events": _events_to_forward_returns(df, events["three_way"]),
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

    all_two_way, all_three_way = [], []
    for r in results:
        if r.get("error"):
            continue
        all_two_way.extend(r["two_way_events"])
        all_three_way.extend(r["three_way_events"])

    print(f"\n=== 回測{months_arg}個月，{len(unique_watchlist(STOCK_NAME_MAP))}檔加總 ===")
    print("（各天期後面的「排除N筆訊號已反轉」：價格跌破頸線或KD365死叉，\n"
          "不算進那個天期的統計）\n")
    _print_stats("A) 均線+MACD兩者同步，KD365不同步（舊版兩者確認）", all_two_way)
    print()
    _print_stats("B) 均線+MACD+KD365三者同步（新版三者確認）", all_three_way)


if __name__ == "__main__":
    main()
