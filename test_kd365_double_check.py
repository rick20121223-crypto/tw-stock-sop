"""
驗證「Double Check」機制（KD(36,5,5)死叉 + MACD死叉，課程原文《中山北路
二段65巷6號.m4a》）是不是真的比單看MACD更能抓到後續走弱，加上均線斜率
最高裁決疊加後會不會更準。

第二版修正（第一版把「MACD頂背離」誤當成Double Check的MACD那一半，已經
改成用真正的MACD死叉，見_step3_macd()）。

第三版修正（這一版）：排除「訊號中途已經反轉」卻還硬算到的天期。原本
不管訊號有沒有在N天內就反轉，一律算「N天後報酬」——但如果KD365在死叉
後第8天就金叉翻回去了，一個照SOP操作的人在第8天就會重新評估，不會傻傻
抱著死叉部位撐到第20天，所以「死叉後20天報酬」這個數字不該算進Double
Check的統計，會錯誤地把訊號早就失效之後的漲跌也算成訊號的功過
（2026-03-23金居(8358)案例就是這樣：死叉後約第10個交易日KD365就金叉
翻多了，但原本的版本還是把20天後報酬算進Double Check失敗案例）。

修正做法：每筆事件額外算「訊號維持了幾個交易日」（A組看MACD自己哪天
golden cross翻回去，B/C組看KD365哪天golden cross翻回去）；某個天期N
只有在「訊號維持天數 > N」(也就是到第N天訊號都還沒反轉)時才算進那個
天期的統計，反轉更早的事件在該天期被排除（但較短天期如果還沒反轉，
一樣算數——同一筆事件,5天可能算、10天/20天被排除)。

比較三組「訊號後N日報酬／下跌比例」，三組互斥（不重疊）：
    A) 只有MACD死叉，KD365沒有同步死叉（單一指標基準組）
    B) MACD死叉 + KD365死叉同時成立（真正的Double Check）
    C) B再加上生死線(KEY_MA)斜率也轉下彎（課程「均線斜率具最高裁決權」，
       Double Check疊加均線轉空才是「必須出場」等級，B但均線未轉下彎的
       部分可能只是「示警/減碼」而已）

用意：如果B比A更有保護力（後續下跌比例更高、平均報酬更低），代表KD365
確實能過濾掉MACD單獨死叉的雜訊；如果C又比B更有保護力，代表均線轉下彎
這層疊加確實能進一步篩出真正該出場的情況，才考慮之後讓這些條件影響
conclusion。

這支腳本只跑日線（跟test_weekly_kd_and_daily_ma35_buy.py的週線部分
分工，60分/5分的分K資料Fugle API有時間深度限制，之後如果要測短線
版本另外處理）。

重用backtest_signal_log.py既有的_forward_returns(price_df, signal_date)：
吃「完整價格DataFrame + 訊號當天日期字串」，回傳{5: 報酬, 10: 報酬,
20: 報酬}（FORWARD_WINDOWS=(5,10,20)，鍵是整數交易日數、不是字串）。

用法（在專案資料夾內執行，需要FinMind token）：
    python3 test_kd365_double_check.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import RULES, _step3_macd, _step_kd365
from stock_core import (KEY_MA, STOCK_NAME_MAP, get_stock_data, ma_slope,
                         run_all_indicators, unique_watchlist)

BACKTEST_MONTHS_DEFAULT = 24
MAX_REVERSAL_LOOKAHEAD = max(FORWARD_WINDOWS)  # 只需要看到最長的天期(20天)就夠


def _find_reversal_day(df: pd.DataFrame, event_idx: int, fast_col: str, slow_col: str,
                        max_days: int = MAX_REVERSAL_LOOKAHEAD):
    """從event_idx後一天開始找，fast_col重新穿越到slow_col之上(golden cross)
    的第一個交易日偏移量(1~max_days)。找不到(資料不夠、或max_days內都還
    沒反轉)就回傳None，代表訊號在max_days交易日內都還維持。"""
    for offset in range(1, max_days + 1):
        idx = event_idx + offset
        if idx >= len(df):
            return None
        fast, slow = df.iloc[idx].get(fast_col), df.iloc[idx].get(slow_col)
        if pd.isna(fast) or pd.isna(slow):
            continue
        if fast > slow:
            return offset
    return None


def _collect_events(df: pd.DataFrame) -> dict:
    """逐日重建MACD死叉(_step3_macd)、KD365死叉(_step_kd365)、生死線
    (KEY_MA)斜率，標出三組事件，彼此互斥：
    A) macd_only：MACD死叉成立、KD365沒有同步死叉
    B) double_check：MACD死叉 + KD365死叉同時成立（B不含在A裡）
    C) double_check_ma_down：B再加上生死線斜率也轉下彎（C是B的子集合，
       故意讓C⊆B，因為C要回答的問題是「B裡面有沒有再疊加均線確認會更準」，
       不是跟A/B做互斥三分）

    每筆事件除了日期，也附上reversal_day（訊號維持的交易日數，A組看MACD
    自己的golden cross、B/C組看KD365的golden cross，None代表max_days內
    都還沒反轉）。

    迴圈起點跳過KD365暖身期（n+m1+m2根K棒，K365/D365前段被種子值50.0
    填滿，太早算交叉會是假訊號）跟均線斜率的lookback視窗。
    """
    kd365_cfg = RULES["indicators"]["kd365"]
    warmup_bars = kd365_cfg["n"] + kd365_cfg["m1"] + kd365_cfg["m2"]
    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
    key_ma_col = KEY_MA["日"]
    start = max(warmup_bars, slope_lookback + 1)

    macd_only, double_check, double_check_ma_down = [], [], []

    for i in range(start, len(df)):
        window = df.iloc[:i + 1]
        latest = window.iloc[-1]

        reasons, caveats = [], []
        _, macd_dead_cross, _ = _step3_macd(window, latest, reasons, caveats)
        if not macd_dead_cross:
            continue

        reasons2, caveats2 = [], []
        _, kd365_cross_down, _ = _step_kd365(window, "日", latest, reasons2, caveats2)

        signal_date = df.iloc[i]["date"]
        if not kd365_cross_down:
            reversal = _find_reversal_day(df, i, "DIF", "MACD_signal")
            macd_only.append({"date": signal_date, "reversal_day": reversal})
            continue

        reversal = _find_reversal_day(df, i, "K365", "D365")
        double_check.append({"date": signal_date, "reversal_day": reversal})

        key_ma_slope = ma_slope(window, key_ma_col, lookback=slope_lookback)
        if key_ma_slope == "下彎":
            double_check_ma_down.append({"date": signal_date, "reversal_day": reversal})

    return {"macd_only": macd_only, "double_check": double_check,
            "double_check_ma_down": double_check_ma_down}


def _events_to_forward_returns(df: pd.DataFrame, events: list) -> list:
    """把每筆事件的N天後報酬算出來，但只有「訊號到第N天都還沒反轉」才保留
    那個天期的數字，反轉更早的天期改成None（讓_print_stats自然排除）。"""
    results = []
    for e in events:
        raw = _forward_returns(df, e["date"])
        reversal = e["reversal_day"]
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
                "macd_only_events": _events_to_forward_returns(df, events["macd_only"]),
                "double_check_events": _events_to_forward_returns(df, events["double_check"]),
                "double_check_ma_down_events": _events_to_forward_returns(
                    df, events["double_check_ma_down"]),
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
        down_ratio = sum(1 for r in rets if r < 0) / len(rets)
        excl_note = f"，排除{excluded}筆訊號已反轉" if excluded else ""
        print(f"  {window}日後：平均報酬{avg:+.2%}，下跌比例{down_ratio:.1%}"
              f"（{len(rets)}筆{excl_note}）")


def main() -> None:
    api_token = sys.argv[1] if len(sys.argv) > 1 else ""
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    all_macd_only, all_double_check, all_double_check_ma_down = [], [], []
    for r in results:
        if r.get("error"):
            continue
        all_macd_only.extend(r["macd_only_events"])
        all_double_check.extend(r["double_check_events"])
        all_double_check_ma_down.extend(r["double_check_ma_down_events"])

    print(f"\n=== 回測{months_arg}個月，{len(unique_watchlist(STOCK_NAME_MAP))}檔加總 ===")
    print("（各天期後面的「排除N筆訊號已反轉」：訊號在那個天期之前就已經\n"
          "golden cross翻回去了，不算進那個天期的統計）\n")
    _print_stats("A) 只有MACD死叉（KD365未同步死叉）", all_macd_only)
    print()
    _print_stats("B) MACD死叉+KD365死叉（Double Check）", all_double_check)
    print()
    _print_stats("C) B再加上生死線斜率也轉下彎（均線最高裁決疊加）", all_double_check_ma_down)


if __name__ == "__main__":
    main()
