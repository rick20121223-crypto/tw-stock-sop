"""
驗證KD(36,5,5)+MACD的Double Check機制，是不是真的比單看MACD背離更可靠。

比較兩組「訊號後N日報酬／下跌比例」，A/B互斥（不重疊）：
    A) 只有MACD頂背離、KD365沒有死叉（純MACD訊號，沒有Double Check確認）
    B) MACD頂背離 + KD365死亡交叉同時成立（Double Check雙重確認）

用意：B組如果比A組保護力更好（後續下跌比例更高、平均報酬更低），
代表Double Check機制真的有篩選效果，才考慮之後讓它影響conclusion。
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
from sop_decision import RULES, _detect_bearish_divergence, _step_kd365
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24


def _collect_events(df: pd.DataFrame) -> dict:
    """逐日重建_step3_macd的背離判斷跟_step_kd365的死叉判斷，標出A組/B組的
    事件日期字串。A組(macd_only)跟B組(double_check)互斥：A組只算「MACD
    背離成立、但KD365沒有死叉」的事件，B組算「MACD背離+KD365死叉同時
    成立」的事件——這樣A/B才是能公平比較的兩群事件，不會B是A的子集合
    （原本的寫法B永遠是A的子集，系統性低估了Double Check的差異）。

    cfg_lookback直接讀sop_rules.yaml的kd365.divergence_lookback，不寫死
    常數，確保這支回測腳本永遠跟它驗證的規則參數同步。迴圈起點額外跳過
    KD365的暖身期（n+m1+m2根K棒）：KD(36,5,5)前面這段K365/D365都還沒有
    真正的RSV可用、被種子值50.0填滿，太早開始算交叉/背離會拿到假訊號。
    """
    kd365_cfg = RULES["indicators"]["kd365"]
    cfg_lookback = kd365_cfg["divergence_lookback"]
    warmup_bars = kd365_cfg["n"] + kd365_cfg["m1"] + kd365_cfg["m2"]
    start = max(cfg_lookback, warmup_bars)
    macd_only_dates, double_check_dates = [], []

    for i in range(start, len(df)):
        window = df.iloc[:i + 1]
        latest = window.iloc[-1]

        macd_div = _detect_bearish_divergence(window, cfg_lookback, column="MACD_hist")
        if not macd_div:
            continue

        reasons, caveats = [], []
        _, kd365_cross_down, _ = _step_kd365(window, "日", latest, reasons, caveats)

        signal_date = df.iloc[i]["date"]
        if kd365_cross_down:
            double_check_dates.append(signal_date)
        else:
            macd_only_dates.append(signal_date)

    return {"macd_only": macd_only_dates, "double_check": double_check_dates}


def _dates_to_forward_returns(df: pd.DataFrame, dates: list) -> list:
    return [_forward_returns(df, d) for d in dates]


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
                "macd_only_events": _dates_to_forward_returns(df, events["macd_only"]),
                "double_check_events": _dates_to_forward_returns(df, events["double_check"]),
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
        if not rets:
            continue
        avg = sum(rets) / len(rets)
        down_ratio = sum(1 for r in rets if r < 0) / len(rets)
        print(f"  {window}日後：平均報酬{avg:+.2%}，下跌比例{down_ratio:.1%}")


def main() -> None:
    api_token = sys.argv[1] if len(sys.argv) > 1 else ""
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    all_macd_only, all_double_check = [], []
    for r in results:
        if r.get("error"):
            continue
        all_macd_only.extend(r["macd_only_events"])
        all_double_check.extend(r["double_check_events"])

    print(f"\n=== 回測{months_arg}個月，{len(unique_watchlist(STOCK_NAME_MAP))}檔加總 ===\n")
    _print_stats("A) 只有MACD頂背離（KD365未死叉）", all_macd_only)
    print()
    _print_stats("B) MACD頂背離+KD365死亡交叉（Double Check）", all_double_check)


if __name__ == "__main__":
    main()
