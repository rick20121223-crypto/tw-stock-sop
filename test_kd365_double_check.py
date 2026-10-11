"""
驗證KD(36,5,5)+MACD的Double Check機制，是不是真的比單看MACD背離更可靠。

比較兩組「訊號後N日報酬／下跌比例」：
    A) 只有MACD頂背離（現有_step3_macd.macd_bearish_divergence單獨成立）
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
from sop_decision import _detect_bearish_divergence, _step_kd365
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24


def _collect_events(df: pd.DataFrame) -> dict:
    """逐日重建_step3_macd的背離判斷跟_step_kd365的死叉判斷，標出A組/B組的事件日期字串。"""
    cfg_lookback = 20  # 對應sop_rules.yaml macd.divergence_lookback/kd365.divergence_lookback
    macd_only_dates, double_check_dates = [], []

    for i in range(cfg_lookback, len(df)):
        window = df.iloc[:i + 1]
        latest = window.iloc[-1]

        macd_div = _detect_bearish_divergence(window, cfg_lookback, column="MACD_hist")
        if not macd_div:
            continue

        reasons, caveats = [], []
        _, kd365_cross_down, _ = _step_kd365(window, "日", latest, reasons, caveats)

        signal_date = df.iloc[i]["date"]
        macd_only_dates.append(signal_date)
        if kd365_cross_down:
            double_check_dates.append(signal_date)

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
    _print_stats("A) 只有MACD頂背離", all_macd_only)
    print()
    _print_stats("B) MACD頂背離+KD365死亡交叉（Double Check）", all_double_check)


if __name__ == "__main__":
    main()
