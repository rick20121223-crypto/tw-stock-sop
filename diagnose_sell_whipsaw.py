"""
檢查「四關價跌破」這類賣出觸發條件會不會太敏感、容易「賣飛」（觸發賣出後
股價其實沒有真的繼續跌，甚至很快就漲回去）。

做法：對核心持股近6個月窗口，逐日重建長期(週+日)訊號，抓出「每一次」
訊號從非賣出減碼轉成賣出減碼的事件（不是只看這次崩盤那一次，整個窗口
內只要有轉折都算），記錄當下是哪個條件新觸發的，再接上
historical_backtest.py已經算好的「未來N個交易日報酬」，依觸發條件分組
比較——如果某個觸發條件事後的平均報酬/正報酬比例偏高，代表那個條件
容易觸發「賣飛」（賣出後股價其實續漲）。

用法（在專案資料夾內執行）：
    python3 diagnose_sell_whipsaw.py [FinMind_Token] [回測月數=6]
"""
import os
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from diagnose_sell_lag import _evaluate_detail, _matched_markers
from historical_backtest import BACKTEST_MONTHS_DEFAULT, WARMUP_CALENDAR_DAYS
from sop_decision import classify_final, combine_timeframes, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist


def find_sell_events(name: str, code: str, market: str, api_token: str,
                      backtest_months: int) -> list:
    """回傳這檔股票在回測窗口內，所有「轉為賣出減碼」事件的清單，每筆帶
    觸發條件跟未來N日報酬。"""
    today = date.today()
    backtest_start = today - timedelta(days=backtest_months * 30)
    fetch_start = (backtest_start - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")

    df_day = get_stock_data(code, market, fetch_start, str(today), api_token)
    if df_day.empty:
        return []
    df_day = run_all_indicators(df_day, "日")
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    as_of_dates = [d for d in df_day["date"] if d >= backtest_start_str]

    events = []
    prev_signal = None
    prev_day_markers, prev_week_markers = set(), set()
    for d in as_of_dates:
        verdict_day, verdict_week, signal = _evaluate_detail(df_day, d)
        if verdict_day is None:
            continue
        day_markers = _matched_markers(verdict_day.reasons)
        week_markers = _matched_markers(verdict_week.reasons)

        if signal == "賣出減碼" and prev_signal is not None and prev_signal != "賣出減碼":
            source = "日" if verdict_day.conclusion == "賣出減碼" else "週"
            markers_now = day_markers if source == "日" else week_markers
            markers_before = prev_day_markers if source == "日" else prev_week_markers
            newly = sorted(markers_now - markers_before) or ["(無匹配關鍵字/綜合分數路徑)"]
            close = float(df_day.loc[df_day["date"] == d, "close"].iloc[-1])
            fwd = _forward_returns(df_day, d)
            events.append({
                "name": name, "code": code, "date": d, "close": close,
                "source": source, "triggers": newly,
                **{f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS},
            })

        prev_signal = signal
        prev_day_markers, prev_week_markers = day_markers, week_markers

    return events


def run(api_token: str, backtest_months: int = BACKTEST_MONTHS_DEFAULT) -> pd.DataFrame:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    all_events = []
    for name, code, market in watchlist:
        try:
            all_events.extend(find_sell_events(name, code, market, api_token, backtest_months))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ {name}（{code}）：{exc}")
    return pd.DataFrame(all_events)


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    df = run(api_token, months_arg)
    if df.empty:
        print("沒有找到任何賣出減碼轉折事件。")
        sys.exit(0)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)

    print(f"=== 全部賣出轉折事件（共{len(df)}筆）===")
    print(df[["name", "date", "source", "triggers"] + [f"fwd_ret_{n}d" for n in FORWARD_WINDOWS]]
          .to_string(index=False))

    # 把每筆事件按「觸發條件」展開（一筆事件可能同時有多個條件一起新觸發）
    exploded = df.explode("triggers")
    print("\n=== 依觸發條件分組的事後報酬（判斷「賣飛」機率）===")
    rows = []
    for trigger, group in exploded.groupby("triggers"):
        row = {"觸發條件": trigger, "事件數": len(group)}
        for n in FORWARD_WINDOWS:
            valid = group[f"fwd_ret_{n}d"].dropna()
            row[f"{n}日_平均報酬%"] = round(valid.mean() * 100, 2) if len(valid) else None
            row[f"{n}日_股價續跌比例%"] = round((valid < 0).mean() * 100, 1) if len(valid) else None
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("事件數", ascending=False)
    print(summary.to_string(index=False))
    print("\n讀法：「股價續跌比例」越高代表這個觸發條件事後真的續跌、賣對的比例越高；")
    print("比例越低（平均報酬是正的）代表這個條件常常觸發完股價就漲回去了，容易賣飛。")
