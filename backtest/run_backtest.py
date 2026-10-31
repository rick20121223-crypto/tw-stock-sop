"""
步驟2/3主流程：抓資料 -> 算指標 -> 算賣出側/買進側事件 -> 落地CSV。

用法：
    python3 -m backtest.run_backtest [FinMind token]

輸出（backtest/output/，不進版控）：
    universe.csv        回測標的清單與抓取狀態
    sell_events.csv      每一筆賣出側訊號觸發事件（含事後統計欄位）
    buy_events.csv       每一筆買進側訊號觸發事件
    lead_lag_pairs.csv   S1/S2/S3a兩兩之間「最近一次對方訊號」的交易日差
                         （正值＝自己比對方晚出現），核心假設的統計依據
    price_gaps.csv       還原權息後仍有單日跳空(>=20%)的個股/日期，可能是
                         減資等本模組無法自動還原的事件，寫進報告限制章節
"""

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from backtest.buy_signals import BuyParams, compute_buy_signals
from backtest.data_fetch import detect_price_gaps, fetch_adjusted_daily
from backtest.events import nearest_lag_days
from backtest.indicators import run_indicators
from backtest.sell_signals import SellParams, compute_sell_signals
from backtest.universe import build_universe

OUTPUT_DIR = Path(__file__).parent / "output"

YEARS_DEFAULT = 3
WARMUP_CALENDAR_DAYS = 240  # 涵蓋MA35/KD36/MACD(12,26,9)的暖身，見validate_indicators.py的收斂測試


def _fetch_start(years: int) -> str:
    start = date.today() - timedelta(days=years * 365 + WARMUP_CALENDAR_DAYS)
    return start.strftime("%Y-%m-%d")


def _analysis_start(years: int) -> str:
    return (date.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")


def _process_one(name: str, code: str, market: str, list_tag: str, api_token: str,
                  years: int, sell_params: SellParams, buy_params: BuyParams):
    try:
        fetch_start = _fetch_start(years)
        analysis_start = _analysis_start(years)
        today_str = date.today().strftime("%Y-%m-%d")
        df = fetch_adjusted_daily(code, market, fetch_start, today_str, api_token)
        if df.empty or len(df) < 60:
            return code, None, None, None, "資料為空或過短"
        df_ind = run_indicators(df)

        sell_events, sell_idx, heads_df = compute_sell_signals(df_ind, sell_params)
        buy_events, buy_idx = compute_buy_signals(df_ind, buy_params)

        if not heads_df.empty:
            heads_df = heads_df[heads_df["peak_date"] >= analysis_start].copy()
            heads_df.insert(0, "list_tag", list_tag)
            heads_df.insert(0, "market", market)
            heads_df.insert(0, "code", code)
            heads_df.insert(0, "name", name)

        # 只保留分析窗口內（近N年）的觸發事件，暖身期間的訊號不計入統計
        if not sell_events.empty:
            sell_events = sell_events[sell_events["date"] >= analysis_start].copy()
            sell_events.insert(0, "list_tag", list_tag)
            sell_events.insert(0, "market", market)
            sell_events.insert(0, "code", code)
            sell_events.insert(0, "name", name)
        if not buy_events.empty:
            buy_events = buy_events[buy_events["date"] >= analysis_start].copy()
            buy_events.insert(0, "list_tag", list_tag)
            buy_events.insert(0, "market", market)
            buy_events.insert(0, "code", code)
            buy_events.insert(0, "name", name)

        gaps = detect_price_gaps(df_ind)
        gap_dates = pd.to_datetime(gaps["date"]) if not gaps.empty else pd.Series([], dtype="datetime64[ns]")
        if not gaps.empty:
            gaps = gaps.copy()
            gaps.insert(0, "name", name)
            gaps.insert(0, "code", code)

        def _flag_near_gap(events: pd.DataFrame) -> pd.DataFrame:
            if events.empty or gap_dates.empty:
                events = events.copy() if not events.empty else events
                if not events.empty:
                    events["near_unadjusted_gap"] = False
                return events
            events = events.copy()
            ev_dates = pd.to_datetime(events["date"])
            near = pd.Series(False, index=events.index)
            for gd in gap_dates:
                near |= (ev_dates - gd).abs() <= pd.Timedelta(days=15)
            events["near_unadjusted_gap"] = near
            return events

        sell_events = _flag_near_gap(sell_events)
        buy_events = _flag_near_gap(buy_events)

        analysis_cutoff = df_ind["date"] >= analysis_start
        lag_rows = []
        s1 = [i for i in sell_idx["S1_MACD死叉"] if analysis_cutoff.iloc[i]]
        s2 = [i for i in sell_idx["S2_KD365死叉"] if analysis_cutoff.iloc[i]]
        s3a = [i for i in sell_idx["S3a_跌破BBI"] if analysis_cutoff.iloc[i]]
        for pair_name, a, b in [("KD365死叉_vs_MACD死叉", s2, s1),
                                 ("MACD死叉_vs_跌破BBI", s1, s3a),
                                 ("KD365死叉_vs_跌破BBI", s2, s3a)]:
            for a_idx, lag in nearest_lag_days(a, b, max_lag=30):
                lag_rows.append({"code": code, "name": name, "pair": pair_name, "lag_days": lag,
                                  "date": df_ind["date"].iloc[a_idx]})
        lag_df = pd.DataFrame(lag_rows)

        return code, sell_events, buy_events, lag_df, gaps, heads_df
    except Exception as exc:  # noqa: BLE001
        return code, None, None, None, f"error: {exc}", None


def run(api_token: str = "", liquidity_top_n: int = 100, years: int = YEARS_DEFAULT,
        sell_params: SellParams = SellParams(), buy_params: BuyParams = BuyParams(),
        max_workers: int = 6, write: bool = True, universe=None):
    if universe is None:
        universe = build_universe(liquidity_top_n=liquidity_top_n)

    sell_frames, buy_frames, lag_frames, gap_frames, heads_frames = [], [], [], [], []
    universe_status = []

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, name, code, market, list_tag, api_token,
                             years, sell_params, buy_params): (name, code, market, list_tag)
            for name, code, market, list_tag in universe
        }
        for fut in concurrent.futures.as_completed(futures):
            name, code, market, list_tag = futures[fut]
            result_code, sell_ev, buy_ev, lag_df, extra, heads_df = fut.result()
            status = "OK"
            if isinstance(extra, str):
                status = extra
            if sell_ev is not None and not sell_ev.empty:
                sell_frames.append(sell_ev)
            if buy_ev is not None and not buy_ev.empty:
                buy_frames.append(buy_ev)
            if lag_df is not None and not lag_df.empty:
                lag_frames.append(lag_df)
            if isinstance(extra, pd.DataFrame) and not extra.empty:
                gap_frames.append(extra)
            if heads_df is not None and not heads_df.empty:
                heads_frames.append(heads_df)
            universe_status.append({"name": name, "code": code, "market": market,
                                     "list_tag": list_tag, "status": status})

    sell_df = pd.concat(sell_frames, ignore_index=True) if sell_frames else pd.DataFrame()
    buy_df = pd.concat(buy_frames, ignore_index=True) if buy_frames else pd.DataFrame()
    lag_df_all = pd.concat(lag_frames, ignore_index=True) if lag_frames else pd.DataFrame()
    gap_df_all = pd.concat(gap_frames, ignore_index=True) if gap_frames else pd.DataFrame()
    heads_df_all = pd.concat(heads_frames, ignore_index=True) if heads_frames else pd.DataFrame()

    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(universe_status).to_csv(OUTPUT_DIR / "universe.csv", index=False)
        sell_df.to_csv(OUTPUT_DIR / "sell_events.csv", index=False)
        buy_df.to_csv(OUTPUT_DIR / "buy_events.csv", index=False)
        lag_df_all.to_csv(OUTPUT_DIR / "lead_lag_pairs.csv", index=False)
        gap_df_all.to_csv(OUTPUT_DIR / "price_gaps.csv", index=False)
        heads_df_all.to_csv(OUTPUT_DIR / "real_heads.csv", index=False)

    n_ok = sum(1 for s in universe_status if s["status"] == "OK")
    print(f"標的總數 {len(universe)}，成功 {n_ok}，失敗 {len(universe) - n_ok}")
    print(f"賣出側事件 {len(sell_df)} 筆，買進側事件 {len(buy_df)} 筆，"
          f"領先/落後配對 {len(lag_df_all)} 筆，跳空紀錄 {len(gap_df_all)} 筆，"
          f"真頭部 {len(heads_df_all)} 個")
    return sell_df, buy_df, lag_df_all, gap_df_all, pd.DataFrame(universe_status), heads_df_all


if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FINMIND_TOKEN", "")
    t0 = time.time()
    run(token)
    print(f"耗時 {time.time() - t0:.1f} 秒")
