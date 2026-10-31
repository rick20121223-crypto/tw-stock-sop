"""
W底 W0~W4 回測主流程：抓資料(沿用run_backtest.py同款快取/還原權息) -> 偵測
W底型態 -> 依W0~W4分組 -> 落地CSV。

用法：python3 -m backtest.run_w_bottom [FinMind token]
"""
import concurrent.futures
import os
import sys
import time
from pathlib import Path

import pandas as pd

from backtest.data_fetch import detect_price_gaps, fetch_adjusted_daily
from backtest.indicators import run_indicators
from backtest.run_backtest import _analysis_start, _fetch_start, YEARS_DEFAULT
from backtest.universe import build_universe
from backtest.w_bottom import WBottomParams
from backtest.w_bottom_signals import WGroupParams, compute_w_bottom_events

OUTPUT_DIR = Path(__file__).parent / "output"


def _process_one(name, code, market, list_tag, api_token, years, w_params, g_params):
    try:
        fetch_start = _fetch_start(years)
        analysis_start = _analysis_start(years)
        from datetime import date
        today_str = date.today().strftime("%Y-%m-%d")
        df = fetch_adjusted_daily(code, market, fetch_start, today_str, api_token)
        if df.empty or len(df) < 60:
            return code, None, "資料為空或過短"
        df_ind = run_indicators(df)
        ev = compute_w_bottom_events(df_ind, w_params, g_params)
        if ev.empty:
            return code, ev, None
        ev = ev[ev["date"] >= analysis_start].copy()
        gaps = detect_price_gaps(df_ind)
        if not gaps.empty:
            gap_dates = pd.to_datetime(gaps["date"])
            ev_dates = pd.to_datetime(ev["date"])
            near = pd.Series(False, index=ev.index)
            for gd in gap_dates:
                near |= (ev_dates - gd).abs() <= pd.Timedelta(days=15)
            ev["near_unadjusted_gap"] = near
        else:
            ev["near_unadjusted_gap"] = False
        ev.insert(0, "list_tag", list_tag)
        ev.insert(0, "market", market)
        ev.insert(0, "code", code)
        ev.insert(0, "name", name)
        return code, ev, None
    except Exception as exc:  # noqa: BLE001
        return code, None, f"error: {exc}"


def run(api_token: str = "", liquidity_top_n: int = 100, years: int = YEARS_DEFAULT,
        w_params: WBottomParams = WBottomParams(), g_params: WGroupParams = WGroupParams(),
        max_workers: int = 6, write: bool = True, universe=None):
    if universe is None:
        universe = build_universe(liquidity_top_n=liquidity_top_n)

    frames = []
    status_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, name, code, market, list_tag, api_token,
                             years, w_params, g_params): (name, code, market, list_tag)
            for name, code, market, list_tag in universe
        }
        for fut in concurrent.futures.as_completed(futures):
            name, code, market, list_tag = futures[fut]
            result_code, ev, err = fut.result()
            status = err if err else "OK"
            if ev is not None and not ev.empty:
                frames.append(ev)
            status_rows.append({"name": name, "code": code, "market": market,
                                 "list_tag": list_tag, "status": status})

    w_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if write:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        w_df.to_csv(OUTPUT_DIR / "w_bottom_events.csv", index=False)
        pd.DataFrame(status_rows).to_csv(OUTPUT_DIR / "w_bottom_universe.csv", index=False)

    n_ok = sum(1 for s in status_rows if s["status"] == "OK")
    print(f"標的總數 {len(universe)}，成功 {n_ok}，失敗 {len(universe) - n_ok}")
    print(f"W底事件 {len(w_df)} 筆")
    return w_df, pd.DataFrame(status_rows)


if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FINMIND_TOKEN", "")
    t0 = time.time()
    run(token)
    print(f"耗時 {time.time() - t0:.1f} 秒")
