"""
項目3：大盤環境控制。把B1~B4(buy_events.csv)、W0~W4(w_bottom_events.csv)
各自接上TAIEX regime與超額報酬，輸出依環境拆分的統計表。

用法：python3 -m backtest.round3_regime_analysis
"""
from pathlib import Path

import pandas as pd

from backtest.market_regime import (attach_regime_and_excess_return,
                                     build_benchmark_regime, fetch_0050_proxy)
from backtest.run_backtest import _fetch_start
from backtest.summarize import summarize_by_regime

OUTPUT_DIR = Path(__file__).parent / "output"


def load_regime_events(api_token: str = ""):
    from datetime import date
    start = _fetch_start(3)
    today = date.today().isoformat()
    proxy = fetch_0050_proxy(start, today, api_token)
    regime_df = build_benchmark_regime(proxy)

    buy = pd.read_csv(OUTPUT_DIR / "buy_events.csv", dtype={"code": str})
    buy = buy[~buy["near_unadjusted_gap"]]
    w = pd.read_csv(OUTPUT_DIR / "w_bottom_events.csv", dtype={"code": str})
    w = w[~w["near_unadjusted_gap"]]

    results = {}
    for regime_col in ["regime_ma60", "regime_ret60"]:
        buy_r = attach_regime_and_excess_return(buy, regime_df, regime_col)
        w_r = attach_regime_and_excess_return(w, regime_df, regime_col)
        results[regime_col] = {"buy": buy_r, "w": w_r}
    return regime_df, results


if __name__ == "__main__":
    regime_df, results = load_regime_events()
    pd.set_option("display.width", 240)
    print("TAIEX regime_ma60分布:\n", regime_df["regime_ma60"].value_counts())
    print("TAIEX regime_ret60分布:\n", regime_df["regime_ret60"].value_counts())
    for regime_col, data in results.items():
        print(f"\n=== {regime_col} - 買進側 ===")
        print(summarize_by_regime(data["buy"]).to_string(index=False))
        print(f"\n=== {regime_col} - W底 ===")
        print(summarize_by_regime(data["w"]).to_string(index=False))
