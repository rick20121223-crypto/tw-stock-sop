"""
第二輪驗證的參數敏感度：N(近N日交叉)=3/5/8、35MA下彎定義=前1日/前3日比較、
真頭部回檔門檻=10%/15%/20%。6個變體（含baseline）各自完整跑一次108檔，
把sell_df/heads_df都存起來（不落地逐筆CSV，資料量太大），供
round2_report.py計算每個變體下的驗證1/2/3結果，檢查結論是否穩定。

用法：python3 -m backtest.round2_sensitivity [FinMind token]
輸出：backtest/output/round2_cache/<variant>.pkl（sell_df, heads_df, buy_df）
"""
import os
import pickle
import sys
from pathlib import Path

from backtest.buy_signals import BuyParams
from backtest.run_backtest import run
from backtest.sell_signals import SellParams
from backtest.universe import build_universe

CACHE_DIR = Path(__file__).parent / "output" / "round2_cache"

VARIANTS = [
    ("baseline_N5_lb3_head15", SellParams()),
    ("N3", SellParams(recent_n=3)),
    ("N8", SellParams(recent_n=8)),
    ("lb1", SellParams(ma35_slope_lookback=1)),
    ("head10", SellParams(head_pullback=0.10)),
    ("head20", SellParams(head_pullback=0.20)),
]


def run_all(api_token: str = "", liquidity_top_n: int = 100):
    universe = build_universe(liquidity_top_n=liquidity_top_n)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for label, sp in VARIANTS:
        cache_file = CACHE_DIR / f"{label}.pkl"
        if cache_file.exists():
            print(f"{label}: 已有快取，略過")
            continue
        sell_df, buy_df, _, _, _, heads_df = run(api_token, sell_params=sp, buy_params=BuyParams(),
                                                   write=False, universe=universe)
        with open(cache_file, "wb") as f:
            pickle.dump({"sell": sell_df, "buy": buy_df, "heads": heads_df}, f)
        print(f"{label}: 已存快取（sell {len(sell_df)} 筆, 真頭部 {len(heads_df)} 個）")


if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FINMIND_TOKEN", "")
    run_all(token)
