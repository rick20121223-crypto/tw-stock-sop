"""
第三輪W底參數敏感度：右底間隔(10~40/5~60)、頸線定義(收盤/最高點)、
近N日金叉(3/5/8)、均線斜率(20MA/35MA)，各自單獨換一組值跑一次，結果快取
成pickle供round3_report.py比較。

用法：python3 -m backtest.round3_w_sensitivity [FinMind token]
"""
import os
import pickle
import sys
from pathlib import Path

from backtest.run_w_bottom import run
from backtest.universe import build_universe
from backtest.w_bottom import WBottomParams
from backtest.w_bottom_signals import WGroupParams

CACHE_DIR = Path(__file__).parent / "output" / "round3_w_cache"

VARIANTS = [
    ("baseline_gap10-40_close_n3_ma20", WBottomParams(), WGroupParams()),
    ("gap5-60", WBottomParams(right_gap_min=5, right_gap_max=60), WGroupParams()),
    ("neckline_high", WBottomParams(neckline_mode="high"), WGroupParams()),
    ("n5", WBottomParams(), WGroupParams(kd_recent_n=5)),
    ("n8", WBottomParams(), WGroupParams(kd_recent_n=8)),
    ("ma35", WBottomParams(), WGroupParams(ma_period=35)),
]


def run_all(api_token: str = "", liquidity_top_n: int = 100):
    universe = build_universe(liquidity_top_n=liquidity_top_n)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for label, wp, gp in VARIANTS:
        cache_file = CACHE_DIR / f"{label}.pkl"
        if cache_file.exists():
            print(f"{label}: 已有快取，略過")
            continue
        w_df, _ = run(api_token, w_params=wp, g_params=gp, write=False, universe=universe)
        with open(cache_file, "wb") as f:
            pickle.dump({"w": w_df}, f)
        print(f"{label}: 已存快取（{len(w_df)} 筆）")


if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FINMIND_TOKEN", "")
    run_all(token)
