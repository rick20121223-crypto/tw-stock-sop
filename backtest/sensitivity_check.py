"""
步驟4要求的參數敏感度檢查：N(近N日內發生過交叉)、日35MA/BBI斜率門檻、
頭部定義的回檔門檻，各換一組值，看賣出側訊號的命中率/假訊號率/20日報酬
排名會不會變。只重算彙總統計（不重新落地逐筆事件CSV），跟
backtest/output/下的baseline結果分開存成sensitivity_summary.csv。

用法：python3 -m backtest.sensitivity_check [FinMind token]
"""
import os
import sys
from pathlib import Path

import pandas as pd

from backtest.buy_signals import BuyParams
from backtest.run_backtest import run
from backtest.sell_signals import SellParams
from backtest.summarize import summarize_sell
from backtest.universe import build_universe

OUTPUT_FILE = Path(__file__).parent / "output" / "sensitivity_summary.csv"

VARIANTS = [
    ("baseline(N=5,斜率門檻=0,頭部回檔=15%)", SellParams()),
    ("N=10", SellParams(recent_n=10)),
    ("斜率門檻=0.3%", SellParams(ma35_slope_threshold=0.003, bbi_slope_threshold=0.003)),
    ("頭部回檔門檻=10%", SellParams(head_pullback=0.10)),
    ("頭部回檔門檻=20%", SellParams(head_pullback=0.20)),
]

KEEP_SIGNALS = ["S1_MACD死叉", "S2_KD365死叉", "S3a_跌破BBI", "S3b_BBI轉下彎",
                "S4a_三線同步狀態", "S4b_近N日三線同步"]
KEEP_COLS = ["變體", "signal", "樣本數", "平均回落幅度%(距前波高點)", "前波高點驗證為真頭部比例%",
             "命中率%(20日內回檔>=10%)", "假訊號率%(20日內創60日新高)", "20日_平均報酬%", "20日_中位數報酬%"]


def run_sensitivity(api_token: str = "", liquidity_top_n: int = 100) -> pd.DataFrame:
    universe = build_universe(liquidity_top_n=liquidity_top_n)  # 固定清單，各變體共用，排除清單抽樣差異干擾
    rows = []
    for label, sp in VARIANTS:
        sell_df, _, _, _, _, _ = run(api_token, sell_params=sp, buy_params=BuyParams(),
                                      write=False, universe=universe)
        summary = summarize_sell(sell_df)
        summary = summary[summary["signal"].isin(KEEP_SIGNALS)].copy()
        summary.insert(0, "變體", label)
        rows.append(summary[KEEP_COLS])
    result = pd.concat(rows, ignore_index=True)
    return result


if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FINMIND_TOKEN", "")
    result = run_sensitivity(token)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)
    print(result.to_string(index=False))
