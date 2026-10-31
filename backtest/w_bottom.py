"""
W底型態偵測（日線，演算法近似，不等同人工看圖）。只用來產生「訊號日」供
事件研究統計，不影響sop_decision.py既有判讀邏輯。

演算法（依使用者2026-10-17規格）：
  左底：N=60日搜尋範圍內的區域低點（前後各5日內收盤最低）。
  右底：晚於左底10~40個交易日的另一個區域低點，且右底收盤>左底收盤。
  頸線：左底與右底之間（不含兩端）的最高收盤價（close模式）或最高點
        （high模式），兩種都支援。
  突破：右底之後第一個收盤價站上頸線的交易日。
  回測不破+收紅K：突破後1~5個交易日內，找第一個「收盤仍在頸線之上」且
        「當日收紅（收盤>開盤）」的交易日當訊號日；過程中只要有任何一天
        收盤跌破頸線，這次突破視為失敗（不另外找下一次突破，見模組
        docstring底下的簡化說明）。

簡化假設（已知跟人工看圈可能不一致，見報告限制章節）：
  - 每個左底只取「最早一個」符合條件的右底，不窮舉所有右底候選。
  - 突破失敗（回測破線）後不會在同一個左右底之間再找下一次突破嘗試。
  - 區域低點用收盤價判斷，不是用最低價(low/min欄位)。
  - 左底候選彼此之間沒有去除重疊/鄰近的機制，同一段走勢可能被重複偵測成
    好幾個略有位移的「W底」。
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class WBottomParams:
    local_min_window: int = 5      # 左右各N日內為最低，才算區域低點
    right_gap_min: int = 10
    right_gap_max: int = 40
    retest_window: int = 5         # 突破後1~5個交易日內找訊號日
    neckline_mode: str = "close"   # "close" 或 "high"
    breakout_search_limit: int = 60  # 右底之後最多搜尋幾天找突破，找不到就放棄


def _local_minima(close: np.ndarray, window: int) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        out[i] = close[i] <= close[lo:hi].min()
    return out


def detect_w_bottoms(df_ind: pd.DataFrame, params: WBottomParams = WBottomParams()) -> List[dict]:
    close = df_ind["close"].to_numpy()
    open_ = df_ind["open"].to_numpy()
    high = df_ind["max"].to_numpy()
    dates = df_ind["date"].to_numpy()
    n = len(close)

    is_min = _local_minima(close, params.local_min_window)
    candidates = np.where(is_min)[0]

    patterns = []
    for left in candidates:
        right_lo = left + params.right_gap_min
        right_hi = left + params.right_gap_max
        right_candidates = candidates[(candidates >= right_lo) & (candidates <= right_hi)]
        right_candidates = right_candidates[close[right_candidates] > close[left]]
        if len(right_candidates) == 0:
            continue
        right = int(right_candidates[0])

        if right - left < 2:
            continue
        if params.neckline_mode == "close":
            neckline = close[left + 1:right].max()
        else:
            neckline = high[left + 1:right].max()

        breakout = None
        search_end = min(n, right + 1 + params.breakout_search_limit)
        for d in range(right + 1, search_end):
            if close[d] > neckline:
                breakout = d
                break
        if breakout is None:
            continue

        signal_idx = None
        broken = False
        for d in range(breakout, min(n, breakout + params.retest_window + 1)):
            if close[d] < neckline:
                broken = True
                break
            if close[d] > open_[d]:
                signal_idx = d
                break
        if broken or signal_idx is None:
            continue

        patterns.append({
            "left_idx": int(left), "left_date": dates[left], "left_price": float(close[left]),
            "right_idx": right, "right_date": dates[right], "right_price": float(close[right]),
            "neckline_price": float(neckline), "neckline_mode": params.neckline_mode,
            "breakout_idx": int(breakout), "breakout_date": dates[breakout],
            "signal_idx": int(signal_idx), "signal_date": dates[signal_idx],
            "signal_close": float(close[signal_idx]),
        })
    return patterns
