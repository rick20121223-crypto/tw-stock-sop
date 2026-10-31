"""
大盤環境分類＋超額報酬計算。

使用者2026-10-17指示：TWSE加權指數官方端點、FinMind全部資料集這個session
都被暫時擋下（分別回HTTP 428/403，見第三輪報告第3節），改用0050
（元大台灣50）當大盤代理指標——0050資料已經在本機快取，不需要再打被擋的
資料源。0050追蹤市值前50大，跟加權指數高度相關但不是同一個東西（不含
中小型股、ETF本身可能有折溢價），報告需要清楚標示這是代理指標，不是
使用者原本指定的TAIEX本身。

0050在2025-06-18有一次1:4分割（18日收盤47.57 vs 前一交易日(06-10)收盤
188.65，比例0.25232），TaiwanStockDividendResult這個除權息資料集不涵蓋
ETF單位分割，本模組的標準還原權息流程不會自動處理，所以用這裡單獨
手動校正一次（乘上比例，分割日之前的價格全部調整），否則2025年中的
60日均線/60日報酬會出現一次假的「崩盤」。

兩種大盤環境分類都提供：
  regime_ma60：代理指標收盤站上60MA且60MA上揚 = 多頭，其餘 = 非多頭。
  regime_ret60：代理指標60日報酬>0 = 多頭，其餘 = 非多頭。
"""

import numpy as np
import pandas as pd

from backtest.data_fetch import fetch_raw_daily
from backtest.indicators import slope_classify

FORWARD_WINDOWS = (5, 10, 20)

# 0050在2025-06-18的1:4分割，手動校正用（見模組docstring）。
PROXY_0050_SPLIT_DATE = "2025-06-18"
PROXY_0050_SPLIT_RATIO = 47.57 / 188.65


def fetch_0050_proxy(start_date: str, end_date: str, api_token: str = "") -> pd.DataFrame:
    """回傳[date, close]，0050原始收盤價，已手動校正2025-06-18的1:4分割。"""
    df = fetch_raw_daily("0050", "TW", start_date, end_date, api_token)
    df = df[["date", "close"]].copy()
    mask = df["date"] < PROXY_0050_SPLIT_DATE
    df.loc[mask, "close"] = df.loc[mask, "close"] * PROXY_0050_SPLIT_RATIO
    return df


def build_benchmark_regime(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """benchmark_df需含[date, close]，可以是真正的大盤指數，也可以是像0050
    這種代理指標——欄位名稱不帶「taiex」字樣，避免跟真正的加權指數混淆。"""
    df = benchmark_df.sort_values("date").reset_index(drop=True).copy()
    df["MA60"] = df["close"].rolling(window=60).mean()
    ma60_slope = slope_classify(df["MA60"], lookback=3, threshold=0.0)
    above_ma60 = df["close"] > df["MA60"]
    df["regime_ma60"] = np.where(above_ma60 & (ma60_slope == "上揚"), "多頭", "非多頭")

    ret60 = df["close"] / df["close"].shift(60) - 1
    df["ret60"] = ret60
    df["regime_ret60"] = np.where(ret60 > 0, "多頭", "非多頭")

    for n in FORWARD_WINDOWS:
        df[f"benchmark_fwd_ret_{n}d"] = df["close"].shift(-n) / df["close"] - 1

    return df[["date", "close", "MA60", "regime_ma60", "ret60", "regime_ret60"] +
              [f"benchmark_fwd_ret_{n}d" for n in FORWARD_WINDOWS]]


def attach_regime_and_excess_return(events: pd.DataFrame, regime_df: pd.DataFrame,
                                     regime_col: str = "regime_ma60") -> pd.DataFrame:
    """把events(已有date欄位、fwd_ret_5d/10d/20d欄位)跟regime_df用date合併，
    加上該日大盤環境分類、以及超額報酬(個股報酬-同期間代理指標報酬)。
    合併不到大盤資料的日期（通常是資料邊界）regime/超額報酬會是NaN。"""
    events = events.copy()
    events["date"] = events["date"].astype(str)
    regime_df = regime_df.copy()
    regime_df["date"] = regime_df["date"].astype(str)
    merged = events.merge(regime_df, on="date", how="left", suffixes=("", "_benchmark"))
    for n in FORWARD_WINDOWS:
        if f"fwd_ret_{n}d" in merged.columns:
            merged[f"excess_ret_{n}d"] = merged[f"fwd_ret_{n}d"] - merged[f"benchmark_fwd_ret_{n}d"]
    merged["regime"] = merged[regime_col]
    return merged
