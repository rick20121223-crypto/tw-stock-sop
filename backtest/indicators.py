"""
指標計算：直接借用 stock_core.py 既有的 MACD/KD/BBI/MA 計算函式（同一套公式、
同一組參數），本模組只加「交叉偵測」「斜率分類（門檻可參數化）」這些回測
專用的輔助函式，不重算一份不一致的指標版本，也不改 stock_core.py。

stock_core.calc_macd() 欄位命名：DIF（差離值）、MACD_signal（訊號線，即一般
看盤軟體/券商App標示的「MACD」那條線）。本模組與報告沿用這組命名，不另外
取別名，避免兩套「MACD」稱呼混淆。
"""

import numpy as np
import pandas as pd

from stock_core import calc_bbi, calc_kd, calc_ma, calc_macd

KD365_PARAMS = dict(n=36, m1=5, m2=5, k_col="K365", d_col="D365")


def run_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在日線 df（含 open/max/min/close）上疊加 MACD(12,26,9)、KD(9,3,3)、
    KD(36,5,5)、BBI(3,6,12,24)、MA35。"""
    df = df.copy()
    df = calc_macd(df)
    df = calc_kd(df)
    df = calc_kd(df, **KD365_PARAMS)
    df = calc_bbi(df)
    df = calc_ma(df, periods=(35,))
    return df


def slope_classify(series: pd.Series, lookback: int, threshold: float = 0.0) -> pd.Series:
    """拿 series 今日值跟 lookback 天前比較，diff > threshold 判「上揚」、
    diff < -threshold 判「下彎」，否則「走平」。threshold 預設 0（嚴格比大小，
    不做雜訊容忍），依使用者規格「門檻做成參數並預設0」。回傳逐日分類的
    Series（字串），資料不足（lookback天內沒有值）回傳 NaN。"""
    diff = series - series.shift(lookback)
    out = pd.Series(np.where(diff > threshold, "上揚",
                     np.where(diff < -threshold, "下彎", "走平")), index=series.index)
    out[diff.isna()] = np.nan
    return out


def crossed_down(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """今天 fast 由上往下穿 slow（死叉發生日）。"""
    return (fast.shift(1) >= slow.shift(1)) & (fast < slow)


def crossed_up(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """今天 fast 由下往上穿 slow（金叉發生日）。"""
    return (fast.shift(1) <= slow.shift(1)) & (fast > slow)


def crossed_down_recently(fast: pd.Series, slow: pd.Series, n: int) -> pd.Series:
    """近 N 個交易日內（含今天）發生過死叉，N 預設見呼叫端。"""
    return crossed_down(fast, slow).rolling(window=n, min_periods=1).max().astype(bool)


def crossed_up_recently(fast: pd.Series, slow: pd.Series, n: int) -> pd.Series:
    return crossed_up(fast, slow).rolling(window=n, min_periods=1).max().astype(bool)
