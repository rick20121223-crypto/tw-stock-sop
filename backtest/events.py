"""
事件研究共用工具：前波高點/頭部標記、未來N日報酬與最大不利波動、
命中率/假訊號率、以及「兩種訊號誰先誰後」的領先/落後天數量測。

全部函式都只往「過去」看（rolling/shift），不會用到未來資料去產生訊號本身；
forward_returns()/hit_rate()/false_signal_rate()/is_real_head() 這類「事後」
統計函式會用到訊號當天「之後」的資料，但那是用來量測訊號「準不準」，不是
拿來產生訊號，不構成look-ahead bias（跟historical_backtest.py docstring裡
講的原則一致）。
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# 前波高點 / 頭部標記
# ------------------------------------------------------------------
def rolling_peak(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """對每一天，往回看window天（含當天）的收盤價最高點與其日期，只用過去
    資料，可以安全地當成「訊號觸發當下看到的前波高點」。"""
    close = df["close"].to_numpy()
    n = len(close)
    peak_price = np.full(n, np.nan)
    peak_idx = np.full(n, -1, dtype=int)
    for i in range(n):
        lo = max(0, i - window + 1)
        seg = close[lo:i + 1]
        j = int(np.argmax(seg))
        peak_price[i] = seg[j]
        peak_idx[i] = lo + j
    out = df.copy()
    out["peak_price"] = peak_price
    out["peak_idx"] = peak_idx
    out["days_since_peak"] = np.arange(n) - peak_idx
    out["pullback_from_peak_pct"] = (peak_price - close) / peak_price * 100
    return out


def is_real_head(df_with_peak: pd.DataFrame, pullback_threshold: float = 0.15,
                  confirm_window: int = 20) -> pd.Series:
    """對df裡每一列當作「候選高點」(peak_idx==自己的index)，檢查高點之後
    confirm_window個交易日內，收盤價是否曾經從高點回落達pullback_threshold
    （預設15%），回傳以「高點所在列」為索引、True/False的Series（只對真的
    是當下窗口內最高點的那幾列有效，其餘列為NaN）。用到高點之後的資料，
    屬於事後驗證用途（見模組docstring）。"""
    close = df_with_peak["close"].to_numpy()
    peak_idx = df_with_peak["peak_idx"].to_numpy()
    n = len(close)
    result = pd.Series(np.nan, index=df_with_peak.index, dtype=object)
    is_peak_row = np.arange(n) == peak_idx
    for i in np.where(is_peak_row)[0]:
        peak_price = close[i]
        future = close[i + 1:i + 1 + confirm_window]
        if len(future) == 0:
            continue
        min_future = future.min()
        result.iloc[i] = bool((peak_price - min_future) / peak_price >= pullback_threshold)
    return result


def extract_real_heads(df_with_peak: pd.DataFrame, peak_is_real_head_filled: pd.Series) -> pd.DataFrame:
    """回傳這支股票全部「驗證為真頭部」的高點清單（去重，一個高點一列），
    欄位：peak_idx, peak_date, peak_price。peak_is_real_head_filled是已經
    往前填值過的Series（同一個高點之後、下一個高點之前都是同一個True/False/
    NaN值，見compute_sell_signals()）。用來當驗證2「以頭部為單位」的覆蓋率/
    精確率分母——沒有任何訊號觸發的真頭部也要算進分母，所以不能只從事件表
    （sell_events）反推，要單獨從完整日線的peak_idx欄位去重取得。"""
    tmp = df_with_peak[["peak_idx", "date", "peak_price"]].copy()
    tmp["is_real_head"] = peak_is_real_head_filled.to_numpy()
    tmp = tmp[tmp["is_real_head"] == True]  # noqa: E712
    if tmp.empty:
        return pd.DataFrame(columns=["peak_idx", "peak_date", "peak_price"])
    dedup = tmp.groupby("peak_idx").first().reset_index()
    return dedup.rename(columns={"date": "peak_date"})[["peak_idx", "peak_date", "peak_price"]]


# ------------------------------------------------------------------
# 未來N日報酬 / 最大不利波動 / 命中率 / 假訊號率
# ------------------------------------------------------------------
def forward_returns(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    close = df["close"]
    out = {}
    for n in windows:
        out[f"fwd_ret_{n}d"] = close.shift(-n) / close - 1
    return pd.DataFrame(out, index=df.index)


def forward_mae(df: pd.DataFrame, horizon: int = 20, direction: str = "long") -> pd.Series:
    """進場後horizon個交易日內最大不利波動（負數，越負越糟）。
    direction="long"（買進持有，看最低點）/"short"（賣出減碼後，看最高點，
    代表「賣太早」的機會成本，仍以負數表示不利幅度的大小）。"""
    close = df["close"].to_numpy()
    n = len(close)
    mae = np.full(n, np.nan)
    for i in range(n):
        window = close[i + 1:i + 1 + horizon]
        if len(window) == 0:
            continue
        if direction == "long":
            mae[i] = window.min() / close[i] - 1
        else:
            mae[i] = -(window.max() / close[i] - 1)
    return pd.Series(mae, index=df.index)


def hit_rate_flags(df: pd.DataFrame, horizon: int = 20, drawdown_threshold: float = 0.10) -> pd.Series:
    """觸發後horizon日內是否出現>=drawdown_threshold的回檔（賣出側命中率用）。"""
    close = df["close"].to_numpy()
    n = len(close)
    hit = np.full(n, np.nan)
    for i in range(n):
        window = close[i + 1:i + 1 + horizon]
        if len(window) == 0:
            continue
        hit[i] = bool((close[i] - window.min()) / close[i] >= drawdown_threshold)
    return pd.Series(hit, index=df.index)


def false_signal_flags(df: pd.DataFrame, high_window: int = 60, horizon: int = 20) -> pd.Series:
    """觸發後horizon日內是否創high_window日新高（賣出側假訊號率用）：
    未來某天的收盤價，本身就是「以該天為基準往回high_window天」的最高點。"""
    close = df["close"].to_numpy()
    n = len(close)
    roll_max = pd.Series(close).rolling(window=high_window, min_periods=1).max().to_numpy()
    flag = np.full(n, np.nan)
    for i in range(n):
        lo, hi = i + 1, min(n, i + 1 + horizon)
        if lo >= hi:
            continue
        made_new_high = False
        for d in range(lo, hi):
            if close[d] >= roll_max[d] - 1e-9:
                made_new_high = True
                break
        flag[i] = made_new_high
    return pd.Series(flag, index=df.index)


# ------------------------------------------------------------------
# 領先/落後天數：不預設誰先誰後，對每個A訊號找最近的B訊號（前後都找），
# 回傳帶正負號的交易日差（正值＝B比A晚出現，即A領先B）
# ------------------------------------------------------------------
def nearest_lag_days(dates_a: List[int], dates_b: List[int], max_lag: int = 30) -> List[tuple]:
    """dates_a/dates_b是交易日「整數索引」（同一檔股票同一份df的row位置，
    不是日期字串），回傳每個a事件的 (a_idx, b_idx-a_idx)，超過max_lag天的
    配對忽略（視為不相關的兩次事件，不要硬湊）。a_idx供呼叫端對回原始日期，
    拿來做分期間（前半/後半）的穩健性檢查。"""
    if not dates_a or not dates_b:
        return []
    b_arr = np.array(sorted(dates_b))
    out = []
    for a in dates_a:
        diffs = b_arr - a
        j = np.argmin(np.abs(diffs))
        if abs(diffs[j]) <= max_lag:
            out.append((a, int(diffs[j])))
    return out
