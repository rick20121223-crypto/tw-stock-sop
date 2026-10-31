"""
W底 W0~W4 分組事件統計。W0是型態本身；W1疊加均線斜率上揚+MACD零軸上；
W2/W3/W4是W1之下按KD365狀態互斥分組（剛金叉/持續K>D但非剛交叉/K<D），
對應但丁老師規則「KD365必須剛金叉才算真正買點，否則降級觀望」。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.events import forward_mae, forward_returns
from backtest.indicators import crossed_up_recently, slope_classify
from backtest.w_bottom import WBottomParams, detect_w_bottoms

FORWARD_WINDOWS = (5, 10, 20)


@dataclass
class WGroupParams:
    ma_period: int = 20          # 20或35
    ma_slope_lookback: int = 3
    ma_slope_threshold: float = 0.0
    kd_recent_n: int = 3


def _false_breakout_flags(close: np.ndarray, signal_idx: int, neckline: float, horizon: int = 10) -> bool:
    end = min(len(close), signal_idx + 1 + horizon)
    window = close[signal_idx + 1:end]
    if len(window) == 0:
        return np.nan
    return bool((window < neckline).any())


def compute_w_bottom_events(df_ind: pd.DataFrame, w_params: WBottomParams,
                             g_params: WGroupParams) -> pd.DataFrame:
    patterns = detect_w_bottoms(df_ind, w_params)
    if not patterns:
        return pd.DataFrame()
    # 不同的左/右底候選可能收斂到同一個訊號日(同一次突破)，去重避免同一個
    # 交易結果被重複計入統計（只保留每個signal_idx第一次偵測到的pattern）。
    seen_idx = set()
    deduped = []
    for p in patterns:
        if p["signal_idx"] in seen_idx:
            continue
        seen_idx.add(p["signal_idx"])
        deduped.append(p)
    patterns = deduped

    ma_col = f"MA{g_params.ma_period}"
    if ma_col not in df_ind.columns:
        from stock_core import calc_ma
        df_ind = calc_ma(df_ind, periods=(g_params.ma_period,))
    ma_slope = slope_classify(df_ind[ma_col], g_params.ma_slope_lookback, g_params.ma_slope_threshold)
    dif_positive = df_ind["DIF"] > 0
    kd_up_recent = crossed_up_recently(df_ind["K365"], df_ind["D365"], g_params.kd_recent_n)
    kd_state_up = df_ind["K365"] > df_ind["D365"]

    close = df_ind["close"].to_numpy()
    fwd = forward_returns(df_ind, windows=FORWARD_WINDOWS)
    mae = {n: forward_mae(df_ind, n, direction="long") for n in FORWARD_WINDOWS}

    rows = []
    for p in patterns:
        i = p["signal_idx"]
        w1 = bool(ma_slope.iloc[i] == "上揚") and bool(dif_positive.iloc[i])
        if w1:
            if kd_up_recent.iloc[i]:
                group = "W2_剛金叉(完整規則)"
            elif kd_state_up.iloc[i]:
                group = "W3_持續K大於D非剛交叉"
            else:
                group = "W4_K小於D尚未金叉"
        else:
            group = None  # 不滿足W1，不屬於W2/W3/W4任何一組

        base_row = {
            "left_date": p["left_date"], "right_date": p["right_date"],
            "neckline_price": p["neckline_price"], "neckline_mode": p["neckline_mode"],
            "signal_idx": i, "date": p["signal_date"], "close": p["signal_close"],
            "false_breakout_10d": _false_breakout_flags(close, i, p["neckline_price"], horizon=10),
        }
        for n in FORWARD_WINDOWS:
            base_row[f"fwd_ret_{n}d"] = fwd[f"fwd_ret_{n}d"].iloc[i]
            base_row[f"mae_{n}d"] = mae[n].iloc[i]

        w0_row = dict(base_row); w0_row["signal"] = "W0_型態本身"
        rows.append(w0_row)
        if w1:
            w1_row = dict(base_row); w1_row["signal"] = "W1_斜率上揚且MACD零軸上"
            rows.append(w1_row)
        if group is not None:
            g_row = dict(base_row); g_row["signal"] = group
            rows.append(g_row)

    return pd.DataFrame(rows)
