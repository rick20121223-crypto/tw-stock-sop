"""
買進側事件定義 B1~B4。第一版只做「一般買進訊號」，W底型態判定不實作
（見使用者規格明確註記，報告需要再次提醒這個限制）。
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest.events import forward_mae, forward_returns
from backtest.indicators import crossed_up, crossed_up_recently, slope_classify

BUY_FORWARD_WINDOWS = (5, 10, 20)


@dataclass
class BuyParams:
    kd_recent_n: int = 3  # B2/B4「近3個交易日內剛金叉」
    ma35_slope_lookback: int = 3
    ma35_slope_threshold: float = 0.0
    bbi_slope_lookback: int = 3
    bbi_slope_threshold: float = 0.0


def _build_event_rows(df: pd.DataFrame, trigger_mask: pd.Series, signal_name: str,
                       aux: Dict[str, pd.Series]) -> pd.DataFrame:
    idx = np.where(trigger_mask.fillna(False).to_numpy())[0]
    if len(idx) == 0:
        return pd.DataFrame()
    rows = []
    for i in idx:
        row = {"signal": signal_name, "row_idx": i, "date": df["date"].iloc[i],
               "close": df["close"].iloc[i]}
        for n in BUY_FORWARD_WINDOWS:
            row[f"fwd_ret_{n}d"] = aux[f"fwd_ret_{n}d"].iloc[i]
            row[f"mae_{n}d"] = aux[f"mae_{n}d"].iloc[i]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_buy_signals(df_ind: pd.DataFrame, params: BuyParams = BuyParams()):
    df = df_ind.copy()
    fwd = forward_returns(df, windows=BUY_FORWARD_WINDOWS)
    aux = {}
    for n in BUY_FORWARD_WINDOWS:
        aux[f"fwd_ret_{n}d"] = fwd[f"fwd_ret_{n}d"]
        aux[f"mae_{n}d"] = forward_mae(df, n, direction="long")

    ma35_slope = slope_classify(df["MA35"], params.ma35_slope_lookback, params.ma35_slope_threshold)
    bbi_slope = slope_classify(df["BBI"], params.bbi_slope_lookback, params.bbi_slope_threshold)

    macd_golden = crossed_up(df["DIF"], df["MACD_signal"])
    b1 = macd_golden & (df["DIF"] > 0)

    b2 = crossed_up_recently(df["K365"], df["D365"], params.kd_recent_n)

    bbi_up_state = ((df["close"] > df["BBI"]) & (bbi_slope == "上揚")).astype(bool)
    b3 = bbi_up_state & (~bbi_up_state.shift(1).fillna(False).astype(bool))

    b4 = (ma35_slope == "上揚") & (df["DIF"] > 0) & b2

    all_events = []
    trigger_idx: Dict[str, List[int]] = {}
    for name, mask in [
        ("B1_MACD零軸上金叉", b1),
        ("B2_KD365剛金叉", b2),
        ("B3_站上BBI且BBI上揚", b3),
        ("B4_三者同步", b4),
    ]:
        ev = _build_event_rows(df, mask, name, aux)
        if not ev.empty:
            all_events.append(ev)
        trigger_idx[name] = np.where(mask.fillna(False).to_numpy())[0].tolist()

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    return events_df, trigger_idx
