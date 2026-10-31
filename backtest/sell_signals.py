"""
賣出側事件定義 S1~S5，見PRODUCT任務規格。所有訊號都只用「當日收盤」判斷，
不用未來資料（run_indicators()本身也只用rolling/ewm，天生不會看未來）。
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from backtest.events import (extract_real_heads, false_signal_flags, forward_mae,
                              forward_returns, hit_rate_flags, is_real_head, rolling_peak)
from backtest.indicators import crossed_down, crossed_down_recently, slope_classify


@dataclass
class SellParams:
    recent_n: int = 5            # 「近N日內發生過交叉」的N（S4b及V3的b版本共用）
    head_window: int = 60        # 前波高點回看天數
    head_pullback: float = 0.15  # 頭部定義：回檔幾%才算真頭部
    head_confirm_window: int = 20
    hit_horizon: int = 20
    hit_drawdown: float = 0.10
    false_signal_window: int = 60
    false_signal_horizon: int = 20
    ma35_slope_lookback: int = 3  # 35MA下彎定義：今日跟前幾日比較（1=前一日、3=前三日）
    ma35_slope_threshold: float = 0.0
    layer_search_window: int = 60  # S5每一層往後找下一層的最長交易日數


SELL_FORWARD_WINDOWS = (5, 10, 20)


def _build_event_rows(df: pd.DataFrame, trigger_mask: pd.Series, signal_name: str,
                       aux: Dict[str, pd.Series]) -> pd.DataFrame:
    idx = np.where(trigger_mask.fillna(False).to_numpy())[0]
    if len(idx) == 0:
        return pd.DataFrame()
    rows = []
    for i in idx:
        row = {
            "signal": signal_name,
            "row_idx": i,
            "date": df["date"].iloc[i],
            "close": df["close"].iloc[i],
            "peak_idx": df["peak_idx"].iloc[i],
            "peak_price": df["peak_price"].iloc[i],
            "days_since_peak": df["days_since_peak"].iloc[i],
            "pullback_from_peak_pct": df["pullback_from_peak_pct"].iloc[i],
            "peak_is_real_head": aux["peak_is_real_head"].iloc[i],
            "hit_20d": aux["hit"].iloc[i],
            "false_signal_20d": aux["false_signal"].iloc[i],
        }
        for n in SELL_FORWARD_WINDOWS:
            row[f"fwd_ret_{n}d"] = aux[f"fwd_ret_{n}d"].iloc[i]
            row[f"mae_{n}d"] = aux[f"mae_{n}d"].iloc[i]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_sell_signals(df_ind: pd.DataFrame, params: SellParams = SellParams()):
    """df_ind: 已經跑過 backtest.indicators.run_indicators() 的日線df。
    回傳 (events_df, trigger_idx, heads_df)：
      events_df：每一筆訊號觸發的事件列（含事後統計欄位）。
      trigger_idx：{signal_name: [row_idx,...]}，供跨訊號領先/落後天數分析用
      （只放「離散事件」S1/S2/S3a/S3b，S4/S5是前面幾個的組合，不重複放）。
      heads_df：這支股票全部「驗證為真頭部」的高點清單（見events.extract_real_heads），
      用來當驗證2精確率/召回率分析的分母，含沒有任何訊號觸發的真頭部。
    """
    df = rolling_peak(df_ind, window=params.head_window)
    peak_head_flags = is_real_head(df, params.head_pullback, params.head_confirm_window)
    # peak_head_flags 只在「高點所在那一列」有值，要往前填給高點之後、直到
    # 下一個高點之前的所有列共用（事件發生在高點之後，問的是「這波高點後來
    # 驗證是不是真頭部」）。
    peak_is_real_head = peak_head_flags.reindex(df.index).copy()
    last_val = None
    filled = []
    for v in peak_head_flags.to_numpy():
        if not (isinstance(v, float) and np.isnan(v)):
            last_val = v
        filled.append(last_val)
    peak_is_real_head = pd.Series(filled, index=df.index)

    fwd = forward_returns(df, windows=SELL_FORWARD_WINDOWS)
    aux = {
        "peak_is_real_head": peak_is_real_head,
        "hit": hit_rate_flags(df, params.hit_horizon, params.hit_drawdown),
        "false_signal": false_signal_flags(df, params.false_signal_window, params.false_signal_horizon),
    }
    for n in SELL_FORWARD_WINDOWS:
        aux[f"fwd_ret_{n}d"] = fwd[f"fwd_ret_{n}d"]
        aux[f"mae_{n}d"] = forward_mae(df, n, direction="short")

    macd_dead = crossed_down(df["DIF"], df["MACD_signal"])
    kd_dead = crossed_down(df["K365"], df["D365"])
    macd_dead_recent = crossed_down_recently(df["DIF"], df["MACD_signal"], params.recent_n)
    kd_dead_recent = crossed_down_recently(df["K365"], df["D365"], params.recent_n)
    ma35_slope = slope_classify(df["MA35"], params.ma35_slope_lookback, params.ma35_slope_threshold)
    bbi_break = crossed_down(df["close"], df["BBI"])
    # S3b：精確公式（今日BBI<前一日BBI，且前一日BBI>=前二日BBI，即「剛」
    # 轉下彎的第一天），依使用者2026-10-16給的明確定義，取代先前用
    # slope_classify(3日比較)的假設版本。
    bbi_turn_down = (df["BBI"] < df["BBI"].shift(1)) & (df["BBI"].shift(1) >= df["BBI"].shift(2))

    macd_state_dead = df["DIF"] < df["MACD_signal"]
    kd_state_dead = df["K365"] < df["D365"]
    ma35_down = (ma35_slope == "下彎")

    def _onset(state: pd.Series) -> pd.Series:
        state = state.astype(bool)
        return state & (~state.shift(1).fillna(False).astype(bool))

    # --- 驗證3的A/B/C/D拆解，(a)狀態版 / (b)近N日版 兩種死叉定義都跑 ---
    # A：MACD死叉單一條件；B：A+35MA下彎；C：A+KD365死叉(不含35MA)；
    # D：三者同時(=S4)。(a)版用「狀態onset」，(b)版用「近N日內發生過交叉」
    # （對單一條件A而言，(b)版本身就是「死叉後N日內的每一天」，跟(a)版的
    # 「死叉當天」天生不同，這點在報告裡要說明，不是計算錯誤）。
    a_a = macd_dead  # = S1，狀態版onset與離散交叉事件等價
    b_a = _onset((macd_state_dead & ma35_down).astype(bool))      # = S4x
    c_a = _onset((macd_state_dead & kd_state_dead).astype(bool))  # 新增：MACD+KD365，無35MA
    d_a = _onset((macd_state_dead & kd_state_dead & ma35_down).astype(bool))  # = S4a

    a_b = macd_dead_recent
    b_b = macd_dead_recent & ma35_down
    c_b = macd_dead_recent & kd_dead_recent                       # 新增：無35MA
    d_b = macd_dead_recent & kd_dead_recent & ma35_down           # = S4b

    all_events = []
    trigger_idx: Dict[str, List[int]] = {}
    for name, mask in [
        ("S1_MACD死叉", macd_dead),
        ("S2_KD365死叉", kd_dead),
        ("S3a_跌破BBI", bbi_break),
        ("S3b_BBI轉下彎", bbi_turn_down),
        ("S4a_三線同步狀態", d_a),
        ("S4b_近N日三線同步", d_b),
        ("S4x_MACD死叉狀態+35MA下彎(無KD365)", b_a),
        ("V3_Ca_MACD死叉狀態+KD365死叉狀態(無35MA)", c_a),
        ("V3_Ab_近N日MACD死叉(單一條件)", a_b),
        ("V3_Bb_近N日MACD死叉+35MA下彎", b_b),
        ("V3_Cb_近N日MACD死叉+KD365死叉(無35MA)", c_b),
    ]:
        ev = _build_event_rows(df, mask, name, aux)
        if not ev.empty:
            all_events.append(ev)
        trigger_idx[name] = np.where(mask.fillna(False).to_numpy())[0].tolist()

    # --- S5 分層出場：KD365死叉(第1層) -> 其後第一次MACD死叉(第2層) ->
    # 其後第一次(S4a狀態 或 跌破BBI)(第3層)。每個episode只往前搜尋一次，
    # 搜不到就停在那一層（代表沒有完整走完三層）。
    kd_idx = trigger_idx["S2_KD365死叉"]
    macd_idx_arr = np.array(trigger_idx["S1_MACD死叉"])
    layer3_mask = d_a | bbi_break
    layer3_idx_arr = np.where(layer3_mask.fillna(False).to_numpy())[0]

    s5_rows = []
    for l1 in kd_idx:
        cand2 = macd_idx_arr[(macd_idx_arr > l1) & (macd_idx_arr <= l1 + params.layer_search_window)]
        l2 = int(cand2.min()) if len(cand2) else None
        l3 = None
        if l2 is not None:
            cand3 = layer3_idx_arr[(layer3_idx_arr > l2) & (layer3_idx_arr <= l2 + params.layer_search_window)]
            l3 = int(cand3.min()) if len(cand3) else None
        for layer_name, idx in [("S5_第1層_KD365死叉", l1), ("S5_第2層_MACD死叉", l2), ("S5_第3層_三線齊或跌破BBI", l3)]:
            if idx is None:
                continue
            row = {
                "signal": layer_name, "row_idx": idx, "date": df["date"].iloc[idx],
                "close": df["close"].iloc[idx], "peak_idx": df["peak_idx"].iloc[idx],
                "peak_price": df["peak_price"].iloc[idx],
                "days_since_peak": df["days_since_peak"].iloc[idx],
                "pullback_from_peak_pct": df["pullback_from_peak_pct"].iloc[idx],
                "peak_is_real_head": peak_is_real_head.iloc[idx],
                "hit_20d": aux["hit"].iloc[idx], "false_signal_20d": aux["false_signal"].iloc[idx],
                "episode_l1_idx": l1, "episode_l2_idx": l2, "episode_l3_idx": l3,
            }
            for n in SELL_FORWARD_WINDOWS:
                row[f"fwd_ret_{n}d"] = aux[f"fwd_ret_{n}d"].iloc[idx]
                row[f"mae_{n}d"] = aux[f"mae_{n}d"].iloc[idx]
            s5_rows.append(row)
    if s5_rows:
        all_events.append(pd.DataFrame(s5_rows))

    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    heads_df = extract_real_heads(df, peak_is_real_head)
    return events_df, trigger_idx, heads_df
