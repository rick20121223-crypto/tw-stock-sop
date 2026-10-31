"""
把 run_backtest.py 落地的事件明細CSV彙總成報告用的統計表。預設排除
near_unadjusted_gap==True的事件（還原權息涵蓋不到的跳空附近，統計會失真，
見price_gaps.csv／報告限制章節），可用include_gap_events=True保留。
"""

from typing import Optional

import numpy as np
import pandas as pd

SELL_WINDOWS = (5, 10, 20)
BUY_WINDOWS = (5, 10, 20)


def _clean(events: pd.DataFrame, include_gap_events: bool) -> pd.DataFrame:
    if events.empty:
        return events
    if not include_gap_events and "near_unadjusted_gap" in events.columns:
        events = events[~events["near_unadjusted_gap"]]
    return events


def summarize_sell(events: pd.DataFrame, by_list_tag: bool = False,
                    include_gap_events: bool = False) -> pd.DataFrame:
    events = _clean(events, include_gap_events)
    if events.empty:
        return pd.DataFrame()
    group_cols = ["signal"] + (["list_tag"] if by_list_tag else [])

    rows = []
    for key, g in events.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["樣本數"] = len(g)
        row["去重股票數"] = g["code"].nunique()
        row["平均回落幅度%(距前波高點)"] = round(g["pullback_from_peak_pct"].mean(), 2)
        row["平均距高點交易日"] = round(g["days_since_peak"].mean(), 1)
        head_flag = g["peak_is_real_head"].dropna()
        row["前波高點驗證為真頭部比例%"] = round(head_flag.mean() * 100, 1) if len(head_flag) else None
        hit = g["hit_20d"].dropna()
        row["命中率%(20日內回檔>=10%)"] = round(hit.mean() * 100, 1) if len(hit) else None
        false_sig = g["false_signal_20d"].dropna()
        row["假訊號率%(20日內創60日新高)"] = round(false_sig.mean() * 100, 1) if len(false_sig) else None
        for n in SELL_WINDOWS:
            ret = g[f"fwd_ret_{n}d"].dropna()
            mae = g[f"mae_{n}d"].dropna()
            row[f"{n}日_平均報酬%"] = round(ret.mean() * 100, 2) if len(ret) else None
            row[f"{n}日_中位數報酬%"] = round(ret.median() * 100, 2) if len(ret) else None
            row[f"{n}日_平均最大不利波動%"] = round(mae.mean() * 100, 2) if len(mae) else None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_regime(events: pd.DataFrame, min_sample: int = 30) -> pd.DataFrame:
    """events需含signal、regime、fwd_ret_5d/10d/20d欄位，excess_ret_5d/10d/20d
    欄位若存在就一併統計（沒有就跳過那幾欄）。依(signal,regime)分組，回傳
    樣本數/報酬(均值中位數)/勝率/超額報酬(均值中位數)/超額勝率。"""
    if events.empty or "regime" not in events.columns:
        return pd.DataFrame()
    events = events.dropna(subset=["regime"])
    has_excess = "excess_ret_20d" in events.columns
    rows = []
    for (signal, regime), g in events.groupby(["signal", "regime"]):
        n = len(g)
        row = {"signal": signal, "regime": regime, "樣本數": n,
               "樣本不足": "是" if n < min_sample else ""}
        for w in (5, 10, 20):
            ret = g[f"fwd_ret_{w}d"].dropna()
            row[f"{w}日_平均報酬%"] = round(ret.mean() * 100, 2) if len(ret) else None
            row[f"{w}日_中位數報酬%"] = round(ret.median() * 100, 2) if len(ret) else None
            row[f"{w}日_勝率%"] = round((ret > 0).mean() * 100, 1) if len(ret) else None
            if has_excess:
                exc = g[f"excess_ret_{w}d"].dropna()
                row[f"{w}日_平均超額報酬%"] = round(exc.mean() * 100, 2) if len(exc) else None
                row[f"{w}日_超額勝率%"] = round((exc > 0).mean() * 100, 1) if len(exc) else None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_w_bottom(events: pd.DataFrame, by_list_tag: bool = False,
                        include_gap_events: bool = False,
                        min_sample: int = 30) -> pd.DataFrame:
    """W0~W4分組統計：樣本數、去重股票數、5/10/20日報酬(均值/中位數)、勝率、
    平均MAE、假突破率(10日內跌破頸線的比例)。樣本數<min_sample的組別標註
    「樣本不足」，不得下結論（使用者2026-10-17規格明確要求）。"""
    events = _clean(events, include_gap_events)
    if events.empty:
        return pd.DataFrame()
    group_cols = ["signal"] + (["list_tag"] if by_list_tag else [])

    rows = []
    for key, g in events.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        n = len(g)
        row["樣本數"] = n
        row["樣本不足"] = "是" if n < min_sample else ""
        row["去重股票數"] = g["code"].nunique()
        fb = g["false_breakout_10d"].dropna()
        row["假突破率%(10日內跌破頸線)"] = round(fb.mean() * 100, 1) if len(fb) else None
        for w in (5, 10, 20):
            ret = g[f"fwd_ret_{w}d"].dropna()
            mae = g[f"mae_{w}d"].dropna()
            row[f"{w}日_平均報酬%"] = round(ret.mean() * 100, 2) if len(ret) else None
            row[f"{w}日_中位數報酬%"] = round(ret.median() * 100, 2) if len(ret) else None
            row[f"{w}日_勝率%(正報酬)"] = round((ret > 0).mean() * 100, 1) if len(ret) else None
            row[f"{w}日_平均最大不利波動%"] = round(mae.mean() * 100, 2) if len(mae) else None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_buy(events: pd.DataFrame, by_list_tag: bool = False,
                   include_gap_events: bool = False) -> pd.DataFrame:
    events = _clean(events, include_gap_events)
    if events.empty:
        return pd.DataFrame()
    group_cols = ["signal"] + (["list_tag"] if by_list_tag else [])

    rows = []
    for key, g in events.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["樣本數"] = len(g)
        row["去重股票數"] = g["code"].nunique()
        for n in BUY_WINDOWS:
            ret = g[f"fwd_ret_{n}d"].dropna()
            mae = g[f"mae_{n}d"].dropna()
            row[f"{n}日_平均報酬%"] = round(ret.mean() * 100, 2) if len(ret) else None
            row[f"{n}日_中位數報酬%"] = round(ret.median() * 100, 2) if len(ret) else None
            row[f"{n}日_勝率%(正報酬)"] = round((ret > 0).mean() * 100, 1) if len(ret) else None
            row[f"{n}日_平均最大不利波動%"] = round(mae.mean() * 100, 2) if len(mae) else None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_lead_lag(lag_df: pd.DataFrame) -> pd.DataFrame:
    """lag_days: 正值代表「自己比對方晚出現」。pair格式"A_vs_B"時，
    lag=B發生列-A發生列，所以lag>0代表A領先B（A先出現），lag<0代表B領先A。"""
    if lag_df.empty:
        return pd.DataFrame()
    rows = []
    for pair, g in lag_df.groupby("pair"):
        lags = g["lag_days"]
        a_name, b_name = pair.split("_vs_")
        rows.append({
            "配對": pair,
            "配對數": len(lags),
            "去重股票數": g["code"].nunique(),
            "平均天數差(正=A先出現)": round(lags.mean(), 2),
            "中位數天數差": round(lags.median(), 1),
            f"{a_name}領先比例%": round((lags > 0).mean() * 100, 1),
            f"{b_name}領先比例%": round((lags < 0).mean() * 100, 1),
            "同日比例%": round((lags == 0).mean() * 100, 1),
        })
    return pd.DataFrame(rows)


DECOMPOSITION_COLS = ["組合", "signal", "樣本數", "平均距高點交易日",
                      "命中率%(20日內回檔>=10%)", "假訊號率%(20日內創60日新高)",
                      "20日_平均報酬%", "20日_中位數報酬%"]


def summarize_decomposition(sell_events: pd.DataFrame, combo_signals, labels) -> pd.DataFrame:
    """驗證3的A/B/C/D拆解表：combo_signals是訊號名稱list（依序對應labels），
    回傳只含這幾個訊號、依labels順序排列的彙總表，用來比較「多加一個條件」
    前後的命中率/假訊號率/20日報酬/平均距高點交易日變化。輸入事件表可以是
    全期間或切半後的子集，呼叫端自己決定要不要先篩日期。"""
    s = summarize_sell(sell_events)
    present = [c for c in combo_signals if c in set(s["signal"])]
    if not present:
        return pd.DataFrame()
    present_labels = [labels[combo_signals.index(c)] for c in present]
    out = s.set_index("signal").loc[present].reset_index()
    out.insert(0, "組合", present_labels)
    return out[DECOMPOSITION_COLS]


def summarize_head_coverage(sell_events: pd.DataFrame, heads_df: pd.DataFrame,
                             signals) -> pd.DataFrame:
    """以「真頭部」為單位的覆蓋率(召回率)/精確率分析（驗證2）：
    - 真頭部總數：heads_df裡這幾檔股票全部驗證為真的頭部個數（分母，含完全
      沒有任何訊號觸發的頭部）。
    - 觸發比例%(召回率)：這個訊號在多少比例的真頭部後來有至少觸發一次
      （用peak_idx比對，同一個頭部到下一個頭部之前都算同一組，不另外設
      時間窗，頭部到頭部之間本來就有天然邊界）。
    - 精確率%：這個訊號「所有」觸發事件裡，有多少比例的peak_is_real_head
      是True（分母是這個訊號的全部觸發次數，不限真頭部，直接用
      sell_events本身的peak_is_real_head欄位）。
    - 首次觸發距高點天數/回落%：對每個「有觸發」的真頭部，取該訊號第一次
      觸發（row_idx最小）那一筆的days_since_peak/pullback_from_peak_pct，
      再取平均/中位數。
    """
    n_heads_total = len(heads_df)
    rows = []
    for signal in signals:
        ev = sell_events[sell_events["signal"] == signal]
        precision_base = ev["peak_is_real_head"].dropna()
        precision = round((precision_base == True).mean() * 100, 1) if len(precision_base) else None  # noqa: E712

        if heads_df.empty or ev.empty:
            rows.append({"訊號": signal, "全部觸發次數": len(ev), "精確率%(真頭部觸發/全部觸發)": precision,
                         "真頭部總數": n_heads_total, "有觸發的真頭部數": 0, "觸發比例%(召回率)": 0.0 if n_heads_total else None,
                         "首次觸發平均距高點交易日": None, "首次觸發中位數距高點交易日": None,
                         "首次觸發平均距高點回落%": None, "首次觸發中位數距高點回落%": None})
            continue

        merged = heads_df.merge(ev, on=["code", "peak_idx"], how="left", suffixes=("", "_ev"))
        first_occ = merged.sort_values("row_idx").groupby(["code", "peak_idx"], as_index=False).first()
        hit_heads = first_occ[first_occ["row_idx"].notna()]
        n_hit = len(hit_heads)
        rows.append({
            "訊號": signal,
            "全部觸發次數": len(ev),
            "精確率%(真頭部觸發/全部觸發)": precision,
            "真頭部總數": n_heads_total,
            "有觸發的真頭部數": n_hit,
            "觸發比例%(召回率)": round(n_hit / n_heads_total * 100, 1) if n_heads_total else None,
            "首次觸發平均距高點交易日": round(hit_heads["days_since_peak"].mean(), 1) if n_hit else None,
            "首次觸發中位數距高點交易日": round(hit_heads["days_since_peak"].median(), 1) if n_hit else None,
            "首次觸發平均距高點回落%": round(hit_heads["pullback_from_peak_pct"].mean(), 2) if n_hit else None,
            "首次觸發中位數距高點回落%": round(hit_heads["pullback_from_peak_pct"].median(), 2) if n_hit else None,
        })
    return pd.DataFrame(rows)


def summarize_s5_layers(sell_events: pd.DataFrame) -> pd.DataFrame:
    """S5分層出場：量測第1層(KD365死叉)->第2層(MACD死叉)、
    第2層->第3層(三線齊或跌破BBI)的實際間隔交易日數（直接回答假設的順序問題，
    但這是「已經先出現KD365死叉」為前提的條件順序，跟summarize_lead_lag()
    的無條件最近配對分析互補，不能只看這個就下結論，見報告限制章節。"""
    ev = sell_events[sell_events["signal"].str.startswith("S5_", na=False)]
    if ev.empty:
        return pd.DataFrame()
    episodes = ev.drop_duplicates(subset=["code", "episode_l1_idx"])
    l2_gap = (episodes["episode_l2_idx"] - episodes["episode_l1_idx"]).dropna()
    l3_gap = (episodes["episode_l3_idx"] - episodes["episode_l2_idx"]).dropna()
    n_l1 = episodes["episode_l1_idx"].notna().sum()
    n_l2 = episodes["episode_l2_idx"].notna().sum()
    n_l3 = episodes["episode_l3_idx"].notna().sum()
    return pd.DataFrame([
        {"層級轉換": "第1層(KD365死叉)->第2層(MACD死叉)", "episode數": n_l1,
         "走到下一層比例%": round(n_l2 / n_l1 * 100, 1) if n_l1 else None,
         "平均間隔交易日": round(l2_gap.mean(), 2) if len(l2_gap) else None,
         "中位數間隔交易日": round(l2_gap.median(), 1) if len(l2_gap) else None},
        {"層級轉換": "第2層(MACD死叉)->第3層(三線齊或跌破BBI)", "episode數": n_l2,
         "走到下一層比例%": round(n_l3 / n_l2 * 100, 1) if n_l2 else None,
         "平均間隔交易日": round(l3_gap.mean(), 2) if len(l3_gap) else None,
         "中位數間隔交易日": round(l3_gap.median(), 1) if len(l3_gap) else None},
    ])
