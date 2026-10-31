"""
但丁老師 SOP 自動判讀模組
--------------------------------
依「四關價 → 均線(含斜率) → MACD → OBV/BBI(/MTM) → 左側平台」的位階順序
逐層檢查，高位階訊號可以否決低位階訊號（尤其：生死線下彎時，任何買訊都
要降級；真正的頂背離必須「MACD背離 + OBV/BBI量價背離」同步出現才算數）。

所有條件與門檻值都放在同目錄的 sop_rules.yaml，改規則只需要改那個檔案，
不用動這裡的程式碼（見 _load_rules() / RULES）。

單一週期判讀：evaluate_timeframe()
跨週期整合（日/週/60分/5分）：combine_timeframes()，採「為日線留倉，
為五分出場」— 長天期結構轉弱可以否決短天期的反彈買訊；短天期轉弱不會
推翻長天期多頭結構，但會提示先幫短打部位出場。

純規則式評分，不做任何預測或保證，僅供輔助判讀。
被 streamlit_app.py / pages/*.py / multi_timeframe_check.py 匯入使用：
    from sop_decision import evaluate_timeframe, combine_timeframes, classify_final
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from stock_core import (
    FAST_MA,
    HALF_YEAR_MA,
    KEY_MA,
    ma_slope,
    normalize_timeframe,
)

_RULES_PATH = Path(__file__).with_name("sop_rules.yaml")


def _load_rules() -> dict:
    with open(_RULES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


RULES = _load_rules()


@dataclass
class Verdict:
    conclusion: str                 # 買進 / 加碼 / 賣出減碼 / 觀望
    confidence: str                 # 高 / 中 / 低
    score: float
    reasons: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    volume_breakout_confirmed: bool = False  # 三角收斂帶量突破確認（見_detect_triangle_consolidation）
    suggested_exposure: Optional[int] = None  # 100/70/50/30/0，None=無法明確分類（見exposure_rules）


# ------------------------------------------------------------------
# Step 1：四關價（最高位階，僅日線概念適用；其餘週期略過）
# ------------------------------------------------------------------
def _step1_four_key_prices(tf: str, latest: pd.Series,
                            reasons: List[str], caveats: List[str]) -> int:
    """回傳 -1（弱勢空方表態）/ 0（中性或不適用）/ +1（偏多）"""
    cfg = RULES["indicators"]["four_key_prices"]
    if tf not in cfg.get("applicable_timeframes", ["日"]):
        caveats.append("四關價是日線概念，本週期略過")
        return 0
    if pd.isna(latest.get("今開")) or pd.isna(latest.get("昨高")) or pd.isna(latest.get("昨低")):
        caveats.append("四關價資料不足，此步驟略過")
        return 0

    if latest["今開"] < latest["昨低"]:
        reasons.append("四關價：今開跌破昨低，格局判定為弱勢空方表態")
        return -1
    if latest["今開"] > latest["昨高"]:
        reasons.append("四關價：今開站上昨高，格局偏多")
        return 1

    # 今開沒跌破昨低，但盤中走弱、今收已經跌破昨低（長黑棒）——原規則
    # 只看今開，這種情況完全沒反應。只提示，不改變四關價分數。
    if cfg.get("close_below_low_caveat", False) and pd.notna(latest.get("close")):
        if latest["close"] < latest["昨低"]:
            caveats.append(
                f"⚠️ 今開雖未跌破昨低，但今收（{latest['close']:.2f}）已跌破昨低"
                f"（{latest['昨低']:.2f}），盤中走弱力道不小，留意隔日是否延續轉弱"
                "（僅提示，不計入四關價分數）"
            )
    return 0


# ------------------------------------------------------------------
# Step 2：均線（方向比價位重要；生死線下彎＋跌破＝硬否決買訊）
# ------------------------------------------------------------------
def _step2_ma(df: pd.DataFrame, tf: str, latest: pd.Series,
              reasons: List[str], caveats: List[str]) -> tuple:
    """回傳 (ma_bias: float, key_ma_down_veto: bool)"""
    cfg = RULES["indicators"]["ma_step"]
    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
    ma_bias = 0.0
    key_ma_down_veto = False

    fast_col, key_col = FAST_MA[tf], KEY_MA[tf]

    if fast_col in latest.index and pd.notna(latest.get(fast_col)):
        fast_slope = ma_slope(df, fast_col, lookback=slope_lookback)
        above_fast = latest["close"] >= latest[fast_col]
        if above_fast and fast_slope == "上揚":
            ma_bias += cfg["fast_up_bonus"]
            reasons.append(f"{fast_col} 上揚且股價站上，短線動能偏多")
            # 日線口訣：站穩5MA只是第一關，能同時挑戰/站上10MA才是止跌
            # 表態更明確的第二關，額外加分（僅日線適用，10MA只在日線計算）。
            second_gate_ma = cfg.get("second_gate_ma")
            if tf == "日" and second_gate_ma and pd.notna(latest.get(second_gate_ma)) \
                    and latest["close"] >= latest[second_gate_ma]:
                ma_bias += cfg["second_gate_bonus"]
                reasons.append(f"同時站上{second_gate_ma}，短線止跌表態更明確")
        elif (not above_fast) and fast_slope == "下彎":
            ma_bias += cfg["fast_down_penalty"]
            reasons.append(f"{fast_col} 下彎且股價跌破，短線動能偏空")
        elif (not above_fast) and fast_slope == "上揚" and cfg.get("fast_cross_under_caveat", False):
            # 股價已經跌破快均線，但快均線斜率還沒反應過來、仍判定上揚
            # ——原規則兩個分支都接不到，完全沒提示。只加caveat，不計分。
            caveats.append(
                f"⚠️ 股價已跌破{fast_col}，但{fast_col}斜率尚未轉下彎（仍上揚），"
                "短線動能出現轉弱跡象，留意是否持續走弱、轉成真正下彎（僅提示，不計分）"
            )
    else:
        caveats.append(f"{fast_col} 資料不足")

    if key_col in latest.index and pd.notna(latest.get(key_col)):
        key_slope = ma_slope(df, key_col, lookback=slope_lookback)
        above_key = latest["close"] >= latest[key_col]
        if key_slope == "下彎":
            ma_bias += cfg["key_down_penalty"]
            reasons.append(f"⚠️ 生死線 {key_col} 已下彎，任何買訊都要降級")
            if not above_key:
                key_ma_down_veto = True
                reasons.append(f"且股價已跌破 {key_col}，結構偏空，反彈視為誘多假動作")
        elif key_slope == "上揚":
            if above_key:
                ma_bias += cfg["key_up_bonus"]
                reasons.append(f"生死線 {key_col} 上揚且站上，中長線結構偏多")
            else:
                caveats.append(
                    f"{key_col} 仍上揚但股價暫時跌破，視為主力假跌破/夾起單洗盤，不判轉空"
                )
    else:
        caveats.append(f"{key_col} 資料不足（可能是資料期間不夠長）")

    return ma_bias, key_ma_down_veto


# ------------------------------------------------------------------
# Step 2 附則（僅週線適用）：長多回檔若落在週5MA～週20MA區間止跌打底，
# 視為解鎖長線佈局的買點。
# 簡化偵測：收盤落在 MA5/MA20 區間內，且最近幾根K棒的低點不再創新低
# （不是教科書式嚴謹的打底型態辨識，僅供參考）。
# ------------------------------------------------------------------
def _step2_weekly_pullback_zone(df: pd.DataFrame, tf: str, latest: pd.Series,
                                 reasons: List[str]) -> float:
    if tf != "週":
        return 0.0

    cfg = RULES["indicators"]["weekly_pullback_zone"]
    ma5, ma20, close = latest.get("MA5"), latest.get("MA20"), latest.get("close")
    if pd.isna(ma5) or pd.isna(ma20) or pd.isna(close) or len(df) < cfg["min_bars"]:
        return 0.0

    lower, upper = min(ma5, ma20), max(ma5, ma20)
    if not (lower <= close <= upper):
        return 0.0

    recent_lows = df["min"].tail(cfg["recent_low_window"])
    if recent_lows.isna().any():
        return 0.0
    stabilizing = recent_lows.iloc[-1] >= recent_lows.iloc[:-1].min()
    if not stabilizing:
        return 0.0

    reasons.append("週線回檔落於週5MA～週20MA區間且止跌打底，屬解鎖長線佈局的買點")
    return cfg["bonus"]


# ------------------------------------------------------------------
# Step 2 附則（僅週線適用）：週KD金叉＋站穩週5MA/週35MA(上揚)，但丁老師
# 原始教材沒有這條，是2026-10用24個月、13檔核心持股歷史回測驗證後新增
# （10日後正報酬比例65.5%，優於無條件基準53.9%）。KD已由stock_core.
# calc_kd()對所有週期計算，這裡不用另外算。
# ------------------------------------------------------------------
def _step2_weekly_kd_golden_cross(df: pd.DataFrame, tf: str, latest: pd.Series,
                                   reasons: List[str]) -> float:
    if tf != "週" or len(df) < 2:
        return 0.0

    cfg = RULES["indicators"]["weekly_kd_golden_cross"]
    k, d = latest.get("K"), latest.get("D")
    prev = df.iloc[-2]
    prev_k, prev_d = prev.get("K"), prev.get("D")
    if pd.isna(k) or pd.isna(d) or pd.isna(prev_k) or pd.isna(prev_d):
        return 0.0
    golden_cross = prev_k <= prev_d and k > d
    if not golden_cross:
        return 0.0

    ma5, ma35, close = latest.get("MA5"), latest.get("MA35"), latest.get("close")
    if pd.isna(ma5) or pd.isna(ma35) or pd.isna(close):
        return 0.0
    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
    above_both = close >= ma5 and close >= ma35
    ma35_rising = ma_slope(df, "MA35", lookback=slope_lookback) == "上揚"

    if above_both and ma35_rising:
        reasons.append("週KD金叉，且同時站穩週5MA/週35MA(上揚)，中長線買點訊號增強")
        return cfg["bonus"]
    return 0.0


# ------------------------------------------------------------------
# Step 2 附則（僅60分線適用）：60分MA20+MA120(烏龜線)+MA240三線同步站上
# 且上揚，但丁老師原始教材沒有這條，是2026-10用lookback 350天、13檔核心
# 持股歷史回測驗證後新增（5根K棒後正報酬比例53.1%，優於無條件基準
# 50.3%）。MA120只為此規則額外計入TIMEFRAME_MA_PERIODS["60分"]，FAST_MA/
# KEY_MA仍維持MA20/MA240不變。
# ------------------------------------------------------------------
def _step2_60min_turtle_confirm(df: pd.DataFrame, tf: str, latest: pd.Series,
                                 reasons: List[str]) -> float:
    if tf != "60分":
        return 0.0

    cfg = RULES["indicators"]["min60_turtle_confirm"]
    close = latest.get("close")
    ma20, ma120, ma240 = latest.get("MA20"), latest.get("MA120"), latest.get("MA240")
    if pd.isna(close) or pd.isna(ma20) or pd.isna(ma120) or pd.isna(ma240):
        return 0.0

    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
    fast_up = close >= ma20 and ma_slope(df, "MA20", lookback=slope_lookback) == "上揚"
    turtle_up = close >= ma120 and ma_slope(df, "MA120", lookback=slope_lookback) == "上揚"
    key_up = close >= ma240 and ma_slope(df, "MA240", lookback=slope_lookback) == "上揚"

    if fast_up and turtle_up and key_up:
        reasons.append("60分MA20+MA120(烏龜線)+MA240三線同步站上且上揚，增強買進訊號")
        return cfg["bonus"]
    return 0.0


# ------------------------------------------------------------------
# Step 2 附則（僅60分線適用）：60分20MA~60MA多空預警區間，2026-10-21課程
# 新增規則（60分MA60是這次新增的均線，原本只有MA20/120/240）。跌破20MA
# 是第一預警（多單第一步撤離/減碼，或空單第一步試打）；跌破60MA是第二
# 預警（空單第二步加碼；若60MA走平/上揚，也可以是空單回補或多單尋求
# 轉折的防守點）。這是剛教的新規則，還沒有backtest驗證過實際預測力，
# 先只附加caveats提示，不計入ma_bias、不影響conclusion——等驗證過再
# 考慮正式納入評分（跟min60_turtle_confirm/weekly_kd_golden_cross走
# 同一套流程）。假設60分正常結構下MA20在MA60之上（跌破60MA通常也已經
# 跌破20MA），所以用elif只報比較嚴重的那一則，不重複兩則訊息。
# ------------------------------------------------------------------
def _step2_60min_warning_zone(df: pd.DataFrame, tf: str, latest: pd.Series,
                               caveats: List[str]) -> None:
    cfg = RULES["indicators"].get("min60_warning_zone", {})
    if tf != "60分" or not cfg.get("enabled", True):
        return

    close = latest.get("close")
    ma20, ma60 = latest.get("MA20"), latest.get("MA60")
    if pd.isna(close) or pd.isna(ma20) or pd.isna(ma60):
        return

    if close < ma60:
        slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
        ma60_slope = ma_slope(df, "MA60", lookback=slope_lookback)
        if ma60_slope in ("上揚", "走平"):
            caveats.append(
                "⚠️ 60分跌破60MA（多空預警區間下緣），但60MA走平/上揚："
                "依但丁老師2026-10-21課程規則，可視為空單回補或多單尋求轉折的防守點"
            )
        else:
            caveats.append(
                "⚠️ 60分跌破60MA（多空預警區間下緣）且60MA同步下彎："
                "依但丁老師2026-10-21課程規則，空單可執行第二步加碼"
            )
    elif close < ma20:
        caveats.append(
            "⚠️ 60分跌破20MA（多空預警區間上緣）：依但丁老師2026-10-21課程"
            "規則，多單宜進行第一步撤離/減碼，或視為空單第一步試打"
        )


# ------------------------------------------------------------------
# 簡化版頂背離偵測：近期股價創高，但 MACD 柱狀圖高點未同步創高。
# 這是簡化偵測，非教科書式嚴謹型態辨識，僅供參考。
# ------------------------------------------------------------------
def _detect_bearish_divergence(df: pd.DataFrame, lookback: int, column: str = "MACD_hist") -> bool:
    if len(df) < lookback or column not in df.columns:
        return False
    window = df.tail(lookback).reset_index(drop=True)
    half = lookback // 2
    prior, recent = window.iloc[:half], window.iloc[half:]
    if prior["close"].isna().all() or recent["close"].isna().all():
        return False

    prior_peak_idx = prior["close"].idxmax()
    recent_peak_idx = recent["close"].idxmax()
    prior_peak_close = prior.loc[prior_peak_idx, "close"]
    recent_peak_close = recent.loc[recent_peak_idx, "close"]
    prior_peak_val = prior.loc[prior_peak_idx, column]
    recent_peak_val = recent.loc[recent_peak_idx, column]

    if pd.isna(prior_peak_val) or pd.isna(recent_peak_val):
        return False

    return bool(recent_peak_close > prior_peak_close and recent_peak_val < prior_peak_val)


# ------------------------------------------------------------------
# Step 3：MACD（落後指標；背離僅供警戒，不單獨判賣）
# ------------------------------------------------------------------
def _step3_macd(df: pd.DataFrame, latest: pd.Series,
                 reasons: List[str], caveats: List[str]) -> tuple:
    """回傳 (macd_bias: float, dead_cross_confirmed: bool, bearish_divergence: bool)"""
    cfg = RULES["indicators"]["macd"]
    macd_bias = 0.0
    dead_cross = False

    if pd.isna(latest.get("DIF")) or pd.isna(latest.get("MACD_signal")):
        caveats.append("MACD 資料不足")
        return macd_bias, dead_cross, False

    if latest["DIF"] > 0:
        macd_bias += cfg["dif_above_zero_bonus"]
        reasons.append("MACD DIF 位於零軸之上")
    else:
        macd_bias += cfg["dif_below_zero_penalty"]
        reasons.append("MACD DIF 位於零軸之下")

    if len(df) >= 2:
        prev = df.iloc[-2]
        if pd.notna(prev.get("DIF")) and pd.notna(prev.get("MACD_signal")):
            golden = prev["DIF"] <= prev["MACD_signal"] and latest["DIF"] > latest["MACD_signal"]
            dead = prev["DIF"] >= prev["MACD_signal"] and latest["DIF"] < latest["MACD_signal"]
            if golden:
                if latest["DIF"] < 0:
                    macd_bias += cfg["golden_cross_below_zero_bonus"]
                    reasons.append("MACD 零軸下黃金交叉，僅屬弱勢反彈，不當作進場訊號")
                else:
                    macd_bias += cfg["golden_cross_above_zero_bonus"]
                    reasons.append("MACD 黃金交叉（零軸之上），動能轉強")
            elif dead:
                macd_bias += cfg["dead_cross_penalty"]
                dead_cross = True
                reasons.append("MACD 出現死亡交叉（DIF 下穿訊號線）")

    bearish_divergence = _detect_bearish_divergence(df, cfg["divergence_lookback"])
    if bearish_divergence:
        reasons.append("⚠️ MACD 出現頂背離跡象（股價創高、柱狀圖未同步創高），先提高警覺、收緊停利")

    # 還沒死叉時，原規則對「紅柱連續縮小」完全沒有反應，要等死叉才示警。
    # 這裡只加caveat提前提示動能正在衰退，不計分、不影響conclusion。
    fade_bars = cfg.get("momentum_fade_bars")
    if fade_bars and not dead_cross and "MACD_hist" in df.columns and len(df) >= fade_bars:
        hist_tail = df["MACD_hist"].tail(fade_bars)
        if hist_tail.notna().all() and hist_tail.iloc[-1] > 0 and hist_tail.is_monotonic_decreasing:
            caveats.append(
                f"⚠️ MACD 柱狀體已連續{fade_bars}天收斂（動能減弱中，尚未死叉），"
                "留意若持續收斂可能轉為死叉（僅提示，不計分）"
            )

    return macd_bias, dead_cross, bearish_divergence


# ------------------------------------------------------------------
# Step 3.5：KD(36,5,5) Double Check（但丁老師課程新增的第三代指標）
# 只適用日/60分/5分，呼叫端須在tf != "週"時才呼叫這個函式——週線維持
# 既有_step2_weekly_kd_golden_cross用的KD(9,3,3)，不套這組參數
# （見sop_rules.yaml kd365區塊註解）。這一輪刻意不讓kd365_bias進評分
# 公式、不讓kd365_cross_down/kd365_bearish_divergence否決conclusion，
# 只用於reasons/caveats文字與suggested_exposure的計算，等
# test_kd365_double_check.py回測驗證過才考慮讓它影響conclusion本身。
# ------------------------------------------------------------------
def _step_kd365(df: pd.DataFrame, tf: str, latest: pd.Series,
                 reasons: List[str], caveats: List[str]) -> tuple:
    """回傳 (kd365_bias: float, kd365_cross_down: bool, kd365_bearish_divergence: bool)"""
    cfg = RULES["indicators"]["kd365"]
    kd365_bias = 0.0
    kd365_cross_down = False

    if pd.isna(latest.get("K365")) or pd.isna(latest.get("D365")):
        caveats.append("KD(36,5,5) 資料不足")
        return kd365_bias, False, False

    k, d = latest["K365"], latest["D365"]

    if len(df) >= 2:
        prev = df.iloc[-2]
        if pd.notna(prev.get("K365")) and pd.notna(prev.get("D365")):
            golden = prev["K365"] <= prev["D365"] and k > d
            dead = prev["K365"] >= prev["D365"] and k < d
            if golden:
                kd365_bias += cfg["golden_cross_bonus"]
                reasons.append("KD(36,5,5) 出現黃金交叉")
            elif dead:
                kd365_bias += cfg["dead_cross_penalty"]
                kd365_cross_down = True
                reasons.append("KD(36,5,5) 出現死亡交叉")

    high_th = cfg["blunted_threshold_high"]
    low_th = cfg["blunted_threshold_low"]
    blunted_bars = cfg["blunted_min_bars"]
    if len(df) >= blunted_bars:
        recent_k = df["K365"].tail(blunted_bars)
        if recent_k.notna().all() and (recent_k > high_th).all():
            reasons.append(f"KD(36,5,5) 連續{blunted_bars}根K棒維持高檔鈍化（>{high_th}）")
        elif recent_k.notna().all() and (recent_k < low_th).all():
            reasons.append(f"KD(36,5,5) 連續{blunted_bars}根K棒維持低檔鈍化（<{low_th}）")

    kd365_bearish_divergence = _detect_bearish_divergence(
        df, cfg["divergence_lookback"], column="K365")
    if kd365_bearish_divergence:
        reasons.append("⚠️ KD(36,5,5) 出現頂背離跡象（股價創高、K值未同步創高），先提高警覺")

    return kd365_bias, kd365_cross_down, kd365_bearish_divergence


# ------------------------------------------------------------------
# Step 4：量能驗證（日/週/60分用 OBV，5分因OBV易鈍化改用BBI）
# ------------------------------------------------------------------
def _step4_volume(df: pd.DataFrame, tf: str, latest: pd.Series, macd_bearish_divergence: bool,
                   reasons: List[str], caveats: List[str]) -> tuple:
    """回傳 (volume_bias: float, volume_bear_confirm: bool)
    volume_bear_confirm=True 代表「MACD背離 + 量能同步走壞」，才是真正可信的頂背離。
    """
    cfg = RULES["indicators"]["volume"]
    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
    volume_bias = 0.0
    volume_bear_confirm = False

    if tf == "5分":
        if "BBI" in latest.index and pd.notna(latest.get("BBI")):
            bbi_slope = ma_slope(df, "BBI", lookback=slope_lookback)
            if bbi_slope == "上揚":
                volume_bias += cfg["rising_bonus"]
                reasons.append("BBI 上揚，短線多方結構延續")
            elif bbi_slope == "下彎":
                volume_bias += cfg["falling_penalty"]
                reasons.append("BBI 下彎")
                if macd_bearish_divergence:
                    volume_bear_confirm = True
                    reasons.append("MACD 頂背離 + BBI 同步走壞，5分線頂背離訊號較可信")
        else:
            caveats.append("BBI 資料不足")
        return volume_bias, volume_bear_confirm

    if pd.isna(latest.get("OBV")):
        caveats.append("此標的無成交量資料，OBV 量價驗證略過")
        return volume_bias, volume_bear_confirm

    obv_slope = ma_slope(df, "OBV_MA", lookback=slope_lookback)
    if obv_slope == "上揚":
        volume_bias += cfg["rising_bonus"]
        reasons.append("OBV 均線上揚，量能同步價格")
    elif obv_slope == "下彎":
        volume_bias += cfg["falling_penalty"]
        reasons.append("OBV 均線下彎，量能未同步價格")
        if macd_bearish_divergence:
            volume_bear_confirm = True
            reasons.append("MACD 頂背離 + OBV 量價背離同步出現，才是較可信的頂背離")

    return volume_bias, volume_bear_confirm


# ------------------------------------------------------------------
# Step 5：MTM（僅 60分線適用，領先示警，比均線更早示警主力撤退）
# ------------------------------------------------------------------
def _step5_mtm(df: pd.DataFrame, tf: str, latest: pd.Series, reasons: List[str]) -> bool:
    cfg = RULES["indicators"]["mtm"]
    if tf not in cfg.get("applicable_timeframes", ["60分"]) or pd.isna(latest.get("MTM")):
        return False

    confirm_bars = cfg.get("confirm_bars", 1)
    if confirm_bars <= 1:
        # 舊版行為（預設）：單根K棒翻負，或近lookback_bars根內曾死叉，就觸發。
        # 多抓一根當shift(1)的參照基準：先tail()再shift()的話，視窗裡第一根
        # 的「前一根」在視窗外看不到，永遠是NaN，等於視窗頭那根的死叉transition
        # 偵測不到、實際lookback少算一根。多抓lookback_bars+1根，讓視窗內
        # 真正要看的lookback_bars根都能拿到正確的前一根做比較。
        recent = df.tail(cfg["lookback_bars"] + 1)
        cross_down = ((recent["MTM"] < 0) & (recent["MTM"].shift(1) >= 0)).any()
        if cross_down or latest["MTM"] < 0:
            reasons.append("MTM 翻空/死叉，60分線短線出場訊號優先示警")
            return True
        return False

    # confirm_bars>1：改用「連續N根都翻負」才算確認翻空，取代舊版的單根
    # 即觸發邏輯——2026-10-07觀察到短期訊號反轉比例飆高（近期單週近7成
    # 異動幾天內又反轉回去），多半是MTM貼著零軸上下小幅震盪被單根雜訊
    # 放大成訊號，故加這個緩衝帶選項；confirm_bars=1時完全不影響舊行為。
    last_n = df["MTM"].tail(confirm_bars)
    confirmed_down = len(last_n) == confirm_bars and (last_n < 0).all()
    if confirmed_down:
        reasons.append(f"MTM 連續{confirm_bars}根翻空，60分線短線出場訊號優先示警（已排除單根雜訊）")
        return True
    return False


# ------------------------------------------------------------------
# Step 5b：CCI（第三代領先指標，與 MTM 並列參考，適用 60分/5分線）
# 二線輔助指標，不計入 score：突破±100後折返穿越視為轉折訊號，跟MTM同等級
# 觸發「觀望」示警，但不凌駕MTM/MACD/均線的既有位階（不能單獨判賣出減碼、
# 也不會因為折返站上-100就把結論升級為買進——SOP裡所有二線指標都只降級
# 不升級，真正要買進仍須四關價/均線/MACD/量能同步確認）。
# 極端值（|CCI|>200）只附加提示，不改變結論。
# ------------------------------------------------------------------
def _step5b_cci(df: pd.DataFrame, tf: str, latest: pd.Series, reasons: List[str],
                 caveats: List[str]) -> tuple:
    """回傳 (cci_exit_alert: bool, cci_bias: float)。
    cci_exit_alert 行為完全不變（只有熊市折返才觸發，二線輔助降級用）。
    cci_bias 是2026-10-31新增，供scoring.mode="weighted_48_20_20"使用：
    重用這裡本來就有算的bull_foldback/bear_foldback狀態，不是另外設計的
    新門檻（+1/-1/0），不計入既有的legacy score公式。
    """
    cfg = RULES["indicators"]["cci"]
    if tf not in cfg.get("applicable_timeframes", []) or pd.isna(latest.get("CCI")):
        return False, 0.0

    signal_threshold = cfg["signal_threshold"]
    extreme_threshold = cfg["extreme_threshold"]
    latest_cci = latest["CCI"]
    confirm_bars = cfg.get("confirm_bars", 1)

    if abs(latest_cci) > extreme_threshold:
        direction = "超買" if latest_cci > 0 else "超賣"
        caveats.append(
            f"CCI達極端值{latest_cci:.0f}（{direction}，突破±{extreme_threshold:.0f}），"
            "需特別關注，但非趨勢反轉保證"
        )

    recent = df.tail(cfg["lookback_bars"])
    prior = recent["CCI"].iloc[:-1]
    if prior.empty:
        return False, 0.0

    if confirm_bars <= 1:
        # 舊版行為（預設）：曾經站上/跌破門檻，這一根折返就立刻觸發
        bear_foldback = (prior >= signal_threshold).any() and latest_cci < signal_threshold
        bull_foldback = (prior <= -signal_threshold).any() and latest_cci > -signal_threshold
    else:
        # confirm_bars>1：改成「連續N根都已折返站穩」才算確認，取代單根
        # 即觸發——2026-10-08短期訊號雜訊反轉事件診斷（diagnose_short_term_
        # whipsaw.py）發現CCI示警出現在30.9%的雜訊反轉裡，僅次於Step6左側
        # 平台，推測是CCI貼著±100來回折返被單根雜訊放大；confirm_bars=1
        # 時完全不影響舊行為。
        was_above = (prior >= signal_threshold).any()
        was_below = (prior <= -signal_threshold).any()
        last_n = df["CCI"].tail(confirm_bars)
        confirmed_below = len(last_n) == confirm_bars and (last_n < signal_threshold).all()
        confirmed_above = len(last_n) == confirm_bars and (last_n > -signal_threshold).all()
        bear_foldback = was_above and confirmed_below
        bull_foldback = was_below and confirmed_above

    if bear_foldback:
        reasons.append(
            f"⚠️ CCI由+{signal_threshold:.0f}之上折返跌破+{signal_threshold:.0f}，短線動能轉弱疑慮，"
            "比照MTM列入領先指標示警（二線輔助，不凌駕均線/MACD）"
        )
        return True, -1.0

    if bull_foldback:
        caveats.append(
            f"CCI由-{signal_threshold:.0f}之下折返站上-{signal_threshold:.0f}，短線止跌訊號，"
            "但屬二線輔助指標，不單獨作為買進依據，仍須均線/MACD/支撐同步確認"
        )
        return False, 1.0

    return False, 0.0


# ------------------------------------------------------------------
# 2026-10-31新增：scoring.mode="weighted_48_20_20"用的正規化/加權計分，
# 對照1030錄音「程式單權重配比」（均線48%／四關價20%／指標組20%）。
# 純函式，不讀取RULES全域狀態，方便獨立測試（見test_scoring_weighted_
# 48_20_20.py）；預設scoring.mode="legacy"時完全不會呼叫到這裡。
# ------------------------------------------------------------------
def _normalize_bias(raw_bias: float, max_abs: float) -> float:
    """把任意範圍的bias正規化到[-1, 1]。max_abs<=0時回傳0，避免除以0。"""
    if not max_abs:
        return 0.0
    return max(-1.0, min(1.0, raw_bias / max_abs))


def _weighted_score_48_20_20(four_key_bias: float, ma_bias: float, macd_bias: float,
                              kd365_bias: float, cci_bias: float, tf: str, cfg: dict) -> float:
    """依均線48%／四關價20%／指標組20%計算score。指標組在CCI不適用的
    週期（週/日，見sop_rules.yaml indicators.cci.applicable_timeframes）
    只用MACD+KD平均，60分/5分才併入CCI做三項平均。
    """
    norm_ma = _normalize_bias(ma_bias, cfg["ma_bias_max_abs"])
    norm_four_key = _normalize_bias(four_key_bias, cfg["four_key_bias_max_abs"])
    norm_macd = _normalize_bias(macd_bias, cfg["macd_bias_max_abs"])
    norm_kd = _normalize_bias(kd365_bias, cfg["kd365_bias_max_abs"])

    if tf in RULES["indicators"]["cci"].get("applicable_timeframes", []):
        norm_cci = _normalize_bias(cci_bias, cfg["cci_bias_max_abs"])
        indicator_group = (norm_macd + norm_kd + norm_cci) / 3
    else:
        indicator_group = (norm_macd + norm_kd) / 2

    core = (norm_ma * cfg["ma_weight_pct"] + norm_four_key * cfg["four_key_weight_pct"]
            + indicator_group * cfg["indicator_group_weight_pct"])
    return core * cfg["scale"]


def _score_thresholds(mode: str, buy_cfg: dict, sell_cfg: dict, weighted_cfg: dict) -> tuple:
    """回傳(buy_threshold, sell_threshold)。weighted_48_20_20模式必須用它
    自己專屬的門檻（weighted_cfg["buy_threshold"]/["sell_threshold"]），
    不能沿用legacy的buy_rules/sell_rules.score_threshold(±3)——兩種公式
    尺度不同，直接共用門檻會讓weighted模式幾乎發不出買進訊號（見
    sop_rules.yaml該區塊註解的校準說明）。
    """
    if mode == "weighted_48_20_20":
        return weighted_cfg["buy_threshold"], weighted_cfg["sell_threshold"]
    return buy_cfg["score_threshold"], sell_cfg["score_threshold"]


# ------------------------------------------------------------------
# 把局部高/低擺動點依價位聚成群集：彼此價位相差在 tolerance 以內的算同一群
# （同一個「反覆被測試」的價位帶），只有群內點數 >= min_touches 才算「整理平台」
# ——這是真正的平台跟「單次孤立擺動點」的關鍵區別：但丁老師目視判讀平台時看的
# 是「這個價位有沒有反覆測試過」，不是「歷史上離現價最近的隨便一個擺動點」
# （後者可能是一年前一根孤立的插針，跟現在的價格結構毫無關係）。
# ------------------------------------------------------------------
def _find_pivot_points(window: pd.DataFrame, pivot_window: int) -> tuple:
    """在 window（需已 reset_index）裡找局部高/低轉折點，回傳 (pivot_highs, pivot_lows)，
    兩者皆為以「window 內位置」為 index 的 pd.Series。中心點兩側資料不足的位置
    （包含最新幾根K棒）不會被 rolling(center=True) 標記為轉折點，天然排除
    「還在形成中、尚未確認的擺動」。"""
    roll_max = window["max"].rolling(window=pivot_window, center=True).max()
    roll_min = window["min"].rolling(window=pivot_window, center=True).min()
    pivot_highs = window.loc[window["max"] == roll_max, "max"]
    pivot_lows = window.loc[window["min"] == roll_min, "min"]
    return pivot_highs, pivot_lows


def _merge_adjacent_pivots(pivots: pd.Series, extreme: str) -> pd.Series:
    """走平的高/低點常常連續好幾根K棒都被 rolling(center=True) 標記為轉折點
    （例如波段最低點附近有兩根K棒同低），這些相鄰的標記其實是同一個擺動，
    只保留其中最極端的一個代表點，避免下游把同一個擺動誤判成兩個獨立轉折。"""
    if pivots.empty:
        return pivots
    positions = pivots.index.tolist()
    groups = [[positions[0]]]
    for pos in positions[1:]:
        if pos - groups[-1][-1] <= 1:
            groups[-1].append(pos)
        else:
            groups.append([pos])
    merged = {}
    for group in groups:
        values = pivots.loc[group]
        rep = values.idxmin() if extreme == "min" else values.idxmax()
        merged[rep] = values.loc[rep]
    return pd.Series(merged).sort_index()


def _cluster_pivot_levels(pivots: pd.Series, tolerance: float, min_touches: int) -> List[float]:
    values = sorted(v for v in pivots.dropna().tolist())
    if not values:
        return []
    clusters: List[List[float]] = [[values[0]]]
    for v in values[1:]:
        cluster_mean = sum(clusters[-1]) / len(clusters[-1])
        if cluster_mean and abs(v - cluster_mean) / cluster_mean <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters if len(c) >= min_touches]


# ------------------------------------------------------------------
# 型態學輔助觸發層（pattern_recognition，見 sop_rules.yaml 同名區塊註解）
# ⚠️ 以下三個偵測函式的輸出永遠不直接產生買賣結論：
#   - W底/M頭：只附加「注意轉折」提醒到 reasons，是否「轉折訊號確認」還是
#     「型態轉折待確認」純粹是措辭差異，最終買賣結論仍由 evaluate_timeframe()
#     既有的四關價/均線/MACD/OBV分層否決制決定；偵測到的頸線位置回傳給
#     呼叫端，併入 Step6 左側平台支撐/壓力的候選來源之一。
#   - 三角收斂：回傳目前狀態供 evaluate_timeframe() 決定是否要把買進/加碼
#     強制降級為觀望（見 watch_rules.triangle_consolidation_downgrade）。
# ------------------------------------------------------------------
def _kd365_crossed_up_recently(df: pd.DataFrame, window: int) -> bool:
    """檢查K365是否在最近window個交易日內（含今天）黃金交叉D365（前一天
    K<=D、當天K>D）。用來判斷W底突破當下KD365是不是「剛好同步轉強」，
    不是「已經維持多頭一陣子」——2026-10-13 backtest(test_kd365_w_bottom_
    buy.py)驗證過這個「剛交叉」版本才有實際預測力，「目前狀態」版本
    （K365>D365現在是不是成立）沒有證據支持，所以這裡故意不是單純比較
    latest一筆的K365/D365（見sop_rules.yaml kd365.w_bottom_fresh_cross_
    window區塊註解）。"""
    if len(df) < 2:
        return False
    start = max(1, len(df) - window)
    for i in range(start, len(df)):
        prev, cur = df.iloc[i - 1], df.iloc[i]
        pk, pd_ = prev.get("K365"), prev.get("D365")
        ck, cd = cur.get("K365"), cur.get("D365")
        if pd.notna(pk) and pd.notna(pd_) and pd.notna(ck) and pd.notna(cd):
            if pk <= pd_ and ck > cd:
                return True
    return False
def _detect_w_bottom(df: pd.DataFrame, fast_slope_up: bool, macd_bias: float,
                      kd365_bullish: bool, reasons: List[str]) -> Optional[Tuple[float, bool]]:
    cfg = RULES.get("pattern_recognition", {}).get("w_bottom", {})
    if not cfg.get("enabled", True):
        return None

    pivot_window = cfg["pivot_window"]
    lookback = cfg.get("lookback_bars")
    tolerance = cfg["neckline_tolerance_pct"]

    window = (df.tail(lookback) if lookback else df).reset_index(drop=True)
    if len(window) < pivot_window * 2 + 3:
        return None

    pivot_highs, pivot_lows = _find_pivot_points(window, pivot_window)
    pivot_lows = _merge_adjacent_pivots(pivot_lows, "min")
    if len(pivot_lows) < 2:
        return None

    low_positions = pivot_lows.index.tolist()
    left_pos, right_pos = low_positions[-2], low_positions[-1]
    left_low, right_low = pivot_lows.loc[left_pos], pivot_lows.loc[right_pos]
    if not (right_low > left_low):  # 右底比左底高才算數
        return None

    between_highs = pivot_highs[(pivot_highs.index > left_pos) & (pivot_highs.index < right_pos)]
    neckline = (between_highs.max() if not between_highs.empty
                else window.loc[left_pos:right_pos, "max"].max())
    if pd.isna(neckline) or neckline <= right_low:
        return None

    after_right = window.loc[right_pos:]
    broke_above = after_right[after_right["close"] > neckline]
    if broke_above.empty:
        return None  # 尚未突破頸線，型態還沒走完，先不觸發提醒

    # 只檢查「突破頸線之後」的拉回有沒有跌破頸線——右底本身收盤本來就在頸線
    # 之下（那正是它被判定為底部的原因），不能拿右底自己去判斷「回測不破」。
    since_breakout = window.loc[broke_above.index[0]:]
    if since_breakout["close"].min() < neckline * (1 - tolerance):
        return None  # 回測時已經跌破頸線，型態失敗

    latest = window.iloc[-1]
    if not (pd.notna(latest.get("open")) and pd.notna(latest.get("close"))
            and latest["close"] > latest["open"]):
        return None  # 還沒出現回測不破後的確認紅K，先不觸發提醒

    synced = fast_slope_up and macd_bias > 0 and kd365_bullish
    if synced:
        reasons.append(
            f"📐 偵測到W底型態（左底約{left_low:.2f}／右底約{right_low:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收紅K，且均線斜率、MACD、KD(36,5,5)三者同步轉多，轉折訊號確認"
            "（輔助提醒，結論仍以四關價/均線/MACD/OBV分層否決制為準）"
        )
    else:
        reasons.append(
            f"📐 偵測到W底型態（左底約{left_low:.2f}／右底約{right_low:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收紅K，但均線斜率/MACD/KD(36,5,5)尚未同步轉多，標記為「型態轉折待確認」"
            "（僅供觀察，非買進依據）"
        )
    return neckline, synced


def _detect_m_top(df: pd.DataFrame, fast_slope_down: bool, macd_bias: float,
                   reasons: List[str]) -> Optional[float]:
    cfg = RULES.get("pattern_recognition", {}).get("m_top", {})
    if not cfg.get("enabled", True):
        return None

    pivot_window = cfg["pivot_window"]
    lookback = cfg.get("lookback_bars")
    tolerance = cfg["neckline_tolerance_pct"]

    window = (df.tail(lookback) if lookback else df).reset_index(drop=True)
    if len(window) < pivot_window * 2 + 3:
        return None

    pivot_highs, pivot_lows = _find_pivot_points(window, pivot_window)
    pivot_highs = _merge_adjacent_pivots(pivot_highs, "max")
    if len(pivot_highs) < 2:
        return None

    high_positions = pivot_highs.index.tolist()
    left_pos, right_pos = high_positions[-2], high_positions[-1]
    left_high, right_high = pivot_highs.loc[left_pos], pivot_highs.loc[right_pos]
    if not (right_high < left_high):  # 右頭比左頭低才算數
        return None

    between_lows = pivot_lows[(pivot_lows.index > left_pos) & (pivot_lows.index < right_pos)]
    neckline = (between_lows.min() if not between_lows.empty
                else window.loc[left_pos:right_pos, "min"].min())
    if pd.isna(neckline) or neckline >= right_high:
        return None

    after_right = window.loc[right_pos:]
    broke_below = after_right[after_right["close"] < neckline]
    if broke_below.empty:
        return None  # 尚未跌破頸線，型態還沒走完，先不觸發提醒

    # 只檢查「跌破頸線之後」的反彈有沒有站回頸線之上——右頭本身收盤本來就在
    # 頸線之上（那正是它被判定為頭部的原因），不能拿右頭自己去判斷「回測不破」。
    since_breakdown = window.loc[broke_below.index[0]:]
    if since_breakdown["close"].max() > neckline * (1 + tolerance):
        return None  # 回測時又站回頸線之上，型態失敗

    latest = window.iloc[-1]
    if not (pd.notna(latest.get("open")) and pd.notna(latest.get("close"))
            and latest["close"] < latest["open"]):
        return None  # 還沒出現回測不破後的確認黑K，先不觸發提醒

    synced = fast_slope_down and macd_bias < 0
    if synced:
        reasons.append(
            f"📐 偵測到M頭型態（左頭約{left_high:.2f}／右頭約{right_high:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收黑K，且均線斜率與MACD同步轉空，轉折訊號確認"
            "（輔助提醒，結論仍以四關價/均線/MACD/OBV分層否決制為準）"
        )
    else:
        reasons.append(
            f"📐 偵測到M頭型態（左頭約{left_high:.2f}／右頭約{right_high:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收黑K，但均線斜率/MACD尚未同步轉空，標記為「型態轉折待確認」"
            "（僅供觀察，非賣出依據）"
        )
    return neckline


def _detect_triangle_consolidation(df: pd.DataFrame, tf: str,
                                    reasons: List[str], caveats: List[str]) -> dict:
    """回傳 {"active": bool, "breakout": None|"up"|"down", "volume_confirmed": bool}。
    active=True 且 volume_confirmed=False 時（不論是還沒突破、或突破了但量能未同步
    的假突破），呼叫端應強制降級觀望；volume_confirmed=True 時解除強制觀望，改由
    既有六步SOP分數/否決邏輯正常決定結論。"""
    cfg = RULES.get("pattern_recognition", {}).get("triangle_consolidation", {})
    inactive = {"active": False, "breakout": None, "volume_confirmed": False}
    if not cfg.get("enabled", True):
        return inactive

    pivot_window = cfg["pivot_window"]
    lookback = cfg.get("lookback_bars")
    min_pivots = cfg.get("min_pivots", 2)
    breakout_tolerance = cfg["breakout_tolerance_pct"]
    range_tolerance = cfg.get("range_tolerance_pct", 0.01)
    compression_ratio = cfg.get("compression_ratio", 0.7)

    window = (df.tail(lookback) if lookback else df).reset_index(drop=True)
    if len(window) < pivot_window * 2 + 3:
        return inactive

    pivot_highs, pivot_lows = _find_pivot_points(window, pivot_window)
    pivot_highs = _merge_adjacent_pivots(pivot_highs, "max")
    pivot_lows = _merge_adjacent_pivots(pivot_lows, "min")
    if len(pivot_highs) < min_pivots or len(pivot_lows) < min_pivots:
        return inactive

    # 收斂區間的邊界（range_high/range_low）要排除最新一根K棒才能算，不然
    # 一旦最新這根K棒帶量突破，它自己的高/低價會被算進「近半段」邊界裡，
    # 邊界跟著突破一起膨脹，永遠不可能判定為「突破」——這跟W底/M頭不能拿
    # 右底/右頭自己判斷「回測不破」是同一類問題。
    body = window.iloc[:-1]
    if len(body) < pivot_window * 2 + 2:
        return inactive

    # 只比較窗口「頭尾兩個轉折點」會漏抓中間曾經走出去（創出更高/更低極值）
    # 又縮回來的情況——那是區間中段暴衝過，不是真正的收斂。改成比較「前半段
    # vs 後半段」的高低點與區間寬度：後半段不能創新高/新低，而且區間寬度要
    # 收斂到前半段的 compression_ratio 以下，才算真正「收斂」。
    mid = len(body) // 2
    earlier_high, earlier_low = body.iloc[:mid]["max"].max(), body.iloc[:mid]["min"].min()
    recent_high, recent_low = body.iloc[mid:]["max"].max(), body.iloc[mid:]["min"].min()
    if pd.isna(earlier_high) or pd.isna(earlier_low) or pd.isna(recent_high) or pd.isna(recent_low):
        return inactive

    highs_not_rising = recent_high <= earlier_high * (1 + range_tolerance)  # 高點不再創高
    lows_not_falling = recent_low >= earlier_low * (1 - range_tolerance)    # 低點不再破低
    earlier_width = earlier_high - earlier_low
    recent_width = recent_high - recent_low
    converging = earlier_width > 0 and recent_width <= earlier_width * compression_ratio
    if not (highs_not_rising and lows_not_falling and converging):
        return inactive

    # 收斂區間的邊界要用「近半段（不含最新K棒）」的高低點，不能用整個回看
    # 窗口（含早半段那段還沒收斂、較寬）的絕對高低，否則就算近半段真的收斂
    # 了，回報出來的邊界跟突破判斷還是會被早半段的寬區間污染。另外「有沒有
    # 收斂」不能只看「比早半段窄」這個相對值，近半段本身相對股價的寬度還是
    # 要夠窄，否則會把「還是很寬、只是沒那麼寬」的區間誤判成三角收斂。
    range_high, range_low = recent_high, recent_low
    if pd.isna(range_high) or pd.isna(range_low) or range_high <= range_low:
        return inactive

    max_range_pct = cfg.get("max_range_pct", 0.20)
    if range_low <= 0 or (range_high - range_low) / range_low > max_range_pct:
        return inactive

    close = window.iloc[-1].get("close")
    breakout = None
    if pd.notna(close):
        if close > range_high * (1 + breakout_tolerance):
            breakout = "up"
        elif close < range_low * (1 - breakout_tolerance):
            breakout = "down"

    if breakout is None:
        caveats.append(
            f"📐 偵測到三角收斂區間（高點約{range_high:.2f}／低點約{range_low:.2f}，高點不再創高、"
            "低點不再破低），依SOP列為觀望、持續監控，等待帶量突破區間"
        )
        return {"active": True, "breakout": None, "volume_confirmed": False}

    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
    use_bbi = tf == "5分"
    vol_col = "BBI" if use_bbi else "OBV_MA"
    vol_label = "BBI" if use_bbi else "OBV"
    vol_slope = ma_slope(df, vol_col, lookback=slope_lookback) if vol_col in df.columns else "資料不足"
    volume_confirmed = (
        (breakout == "up" and vol_slope == "上揚")
        or (breakout == "down" and vol_slope == "下彎")
    )
    direction_label = "向上" if breakout == "up" else "向下"
    boundary = range_high if breakout == "up" else range_low

    if volume_confirmed:
        reasons.append(
            f"📐 三角收斂區間{direction_label}突破（區間邊界約{boundary:.2f}），且{vol_label}同步"
            f"{'放大' if breakout == 'up' else '轉弱'}，確認帶量突破，解除觀望標記，"
            "改依六步SOP判讀邏輯正常決定結論"
        )
    else:
        caveats.append(
            f"📐 三角收斂區間看似{direction_label}突破（區間邊界約{boundary:.2f}），但{vol_label}"
            f"未同步{'放大' if breakout == 'up' else '轉弱'}，判定為假突破，暫維持觀望、持續監控"
        )

    return {"active": True, "breakout": breakout, "volume_confirmed": volume_confirmed}


# ------------------------------------------------------------------
# Step 6：左側平台支撐/壓力校正
# 偵測「最近一段歷史裡反覆測試同一價位」的整理箱體（見 _cluster_pivot_levels），
# 只看 platform_lookback_bars 這段最近的歷史，避免抓到很久以前、跟現在價格結構
# 無關的孤立擺動點。
# - 拉回測到左側平台支撐未破 + 短線指標（快均線）轉向 → 相對安全買點，加分
# - 反彈碰到左側平台壓力 + 今開無法過昨高（四關價否決）→ 應獲利了結，減分
# ------------------------------------------------------------------
def _find_platform_support_resistance(df: pd.DataFrame, tf: str, latest: pd.Series,
                                       w_bottom_neckline: Optional[float] = None,
                                       m_top_neckline: Optional[float] = None,
                                       ) -> Optional[Tuple[List[Tuple[float, str]], List[Tuple[float, str]]]]:
    """
    找出左側支撐/壓力候選水位，回傳 (support_levels, resistance_levels)，
    每個元素是 (價位, 來源說明) tuple；資料不足時回傳 None。從
    _step6_left_side_platform()抽出來的共用邏輯，_detect_platform_second_leg()
    (2026-10-23新增的「兩次腳未破」廣義化版本)也需要同一份候選水位，
    避免兩邊各自重算一次pivot。
    """
    cfg = RULES["indicators"]["left_side_platform"]
    exclude_recent = cfg["exclude_recent_bars"]
    pivot_window = cfg["pivot_window"]
    min_left_bars = cfg["min_left_bars"]
    tolerance = cfg["tolerance_pct"]
    lookback = cfg.get("platform_lookback_bars", {}).get(tf)
    min_touches = cfg.get("platform_min_touches", 2)

    if len(df) <= exclude_recent or pd.isna(latest.get("close")):
        return None

    available_left = df.iloc[:-exclude_recent]
    left = available_left.tail(lookback) if lookback else available_left
    if len(left) < min_left_bars:
        return None

    left = left.reset_index(drop=True)
    pivot_highs, pivot_lows = _find_pivot_points(left, pivot_window)

    support_levels = [(s, "整理平台（近期多次觸碰確認）")
                       for s in _cluster_pivot_levels(pivot_lows, tolerance, min_touches)]
    resistance_levels = [(r, "整理平台（近期多次觸碰確認）")
                          for r in _cluster_pivot_levels(pivot_highs, tolerance, min_touches)]
    # 2026-10-20：「整理平台」要求同一價位至少觸碰platform_min_touches次，
    # 單次孤立的歷史明顯高/低點（陽明2609案例的前高67.3）會被濾掉、完全
    # 進不了候選清單。但「前高/前低」在但丁老師原始概念裡不要求反覆測試，
    # 這裡額外加回看窗口內最高的轉折高點／最低的轉折低點各一個，只需出現
    # 一次即可；若跟既有整理平台價位已經很接近（容忍帶內）就不重複加入。
    if cfg.get("single_touch_prior_level", {}).get("enabled", True):
        if not pivot_lows.empty:
            prior_low = pivot_lows.min()
            if not any(abs(prior_low - s) / s <= tolerance for s, _ in support_levels):
                support_levels.append((prior_low, "前低（單次觸及）"))
        if not pivot_highs.empty:
            prior_high = pivot_highs.max()
            if not any(abs(prior_high - r) / r <= tolerance for r, _ in resistance_levels):
                resistance_levels.append((prior_high, "前高（單次觸及）"))
    # 型態學輔助觸發層併入的頸線候選（見 pattern_recognition），與既有K棒
    # 整理平台並列使用，尤其在缺乏明顯歷史平台時作為補充依據。
    if w_bottom_neckline is not None:
        support_levels.append((w_bottom_neckline, "W底頸線"))
    if m_top_neckline is not None:
        resistance_levels.append((m_top_neckline, "M頭頸線"))

    return support_levels, resistance_levels


def _step6_left_side_platform(df: pd.DataFrame, tf: str, latest: pd.Series,
                               fast_slope_up: bool,
                               reasons: List[str], caveats: List[str],
                               w_bottom_neckline: Optional[float] = None,
                               m_top_neckline: Optional[float] = None,
                               ) -> Tuple[float, Optional[List[Tuple[float, str]]]]:
    """回傳 (bias, support_levels)——support_levels給呼叫端的
    _detect_platform_second_leg()重用，不用再重算一次pivot（資料不足時為
    None，呼叫端應該跳過兩次腳偵測）。"""
    cfg = RULES["indicators"]["left_side_platform"]
    levels = _find_platform_support_resistance(df, tf, latest, w_bottom_neckline, m_top_neckline)
    if levels is None:
        caveats.append("左側K棒資料不足，Step6左側平台校正略過")
        return 0.0, None
    support_levels, resistance_levels = levels

    close = latest["close"]
    confirm_bars = cfg.get("confirm_bars", 1)
    tolerance = cfg["tolerance_pct"]

    bias = 0.0
    candidate_supports = [(s, src) for s, src in support_levels if s <= close]
    if candidate_supports:
        support, support_src = max(candidate_supports, key=lambda item: item[0])
        if confirm_bars <= 1:
            dist = (close - support) / support if support else float("inf")
            near_support = 0 <= dist <= tolerance
        else:
            # 連續confirm_bars根收盤價都落在容忍帶內，才算「真的貼著支撐
            # 整理」，不是單根雜訊越過帶子邊緣——2026-10-08短期雜訊反轉
            # 診斷（diagnose_short_term_whipsaw.py）發現Step6出現在49.7%
            # 的雜訊反轉事件中，是目前最大宗來源：股價貼著支撐/壓力整理
            # 時，單根K棒的微小波動就會讓dist在容忍帶邊緣反覆跨進跨出，
            # 直接帶動bias/score來回翻動。confirm_bars=1時完全不影響舊行為。
            recent_closes = df["close"].tail(confirm_bars)
            if support and len(recent_closes) == confirm_bars:
                dists = (recent_closes - support) / support
                near_support = bool(((dists >= 0) & (dists <= tolerance)).all())
            else:
                near_support = False
        if near_support:
            if fast_slope_up:
                bias += cfg["support_bonus"]
                reasons.append(f"左側{support_src}支撐約 {support:.2f}附近拉回未破，且短線指標轉向，相對安全買點")
            else:
                caveats.append(f"股價貼近左側{support_src}支撐約 {support:.2f}，但短線指標尚未轉向，先觀察不急著進場")

    candidate_resistance = [(r, src) for r, src in resistance_levels if r >= close]
    if candidate_resistance:
        resistance, resistance_src = min(candidate_resistance, key=lambda item: item[0])
        today_high = latest.get("max", close)
        dist = (resistance - today_high) / resistance if resistance else float("inf")
        near_resistance = -tolerance <= dist <= tolerance  # 含今日已觸及/接近壓力區
        today_open, yesterday_high = latest.get("今開"), latest.get("昨高")
        failed_to_break_prev_high = (
            tf == "日" and pd.notna(today_open) and pd.notna(yesterday_high)
            and today_open < yesterday_high
        )
        if near_resistance and failed_to_break_prev_high:
            bias += cfg["resistance_penalty"]
            reasons.append(f"反彈碰到左側{resistance_src}壓力約 {resistance:.2f}，且今開未過昨高，宜獲利了結不凹單")
        elif near_resistance:
            caveats.append(f"股價接近左側{resistance_src}壓力區約 {resistance:.2f}，留意反彈受阻風險")

    return bias, support_levels


# ------------------------------------------------------------------
# 共用演算法：「兩次腳測試同一支撐、未破」偵測。window內「min <=
# reference*near_tolerance」的K棒視為測到支撐，依leg_split_rebound的
# 反彈幅度分段成不同「腳」；若最後一段夠新(recent_bars內)、且最後一腳
# 低點沒有跌破第一腳低點超過undercut_tolerance，視為「兩次腳未破」。
# reference可以是逐K棒變動的均線(pd.Series，例如MA120)，也可以是固定
# 價位（例如Step6找到的左側平台支撐）——供_detect_ma120_second_leg()
# 跟_detect_platform_second_leg()共用，2026-10-23從原本只給MA120用的
# _detect_ma120_second_leg()抽出來。
# ------------------------------------------------------------------
def _legs_untested_check(window: pd.DataFrame, reference: pd.Series,
                          leg_split_rebound: float, undercut_tolerance: float,
                          recent_bars: int, near_tolerance: float) -> bool:
    near_support = (window["min"] <= reference * near_tolerance) & pd.notna(reference)
    touch_idxs = window.index[near_support].tolist()
    if len(touch_idxs) < 2:
        return False

    legs = [[touch_idxs[0]]]
    for idx in touch_idxs[1:]:
        base_low = window.loc[legs[-1][-1], "min"]
        between_high = window.loc[legs[-1][-1]:idx, "max"].max()
        if pd.notna(between_high) and pd.notna(base_low) and between_high >= base_low * leg_split_rebound:
            legs.append([idx])
        else:
            legs[-1].append(idx)

    if len(legs) < 2:
        return False

    leg1_low = window.loc[legs[0], "min"].min()
    leg2_low = window.loc[legs[-1], "min"].min()
    latest_idx = window.index[-1]
    is_recent = (latest_idx - legs[-1][-1]) <= recent_bars
    undercut = pd.notna(leg1_low) and pd.notna(leg2_low) and leg2_low < leg1_low * undercut_tolerance

    return is_recent and not undercut


# ------------------------------------------------------------------
# 特殊情境：大跌測到半年線(MA120)打出「第二隻腳未破」→「帶著鋼盔慢買」
# 簡化偵測：近期K棒裡找出兩段「最低價貼近/跌破MA120」的區間（中間曾經
# 反彈拉開至少5%），若第二段的低點沒有明顯跌破第一段低點，視為第二隻腳
# 未破。僅日線適用（半年線＝120個交易日）。
# ------------------------------------------------------------------
def _detect_ma120_second_leg(df: pd.DataFrame, tf: str) -> bool:
    cfg = RULES["indicators"]["ma120_second_leg"]
    if tf not in cfg.get("applicable_timeframes", ["日"]):
        return False

    ma_col = HALF_YEAR_MA.get(tf)
    if ma_col is None or ma_col not in df.columns or len(df) < cfg["min_bars"]:
        return False

    window = df.tail(cfg["window_bars"]).reset_index(drop=True)
    ma = window[ma_col]
    if ma.isna().all():
        return False

    return _legs_untested_check(window, ma, cfg["leg_split_rebound"], cfg["undercut_tolerance"],
                                 cfg["recent_bars"], cfg["near_support_tolerance"])


# ------------------------------------------------------------------
# 2026-10-31新增：「夾起單洗盤」(假跌破生死線MA35後迅速站回) + 半年線
# (MA120)兩次腳未破同時成立時加分。見sop_rules.yaml indicators.
# washout_recovery_second_leg區塊註解——任一條件單獨使用都沒有優於無
# 條件基準的backtest證據，只有兩者同時成立才加分，只適用日線。
# _detect_washout_recovery()拆成獨立函式，供bear_regime_fake_rally_
# guard的放行條件也重用（見該區塊註解），不是只有這裡的加分項用得到。
# ------------------------------------------------------------------
def _detect_washout_recovery(df: pd.DataFrame, tf: str, latest: pd.Series, lookback: int) -> bool:
    if tf != "日" or "MA35" not in df.columns or len(df) < lookback + 1 or pd.isna(latest.get("MA35")):
        return False
    prior = df.iloc[-(lookback + 1):-1]
    was_below_recently = (prior["close"] < prior["MA35"]).any()
    now_above = latest["close"] >= latest["MA35"]
    return bool(was_below_recently and now_above)


def _step2_washout_recovery_with_support(df: pd.DataFrame, tf: str, latest: pd.Series,
                                          ma120_second_leg: bool, reasons: List[str]) -> float:
    cfg = RULES["indicators"].get("washout_recovery_second_leg", {})
    if not cfg.get("enabled", True) or not ma120_second_leg:
        return 0.0

    lookback = cfg.get("lookback_bars", 3)
    if _detect_washout_recovery(df, tf, latest, lookback):
        reasons.append(
            "偵測到假跌破生死線(MA35)後迅速站回，且半年線(MA120)兩次腳未破的"
            "「夾起單洗盤」情境，回測顯示10日後報酬/勝率優於無條件基準，加分"
        )
        return cfg.get("bonus", 1)
    return 0.0


# ------------------------------------------------------------------
# 2026-10-23新增：「兩次腳未破」廣義化版本。原本_detect_ma120_second_leg
# 只認「半年線(MA120)」當支撐參考、只支援日線，但但丁老師課程講的其實
# 更廣義——「左側歷史平台區（或前低）打出兩次腳，底不破就不能砍單」，
# 不限定要是MA120。這裡改用Step6（_find_platform_support_resistance）
# 已經找到的左側支撐候選（整理平台或單次觸及前低）當參考價位，套用同一
# 套_legs_untested_check()演算法，且不限timeframe（週/日/60分/5分都適用，
# 沿用left_side_platform.platform_lookback_bars各週期已有的回看範圍）。
# 跟原本的ma120版本一樣，目前只附加caveats提示，不直接否決賣出減碼、
# 也不直接升級買進/加碼——還沒backtest驗證過，獨立開關
# platform_second_leg.enabled，不用時可以直接關掉不影響其他規則。
# ------------------------------------------------------------------
def _detect_platform_second_leg(df: pd.DataFrame, tf: str, latest: pd.Series,
                                 support_levels: Optional[List[Tuple[float, str]]]) -> Optional[float]:
    """回傳偵測到「兩次腳未破」的支撐價位，沒偵測到回傳None——呼叫端決定
    要不要提示（跟_detect_ma120_second_leg同一套模式：只在conclusion不是
    賣出減碼/加碼時才附加caveat，見evaluate_timeframe下方
    special_notes.platform_second_leg_hint區塊）。"""
    cfg = RULES["indicators"].get("platform_second_leg", {})
    if not cfg.get("enabled", True) or not support_levels:
        return None

    min_bars = cfg.get("min_bars", 20)
    window_bars = cfg.get("window_bars", 60)
    if len(df) < min_bars:
        return None

    window = df.tail(window_bars).reset_index(drop=True)
    if len(window) < min_bars:
        return None

    close = latest.get("close")
    if pd.isna(close):
        return None

    # 只檢查「目前價格以下」的支撐水位（跟Step6 candidate_supports同一個
    # 篩選邏輯），由近到遠依序檢查，抓到第一個成立的就回報並結束。
    candidates = sorted({level for level, _ in support_levels if level <= close}, reverse=True)
    for level in candidates:
        reference = pd.Series(level, index=window.index)
        if _legs_untested_check(window, reference, cfg["leg_split_rebound"],
                                 cfg["undercut_tolerance"], cfg["recent_bars"],
                                 cfg["near_support_tolerance"]):
            return level
    return None


# ------------------------------------------------------------------
# 乖離率過大：股價與快/慢均線同方向乖離都超過門檻 → 不論多空方向，
# 嚴禁在此追高或摸底（只降級買訊，不用來加重賣訊，避免暴跌時被雙重扣分）。
# ------------------------------------------------------------------
def _check_extreme_deviation(tf: str, latest: pd.Series, reasons: List[str]) -> bool:
    fast_col, key_col = FAST_MA.get(tf), KEY_MA.get(tf)
    threshold = RULES["indicators"]["deviation_threshold"].get(tf)
    if not fast_col or not key_col or threshold is None:
        return False
    close, fast, key = latest.get("close"), latest.get(fast_col), latest.get(key_col)
    if pd.isna(close) or pd.isna(fast) or pd.isna(key) or fast == 0 or key == 0:
        return False

    fast_dev = (close - fast) / fast
    key_dev = (close - key) / key
    same_direction = (fast_dev > 0 and key_dev > 0) or (fast_dev < 0 and key_dev < 0)
    if same_direction and abs(fast_dev) > threshold and abs(key_dev) > threshold:
        reasons.append(
            f"⚠️ 股價與 {fast_col}/{key_col} 乖離過大（{fast_dev:+.0%} / {key_dev:+.0%}），"
            "不論多空方向，嚴禁在此追高或摸底"
        )
        return True
    return False


# ------------------------------------------------------------------
# 新規則②：週線短期噴出型態偵測。單一波段漲幅遠超歷史正常區間、且明顯脫
# 離所有均線時，即使短週期指標翻多，仍強制降級為觀望；只要股價尚未站回
# 週5MA～週20MA之間，重新計算時仍會持續偵測為噴出，維持強制觀望。
# ------------------------------------------------------------------
def _check_weekly_blowoff(tf: str, df: pd.DataFrame, latest: pd.Series,
                           reasons: List[str], caveats: List[str]) -> bool:
    cfg = RULES["watch_rules"]["weekly_blowoff_guard"]
    if not cfg.get("enabled") or tf not in cfg.get("applicable_timeframes", ["週"]):
        return False

    lookback = cfg["leg_lookback_weeks"]
    if len(df) < lookback + 1:
        return False

    window = df.tail(lookback + 1)
    close = latest.get("close")
    low_base = window["min"].iloc[:-1].min()
    if pd.isna(close) or pd.isna(low_base) or low_base <= 0:
        return False

    leg_gain = (close - low_base) / low_base
    if leg_gain < cfg["leg_gain_threshold"]:
        return False

    ma_cols = cfg.get("deviation_ma_columns", [])
    deviation_threshold = cfg["deviation_threshold"]
    devs = []
    for col in ma_cols:
        val = latest.get(col)
        if pd.isna(val) or val == 0:
            return False  # 均線資料不足，無法確認脫離所有均線，略過偵測
        devs.append((close - val) / val)

    if devs and all(d > deviation_threshold for d in devs):
        message = cfg["message_template"].format(
            lookback=lookback, gain=leg_gain, ma_cols="/".join(ma_cols)
        )
        reasons.append(f"⚠️ {message}")
        return True

    return False


# ------------------------------------------------------------------
# 新規則①：買進/加碼的三項同時條件（均線上揚、MACD/BBI同步轉向、支撐未
# 破）只要有任一項尚未確認，一律判定為觀望，不可輸出模糊結論。
# ------------------------------------------------------------------
def _apply_buy_add_sync_gate(tf: str, conclusion: str, fast_slope_up: bool,
                              macd_bias: float, volume_bias: float,
                              key_ma_down_veto: bool, four_key_broken: bool,
                              caveats: List[str]) -> str:
    cfg = RULES["watch_rules"]["buy_add_sync_gate"]
    if not cfg.get("enabled") or conclusion not in ("買進", "加碼"):
        return conclusion
    if tf not in cfg.get("apply_timeframes", []):
        return conclusion

    checks = {
        "ma_slope_up": fast_slope_up,
        "macd_volume_sync_up": macd_bias > 0 and volume_bias > 0,
        "support_intact": not key_ma_down_veto and not four_key_broken,
    }
    required = cfg.get("require", [])
    failed = [name for name in required if not checks.get(name, True)]
    if not failed:
        return conclusion

    caveats.append(f"{cfg['message']}（未確認：{'、'.join(failed)}；原結論：{conclusion}）")
    return cfg.get("downgrade_to", "觀望")


# ------------------------------------------------------------------
# 主入口：單一週期綜合判讀
# ------------------------------------------------------------------
def evaluate_timeframe(df: pd.DataFrame, timeframe_label: str) -> Verdict:
    """
    依 SOP 的位階否決邏輯（四關價 → 均線 → MACD → OBV/BBI/MTM）給出
    「買進／加碼／賣出減碼／觀望」結論，而非單純把各指標分數加總後看門檻。
    """
    if df.empty:
        return Verdict(conclusion="觀望", confidence="低", score=0.0,
                        reasons=["資料為空，無法判讀"], caveats=[])

    scoring_cfg = RULES["indicators"]["scoring"]
    buy_cfg = RULES["buy_rules"]
    add_cfg = RULES["add_rules"]
    watch_cfg = RULES["watch_rules"]
    sell_cfg = RULES["sell_rules"]
    slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]

    tf = normalize_timeframe(timeframe_label)
    latest = df.iloc[-1]
    reasons: List[str] = []
    caveats: List[str] = []

    four_key_bias = _step1_four_key_prices(tf, latest, reasons, caveats)
    ma_bias, key_ma_down_veto = _step2_ma(df, tf, latest, reasons, caveats)
    ma_bias += _step2_weekly_pullback_zone(df, tf, latest, reasons)
    ma_bias += _step2_weekly_kd_golden_cross(df, tf, latest, reasons)
    ma_bias += _step2_60min_turtle_confirm(df, tf, latest, reasons)
    _step2_60min_warning_zone(df, tf, latest, caveats)
    macd_bias, dead_cross, macd_bearish_divergence = _step3_macd(df, latest, reasons, caveats)
    # KD(36,5,5) Double Check：只適用日/60分/5分，週線維持既有KD(9,3,3)
    # 規則不變（見sop_rules.yaml kd365區塊註解）
    if tf != "週":
        kd365_bias, kd365_cross_down, kd365_bearish_divergence = _step_kd365(
            df, tf, latest, reasons, caveats)
    else:
        kd365_bias, kd365_cross_down, kd365_bearish_divergence = 0.0, False, False
    volume_bias, volume_bear_confirm = _step4_volume(
        df, tf, latest, macd_bearish_divergence, reasons, caveats)
    mtm_exit_alert = _step5_mtm(df, tf, latest, reasons)
    cci_exit_alert, cci_bias = _step5b_cci(df, tf, latest, reasons, caveats)

    fast_col = FAST_MA.get(tf)
    fast_slope = ma_slope(df, fast_col, lookback=slope_lookback) if fast_col else "資料不足"
    fast_slope_up = fast_slope == "上揚"
    fast_slope_down = fast_slope == "下彎"

    # ---- 型態學輔助觸發層（見 sop_rules.yaml pattern_recognition）----
    # 只附加提醒/頸線候選，不直接產生買賣結論；三角收斂的強制觀望在下方
    # 與其他 watch_rules 降級規則一起套用。
    # KD365最近是否「剛黃金交叉」（不是「目前是不是多頭狀態」）——
    # 2026-10-13 backtest驗證過兩種操作化方式：「目前狀態」版沒有證據
    # 支持，「剛交叉」版三個天期都明顯勝出，見sop_rules.yaml kd365.
    # w_bottom_fresh_cross_window區塊註解。排除週線，跟KD365系列既有
    # 規則（週線維持原本KD(9,3,3)）一致。
    kd365_bullish = tf != "週" and _kd365_crossed_up_recently(
        df, RULES["indicators"]["kd365"]["w_bottom_fresh_cross_window"])
    w_bottom_result = _detect_w_bottom(df, fast_slope_up, macd_bias, kd365_bullish, reasons)
    w_bottom_neckline = w_bottom_result[0] if w_bottom_result else None
    w_bottom_pending = bool(w_bottom_result) and not w_bottom_result[1]
    m_top_neckline = _detect_m_top(df, fast_slope_down, macd_bias, reasons)
    triangle = _detect_triangle_consolidation(df, tf, reasons, caveats)
    triangle_force_watch = triangle["active"] and not triangle["volume_confirmed"]
    triangle_breakout_confirmed = bool(triangle["active"] and triangle["volume_confirmed"])

    platform_bias, platform_support_levels = _step6_left_side_platform(
        df, tf, latest, fast_slope_up, reasons, caveats,
        w_bottom_neckline=w_bottom_neckline, m_top_neckline=m_top_neckline)
    ma120_second_leg = _detect_ma120_second_leg(df, tf)
    platform_second_leg_level = _detect_platform_second_leg(df, tf, latest, platform_support_levels)
    ma_bias += _step2_washout_recovery_with_support(df, tf, latest, ma120_second_leg, reasons)
    extreme_deviation = _check_extreme_deviation(tf, latest, reasons)
    weekly_blowoff = _check_weekly_blowoff(tf, df, latest, reasons, caveats)

    # 2026-10-31新增：scoring.mode="weighted_48_20_20"時改用均線48%／
    # 四關價20%／指標組20%（見_weighted_score_48_20_20註解）；OBV/左側
    # 平台(volume_bias/platform_bias)不在但丁老師這次講的三層結構裡，
    # 兩種mode都維持原本的額外加分方式不變。預設仍是legacy，行為不變。
    if scoring_cfg.get("mode", "legacy") == "weighted_48_20_20":
        core_score = _weighted_score_48_20_20(
            four_key_bias, ma_bias, macd_bias, kd365_bias, cci_bias, tf,
            scoring_cfg["weighted_48_20_20"])
        score = core_score + volume_bias + platform_bias
    else:
        score = (four_key_bias * scoring_cfg["four_key_weight"] + ma_bias * scoring_cfg["ma_weight"]
                 + macd_bias + volume_bias + platform_bias)

    # ---- 判定「賣出減碼」：符合任一條件即可（見 SOP 規則）----
    four_key_broken = tf == "日" and four_key_bias == -1
    dead_cross_confirmed = dead_cross and ma_bias < 0
    # 2026-10-07：volume_bear_confirm（MACD頂背離+OBV同步走壞）原本不管均線
    # 方向，單獨就能跳過均線否決硬判賣出減碼——跟dead_cross_confirmed要求
    # ma_bias<0才算確認的作法不一致，也跟但丁老師「均線斜率具最終裁決權，
    # 一代指標未破壞不可單憑二三代指標背離盲目全面砍單」的原則衝突。
    # 2026-10-05金像電(2368)案例：10/2收盤觸發此規則判賣出減碼，均線(MA35)
    # 當時仍上揚且站上，隔一個交易日卻在PCB族群消息面帶動下鎖漲停，促成
    # 用test_volume_bear_confirm_ma_gate(_wide).py回測驗證（60檔/36個月/
    # 138筆事件）：均線仍偏多時單獨出現這個訊號，20日後續跌45.3%、上漲
    # 54.7%（接近銅板），但兩端都有肥尾（最差-47%、最佳+75.4%）——不是
    # 「大多數是假警報可以忽略」，也不是「真的要全賣」，是分佈兩端都很
    # 極端，所以改成「分級減碼」而非全有全無：均線同步偏空才算全賣的
    # 確認；均線仍偏多時，降級為「賣出減碼」但在caveats註明建議只先減碼
    # 50%、保留一半觀察（conclusion仍是「賣出減碼」四字不變，只是在caveats
    # 加註細節——combine_timeframes/各pages對conclusion做的是精確字串比對，
    # 不能在這裡夾帶括號文字，否則會讓長短天期否決/UI判色失效）。
    volume_bear_confirm_ma_synced = volume_bear_confirm and ma_bias < 0
    volume_bear_confirm_requires_ma_sync = sell_cfg.get("volume_bear_confirm_requires_ma_sync", True)
    volume_bear_confirm_partial = (
        volume_bear_confirm and volume_bear_confirm_requires_ma_sync and not volume_bear_confirm_ma_synced
    )
    volume_bear_confirm_for_hard_sell = (
        volume_bear_confirm_ma_synced if volume_bear_confirm_requires_ma_sync else volume_bear_confirm
    )
    # ---- KD365 Double Check（賣出側）：MACD死叉+KD365死叉+生死線斜率
    # 轉下彎，同時成立時否決conclusion，視同hard_sell位階。key_slope_down
    # 故意不重用key_ma_down_veto——後者還要求股價已跌破生死線，比這裡
    # 實際驗證過的條件（純粹看斜率方向）更嚴格，見sop_rules.yaml
    # kd365_veto_enabled區塊註解與對應回測（test_kd365_double_check.py）。
    kd365_veto_enabled = sell_cfg.get("kd365_veto_enabled", True)
    key_slope_down = ma_slope(df, KEY_MA[tf], lookback=slope_lookback) == "下彎"
    kd365_double_check_confirmed = dead_cross and kd365_cross_down and key_slope_down
    if kd365_double_check_confirmed:
        reasons.append(
            "⚠️ KD(36,5,5)死叉 + MACD死叉（Double Check）同時成立，且生死線已轉"
            "下彎，課程定義的「必須離場」等級"
        )

    hard_sell = (
        (key_ma_down_veto and sell_cfg.get("key_ma_down_veto", True))
        or (four_key_broken and sell_cfg.get("four_key_broken", True))
        or (dead_cross_confirmed and sell_cfg.get("dead_cross_confirmed", True))
        or (volume_bear_confirm_for_hard_sell and sell_cfg.get("volume_bear_confirm", True))
        or (kd365_double_check_confirmed and kd365_veto_enabled)
    )
    # 待驗證（2026-10-08）：test_leading_indicator_protective_value.py用13檔
    # 60分K(約120天)比較四類降級訊號觸發後的保護力，結果跟SOP位階預設的
    # 「hard_sell權威最高、最可信」相反——hard_sell(這裡)4根K棒後下跌比例
    # 41.0%、平均報酬+1.00%(還是漲的)，比純MTM/CCI示警(leading_indicator_
    # exit_alert，下方)的51.4%/+0.12%還弱，可能是hard_sell的條件本質上是
    # 落後指標的「再確認」，訊號出現時這段跌勢常常已經走完、容易迎來技術
    # 反彈。但這次測試期間連無條件基準都是正報酬(偏多格局)，逆勢賣出訊號
    # 本來就容易在多頭期被軋，不能排除是市況偏多稀釋了hard_sell的保護力，
    # 不是規則設計本身的問題——要拉長回測期、涵蓋過一段像樣的回檔/空頭，
    # 才能真的下定論。在那之前不調整位階順序，均線/四關價仍視為最終裁決權。

    # ---- 只有 MACD 單獨背離、均線與量能都還沒同步走壞：先觀望不追空 ----
    # （均線仍偏多但OBV已同步背離的情況，走volume_bear_confirm_partial那條
    # 「分級減碼」，不要落到這條純觀望——所以這裡也要排除掉它）
    caution_only = (
        watch_cfg.get("caution_only_on_macd_divergence", True)
        and macd_bearish_divergence and not volume_bear_confirm_for_hard_sell
        and not volume_bear_confirm_partial and not key_ma_down_veto
    )

    strong_bullish = (
        ma_bias >= add_cfg["ma_bias_min"] and macd_bias >= add_cfg["macd_bias_min"]
        and four_key_bias >= add_cfg["four_key_bias_min"] and volume_bias >= add_cfg["volume_bias_min"]
    )

    # ---- 領先指標示警（MTM／CCI 並列，同等級，任一觸發都直接壓成觀望）----
    # 2026-10-23：「MTM絕不能用於中繼段/盤整」——目前仍處於三角收斂整理
    # 區間、且還沒帶量突破確認時（跟triangle_force_watch同一個條件：
    # active且未volume_confirmed，一旦突破確認就不算「中繼」了，MTM該
    # 恢復正常作用），MTM翻空不觸發示警降級，避免盤整期雙巴雜訊反覆打架
    # （見sop_rules.yaml indicators.mtm.suppress_during_consolidation
    # 區塊註解）。MTM本身的reasons文字照常附加，只是不讓它驅動conclusion。
    mtm_suppressed_by_consolidation = (
        triangle_force_watch and RULES["indicators"]["mtm"].get("suppress_during_consolidation", True)
    )
    mtm_alert_active = (
        tf == "60分" and mtm_exit_alert and watch_cfg.get("mtm_60m_exit_downgrade", True)
        and not mtm_suppressed_by_consolidation
    )
    if tf == "60分" and mtm_exit_alert and mtm_suppressed_by_consolidation:
        caveats.append("MTM翻空訊號出現在三角收斂整理區間內，依但丁老師規則不採用，暫不以此降級")
    cci_alert_active = cci_exit_alert and watch_cfg.get("cci_exit_alert_downgrade", True)
    leading_indicator_exit_alert = (mtm_alert_active or cci_alert_active) and not key_ma_down_veto

    if hard_sell:
        conclusion = "賣出減碼"
    elif volume_bear_confirm_partial:
        # 均線仍偏多，頂背離+OBV只能算「分級減碼」示警，不是全賣確認
        conclusion = "賣出減碼"
        caveats.append(
            "頂背離+OBV同步走弱，但生死線仍偏多（均線未破壞），回測（60檔/36個月/"
            "138筆事件）顯示這類訊號後20日續跌45%、上漲55%，且兩端皆有肥尾"
            "（最差-47%、最佳+75%）：建議先減碼50%觀察，不要在均線未破壞前全數出清，"
            "待均線也轉空再出清剩餘部位"
        )
    elif leading_indicator_exit_alert:
        conclusion = "觀望"
        if mtm_alert_active:
            caveats.append("60分 MTM 已示警，建議短線先出場觀察，暫不否定較長天期結構")
        if cci_alert_active:
            caveats.append(f"{tf} CCI 已示警（由+100折返），建議短線先出場觀察，暫不否定較長天期結構")
    elif caution_only:
        conclusion = "觀望"
        caveats.append("MACD出現警戒訊號，但均線與量能尚未同步走壞，先觀望、不追空")
    elif strong_bullish:
        conclusion = "加碼" if (ma_bias >= add_cfg["addon_ma_bias"] and volume_bias > 0) else "買進"
    else:
        buy_threshold, sell_threshold = _score_thresholds(
            scoring_cfg.get("mode", "legacy"), buy_cfg, sell_cfg, scoring_cfg["weighted_48_20_20"])
        if score >= buy_threshold:
            conclusion = "買進"
        elif score <= sell_threshold:
            conclusion = "賣出減碼"
        else:
            conclusion = "觀望"

    # ---- 2026-10-31新增：空頭環境假反彈防護（見sop_rules.yaml watch_rules.
    # bear_regime_fake_rally_guard區塊註解）。能走到這裡conclusion仍是
    # 買進/加碼，代表股價其實還站在生死線之上（否則key_ma_down_veto已經
    # 先判賣出減碼），防的是「均線本身還沒轉向、站上去的這段隨時可能被
    # 拉回去」的假反彈，不是「反彈測試均線被打回」這種還沒做偵測的進階
    # 型態。----
    bear_guard_cfg = watch_cfg.get("bear_regime_fake_rally_guard", {})
    if bear_guard_cfg.get("enabled", True) and key_slope_down and conclusion in ("買進", "加碼"):
        # 2026-10-31補強：日線的話，光有「兩次腳未破」不夠——backtest顯示
        # 單獨使用任一條件都沒有優於無條件基準的證據，只有「夾起單+兩次
        # 腳未破」同時成立才有效（見indicators.washout_recovery_second_leg
        # 區塊註解），所以日線這裡也要求夾起單同時成立才放行小量試單。
        # 2026-10-31再次收緊：日線只認ma120_second_leg（半年線版，已驗證
        # 夾起單+它同時成立時10/20日報酬勝率都優於基準），不含
        # platform_second_leg_level（廣義版）——另外用test_washout_
        # recovery_platform_second_leg_backtest.py測過「夾起單+廣義版兩
        # 次腳未破」，5/10日報酬勝率都比基準差，沒有證據支持，不要混進
        # 來稀釋掉已驗證的訊號品質。週/60分/5分目前沒有對應的夾起單
        # backtest，維持原本只看支撐訊號（含廣義版）的寬鬆條件，不隨日線
        # 一起收緊。
        if tf == "日":
            washout_lookback = RULES["indicators"].get(
                "washout_recovery_second_leg", {}).get("lookback_bars", 3)
            has_left_side_support = ma120_second_leg and _detect_washout_recovery(
                df, tf, latest, washout_lookback)
        else:
            has_left_side_support = ma120_second_leg or platform_second_leg_level is not None
        held_above_open = pd.notna(latest.get("今開")) and latest["close"] >= latest["今開"]
        if has_left_side_support and held_above_open:
            if conclusion == "加碼":
                conclusion = "買進"
                caveats.append(
                    "生死線仍下彎，但左側平台/半年線打出兩次腳未破且今日收盤站穩今開，"
                    "依但丁老師規則先以小量試單（5-10%基本部位）處理，不升級為加碼"
                )
        else:
            conclusion = "觀望"
            caveats.append(
                "生死線仍下彎壓制，依但丁老師規則視為破波段假反彈/誘多陷阱，"
                "嚴禁波段重押或順勢加碼（若左側平台/半年線有兩次腳未破的底部支撐"
                "且今日站穩今開才考慮小量試單）"
            )
            caveats.append("若已持有多單，逢弱勢反彈建議執行減碼/買保險，暫不追高")
            caveats.append("均線本身仍下彎尚未轉向，若股價撐不住再度跌破生死線，可視為順勢加碼空點的訊號")

    # ---- 乖離過大：不論多空方向，嚴禁在此追高摸底，買訊一律降級觀望 ----
    if extreme_deviation and watch_cfg.get("extreme_deviation_downgrade", True) and conclusion in ("買進", "加碼"):
        caveats.append(f"雖符合買進/加碼條件，但乖離過大，先降級為觀望，等乖離收斂再說（原結論：{conclusion}）")
        conclusion = "觀望"

    # ---- 新規則①：買進/加碼三項同時條件，任一項未確認一律降級觀望 ----
    conclusion = _apply_buy_add_sync_gate(
        tf, conclusion, fast_slope_up, macd_bias, volume_bias,
        key_ma_down_veto, four_key_broken, caveats)

    # ---- 新規則②：週線短期噴出型態，強制降級為觀望，須站回週5MA~週20MA才重新評估 ----
    if weekly_blowoff and conclusion in ("買進", "加碼"):
        conclusion = "觀望"

    # ---- 型態學輔助觸發層：三角收斂尚未帶量突破（或假突破）→ 買進/加碼強制降級觀望，
    # 不凌駕生死線下彎/四關價破底等硬否決產生的「賣出減碼」----
    if triangle_force_watch and watch_cfg.get("triangle_consolidation_downgrade", True) \
            and conclusion in ("買進", "加碼"):
        conclusion = "觀望"

    # ---- 型態學輔助觸發層：W底尚未經均線/MACD/KD365同步確認 → 買進/加碼
    # 強制降級觀望，不凌駕生死線下彎/四關價破底等硬否決產生的「賣出減碼」----
    if w_bottom_pending and watch_cfg.get("w_bottom_kd365_downgrade", True) \
            and conclusion in ("買進", "加碼"):
        conclusion = "觀望"

    # ---- 特殊情境：大跌測到半年線(MA120)第二隻腳未破，只在非賣出/加碼時提示 ----
    if ma120_second_leg and RULES["special_notes"].get("ma120_second_leg_hint", True) \
            and conclusion not in ("賣出減碼", "加碼"):
        caveats.append(
            "偵測到大跌測到半年線(MA120)打出第二隻腳未破的特殊情境：可考慮「帶著鋼盔慢買」——"
            "先進 5-10% 基本部位，待5分/60分指標確認低點墊高、均線轉平緩後，每日加碼約5%"
            "（僅供分批佈局參考，非立即買進訊號）"
        )

    # ---- 2026-10-23新增：左側平台/前低兩次腳未破（ma120_second_leg的廣義版，
    # 不限定MA120、不限定日線），只在非賣出/加碼時提示 ----
    if platform_second_leg_level is not None and RULES["special_notes"].get("platform_second_leg_hint", True) \
            and conclusion not in ("賣出減碼", "加碼"):
        caveats.append(
            f"📐 偵測到左側平台/前低約{platform_second_leg_level:.2f}打出兩次腳、底部未實質跌破"
            "（依但丁老師規則，底不破不能砍單，僅供觀察參考，非買進依據）"
        )

    abs_score = abs(score)
    if key_ma_down_veto or volume_bear_confirm_for_hard_sell \
            or (ma_bias >= add_cfg["addon_ma_bias"] and volume_bias > 0):
        confidence = "高"
    elif abs_score >= scoring_cfg["confidence_mid_abs_score"]:
        confidence = "中"
    else:
        confidence = "低" if abs_score < scoring_cfg["confidence_low_abs_score"] else "中"

    # ---- 建議曝險（5級，待backtest驗證微調，見sop_rules.yaml exposure_rules）----
    # 這一輪刻意不讓這個欄位影響conclusion——只是把但丁老師課程的部位
    # 控管概念，用既有已經算出來的條件變數做映射，不是重新設計一套邏輯。
    # 週線/月線固定None、不分類：跟「週線與月線完全不動」的範圍限制一致，
    # 不要讓一個沒人設計過的部位建議附掛在週線輸出上（最終review發現，
    # 原本的映射邏輯沒有排除週線，週線會意外拿到0/50/70/100其中一個值）。
    exposure_cfg = RULES.get("exposure_rules", {})
    kd365_macd_double_bear = kd365_cross_down and macd_bearish_divergence
    if tf == "週":
        suggested_exposure = None
    elif hard_sell:
        suggested_exposure = 0
    elif volume_bear_confirm_partial or kd365_macd_double_bear:
        suggested_exposure = 30
    elif caution_only or leading_indicator_exit_alert or kd365_cross_down or kd365_bearish_divergence:
        suggested_exposure = 50
    elif conclusion == "觀望" and not key_ma_down_veto:
        suggested_exposure = exposure_cfg.get("watch_support_held", 70)
    elif conclusion in ("買進", "加碼"):
        suggested_exposure = 100
    elif conclusion == "賣出減碼":
        # 經score門檻判成賣出減碼、但沒有觸發hard_sell/雙重示警的較弱賣出
        # 訊號，原本會落到最後的None——一個「賣出減碼」結論卻配「無法
        # 分類」的曝險建議，使用者看了會當成bug。改給明確的低曝險預設值。
        suggested_exposure = exposure_cfg.get("sell_default", 30)
    else:
        suggested_exposure = None

    if not reasons:
        reasons.append("各項指標訊號不明顯，暫無明確方向")

    return Verdict(conclusion=conclusion, confidence=confidence, score=score,
                    reasons=reasons, caveats=caveats,
                    volume_breakout_confirmed=triangle_breakout_confirmed,
                    suggested_exposure=suggested_exposure)


# ------------------------------------------------------------------
# 跨週期整合：「為日線留倉，為五分出場」
# 長天期（週/日）結構轉弱 → 否決短天期反彈買訊，直接判賣出/減碼
# 短天期（60分/5分）轉弱 → 不推翻長天期多頭結構，但提示先讓短打部位出場
# 這是原始版本（週/日平權、60分/5分平權），見sop_rules.yaml
# watch_rules.combine_timeframes_mode註解——目前仍是正式預設("legacy")。
# ------------------------------------------------------------------
def _combine_timeframes_legacy(verdicts: Dict[str, Verdict], detail: dict) -> dict:
    long_tfs = [tf for tf in ("週", "日") if tf in verdicts]
    short_tfs = [tf for tf in ("60分", "5分") if tf in verdicts]

    # 長天期任一個結構已轉弱 → 直接否決，短天期反彈視為誘多
    for tf in long_tfs:
        if verdicts[tf].conclusion == "賣出減碼":
            return {
                "最終建議": f"賣出減碼（{tf}結構已轉弱，短線反彈視為誘多假動作，不追）",
                "各週期明細": detail,
            }

    long_bullish = any(verdicts[tf].conclusion in ("買進", "加碼") for tf in long_tfs)
    long_watch_only = bool(long_tfs) and all(verdicts[tf].conclusion == "觀望" for tf in long_tfs)
    short_bear_alert = any(verdicts[tf].conclusion == "賣出減碼" for tf in short_tfs)

    if long_bullish and short_bear_alert:
        return {
            "最終建議": "長線續抱、短線先出場（為日線留倉，為五分出場）",
            "各週期明細": detail,
        }

    if long_bullish:
        all_bullish = all(verdicts[tf].conclusion in ("買進", "加碼") for tf in verdicts)
        add_cfg = RULES["add_rules"]
        ambiguous_cfg = RULES["watch_rules"]["combine_timeframes_ambiguous_guard"]
        if all_bullish:
            final = add_cfg["weekly_all_synced_label"]
        elif ambiguous_cfg.get("forbid_ambiguous_partial_bullish", True):
            # 新規則①的跨週期版本：長天期翻多但各週期未全數同步時，
            # 依SOP不可輸出「買進（部分週期訊號仍待確認）」這類模糊結論，一律觀望。
            final = ambiguous_cfg["replacement_conclusion"]
        else:
            final = "買進（部分週期訊號仍待確認）"
        return {"最終建議": final, "各週期明細": detail}

    if long_watch_only:
        return {"最終建議": "觀望（長天期尚未表態，不追短線訊號）", "各週期明細": detail}

    # 只有短天期資料可用（沒有日/週資料能交叉驗證）
    if not long_tfs and short_tfs:
        if short_bear_alert:
            return {"最終建議": "短線賣出減碼（無長天期資料交叉驗證，僅供短打參考）", "各週期明細": detail}
        if all(verdicts[tf].conclusion in ("買進", "加碼") for tf in short_tfs):
            return {"最終建議": "短線買進（無長天期資料交叉驗證，僅供短打參考）", "各週期明細": detail}

    return {"最終建議": "觀望", "各週期明細": detail}


# ------------------------------------------------------------------
# 跨週期整合（實驗模式）："day_60m_primary"——日線為長天期主要判斷依據，
# 週線降為輔助；60分為短天期主要判斷依據，5分降為輔助。輔助週期只能
# 「降級」（把主要週期的買訊往下拉），不能單獨把主要週期的觀望拉成買進
# ——跟既有MTM/CCI「二線輔助只降級不升級」同一套原則。2026-10-23陽明
# (2609)/安葆(7792)兩次案例之後新增，呼應2026-10-21課程「60MA是為中長線
# 交易者設計」把60分地位拉高、淡化5分的方向。還沒有backtest/對照驗證過，
# 見test_combine_weighting_compare.py，mode切成這個之前應該先跑過比較。
# ------------------------------------------------------------------
def _combine_timeframes_day_60m_primary(verdicts: Dict[str, Verdict], detail: dict, cfg: dict) -> dict:
    """
    2026-10-23修訂：輔助週期（週/5分）只拿掉「升級」的權力（不能單獨把主要
    週期的觀望拉成買進，也不能單獨把買進湊成加碼），**保留「降級/否決」
    的權力**（週/5分任一個賣出減碼，一樣可以直接否決/觸發短線先出場）——
    比最早那版更貼近但丁老師「長天期否決短天期」的原始精神，只是把「誰
    有資格發動偏多訊號」這件事從「週或日任一個」收斂成「日線自己」。
    """
    primary_long = cfg.get("primary_long", "日")
    secondary_long = cfg.get("secondary_long", "週")
    primary_short = cfg.get("primary_short", "60分")
    secondary_short = cfg.get("secondary_short", "5分")

    p_long = verdicts.get(primary_long)
    s_long = verdicts.get(secondary_long)
    p_short = verdicts.get(primary_short)
    s_short = verdicts.get(secondary_short)

    if p_long is None:
        # 2026-10-23修bug：沒有日線資料不代表「完全無法判讀」——
        # pages/1_短線進場.py只做60分/5分短線分析，呼叫combine_timeframes()
        # 時本來就只會傳60分/5分，不會有日/週。這裡改成退回「短天期
        # fallback」：60分當主要依據，5分一樣保留否決力，不要因為日線
        # 不存在就把實際算出來的60分/5分結果整個蓋掉（切換成day_60m_
        # primary當天才發現：原本直接回傳「資料不足」，等於把短線進場頁
        # 全部股票打成觀望，是真的production regression，不是防呆分支）。
        if p_short is None:
            return {"最終建議": "觀望（資料不足，無法判讀）", "各週期明細": detail}
        s_short_bear = s_short is not None and s_short.conclusion == "賣出減碼"
        if p_short.conclusion == "賣出減碼" or s_short_bear:
            return {
                "最終建議": "短線賣出減碼（無長天期資料交叉驗證，僅供短打參考）",
                "各週期明細": detail,
            }
        if p_short.conclusion in ("買進", "加碼"):
            return {
                "最終建議": f"短線{p_short.conclusion}（無長天期資料交叉驗證，僅供短打參考）",
                "各週期明細": detail,
            }
        return {"最終建議": "觀望", "各週期明細": detail}

    # 長天期否決權：日／週任一個賣出減碼就直接否決，週保留跟legacy一樣的
    # 否決力，只是不再有「單獨把日線觀望拉成買進」的升級力。
    if p_long.conclusion == "賣出減碼":
        return {
            "最終建議": f"賣出減碼（{primary_long}結構已轉弱，短線反彈視為誘多假動作，不追）",
            "各週期明細": detail,
        }
    if s_long is not None and s_long.conclusion == "賣出減碼":
        return {
            "最終建議": f"賣出減碼（{secondary_long}結構已轉弱，短線反彈視為誘多假動作，不追）",
            "各週期明細": detail,
        }

    p_long_bullish = p_long.conclusion in ("買進", "加碼")

    # 短天期示警：60分／5分任一個賣出減碼都能觸發「短線先出場」，5分保留
    # 跟legacy一樣的降級力，只是不再有單獨升級的力量。
    short_bear_alert = (
        (p_short is not None and p_short.conclusion == "賣出減碼")
        or (s_short is not None and s_short.conclusion == "賣出減碼")
    )

    if p_long_bullish and short_bear_alert:
        return {
            "最終建議": f"長線續抱、短線先出場（{primary_long}為主）",
            "各週期明細": detail,
        }

    if p_long_bullish:
        # 全部週期同步翻多才給強化版「加碼」標籤——週/60分/5分在這裡只是
        # confirm，不是gatekeeper：日線沒先自己達到「加碼」等級，這幾個
        # 輔助週期再怎麼偏多也沒用，呼應「只能降級不能升級」的原則。
        all_long_bullish = s_long is None or s_long.conclusion in ("買進", "加碼")
        all_short_bullish = (
            (p_short is None or p_short.conclusion in ("買進", "加碼"))
            and (s_short is None or s_short.conclusion in ("買進", "加碼"))
        )
        if p_long.conclusion == "加碼" and all_long_bullish and all_short_bullish:
            # 文字只列實際有資料、真的檢查過的週期，不要連沒丟進來的週期
            # 也宣稱「同步翻多」（例如diagnose_sell_lag.py只傳日+週，不該
            # 顯示成連60分/5分都同步翻多）。
            synced_tfs = "+".join(
                tf for tf, v in ((primary_long, p_long), (secondary_long, s_long),
                                  (primary_short, p_short), (secondary_short, s_short))
                if v is not None
            )
            return {
                "最終建議": f"加碼（{synced_tfs}同步翻多，以{primary_long}為主）",
                "各週期明細": detail,
            }
        return {"最終建議": p_long.conclusion, "各週期明細": detail}

    if p_long.conclusion == "觀望":
        return {
            "最終建議": f"觀望（{primary_long}尚未表態，不追{secondary_long}/短線訊號）",
            "各週期明細": detail,
        }

    return {"最終建議": "觀望", "各週期明細": detail}


def combine_timeframes(verdicts: Dict[str, Verdict]) -> dict:
    if not verdicts:
        return {"最終建議": "資料不足，無法判讀", "各週期明細": {}}

    detail = {
        tf: {"結論": v.conclusion, "信心": v.confidence, "理由": v.reasons, "但書": v.caveats,
             "帶量突破": v.volume_breakout_confirmed, "建議曝險": v.suggested_exposure}
        for tf, v in verdicts.items()
    }

    if len(verdicts) == 1:
        only_tf = next(iter(verdicts))
        return {"最終建議": verdicts[only_tf].conclusion, "各週期明細": detail}

    mode_cfg = RULES["watch_rules"].get("combine_timeframes_mode", {})
    if mode_cfg.get("mode") == "day_60m_primary":
        return _combine_timeframes_day_60m_primary(verdicts, detail, mode_cfg)
    return _combine_timeframes_legacy(verdicts, detail)


# ------------------------------------------------------------------
# 把 evaluate_timeframe 的 conclusion 或 combine_timeframes 的「最終建議」
# （可能帶括號附註，例如「長線續抱、短線先出場（為日線留倉，為五分出場）」）
# 正規化成四個分類之一，方便UI上色/排序。
# ------------------------------------------------------------------
def classify_final(text: str) -> str:
    if "賣出" in text or "減碼" in text:
        return "賣出減碼"
    if "加碼" in text:
        return "加碼"
    if "買進" in text or "留倉" in text:
        return "買進"
    return "觀望"
