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
from typing import Dict, List, Optional

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
# 簡化版頂背離偵測：近期股價創高，但 MACD 柱狀圖高點未同步創高。
# 這是簡化偵測，非教科書式嚴謹型態辨識，僅供參考。
# ------------------------------------------------------------------
def _detect_bearish_divergence(df: pd.DataFrame, lookback: int) -> bool:
    if len(df) < lookback or "MACD_hist" not in df.columns:
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
    prior_peak_hist = prior.loc[prior_peak_idx, "MACD_hist"]
    recent_peak_hist = recent.loc[recent_peak_idx, "MACD_hist"]

    if pd.isna(prior_peak_hist) or pd.isna(recent_peak_hist):
        return False

    return recent_peak_close > prior_peak_close and recent_peak_hist < prior_peak_hist


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

    return macd_bias, dead_cross, bearish_divergence


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

    recent = df.tail(cfg["lookback_bars"])
    cross_down = ((recent["MTM"] < 0) & (recent["MTM"].shift(1) >= 0)).any()
    if cross_down or latest["MTM"] < 0:
        reasons.append("MTM 翻空/死叉，60分線短線出場訊號優先示警")
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
def _step5b_cci(df: pd.DataFrame, tf: str, latest: pd.Series, reasons: List[str], caveats: List[str]) -> bool:
    cfg = RULES["indicators"]["cci"]
    if tf not in cfg.get("applicable_timeframes", []) or pd.isna(latest.get("CCI")):
        return False

    signal_threshold = cfg["signal_threshold"]
    extreme_threshold = cfg["extreme_threshold"]
    latest_cci = latest["CCI"]

    if abs(latest_cci) > extreme_threshold:
        direction = "超買" if latest_cci > 0 else "超賣"
        caveats.append(
            f"CCI達極端值{latest_cci:.0f}（{direction}，突破±{extreme_threshold:.0f}），"
            "需特別關注，但非趨勢反轉保證"
        )

    recent = df.tail(cfg["lookback_bars"])
    prior = recent["CCI"].iloc[:-1]
    if prior.empty:
        return False

    bear_foldback = (prior >= signal_threshold).any() and latest_cci < signal_threshold
    bull_foldback = (prior <= -signal_threshold).any() and latest_cci > -signal_threshold

    if bear_foldback:
        reasons.append(
            f"⚠️ CCI由+{signal_threshold:.0f}之上折返跌破+{signal_threshold:.0f}，短線動能轉弱疑慮，"
            "比照MTM列入領先指標示警（二線輔助，不凌駕均線/MACD）"
        )
        return True

    if bull_foldback:
        caveats.append(
            f"CCI由-{signal_threshold:.0f}之下折返站上-{signal_threshold:.0f}，短線止跌訊號，"
            "但屬二線輔助指標，不單獨作為買進依據，仍須均線/MACD/支撐同步確認"
        )

    return False


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
def _detect_w_bottom(df: pd.DataFrame, fast_slope_up: bool, macd_bias: float,
                      reasons: List[str]) -> Optional[float]:
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

    synced = fast_slope_up and macd_bias > 0
    if synced:
        reasons.append(
            f"📐 偵測到W底型態（左底約{left_low:.2f}／右底約{right_low:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收紅K，且均線斜率與MACD同步轉多，轉折訊號確認"
            "（輔助提醒，結論仍以四關價/均線/MACD/OBV分層否決制為準）"
        )
    else:
        reasons.append(
            f"📐 偵測到W底型態（左底約{left_low:.2f}／右底約{right_low:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收紅K，但均線斜率/MACD尚未同步轉多，標記為「型態轉折待確認」"
            "（僅供觀察，非買進依據）"
        )
    return neckline


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
def _step6_left_side_platform(df: pd.DataFrame, tf: str, latest: pd.Series,
                               fast_slope_up: bool,
                               reasons: List[str], caveats: List[str],
                               w_bottom_neckline: Optional[float] = None,
                               m_top_neckline: Optional[float] = None) -> float:
    cfg = RULES["indicators"]["left_side_platform"]
    exclude_recent = cfg["exclude_recent_bars"]
    pivot_window = cfg["pivot_window"]
    min_left_bars = cfg["min_left_bars"]
    tolerance = cfg["tolerance_pct"]
    lookback = cfg.get("platform_lookback_bars", {}).get(tf)
    min_touches = cfg.get("platform_min_touches", 2)

    if len(df) <= exclude_recent or pd.isna(latest.get("close")):
        caveats.append("左側K棒資料不足，Step6左側平台校正略過")
        return 0.0

    available_left = df.iloc[:-exclude_recent]
    left = available_left.tail(lookback) if lookback else available_left
    if len(left) < min_left_bars:
        caveats.append("左側K棒資料不足，Step6左側平台校正略過")
        return 0.0

    left = left.reset_index(drop=True)
    pivot_highs, pivot_lows = _find_pivot_points(left, pivot_window)

    close = latest["close"]
    support_levels = [(s, "整理平台（近期多次觸碰確認）")
                       for s in _cluster_pivot_levels(pivot_lows, tolerance, min_touches)]
    resistance_levels = [(r, "整理平台（近期多次觸碰確認）")
                          for r in _cluster_pivot_levels(pivot_highs, tolerance, min_touches)]
    # 型態學輔助觸發層併入的頸線候選（見 pattern_recognition），與既有K棒
    # 整理平台並列使用，尤其在缺乏明顯歷史平台時作為補充依據。
    if w_bottom_neckline is not None:
        support_levels.append((w_bottom_neckline, "W底頸線"))
    if m_top_neckline is not None:
        resistance_levels.append((m_top_neckline, "M頭頸線"))

    bias = 0.0
    candidate_supports = [(s, src) for s, src in support_levels if s <= close]
    if candidate_supports:
        support, support_src = max(candidate_supports, key=lambda item: item[0])
        dist = (close - support) / support if support else float("inf")
        if 0 <= dist <= tolerance:
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

    return bias


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

    near_support = (window["min"] <= ma * cfg["near_support_tolerance"]) & pd.notna(ma)
    touch_idxs = window.index[near_support].tolist()
    if len(touch_idxs) < 2:
        return False

    legs = [[touch_idxs[0]]]
    for idx in touch_idxs[1:]:
        base_low = window.loc[legs[-1][-1], "min"]
        between_high = window.loc[legs[-1][-1]:idx, "max"].max()
        if pd.notna(between_high) and pd.notna(base_low) and between_high >= base_low * cfg["leg_split_rebound"]:
            legs.append([idx])
        else:
            legs[-1].append(idx)

    if len(legs) < 2:
        return False

    leg1_low = window.loc[legs[0], "min"].min()
    leg2_low = window.loc[legs[-1], "min"].min()
    latest_idx = window.index[-1]
    is_recent = (latest_idx - legs[-1][-1]) <= cfg["recent_bars"]
    undercut = pd.notna(leg1_low) and pd.notna(leg2_low) and leg2_low < leg1_low * cfg["undercut_tolerance"]

    return is_recent and not undercut


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
    macd_bias, dead_cross, macd_bearish_divergence = _step3_macd(df, latest, reasons, caveats)
    volume_bias, volume_bear_confirm = _step4_volume(
        df, tf, latest, macd_bearish_divergence, reasons, caveats)
    mtm_exit_alert = _step5_mtm(df, tf, latest, reasons)
    cci_exit_alert = _step5b_cci(df, tf, latest, reasons, caveats)

    fast_col = FAST_MA.get(tf)
    fast_slope = ma_slope(df, fast_col, lookback=slope_lookback) if fast_col else "資料不足"
    fast_slope_up = fast_slope == "上揚"
    fast_slope_down = fast_slope == "下彎"

    # ---- 型態學輔助觸發層（見 sop_rules.yaml pattern_recognition）----
    # 只附加提醒/頸線候選，不直接產生買賣結論；三角收斂的強制觀望在下方
    # 與其他 watch_rules 降級規則一起套用。
    w_bottom_neckline = _detect_w_bottom(df, fast_slope_up, macd_bias, reasons)
    m_top_neckline = _detect_m_top(df, fast_slope_down, macd_bias, reasons)
    triangle = _detect_triangle_consolidation(df, tf, reasons, caveats)
    triangle_force_watch = triangle["active"] and not triangle["volume_confirmed"]

    platform_bias = _step6_left_side_platform(
        df, tf, latest, fast_slope_up, reasons, caveats,
        w_bottom_neckline=w_bottom_neckline, m_top_neckline=m_top_neckline)
    ma120_second_leg = _detect_ma120_second_leg(df, tf)
    extreme_deviation = _check_extreme_deviation(tf, latest, reasons)
    weekly_blowoff = _check_weekly_blowoff(tf, df, latest, reasons, caveats)

    score = (four_key_bias * scoring_cfg["four_key_weight"] + ma_bias * scoring_cfg["ma_weight"]
             + macd_bias + volume_bias + platform_bias)

    # ---- 判定「賣出減碼」：符合任一條件即可（見 SOP 規則）----
    four_key_broken = tf == "日" and four_key_bias == -1
    dead_cross_confirmed = dead_cross and ma_bias < 0
    hard_sell = (
        (key_ma_down_veto and sell_cfg.get("key_ma_down_veto", True))
        or (four_key_broken and sell_cfg.get("four_key_broken", True))
        or (dead_cross_confirmed and sell_cfg.get("dead_cross_confirmed", True))
        or (volume_bear_confirm and sell_cfg.get("volume_bear_confirm", True))
    )

    # ---- 只有 MACD 單獨背離、均線與量能都還沒同步走壞：先觀望不追空 ----
    caution_only = (
        watch_cfg.get("caution_only_on_macd_divergence", True)
        and macd_bearish_divergence and not volume_bear_confirm and not key_ma_down_veto
    )

    strong_bullish = (
        ma_bias >= add_cfg["ma_bias_min"] and macd_bias >= add_cfg["macd_bias_min"]
        and four_key_bias >= add_cfg["four_key_bias_min"] and volume_bias >= add_cfg["volume_bias_min"]
    )

    # ---- 領先指標示警（MTM／CCI 並列，同等級，任一觸發都直接壓成觀望）----
    mtm_alert_active = tf == "60分" and mtm_exit_alert and watch_cfg.get("mtm_60m_exit_downgrade", True)
    cci_alert_active = cci_exit_alert and watch_cfg.get("cci_exit_alert_downgrade", True)
    leading_indicator_exit_alert = (mtm_alert_active or cci_alert_active) and not key_ma_down_veto

    if hard_sell:
        conclusion = "賣出減碼"
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
    elif score >= buy_cfg["score_threshold"]:
        conclusion = "買進"
    elif score <= sell_cfg["score_threshold"]:
        conclusion = "賣出減碼"
    else:
        conclusion = "觀望"

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

    # ---- 特殊情境：大跌測到半年線(MA120)第二隻腳未破，只在非賣出/加碼時提示 ----
    if ma120_second_leg and RULES["special_notes"].get("ma120_second_leg_hint", True) \
            and conclusion not in ("賣出減碼", "加碼"):
        caveats.append(
            "偵測到大跌測到半年線(MA120)打出第二隻腳未破的特殊情境：可考慮「帶著鋼盔慢買」——"
            "先進 5-10% 基本部位，待5分/60分指標確認低點墊高、均線轉平緩後，每日加碼約5%"
            "（僅供分批佈局參考，非立即買進訊號）"
        )

    abs_score = abs(score)
    if key_ma_down_veto or volume_bear_confirm or (ma_bias >= add_cfg["addon_ma_bias"] and volume_bias > 0):
        confidence = "高"
    elif abs_score >= scoring_cfg["confidence_mid_abs_score"]:
        confidence = "中"
    else:
        confidence = "低" if abs_score < scoring_cfg["confidence_low_abs_score"] else "中"

    if not reasons:
        reasons.append("各項指標訊號不明顯，暫無明確方向")

    return Verdict(conclusion=conclusion, confidence=confidence, score=score,
                    reasons=reasons, caveats=caveats)


# ------------------------------------------------------------------
# 跨週期整合：「為日線留倉，為五分出場」
# 長天期（週/日）結構轉弱 → 否決短天期反彈買訊，直接判賣出/減碼
# 短天期（60分/5分）轉弱 → 不推翻長天期多頭結構，但提示先讓短打部位出場
# ------------------------------------------------------------------
def combine_timeframes(verdicts: Dict[str, Verdict]) -> dict:
    if not verdicts:
        return {"最終建議": "資料不足，無法判讀", "各週期明細": {}}

    detail = {
        tf: {"結論": v.conclusion, "信心": v.confidence, "理由": v.reasons, "但書": v.caveats}
        for tf, v in verdicts.items()
    }

    if len(verdicts) == 1:
        only_tf = next(iter(verdicts))
        return {"最終建議": verdicts[only_tf].conclusion, "各週期明細": detail}

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
