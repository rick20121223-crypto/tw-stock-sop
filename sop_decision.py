"""
但丁老師 SOP 自動判讀模組
--------------------------------
依「四關價 → 均線(含斜率) → MACD → OBV/BBI(/MTM)」的位階順序逐層檢查，
高位階訊號可以否決低位階訊號（尤其：生死線下彎時，任何買訊都要降級；
真正的頂背離必須「MACD背離 + OBV/BBI量價背離」同步出現才算數）。

單一週期判讀：evaluate_timeframe()
跨週期整合（日/週/60分/5分）：combine_timeframes()，採「為日線留倉，
為五分出場」— 長天期結構轉弱可以否決短天期的反彈買訊；短天期轉弱不會
推翻長天期多頭結構，但會提示先幫短打部位出場。

純規則式評分，不做任何預測或保證，僅供輔助判讀。
被 streamlit_app.py / pages/*.py / multi_timeframe_check.py 匯入使用：
    from sop_decision import evaluate_timeframe, combine_timeframes
"""

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from stock_core import FAST_MA, KEY_MA, ma_slope, normalize_timeframe


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
    if tf != "日":
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
    ma_bias = 0.0
    key_ma_down_veto = False

    fast_col, key_col = FAST_MA[tf], KEY_MA[tf]

    if fast_col in latest.index and pd.notna(latest.get(fast_col)):
        fast_slope = ma_slope(df, fast_col)
        above_fast = latest["close"] >= latest[fast_col]
        if above_fast and fast_slope == "上揚":
            ma_bias += 1
            reasons.append(f"{fast_col} 上揚且股價站上，短線動能偏多")
        elif (not above_fast) and fast_slope == "下彎":
            ma_bias -= 1
            reasons.append(f"{fast_col} 下彎且股價跌破，短線動能偏空")
    else:
        caveats.append(f"{fast_col} 資料不足")

    if key_col in latest.index and pd.notna(latest.get(key_col)):
        key_slope = ma_slope(df, key_col)
        above_key = latest["close"] >= latest[key_col]
        if key_slope == "下彎":
            ma_bias -= 2
            reasons.append(f"⚠️ 生死線 {key_col} 已下彎，任何買訊都要降級")
            if not above_key:
                key_ma_down_veto = True
                reasons.append(f"且股價已跌破 {key_col}，結構偏空，反彈視為誘多假動作")
        elif key_slope == "上揚" and above_key:
            ma_bias += 2
            reasons.append(f"生死線 {key_col} 上揚且站上，中長線結構偏多")
    else:
        caveats.append(f"{key_col} 資料不足（可能是資料期間不夠長）")

    return ma_bias, key_ma_down_veto


# ------------------------------------------------------------------
# 簡化版頂背離偵測：近期股價創高，但 MACD 柱狀圖高點未同步創高。
# 這是簡化偵測，非教科書式嚴謹型態辨識，僅供參考。
# ------------------------------------------------------------------
def _detect_bearish_divergence(df: pd.DataFrame, lookback: int = 20) -> bool:
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
    macd_bias = 0.0
    dead_cross = False

    if pd.isna(latest.get("DIF")) or pd.isna(latest.get("MACD_signal")):
        caveats.append("MACD 資料不足")
        return macd_bias, dead_cross, False

    if latest["DIF"] > 0:
        macd_bias += 1
        reasons.append("MACD DIF 位於零軸之上")
    else:
        macd_bias -= 1
        reasons.append("MACD DIF 位於零軸之下")

    if len(df) >= 2:
        prev = df.iloc[-2]
        if pd.notna(prev.get("DIF")) and pd.notna(prev.get("MACD_signal")):
            golden = prev["DIF"] <= prev["MACD_signal"] and latest["DIF"] > latest["MACD_signal"]
            dead = prev["DIF"] >= prev["MACD_signal"] and latest["DIF"] < latest["MACD_signal"]
            if golden:
                if latest["DIF"] < 0:
                    macd_bias += 1
                    reasons.append("MACD 零軸下黃金交叉，僅屬弱勢反彈，不當作進場訊號")
                else:
                    macd_bias += 2
                    reasons.append("MACD 黃金交叉（零軸之上），動能轉強")
            elif dead:
                macd_bias -= 2
                dead_cross = True
                reasons.append("MACD 出現死亡交叉（DIF 下穿訊號線）")

    bearish_divergence = _detect_bearish_divergence(df)
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
    volume_bias = 0.0
    volume_bear_confirm = False

    if tf == "5分":
        if "BBI" in latest.index and pd.notna(latest.get("BBI")):
            bbi_slope = ma_slope(df, "BBI")
            if bbi_slope == "上揚":
                volume_bias += 1
                reasons.append("BBI 上揚，短線多方結構延續")
            elif bbi_slope == "下彎":
                volume_bias -= 1
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

    obv_slope = ma_slope(df, "OBV_MA")
    if obv_slope == "上揚":
        volume_bias += 1
        reasons.append("OBV 均線上揚，量能同步價格")
    elif obv_slope == "下彎":
        volume_bias -= 1
        reasons.append("OBV 均線下彎，量能未同步價格")
        if macd_bearish_divergence:
            volume_bear_confirm = True
            reasons.append("MACD 頂背離 + OBV 量價背離同步出現，才是較可信的頂背離")

    return volume_bias, volume_bear_confirm


# ------------------------------------------------------------------
# Step 5：MTM（僅 60分線適用，領先示警，比均線更早示警主力撤退）
# ------------------------------------------------------------------
def _step5_mtm(df: pd.DataFrame, tf: str, latest: pd.Series, reasons: List[str]) -> bool:
    if tf != "60分" or pd.isna(latest.get("MTM")):
        return False

    recent = df.tail(3)
    cross_down = ((recent["MTM"] < 0) & (recent["MTM"].shift(1) >= 0)).any()
    if cross_down or latest["MTM"] < 0:
        reasons.append("MTM 翻空/死叉，60分線短線出場訊號優先示警")
        return True
    return False


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

    tf = normalize_timeframe(timeframe_label)
    latest = df.iloc[-1]
    reasons: List[str] = []
    caveats: List[str] = []

    four_key_bias = _step1_four_key_prices(tf, latest, reasons, caveats)
    ma_bias, key_ma_down_veto = _step2_ma(df, tf, latest, reasons, caveats)
    macd_bias, dead_cross, macd_bearish_divergence = _step3_macd(df, latest, reasons, caveats)
    volume_bias, volume_bear_confirm = _step4_volume(
        df, tf, latest, macd_bearish_divergence, reasons, caveats)
    mtm_exit_alert = _step5_mtm(df, tf, latest, reasons)

    score = four_key_bias * 2 + ma_bias * 2 + macd_bias + volume_bias

    # ---- 判定「賣出減碼」：符合任一條件即可（見 SOP 規則）----
    four_key_broken = tf == "日" and four_key_bias == -1
    dead_cross_confirmed = dead_cross and ma_bias < 0
    hard_sell = key_ma_down_veto or four_key_broken or dead_cross_confirmed or volume_bear_confirm

    # ---- 只有 MACD 單獨背離、均線與量能都還沒同步走壞：先觀望不追空 ----
    caution_only = macd_bearish_divergence and not volume_bear_confirm and not key_ma_down_veto

    if hard_sell:
        conclusion = "賣出減碼"
    elif tf == "60分" and mtm_exit_alert and not key_ma_down_veto:
        conclusion = "觀望"
        caveats.append("60分 MTM 已示警，建議短線先出場觀察，暫不否定較長天期結構")
    elif caution_only:
        conclusion = "觀望"
        caveats.append("MACD出現警戒訊號，但均線與量能尚未同步走壞，先觀望、不追空")
    elif ma_bias >= 3 and macd_bias >= 1 and four_key_bias >= 0 and volume_bias >= 0:
        conclusion = "加碼" if (ma_bias >= 4 and volume_bias > 0) else "買進"
    elif score >= 3:
        conclusion = "買進"
    elif score <= -3:
        conclusion = "賣出減碼"
    else:
        conclusion = "觀望"

    abs_score = abs(score)
    if key_ma_down_veto or volume_bear_confirm or (ma_bias >= 4 and volume_bias > 0):
        confidence = "高"
    elif abs_score >= 4:
        confidence = "中"
    else:
        confidence = "低" if abs_score < 2 else "中"

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
        return {
            "最終建議": "加碼（各週期已同步翻多）" if all_bullish else "買進（部分週期訊號仍待確認）",
            "各週期明細": detail,
        }

    if long_watch_only:
        return {"最終建議": "觀望（長天期尚未表態，不追短線訊號）", "各週期明細": detail}

    # 只有短天期資料可用（沒有日/週資料能交叉驗證）
    if not long_tfs and short_tfs:
        if short_bear_alert:
            return {"最終建議": "短線賣出減碼（無長天期資料交叉驗證，僅供短打參考）", "各週期明細": detail}
        if all(verdicts[tf].conclusion in ("買進", "加碼") for tf in short_tfs):
            return {"最終建議": "短線買進（無長天期資料交叉驗證，僅供短打參考）", "各週期明細": detail}

    return {"最終建議": "觀望", "各週期明細": detail}
