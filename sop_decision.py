"""
但丁老師 SOP 自動判讀模組
--------------------------------
根據 stock_core.run_all_indicators() 產出的指標欄位
（四關價、均線MA、MTM、MACD、OBV、CCI、KD），綜合評分後
給出「買進／加碼／賣出減碼／觀望」的結論、信心度與理由。

純規則式評分，不做任何預測或保證，僅供輔助判讀。
被 streamlit_app.py 匯入使用：
    from sop_decision import evaluate_timeframe
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from stock_core import ma_slope


@dataclass
class Verdict:
    conclusion: str                 # 買進 / 加碼 / 賣出減碼 / 觀望
    confidence: str                 # 高 / 中 / 低
    score: float
    reasons: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
# 各指標的評分小函式：正分＝偏多，負分＝偏空
# ------------------------------------------------------------------
def _score_four_key_prices(timeframe_label: str, latest: pd.Series,
                            reasons: List[str], caveats: List[str]) -> float:
    """四關價：SOP 裡最高位階指標，僅日線適用"""
    if timeframe_label != "日線":
        caveats.append("非日線資料，四關價步驟略過")
        return 0.0
    if pd.isna(latest.get("今開")) or pd.isna(latest.get("昨高")) or pd.isna(latest.get("昨低")):
        caveats.append("資料不足，四關價步驟略過")
        return 0.0

    if latest["今開"] > latest["昨高"]:
        reasons.append("今開站上昨高，格局偏強")
        return 1.0
    if latest["今開"] < latest["昨低"]:
        reasons.append("今開跌破昨低，格局偏弱")
        return -1.0
    return 0.0


def _score_ma(df: pd.DataFrame, latest: pd.Series, reasons: List[str]) -> float:
    """均線：第一代指標，排列方向 + 斜率比價位本身更重要"""
    score = 0.0
    ma_cols = ("MA5", "MA10", "MA20")
    has_all = all(c in latest.index and pd.notna(latest[c]) for c in ma_cols)

    if has_all:
        if latest["MA5"] > latest["MA10"] > latest["MA20"]:
            score += 2
            reasons.append("均線呈多頭排列（MA5>MA10>MA20）")
        elif latest["MA5"] < latest["MA10"] < latest["MA20"]:
            score -= 2
            reasons.append("均線呈空頭排列（MA5<MA10<MA20）")

    if "MA5" in df.columns and pd.notna(latest.get("MA5")):
        slope = ma_slope(df, "MA5")
        above = latest["close"] >= latest["MA5"]
        if above and slope == "上揚":
            score += 2
            reasons.append("股價站上 MA5 且 MA5 上揚")
        elif not above and slope == "下彎":
            score -= 2
            reasons.append("股價跌破 MA5 且 MA5 下彎")

    return score


def _score_macd(df: pd.DataFrame, latest: pd.Series, reasons: List[str]) -> float:
    """MACD：零軸位置 + 黃金/死亡交叉"""
    score = 0.0
    if pd.isna(latest.get("DIF")) or pd.isna(latest.get("MACD_signal")):
        return score

    if latest["DIF"] > 0:
        score += 1
        reasons.append("MACD DIF 位於零軸之上")
    else:
        score -= 1
        reasons.append("MACD DIF 位於零軸之下")

    if len(df) >= 2:
        prev = df.iloc[-2]
        if pd.notna(prev.get("DIF")) and pd.notna(prev.get("MACD_signal")):
            golden_cross = prev["DIF"] <= prev["MACD_signal"] and latest["DIF"] > latest["MACD_signal"]
            dead_cross = prev["DIF"] >= prev["MACD_signal"] and latest["DIF"] < latest["MACD_signal"]
            if golden_cross:
                score += 2
                reasons.append("MACD 出現黃金交叉（DIF 上穿訊號線）")
            elif dead_cross:
                score -= 2
                reasons.append("MACD 出現死亡交叉（DIF 下穿訊號線）")

    return score


def _score_mtm(df: pd.DataFrame, latest: pd.Series, reasons: List[str]) -> float:
    """MTM：動量方向 + 近期翻多/翻空"""
    score = 0.0
    if pd.isna(latest.get("MTM")):
        return score

    score += 1 if latest["MTM"] > 0 else -1

    recent = df.tail(3)
    cross_up = ((recent["MTM"] > 0) & (recent["MTM"].shift(1) <= 0)).any()
    cross_down = ((recent["MTM"] < 0) & (recent["MTM"].shift(1) >= 0)).any()
    if cross_up:
        score += 2
        reasons.append("MTM 近期翻多（由負轉正）")
    elif cross_down:
        score -= 2
        reasons.append("MTM 近期翻空（由正轉負）")

    return score


def _score_cci(latest: pd.Series, reasons: List[str]) -> float:
    """CCI：三代領先指標，配合 MTM 判斷零軸突破/跌破"""
    if pd.isna(latest.get("CCI")):
        return 0.0
    if latest["CCI"] > 100:
        reasons.append("CCI 站上 +100，強勢突破區")
        return 2.0
    if latest["CCI"] < -100:
        reasons.append("CCI 跌破 -100，弱勢跌破區")
        return -2.0
    return 0.0


def _score_obv(df: pd.DataFrame, latest: pd.Series,
               reasons: List[str], caveats: List[str]) -> float:
    """OBV：量價驗證，指數無成交量資料時略過"""
    if pd.isna(latest.get("OBV")):
        caveats.append("此標的無成交量資料，OBV 量價驗證略過")
        return 0.0

    slope = ma_slope(df, "OBV_MA")
    if slope == "上揚":
        reasons.append("OBV 均線上揚，量能同步價格")
        return 1.0
    if slope == "下彎":
        reasons.append("OBV 均線下彎，量能未同步價格")
        return -1.0
    return 0.0


# ------------------------------------------------------------------
# 主入口：綜合評分 → 結論
# ------------------------------------------------------------------
def evaluate_timeframe(df: pd.DataFrame, timeframe_label: str) -> Verdict:
    """
    根據單一週期（日線/60分線/5分線）已跑過 run_all_indicators() 的資料，
    綜合評分給出 SOP 結論。

    評分規則（僅供輔助判讀，非投資建議）：
        score >= 6   → 加碼
        score >= 3   → 買進
        score <= -3  → 賣出減碼
        其餘          → 觀望
    信心度依 |score| 分高/中/低三級。
    """
    if df.empty:
        return Verdict(conclusion="觀望", confidence="低", score=0.0,
                        reasons=["資料為空，無法判讀"], caveats=[])

    latest = df.iloc[-1]
    reasons: List[str] = []
    caveats: List[str] = []

    score = 0.0
    score += _score_four_key_prices(timeframe_label, latest, reasons, caveats)
    score += _score_ma(df, latest, reasons)
    score += _score_macd(df, latest, reasons)
    score += _score_mtm(df, latest, reasons)
    score += _score_cci(latest, reasons)
    score += _score_obv(df, latest, reasons, caveats)

    if score >= 6:
        conclusion = "加碼"
    elif score >= 3:
        conclusion = "買進"
    elif score <= -3:
        conclusion = "賣出減碼"
    else:
        conclusion = "觀望"

    abs_score = abs(score)
    if abs_score >= 8:
        confidence = "高"
    elif abs_score >= 4:
        confidence = "中"
    else:
        confidence = "低"

    if not reasons:
        reasons.append("各項指標訊號不明顯，暫無明確方向")

    return Verdict(conclusion=conclusion, confidence=confidence, score=score,
                    reasons=reasons, caveats=caveats)
