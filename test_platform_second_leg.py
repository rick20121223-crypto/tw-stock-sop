"""
測試「左側平台/前低兩次腳未破」的廣義版規則（見 sop_rules.yaml
indicators.platform_second_leg、sop_decision.py _detect_platform_second_leg()）。

背景：原本的_detect_ma120_second_leg()只認「半年線(MA120)」當支撐參考、
只支援日線，但但丁老師課程講的是更廣義的「左側歷史平台區（或前低）打出
兩次腳，底不破就不能砍單」。2026-10-23改成重用Step6
（_find_platform_support_resistance）已經找到的左側支撐候選，套用同一套
「兩腳未破」演算法（_legs_untested_check，從_detect_ma120_second_leg()
抽出來），不限定MA120、不限定日線。

這條規則目前只附加caveats提示（跟既有ma120版本一樣），不直接否決賣出
減碼、不直接升級買進/加碼——還沒backtest驗證過，先求行為正確。

用法（在專案資料夾內執行，不需要API）：
    python3 test_platform_second_leg.py
"""
import pandas as pd

from sop_decision import RULES, _detect_platform_second_leg


def _build_double_leg_case():
    """合成60根日K：緩降後在~81附近打出兩次腳（中間反彈到90），第二腳
    (~80.5)沒有跌破第一腳(~81)超過2%，且發生在最近3根K棒內。"""
    n = 60
    closes = [110 - i * 0.7 for i in range(40)]
    closes += [81.5, 81.2, 81.0, 81.3, 81.6]      # idx40-44 第一腳觸及支撐~81
    closes += [83, 86, 89, 90.5, 90.0, 88.0]       # idx45-50 反彈到90(>=81*1.05)
    closes += [85, 83, 81.8, 80.8, 80.5, 80.7]     # idx51-56 第二腳觸及支撐~80.5(未破)
    closes += [82.0, 83.0, 83.5]                    # idx57-59 最近幾根
    assert len(closes) == n

    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n),
        "open": closes, "max": highs, "min": lows, "close": closes,
    })
    return df


def test_detects_double_leg_on_platform_support():
    df = _build_double_leg_case()
    latest = df.iloc[-1]
    support_levels = [(81.0, "整理平台（近期多次觸碰確認）")]

    RULES["indicators"]["platform_second_leg"]["enabled"] = True
    level = _detect_platform_second_leg(df, "日", latest, support_levels)

    assert level is not None, "預期偵測到兩次腳未破，實際沒有"
    assert abs(level - 81.0) < 0.01, f"預期偵測到的支撐價位是81.0附近，實際={level}"
    print(f"✅ 正確偵測到兩次腳未破的平台支撐：約{level}")


def test_disabled_flag_suppresses_detection():
    df = _build_double_leg_case()
    latest = df.iloc[-1]
    support_levels = [(81.0, "整理平台（近期多次觸碰確認）")]

    RULES["indicators"]["platform_second_leg"]["enabled"] = False
    try:
        level = _detect_platform_second_leg(df, "日", latest, support_levels)
        assert level is None, f"關閉開關後不該偵測到任何東西，實際={level}"
        print("✅ 關閉開關後，正確不觸發偵測")
    finally:
        RULES["indicators"]["platform_second_leg"]["enabled"] = True


def test_no_support_levels_returns_none():
    df = _build_double_leg_case()
    latest = df.iloc[-1]
    level = _detect_platform_second_leg(df, "日", latest, [])
    assert level is None, f"沒有支撐候選時應該回傳None，實際={level}"
    print("✅ 沒有Step6支撐候選時，正確不觸發偵測")


def test_applies_to_non_daily_timeframe():
    """廣義版沒有applicable_timeframes限制，週線也該適用（原本的
    ma120_second_leg只支援日線）。"""
    df = _build_double_leg_case()
    latest = df.iloc[-1]
    support_levels = [(81.0, "整理平台（近期多次觸碰確認）")]

    level = _detect_platform_second_leg(df, "週", latest, support_levels)
    assert level is not None, "廣義版應該不限timeframe，週線也該偵測到"
    print(f"✅ 週線（非日線）一樣能偵測到兩次腳未破：約{level}")


def _build_undercut_case():
    """跟_build_double_leg_case()同一個骨架，但第二腳整段（不是單一根K棒）
    都明顯跌破第一腳。單一根K棒的孤立深跌會被_legs_untested_check()的leg
    分段邏輯拆成獨立一腳、反而被忽略不比較——這是演算法本身既有的特性
    （沿用自原本的_detect_ma120_second_leg()，不是這次新增的行為），所以
    這裡故意讓「跌破」橫跨好幾根K棒，確保它會被算進最後一腳的範圍內。"""
    n = 60
    closes = [110 - i * 0.7 for i in range(40)]
    closes += [81.5, 81.2, 81.0, 81.3, 81.6]
    closes += [83, 86, 89, 90.5, 90.0, 88.0]
    closes += [85, 83, 76.5, 75.8, 75.5, 75.7]   # 第二腳整段明顯跌破~81*0.98=79.38
    closes += [77.0, 78.0, 78.5]
    assert len(closes) == n

    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n),
        "open": closes, "max": highs, "min": lows, "close": closes,
    })


def test_undercut_breaks_pattern():
    """第二腳如果真的跌破第一腳超過2%，就不該算「未破」。"""
    df = _build_undercut_case()
    latest = df.iloc[-1]
    support_levels = [(81.0, "整理平台（近期多次觸碰確認）")]

    level = _detect_platform_second_leg(df, "日", latest, support_levels)
    assert level is None, f"第二腳已經明顯跌破，不該判定為兩次腳未破，實際={level}"
    print("✅ 第二腳明顯跌破時，正確不判定為「兩次腳未破」")


if __name__ == "__main__":
    test_detects_double_leg_on_platform_support()
    test_disabled_flag_suppresses_detection()
    test_no_support_levels_returns_none()
    test_applies_to_non_daily_timeframe()
    test_undercut_breaks_pattern()
    print()
    print("全部通過。")
