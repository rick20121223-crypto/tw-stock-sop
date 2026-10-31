"""
測試2026-10-31新增的「夾起單洗盤 + 半年線(MA120)兩次腳未破」加分項
（indicators.washout_recovery_second_leg，見sop_decision.py
_step2_washout_recovery_with_support()）。

背景：test_washout_recovery_buy.py只測「單純夾起單」（假跌破生死線MA35
後迅速站回），24個月/13檔回測結果比無條件基準還差，所以沒有接進評分。
但原始SOP描述裡「左側平台支撐未破」跟「夾起單」本來就是綁在一起講的，
不是獨立訊號——另外用test_washout_recovery_with_support_backtest.py
驗證過組合版（夾起單 AND 兩次腳未破同時成立），10日報酬4.05%/勝率
63.1%明顯優於無條件基準3.50%/57.0%，單獨使用任一條件都沒有這個優勢，
所以這次只把「兩者同時成立」接進加分，不是任一個單獨就加分。

只適用日線（MA35/MA120皆為日線生死線概念，跟ma120_second_leg本身的
applicable_timeframes一致）。

用合成資料，不需要API，執行：
    python3 test_washout_recovery_second_leg_bonus.py
"""
import pandas as pd

from sop_decision import RULES, _step2_washout_recovery_with_support
from stock_core import run_all_indicators


def _build_df(closes, ma35_override=None):
    n = len(closes)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "open": closes, "max": [c + 0.3 for c in closes], "min": [c - 0.3 for c in closes],
        "close": closes, "volume": [1000] * n,
    })
    df = run_all_indicators(df, "日")
    if ma35_override is not None:
        df["MA35"] = ma35_override
    return df


def test_no_bonus_when_second_leg_not_detected():
    closes = [100.0] * 300
    df = _build_df(closes)
    reasons = []
    bonus = _step2_washout_recovery_with_support(df, "日", df.iloc[-1], ma120_second_leg=False, reasons=reasons)
    assert bonus == 0.0, f"沒有兩次腳未破時不該加分，實際={bonus}"
    assert not reasons
    print("✅ 沒有「兩次腳未破」時不加分（即使夾起單本身成立）")


def test_no_bonus_when_second_leg_true_but_no_washout():
    n = 300
    closes = [100.0] * n
    ma35 = [100.0] * n  # 股價一直貼在MA35上，從未跌破也沒有拉回站回的動作
    df = _build_df(closes, ma35_override=pd.Series(ma35))
    reasons = []
    bonus = _step2_washout_recovery_with_support(df, "日", df.iloc[-1], ma120_second_leg=True, reasons=reasons)
    assert bonus == 0.0, f"有兩次腳未破但沒有夾起單動作時不該加分，實際={bonus}"
    print("✅ 有「兩次腳未破」但沒有夾起單動作時不加分")


def test_bonus_when_both_washout_and_second_leg_true():
    n = 300
    closes = [100.0] * n
    ma35 = [100.0] * n
    # 最近3天內有一天收盤跌破MA35(99 < 100)，今天收回站上(close>=MA35)
    closes[-2] = 99.0
    ma35_series = pd.Series(ma35)
    df = _build_df(closes, ma35_override=ma35_series)
    assert df["close"].iloc[-2] < df["MA35"].iloc[-2], "測試資料要先確認真的跌破過"
    assert df["close"].iloc[-1] >= df["MA35"].iloc[-1], "測試資料要先確認今天收回站上"
    reasons = []
    bonus = _step2_washout_recovery_with_support(df, "日", df.iloc[-1], ma120_second_leg=True, reasons=reasons)
    cfg = RULES["indicators"]["washout_recovery_second_leg"]
    assert bonus == cfg["bonus"], f"兩者同時成立應該加{cfg['bonus']}分，實際={bonus}"
    assert any("夾起單" in r for r in reasons)
    print(f"✅ 夾起單+兩次腳未破同時成立時加分 {bonus}，reasons有說明")


def test_not_applicable_outside_daily_timeframe():
    n = 300
    closes = [100.0] * n
    closes[-2] = 99.0
    df = _build_df(closes, ma35_override=pd.Series([100.0] * n))
    reasons = []
    bonus = _step2_washout_recovery_with_support(df, "60分", df.iloc[-1], ma120_second_leg=True, reasons=reasons)
    assert bonus == 0.0, f"非日線不該加分，實際={bonus}"
    print("✅ 非日線（60分）不套用這條加分")


def test_disabled_flag_suppresses_bonus():
    n = 300
    closes = [100.0] * n
    closes[-2] = 99.0
    df = _build_df(closes, ma35_override=pd.Series([100.0] * n))
    RULES["indicators"]["washout_recovery_second_leg"]["enabled"] = False
    try:
        reasons = []
        bonus = _step2_washout_recovery_with_support(df, "日", df.iloc[-1], ma120_second_leg=True, reasons=reasons)
    finally:
        RULES["indicators"]["washout_recovery_second_leg"]["enabled"] = True
    assert bonus == 0.0, f"關閉開關後不該加分，實際={bonus}"
    print("✅ 開關關閉時不加分")


if __name__ == "__main__":
    test_no_bonus_when_second_leg_not_detected()
    test_no_bonus_when_second_leg_true_but_no_washout()
    test_bonus_when_both_washout_and_second_leg_true()
    test_not_applicable_outside_daily_timeframe()
    test_disabled_flag_suppresses_bonus()
    print()
    print("全部通過。")
