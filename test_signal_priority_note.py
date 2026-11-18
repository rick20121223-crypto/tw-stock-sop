"""
測試 2026-11-18 新增的「CCI最快示警→MTM同步確認→KD365雙重確認→均線
裁決」訊號時序提示（見 sop_rules.yaml indicators.signal_priority_note、
sop_decision.py evaluate_timeframe()）。

這條規則確認過不影響判讀（conclusion/score完全不變），只在caveats裡附加
一句純文字提示，而且只有CCI/MTM/KD365/均線同時適用的60分線才顯示
（MTM只支援60分，見indicators.mtm.applicable_timeframes，其他週期本來
就不是四個指標同時運作，顯示這句時序提示沒有意義）。

用法（在專案資料夾內執行，不需要API）：
    python3 test_signal_priority_note.py
"""
import pandas as pd

from sop_decision import RULES, evaluate_timeframe
from stock_core import run_all_indicators

NOTE_TEXT = RULES["indicators"]["signal_priority_note"]["text"]


def _build_df(tf: str, n: int = 300):
    closes = [100 + i * 0.05 for i in range(n)]
    freq = "h" if tf == "60分" else "D"
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq=freq),
        "open": closes, "max": [c + 0.3 for c in closes], "min": [c - 0.3 for c in closes],
        "close": closes, "volume": [1000] * n,
    })
    return run_all_indicators(df, tf)


def test_note_appears_on_60min():
    df = _build_df("60分")
    v = evaluate_timeframe(df, "60分")
    assert NOTE_TEXT in v.caveats, f"60分線應該要有訊號時序提示，實際caveats={v.caveats}"
    print("✅ 60分線正確附加訊號時序提示")


def test_note_absent_on_daily():
    """日線MTM/CCI都不適用，不該顯示這句時序提示（顯示了也沒意義）。"""
    df = _build_df("日")
    v = evaluate_timeframe(df, "日")
    assert NOTE_TEXT not in v.caveats, f"日線不該出現訊號時序提示，實際caveats={v.caveats}"
    print("✅ 日線正確不附加訊號時序提示")


def test_disabled_flag_suppresses_note():
    df = _build_df("60分")
    RULES["indicators"]["signal_priority_note"]["enabled"] = False
    try:
        v = evaluate_timeframe(df, "60分")
        assert NOTE_TEXT not in v.caveats, f"關閉開關後不該出現提示，實際caveats={v.caveats}"
        print("✅ 關閉開關後，60分線正確不附加提示")
    finally:
        RULES["indicators"]["signal_priority_note"]["enabled"] = True  # 還原，避免影響其他測試


def test_does_not_affect_conclusion_or_score():
    """規則確認過只進caveats，驗證關掉/開啟開關時conclusion跟score完全一樣。"""
    df = _build_df("60分")

    RULES["indicators"]["signal_priority_note"]["enabled"] = True
    v_on = evaluate_timeframe(df, "60分")
    RULES["indicators"]["signal_priority_note"]["enabled"] = False
    v_off = evaluate_timeframe(df, "60分")
    RULES["indicators"]["signal_priority_note"]["enabled"] = True

    assert v_on.conclusion == v_off.conclusion, "這條規則不該影響conclusion"
    assert v_on.score == v_off.score, "這條規則不該影響score（純文字提示）"
    print(f"✅ 開關不影響conclusion({v_on.conclusion})/score({v_on.score})，符合「不影響判讀」的確認")


if __name__ == "__main__":
    test_note_appears_on_60min()
    test_note_absent_on_daily()
    test_disabled_flag_suppresses_note()
    test_does_not_affect_conclusion_or_score()
    print()
    print("全部通過。")
