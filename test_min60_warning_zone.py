"""
測試 2026-10-21 課程新增的「60分20MA~60MA多空預警區間」規則（見
sop_rules.yaml indicators.min60_warning_zone、sop_decision.py
_step2_60min_warning_zone()）。

這條規則目前只附加caveats提示，不計入ma_bias、不影響conclusion——還沒
用backtest驗證過實際預測力，先求行為正確（跌破20MA=第一預警，跌破60MA=
第二預警），之後驗證過有效才考慮正式納入評分。

用法（在專案資料夾內執行，不需要API）：
    python3 test_min60_warning_zone.py
"""
import pandas as pd

from sop_decision import RULES, evaluate_timeframe
from stock_core import run_all_indicators


def _build_df(extra_closes):
    n = 300
    closes = [100 + i * 0.05 for i in range(n - len(extra_closes))]
    closes += extra_closes
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="h"),
        "open": closes, "max": [c + 0.3 for c in closes], "min": [c - 0.3 for c in closes],
        "close": closes, "volume": [1000] * n,
    })
    return run_all_indicators(df, "60分")


def test_first_warning_break_ma20_only():
    df = _build_df([100 + (300 - 3) * 0.05 - 0.5, 100 + (300 - 3) * 0.05 - 1.0, 100 + (300 - 3) * 0.05 - 1.3])
    latest = df.iloc[-1]
    assert latest["close"] < latest["MA20"], "這個案例應該要跌破MA20"
    assert latest["close"] >= latest["MA60"], "這個案例不該跌破MA60（才算第一預警）"

    v = evaluate_timeframe(df, "60分")
    warnings = [c for c in v.caveats if "預警" in c]
    assert len(warnings) == 1 and "20MA" in warnings[0] and "60MA" not in warnings[0].split("：")[0], \
        f"預期只出現第一預警(跌破20MA)，實際={warnings}"
    print("✅ 跌破20MA但未跌破60MA時，正確觸發第一預警：", warnings[0])


def test_second_warning_break_ma60_falling():
    extra = [100 + (300 - 10) * 0.05 - i * 1.5 for i in range(1, 11)]
    df = _build_df(extra)
    latest = df.iloc[-1]
    assert latest["close"] < latest["MA60"], "這個案例應該要跌破MA60"

    v = evaluate_timeframe(df, "60分")
    warnings = [c for c in v.caveats if "預警" in c]
    assert len(warnings) == 1 and "60MA" in warnings[0] and "第二步加碼" in warnings[0], \
        f"預期出現第二預警(60MA同步下彎、空單可加碼)，實際={warnings}"
    print("✅ 跌破60MA且60MA同步下彎時，正確觸發第二預警（空單加碼版本）：", warnings[0])


def test_disabled_flag_suppresses_warning():
    extra = [100 + (300 - 10) * 0.05 - i * 1.5 for i in range(1, 11)]
    df = _build_df(extra)

    RULES["indicators"]["min60_warning_zone"]["enabled"] = False
    try:
        v = evaluate_timeframe(df, "60分")
        warnings = [c for c in v.caveats if "預警" in c]
        assert not warnings, f"關閉開關後不該有任何預警提示，實際={warnings}"
        print("✅ 關閉開關後，正確不觸發任何預警提示")
    finally:
        RULES["indicators"]["min60_warning_zone"]["enabled"] = True  # 還原，避免影響其他測試


def test_does_not_affect_conclusion_or_score():
    """規則目前只進caveats，驗證關掉/開啟開關時conclusion跟score完全一樣。"""
    extra = [100 + (300 - 10) * 0.05 - i * 1.5 for i in range(1, 11)]
    df = _build_df(extra)

    RULES["indicators"]["min60_warning_zone"]["enabled"] = True
    v_on = evaluate_timeframe(df, "60分")
    RULES["indicators"]["min60_warning_zone"]["enabled"] = False
    v_off = evaluate_timeframe(df, "60分")
    RULES["indicators"]["min60_warning_zone"]["enabled"] = True

    assert v_on.conclusion == v_off.conclusion, "這條規則不該影響conclusion"
    assert v_on.score == v_off.score, "這條規則不該影響score（目前只進caveats）"
    print(f"✅ 開關不影響conclusion({v_on.conclusion})/score({v_on.score})，符合「先觀察不評分」的設計")


if __name__ == "__main__":
    test_first_warning_break_ma20_only()
    test_second_warning_break_ma60_falling()
    test_disabled_flag_suppresses_warning()
    test_does_not_affect_conclusion_or_score()
    print()
    print("全部通過。")
