"""
測試「MTM不能用於中繼段/盤整」規則（見 sop_rules.yaml
indicators.mtm.suppress_during_consolidation、sop_decision.py
evaluate_timeframe() 裡的 mtm_suppressed_by_consolidation）。

背景：2026-10-21課程提到MTM絕不能用於中繼段/高檔震盪盤，會引發頻繁
雙巴假訊號、嚴重凹單。系統裡剛好已經有三角收斂整理區間的偵測
（_detect_triangle_consolidation），這裡重用同一套判斷：仍處於收斂
整理區間、且還沒帶量突破確認時（跟triangle_force_watch同一個條件），
MTM翻空不觸發「短線先出場」示警降級——MTM本身的reasons文字仍然照常
附加（保留透明度），只是不讓它驅動conclusion，並改在caveats說明被
忽略的原因。

用法（在專案資料夾內執行，不需要API）：
    python3 test_mtm_consolidation_suppress.py
"""
import numpy as np
import pandas as pd

from sop_decision import RULES, evaluate_timeframe
from stock_core import run_all_indicators


def _build_consolidation_with_mtm_dip():
    """110根上升趨勢後，接40根窄幅震盪（早半段震幅較大、後半段收斂），
    構成一個未突破的三角收斂區間，且震盪尾段MTM翻負。"""
    n = 150
    closes = [100 + i * 0.3 for i in range(110)]
    base = closes[-1]
    earlier = [base + 6 * np.sin(i * 0.9) for i in range(20)]
    recent = [base + 0.8 * np.sin(i * 0.9) for i in range(20)]
    closes += earlier + recent
    highs = [c + 0.3 for c in closes]
    lows = [c - 0.3 for c in closes]
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="h"),
        "open": closes, "max": highs, "min": lows, "close": closes,
        "volume": [1000] * n,
    })
    return run_all_indicators(df, "60分")


def test_suppressed_when_enabled_in_consolidation():
    df = _build_consolidation_with_mtm_dip()
    assert df["MTM"].iloc[-1] < 0, "測試資料本身應該要讓MTM翻負（前提不成立）"

    RULES["indicators"]["mtm"]["suppress_during_consolidation"] = True
    v = evaluate_timeframe(df, "60分")

    assert any("MTM" in r for r in v.reasons), "MTM本身的reasons文字應該照常附加"
    suppress_caveats = [c for c in v.caveats if "不採用" in c]
    assert suppress_caveats, f"預期出現『不採用』的抑制說明caveat，實際caveats={v.caveats}"
    exit_caveats = [c for c in v.caveats if "已示警，建議短線先出場觀察" in c]
    assert not exit_caveats, f"啟用抑制時不該出現正常的MTM示警降級caveat，實際={exit_caveats}"
    print(f"✅ 三角收斂整理區間內，MTM翻空被正確抑制：{suppress_caveats[0]}")


def test_not_suppressed_when_disabled():
    df = _build_consolidation_with_mtm_dip()

    RULES["indicators"]["mtm"]["suppress_during_consolidation"] = False
    try:
        v = evaluate_timeframe(df, "60分")
        exit_caveats = [c for c in v.caveats if "已示警，建議短線先出場觀察" in c]
        assert exit_caveats, f"關閉開關後應該退回正常MTM示警降級，實際caveats={v.caveats}"
        suppress_caveats = [c for c in v.caveats if "不採用" in c]
        assert not suppress_caveats, f"關閉開關後不該出現抑制說明，實際={suppress_caveats}"
        print(f"✅ 關閉開關後，正確退回舊行為（正常MTM示警降級）：{exit_caveats[0]}")
    finally:
        RULES["indicators"]["mtm"]["suppress_during_consolidation"] = True  # 還原


if __name__ == "__main__":
    test_suppressed_when_enabled_in_consolidation()
    test_not_suppressed_when_disabled()
    print()
    print("全部通過。")
