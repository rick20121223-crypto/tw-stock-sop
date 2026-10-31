"""
測試2026-10-31新增的「均線48%／四關價20%／指標組(MACD+KD+CCI)20%」加權計分
模式（weighted_48_20_20），對照但丁老師課程錄音(1030)分享的程式單權重配比邏輯
（見sop_rules.yaml indicators.scoring區塊註解）。

預設scoring.mode維持"legacy"（現有公式，完全不變），只有手動切成
"weighted_48_20_20"才會套用這組新公式。這支測試驗證：
1. 正規化/加權計分的純函式行為正確。
2. CCI新增的cci_bias（重用現有bull_foldback/bear_foldback狀態，不改動
   cci_exit_alert既有的降級行為）。
3. 預設legacy模式下，mode維持原值不變；切換mode後score確實改變
   （確認新公式真的被套用，而非誤判預設行為已經走新路徑）。

用法（在專案資料夾內執行，不需要API）：
    python3 test_scoring_weighted_48_20_20.py
"""
import pandas as pd

from sop_decision import RULES, _normalize_bias, _weighted_score_48_20_20, _step5b_cci, evaluate_timeframe
from stock_core import run_all_indicators


def test_normalize_bias_clips_to_unit_range():
    assert _normalize_bias(2.0, 1.0) == 1.0, "超過max_abs應該clip到+1"
    assert _normalize_bias(-2.0, 1.0) == -1.0, "超過max_abs應該clip到-1"
    assert _normalize_bias(0.5, 1.0) == 0.5
    assert _normalize_bias(1.0, 0.0) == 0.0, "max_abs為0時應回傳0，避免除以0"
    print("✅ _normalize_bias 正確clip到[-1, 1]")


def test_weighted_score_uses_two_way_indicator_group_without_cci():
    cfg = RULES["indicators"]["scoring"]["weighted_48_20_20"]
    # 日線：均線全多、四關價全多、MACD全多、KD全空 → 指標組(日線只有MACD+KD)=(1-1)/2=0
    score = _weighted_score_48_20_20(
        four_key_bias=1, ma_bias=cfg["ma_bias_max_abs"], macd_bias=cfg["macd_bias_max_abs"],
        kd365_bias=-cfg["kd365_bias_max_abs"], cci_bias=0, tf="日", cfg=cfg)
    expected = (0.48 * 1 + 0.20 * 1 + 0.20 * 0) * cfg["scale"]
    assert abs(score - expected) < 1e-9, f"日線不該用到CCI，預期{expected}，實際{score}"
    print(f"✅ 日線指標組只用MACD+KD平均（不含CCI），score={score}")


def test_weighted_score_includes_cci_on_60min():
    cfg = RULES["indicators"]["scoring"]["weighted_48_20_20"]
    # 60分：均線/四關價/MACD/KD都全多，CCI全空 → 指標組=(1+1-1)/3
    score = _weighted_score_48_20_20(
        four_key_bias=1, ma_bias=cfg["ma_bias_max_abs"], macd_bias=cfg["macd_bias_max_abs"],
        kd365_bias=cfg["kd365_bias_max_abs"], cci_bias=-cfg["cci_bias_max_abs"], tf="60分", cfg=cfg)
    indicator_group = (1 + 1 - 1) / 3
    expected = (0.48 * 1 + 0.20 * 1 + 0.20 * indicator_group) * cfg["scale"]
    assert abs(score - expected) < 1e-9, f"60分指標組應含CCI，預期{expected}，實際{score}"
    print(f"✅ 60分指標組含CCI三項平均，score={score}")


def _build_cci_df(cci_values):
    n = len(cci_values)
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="h"),
        "open": [100] * n, "max": [101] * n, "min": [99] * n,
        "close": [100] * n, "volume": [1000] * n,
        "CCI": cci_values,
    })


def test_cci_bias_bear_foldback():
    # 注意：正式config的lookback_bars(3)跟confirm_bars(3)目前相等，兩個檢查
    # 窗口會重疊，導致bear/bull_foldback在confirm_bars>1分支下實際上永遠
    # 觸發不了（已回報給使用者、另開task追蹤，非本次任務範圍）。這裡暫時
    # 調大lookback_bars，只為了獨立驗證cci_bias接線本身邏輯正確，跟正式
    # confirm_bars/lookback_bars的數值校正無關。
    cci_cfg = RULES["indicators"]["cci"]
    orig_lookback = cci_cfg["lookback_bars"]
    cci_cfg["lookback_bars"] = 5
    try:
        values = [50] * 5 + [120, 50, 50, 50, 50]
        df = _build_cci_df(values)
        reasons, caveats = [], []
        cci_exit_alert, cci_bias = _step5b_cci(df, "60分", df.iloc[-1], reasons, caveats)
    finally:
        cci_cfg["lookback_bars"] = orig_lookback
    assert cci_exit_alert is True, "由+100之上折返跌破應觸發既有的cci_exit_alert（行為不變）"
    assert cci_bias == -1, f"熊市折返應給cci_bias=-1，實際={cci_bias}"
    print("✅ CCI熊市折返：cci_exit_alert=True（行為不變），cci_bias=-1（新增）")


def test_cci_bias_bull_foldback():
    cci_cfg = RULES["indicators"]["cci"]
    orig_lookback = cci_cfg["lookback_bars"]
    cci_cfg["lookback_bars"] = 5
    try:
        values = [-50] * 5 + [-120, -50, -50, -50, -50]
        df = _build_cci_df(values)
        reasons, caveats = [], []
        cci_exit_alert, cci_bias = _step5b_cci(df, "60分", df.iloc[-1], reasons, caveats)
    finally:
        cci_cfg["lookback_bars"] = orig_lookback
    assert cci_exit_alert is False, "牛市折返不改變既有的cci_exit_alert行為（仍只是caveat提示）"
    assert cci_bias == 1, f"牛市折返應給cci_bias=1，實際={cci_bias}"
    print("✅ CCI牛市折返：cci_exit_alert=False（行為不變），cci_bias=1（新增）")


def test_cci_bias_neutral_when_no_foldback():
    df = _build_cci_df([50] * 11)
    reasons, caveats = [], []
    cci_exit_alert, cci_bias = _step5b_cci(df, "60分", df.iloc[-1], reasons, caveats)
    assert cci_exit_alert is False
    assert cci_bias == 0
    print("✅ CCI無折返時cci_bias=0")


def test_default_mode_is_legacy_and_switching_changes_score():
    n = 300
    closes = [100 + i * 0.3 for i in range(n)]
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "open": closes, "max": [c + 0.5 for c in closes], "min": [c - 0.5 for c in closes],
        "close": closes, "volume": [1000] * n,
    })
    df_i = run_all_indicators(df, "日")

    assert RULES["indicators"]["scoring"].get("mode", "legacy") == "legacy", \
        "預設mode必須維持legacy，確保改動前的行為不變"

    v_legacy = evaluate_timeframe(df_i.copy(), "日")

    RULES["indicators"]["scoring"]["mode"] = "weighted_48_20_20"
    try:
        v_weighted = evaluate_timeframe(df_i.copy(), "日")
    finally:
        RULES["indicators"]["scoring"]["mode"] = "legacy"

    assert v_weighted.score != v_legacy.score, \
        "切換成weighted_48_20_20後score應該跟legacy公式不同（確認新公式真的被套用）"
    print(f"✅ legacy score={v_legacy.score}, weighted_48_20_20 score={v_weighted.score}（確認mode切換有效）")


if __name__ == "__main__":
    test_normalize_bias_clips_to_unit_range()
    test_weighted_score_uses_two_way_indicator_group_without_cci()
    test_weighted_score_includes_cci_on_60min()
    test_cci_bias_bear_foldback()
    test_cci_bias_bull_foldback()
    test_cci_bias_neutral_when_no_foldback()
    test_default_mode_is_legacy_and_switching_changes_score()
    print()
    print("全部通過。")
