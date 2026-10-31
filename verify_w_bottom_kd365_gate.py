"""
驗證KD365 Double Check降級conclusion（買進側Phase 3）：
W底型態偵測到、但均線斜率/MACD/KD(36,5,5)尚未同步確認時，買進/加碼結論
強制降級為觀望。

用monkeypatch直接把evaluate_timeframe()裡其他的輔助step函式（_step1_
four_key_prices等）設成中性/有利值，讓測試只隔離觀察「這次新增的W底
降級規則」造成的結論變化——不需要手工構造完整的W底價格型態（型態偵測
本身已經在verify_w_bottom_kd365.py、test_kd365_w_bottom_buy.py驗證過，
這裡只測「偵測結果接進conclusion」這一段新邏輯）。

MA5/MA35欄位改用真實欄位覆寫（不monkeypatch），因為evaluate_timeframe()
裡的fast_slope_up/key_slope_down是直接呼叫ma_slope(df, ...)算出來的，
不是從_step2_ma()的回傳值來——monkeypatch _step2_ma()本身不會影響這兩個
變數，這兩組欄位分別需要分開處理才能精準控制。

用合成資料，不需要FinMind token，執行：
    python3 verify_w_bottom_kd365_gate.py
"""
import contextlib
import numpy as np
import pandas as pd

import sop_decision as sd
from sop_decision import RULES
from stock_core import run_all_indicators

_NEUTRALIZED_STEPS = [
    "_step1_four_key_prices", "_step2_ma", "_step3_macd", "_step4_volume",
    "_step5_mtm", "_step5b_cci", "_detect_m_top",
    "_detect_triangle_consolidation", "_detect_ma120_second_leg",
    "_check_extreme_deviation", "_check_weekly_blowoff",
    "_step2_weekly_pullback_zone", "_step2_weekly_kd_golden_cross",
    "_step2_60min_turtle_confirm", "_step_kd365", "_detect_w_bottom",
]


def _make_base_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    volume = rng.integers(1000, 5000, n)
    return pd.DataFrame({"date": dates, "open": open_, "max": high,
                          "min": low, "close": close, "volume": volume})


def _make_buy_ready_df() -> pd.DataFrame:
    """構造一組「除了W底規則以外，其他條件都確認買進/加碼」的日線資料：
    MA5/MA35都上揚且站上，讓fast_slope_up/key_slope_down這兩個在
    evaluate_timeframe()裡直接用ma_slope()算出來的真實變數得到正確方向。"""
    df = run_all_indicators(_make_base_df(), "日")
    lookback = RULES["indicators"]["ma_slope"]["lookback"]
    idx_old, idx_last = df.index[-1 - lookback], df.index[-1]
    df.loc[idx_old, "MA5"] = 100.0
    df.loc[idx_last, "MA5"] = 110.0
    df.loc[idx_old, "MA35"] = 95.0
    df.loc[idx_last, "MA35"] = 100.0
    df.loc[idx_last, "close"] = 115.0
    return df


def _make_sell_ready_df() -> pd.DataFrame:
    """構造一組已經觸發key_ma_down_veto（生死線下彎且跌破）的日線資料，
    用來驗證新規則不會把已經判定「賣出減碼」的結論動到（不升級、不覆蓋）。"""
    df = run_all_indicators(_make_base_df(), "日")
    lookback = RULES["indicators"]["ma_slope"]["lookback"]
    idx_old, idx_last = df.index[-1 - lookback], df.index[-1]
    df.loc[idx_old, "MA35"] = 110.0
    df.loc[idx_last, "MA35"] = 100.0
    df.loc[idx_last, "close"] = 90.0
    df.loc[idx_old, "MA5"] = 100.0
    df.loc[idx_last, "MA5"] = 95.0
    return df


@contextlib.contextmanager
def _neutralize_other_steps(w_bottom_return, ma_step_return=(4.0, False)):
    originals = {name: getattr(sd, name) for name in _NEUTRALIZED_STEPS}
    sd._step1_four_key_prices = lambda *a, **kw: 0
    sd._step2_ma = lambda *a, **kw: ma_step_return
    sd._step3_macd = lambda *a, **kw: (1.0, False, False)
    sd._step4_volume = lambda *a, **kw: (1.0, False)
    sd._step_kd365 = lambda *a, **kw: (0.0, False, False)
    sd._step5_mtm = lambda *a, **kw: False
    sd._step5b_cci = lambda *a, **kw: (False, 0.0)
    sd._detect_w_bottom = lambda *a, **kw: w_bottom_return
    sd._detect_m_top = lambda *a, **kw: None
    sd._detect_triangle_consolidation = lambda *a, **kw: {
        "active": False, "breakout": None, "volume_confirmed": False}
    sd._detect_ma120_second_leg = lambda *a, **kw: False
    sd._check_extreme_deviation = lambda *a, **kw: False
    sd._check_weekly_blowoff = lambda *a, **kw: False
    sd._step2_weekly_pullback_zone = lambda *a, **kw: 0.0
    sd._step2_weekly_kd_golden_cross = lambda *a, **kw: 0.0
    sd._step2_60min_turtle_confirm = lambda *a, **kw: 0.0
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(sd, name, fn)


def test_pending_w_bottom_downgrades_buy_to_watch() -> None:
    df = _make_buy_ready_df()
    with _neutralize_other_steps(w_bottom_return=(100.0, False)):
        verdict = sd.evaluate_timeframe(df, "日")
    assert verdict.conclusion == "觀望", verdict.conclusion
    print("OK: W底偵測到但均線/MACD/KD365未同步確認時，買進/加碼結論被強制降級為觀望")


def test_synced_w_bottom_does_not_downgrade() -> None:
    df = _make_buy_ready_df()
    with _neutralize_other_steps(w_bottom_return=(100.0, True)):
        verdict = sd.evaluate_timeframe(df, "日")
    assert verdict.conclusion == "加碼", verdict.conclusion
    print("OK: W底三者同步確認時，結論不受影響，維持加碼")


def test_no_w_bottom_detected_does_not_downgrade() -> None:
    df = _make_buy_ready_df()
    with _neutralize_other_steps(w_bottom_return=None):
        verdict = sd.evaluate_timeframe(df, "日")
    assert verdict.conclusion == "加碼", verdict.conclusion
    print("OK: 完全沒有偵測到W底型態時，這條新規則不會誤觸發")


def test_toggle_off_disables_downgrade() -> None:
    df = _make_buy_ready_df()
    original = RULES["watch_rules"].get("w_bottom_kd365_downgrade", True)
    RULES["watch_rules"]["w_bottom_kd365_downgrade"] = False
    try:
        with _neutralize_other_steps(w_bottom_return=(100.0, False)):
            verdict = sd.evaluate_timeframe(df, "日")
    finally:
        RULES["watch_rules"]["w_bottom_kd365_downgrade"] = original
    assert verdict.conclusion == "加碼", verdict.conclusion
    print("OK: w_bottom_kd365_downgrade=False時，待確認的W底不會強制降級（開關生效）")


def test_hard_sell_not_overridden_by_pending_w_bottom() -> None:
    df = _make_sell_ready_df()
    with _neutralize_other_steps(w_bottom_return=(100.0, False), ma_step_return=(-2.0, True)):
        verdict = sd.evaluate_timeframe(df, "日")
    assert verdict.conclusion == "賣出減碼", verdict.conclusion
    print("OK: 已經判定賣出減碼時，待確認的W底不會把結論動到（不升級、不覆蓋）")


def main() -> None:
    test_pending_w_bottom_downgrades_buy_to_watch()
    test_synced_w_bottom_does_not_downgrade()
    test_no_w_bottom_detected_does_not_downgrade()
    test_toggle_off_disables_downgrade()
    test_hard_sell_not_overridden_by_pending_w_bottom()
    print("\n全部通過")


if __name__ == "__main__":
    main()
