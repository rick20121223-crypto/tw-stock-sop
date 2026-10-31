"""
測試2026-10-31新增的「空頭環境假反彈防護」(watch_rules.bear_regime_fake_
rally_guard)：依1030錄音「程式單權重配比」，均線（生死線）斜率具最終裁決
權，仍下彎壓制時，短線指標轉強多屬「破波段假反彈」，嚴禁波段重押/順勢
加碼。

能走到這條規則（conclusion仍是買進/加碼）時，股價其實已經站在生死線之上
（否則key_ma_down_veto會先把conclusion判成賣出減碼）——所以這裡防的是
「均線本身還沒轉向、站上去的這段隨時可能被拉回去」的假反彈，不是「反彈
測試均線但沒站上」那種進階型態（那個還沒做，見討論記錄）。

兩個例外（經使用者討論確認）：
1. 左側平台/半年線打出兩次腳未破（既有ma120_second_leg/platform_
   second_leg_level偵測）且今日收盤站穩今開 → 放行小量試單：加碼降級
   為買進，不整個打回觀望。
2. 沒有左側支撐，或支撐有但今日沒站穩今開 → 降級為觀望，並附上多單
   （逢弱勢反彈建議減碼/買保險）跟空單（均線未轉向、若撐不住再跌破
   可視為順勢加碼空點）的雙向提示——這兩句話只描述「均線還沒轉向」這個
   已確認的事實，不斷言「反彈測試均線被打回」這個還沒驗證的價格動作。

用monkeypatch控制其他step函式（同verify_w_bottom_kd365_gate.py的作法），
只隔離觀察這條新規則本身的行為。MA35欄位改用真實欄位覆寫來控制
key_slope_down（跟該檔案同理：這個值是evaluate_timeframe()直接呼叫
ma_slope(df,...)算出來的，不是從_step2_ma()回傳值來）。

用合成資料，不需要API，執行：
    python3 test_bear_regime_fake_rally_guard.py
"""
import contextlib

import numpy as np
import pandas as pd

import sop_decision as sd
from sop_decision import RULES
from stock_core import run_all_indicators

_NEUTRALIZED_STEPS = [
    "_step1_four_key_prices", "_step2_ma", "_step3_macd", "_step4_volume",
    "_step5_mtm", "_step5b_cci", "_detect_w_bottom", "_detect_m_top",
    "_detect_triangle_consolidation", "_check_extreme_deviation",
    "_check_weekly_blowoff", "_step2_weekly_pullback_zone",
    "_step2_weekly_kd_golden_cross", "_step2_60min_turtle_confirm", "_step_kd365",
    "_detect_ma120_second_leg", "_detect_platform_second_leg",
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


def _make_bear_slope_but_above_key_df(today_above_open: bool = True,
                                       inject_washout: bool = True) -> pd.DataFrame:
    """構造「生死線(MA35)下彎、但股價目前仍站在MA35之上」的日線資料
    （key_ma_down_veto不觸發的前提，否則根本走不到這條新規則）。快均線
    (MA5)設成上揚，讓既有的buy_add_sync_gate(ma_slope_up這項)不會額外
    把結論打回觀望，干擾這條新規則本身的行為觀察——短線反彈但生死線
    未轉，正好就是這條規則要測的情境。

    inject_washout=True時，額外在「夾起單」lookback視窗內明確安排一天
    close<MA35、今天收回站上，確保washout_recovery真的成立；=False時
    把視窗內的MA35/close都設成close明顯站在MA35之上（不是貼著邊緣），
    確保washout_recovery確實不成立，不是靠隨機base資料湊出來的（原始
    版本沒控制這段，idx_old的MA35=110剛好比隨機資料的close還高，意外
    讓washout條件湊巧成立，沒測到真正要測的情境）。
    today_above_open控制今天收盤是否站穩今開（refinement 1的同日確認鍵）。"""
    df = run_all_indicators(_make_base_df(), "日")
    lookback = RULES["indicators"]["ma_slope"]["lookback"]
    idx_old, idx_last = df.index[-1 - lookback], df.index[-1]

    # 先處理夾起單視窗（offset=1..washout_lookback，相對idx_last），idx_old
    # 剛好可能落在這個視窗內（兩者lookback數字恰好都是3）——所以idx_old/
    # idx_last的均線斜率定義值一定要在這段迴圈「之後」才設，確保不會被蓋掉。
    washout_lookback = RULES["indicators"]["washout_recovery_second_leg"]["lookback_bars"]
    for offset in range(1, washout_lookback + 1):
        idx = df.index[-1 - offset]
        df.loc[idx, "MA35"] = 100.0
        df.loc[idx, "close"] = 90.0 if (inject_washout and offset == 1) else 103.0

    df.loc[idx_old, "MA35"] = 110.0
    df.loc[idx_old, "MA5"] = 95.0
    df.loc[idx_old, "close"] = 115.0   # 明確站在MA35(110)之上，避免跟隨機base資料湊出意外的washout
    df.loc[idx_last, "MA35"] = 100.0   # MA35（生死線）下彎
    df.loc[idx_last, "MA5"] = 103.0    # MA5（快均線）上揚，短線反彈
    df.loc[idx_last, "close"] = 105.0  # 股價仍站在MA35(100)之上，不觸發veto
    df.loc[idx_last, "今開"] = 106.0 if not today_above_open else 103.0
    return df


@contextlib.contextmanager
def _neutralize_other_steps(conclusion_seed: str, ma120_second_leg=False,
                             platform_second_leg_level=None):
    """conclusion_seed="買進"時用一般score門檻湊出買進；"加碼"時用
    strong_bullish條件湊出加碼。其餘step全部中性化，只留這條新規則本身
    的行為可觀察。"""
    originals = {name: getattr(sd, name) for name in _NEUTRALIZED_STEPS}
    sd._step1_four_key_prices = lambda *a, **kw: 0
    if conclusion_seed == "加碼":
        # strong_bullish: ma_bias>=3, macd_bias>=1, four_key_bias>=0, volume_bias>=0
        # 且ma_bias>=addon_ma_bias(4) and volume_bias>0 才會是「加碼」而非「買進」
        sd._step2_ma = lambda *a, **kw: (4.0, False)
        sd._step4_volume = lambda *a, **kw: (1.0, False)
    else:
        sd._step2_ma = lambda *a, **kw: (0.0, False)
        sd._step4_volume = lambda *a, **kw: (1.0, False)
    sd._step3_macd = lambda *a, **kw: (3.0, False, False)
    sd._step_kd365 = lambda *a, **kw: (1.0, False, False)
    sd._step5_mtm = lambda *a, **kw: False
    sd._step5b_cci = lambda *a, **kw: (False, 1.0)
    sd._detect_w_bottom = lambda *a, **kw: None
    sd._detect_m_top = lambda *a, **kw: None
    sd._detect_triangle_consolidation = lambda *a, **kw: {
        "active": False, "breakout": None, "volume_confirmed": False}
    sd._check_extreme_deviation = lambda *a, **kw: False
    sd._check_weekly_blowoff = lambda *a, **kw: False
    sd._step2_weekly_pullback_zone = lambda *a, **kw: 0.0
    sd._step2_weekly_kd_golden_cross = lambda *a, **kw: 0.0
    sd._step2_60min_turtle_confirm = lambda *a, **kw: 0.0
    sd._detect_ma120_second_leg = lambda *a, **kw: ma120_second_leg
    sd._detect_platform_second_leg = lambda *a, **kw: platform_second_leg_level
    try:
        yield
    finally:
        for name, fn in originals.items():
            setattr(sd, name, fn)


def test_no_support_downgrades_buy_to_watch_with_dual_caveats():
    df = _make_bear_slope_but_above_key_df(today_above_open=True)
    with _neutralize_other_steps("買進"):
        v = sd.evaluate_timeframe(df, "日")
    assert v.conclusion == "觀望", f"沒有左側支撐時，均線下彎應該把買進降級為觀望，實際={v.conclusion}"
    joined = " ".join(v.caveats)
    assert "假反彈" in joined or "誘多陷阱" in joined
    assert "減碼" in joined and ("買保險" in joined or "保險" in joined), "應該附上多單提示"
    assert "空單" in joined or "空點" in joined, "應該附上空單提示"
    print("✅ 無左側支撐：買進->觀望，且附上多單/空單雙向提示")


def test_second_leg_without_washout_no_longer_exempts_on_daily():
    """2026-10-31補強：日線的話，光有「兩次腳未破」不夠，backtest顯示
    單獨使用任一條件都沒有優於無條件基準的證據，必須夾起單也同時成立
    才放行小量試單——沒有夾起單時，即使ma120_second_leg=True，也該
    降級為觀望，不是買進。"""
    df = _make_bear_slope_but_above_key_df(today_above_open=True, inject_washout=False)
    with _neutralize_other_steps("加碼", ma120_second_leg=True):
        v = sd.evaluate_timeframe(df, "日")
    assert v.conclusion == "觀望", \
        f"日線只有兩次腳未破、沒有夾起單時，應該降級為觀望（收緊後的標準），實際={v.conclusion}"
    print("✅ 日線：只有兩次腳未破、沒有夾起單時，不再放行小量試單（已改成要求兩者同時成立）")


def test_support_and_held_open_caps_add_to_buy():
    df = _make_bear_slope_but_above_key_df(today_above_open=True)
    with _neutralize_other_steps("加碼", ma120_second_leg=True):
        v = sd.evaluate_timeframe(df, "日")
    assert v.conclusion == "買進", f"有左側支撐且站穩今開，加碼應該降級為買進（不是觀望），實際={v.conclusion}"
    assert any("小量試單" in c or "5" in c for c in v.caveats), f"應該有小量試單的caveat，實際={v.caveats}"
    print("✅ 有左側支撐+站穩今開：加碼->買進（小量試單），不是觀望")


def test_support_without_holding_open_still_downgrades_to_watch():
    df = _make_bear_slope_but_above_key_df(today_above_open=False)
    with _neutralize_other_steps("加碼", ma120_second_leg=True):
        v = sd.evaluate_timeframe(df, "日")
    assert v.conclusion == "觀望", \
        f"有左側支撐但沒站穩今開（同日確認鍵失敗），仍應該降級為觀望，實際={v.conclusion}"
    print("✅ 有左側支撐但沒站穩今開：仍降級為觀望（同日確認鍵生效）")


def test_platform_second_leg_alone_no_longer_counts_as_support_on_daily():
    """2026-10-31再次收緊：另外用test_washout_recovery_platform_second_
    leg_backtest.py測過「夾起單+廣義版兩次腳未破(platform_second_leg_
    level)」，5/10日報酬勝率都比無條件基準差，沒有證據支持——日線這裡
    只認ma120_second_leg（已驗證的半年線版），platform_second_leg_level
    單獨出現（沒有ma120_second_leg）不該再被當成支撐，即使夾起單也成立。"""
    df = _make_bear_slope_but_above_key_df(today_above_open=True, inject_washout=True)
    with _neutralize_other_steps("買進", platform_second_leg_level=63.5):
        v = sd.evaluate_timeframe(df, "日")
    assert v.conclusion == "觀望", \
        f"日線只有platform_second_leg_level、沒有ma120_second_leg時，不該放行小量試單，實際={v.conclusion}"
    print("✅ 日線：只有廣義版兩次腳未破(platform_second_leg_level)、沒有ma120_second_leg時，不再算有支撐")


def test_disabled_flag_skips_guard():
    df = _make_bear_slope_but_above_key_df(today_above_open=True)
    RULES["watch_rules"]["bear_regime_fake_rally_guard"]["enabled"] = False
    try:
        with _neutralize_other_steps("買進"):
            v = sd.evaluate_timeframe(df, "日")
    finally:
        RULES["watch_rules"]["bear_regime_fake_rally_guard"]["enabled"] = True
    assert v.conclusion == "買進", f"關閉開關後不該套用這條規則，實際={v.conclusion}"
    print("✅ 開關關閉時，不套用這條規則（維持原本的買進）")


def test_guard_does_not_touch_existing_sell_conclusion():
    """key_ma_down_veto已經把conclusion判成賣出減碼時，這條新規則不該去動它
    （它的觸發條件本來就排除了賣出減碼/觀望，這裡額外驗證一次迴歸安全性）。"""
    df = run_all_indicators(_make_base_df(), "日")
    lookback = RULES["indicators"]["ma_slope"]["lookback"]
    idx_old, idx_last = df.index[-1 - lookback], df.index[-1]
    df.loc[idx_old, "MA35"] = 110.0
    df.loc[idx_last, "MA35"] = 100.0
    df.loc[idx_last, "close"] = 90.0  # 跌破MA35 -> key_ma_down_veto觸發
    with _neutralize_other_steps("買進"):
        sd._step2_ma = lambda *a, **kw: (0.0, True)  # 模擬key_ma_down_veto=True
        v = sd.evaluate_timeframe(df, "日")
    assert v.conclusion == "賣出減碼", f"key_ma_down_veto應該維持賣出減碼不受影響，實際={v.conclusion}"
    print("✅ 已經判定賣出減碼時，這條新規則不會去動它")


if __name__ == "__main__":
    test_no_support_downgrades_buy_to_watch_with_dual_caveats()
    test_second_leg_without_washout_no_longer_exempts_on_daily()
    test_support_and_held_open_caps_add_to_buy()
    test_support_without_holding_open_still_downgrades_to_watch()
    test_platform_second_leg_alone_no_longer_counts_as_support_on_daily()
    test_disabled_flag_skips_guard()
    test_guard_does_not_touch_existing_sell_conclusion()
    print()
    print("全部通過。")
