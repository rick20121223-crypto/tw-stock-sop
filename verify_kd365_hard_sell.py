"""
驗證KD365 Double Check否決conclusion（賣出側）：
1. MACD死叉 + KD365死叉 + 生死線(MA35)斜率轉下彎，三個條件同時成立時，
   conclusion變成「賣出減碼」，而且reasons裡有Double Check的標記文字。
2. 只有其中兩項成立（生死線沒有轉下彎）時，Double Check標記文字不會出現
   在reasons裡——驗證「三個條件缺一不可」的AND邏輯。
3. 三個條件都成立，但sop_rules.yaml的kd365_veto_enabled設為False時，
   不會因為這個條件被判「賣出減碼」（直接改RULES這個已載入的dict做測試，
   不用改檔案、跑完記得還原，避免影響同一個Python process裡其他驗證）。

用合成資料，不需要FinMind token，執行：
    python3 verify_kd365_hard_sell.py
"""
import numpy as np
import pandas as pd

from sop_decision import RULES, evaluate_timeframe
from stock_core import run_all_indicators

DOUBLE_CHECK_MARKER = "Double Check"


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


def _force_conditions(df: pd.DataFrame, macd_dead_cross: bool,
                       kd365_dead_cross: bool, key_slope_down: bool) -> pd.DataFrame:
    """直接覆寫最後幾筆K棒的DIF/MACD_signal、K365/D365、MA35欄位，精準
    構造出三個條件各自成立或不成立的情境，不受隨機漫步資料本身影響。"""
    df = df.copy()
    idx_prev, idx_last = df.index[-2], df.index[-1]

    if macd_dead_cross:
        df.loc[idx_prev, ["DIF", "MACD_signal"]] = [1.0, 0.5]
        df.loc[idx_last, ["DIF", "MACD_signal"]] = [0.3, 0.6]
    else:
        df.loc[idx_prev, ["DIF", "MACD_signal"]] = [0.5, 1.0]
        df.loc[idx_last, ["DIF", "MACD_signal"]] = [0.9, 0.3]
    df.loc[idx_last, "MACD_hist"] = df.loc[idx_last, "DIF"] - df.loc[idx_last, "MACD_signal"]

    if kd365_dead_cross:
        df.loc[idx_prev, ["K365", "D365"]] = [60.0, 55.0]
        df.loc[idx_last, ["K365", "D365"]] = [50.0, 56.0]
    else:
        df.loc[idx_prev, ["K365", "D365"]] = [55.0, 60.0]
        df.loc[idx_last, ["K365", "D365"]] = [70.0, 56.0]

    lookback = RULES["indicators"]["ma_slope"]["lookback"]
    idx_old = df.index[-1 - lookback]
    if key_slope_down:
        df.loc[idx_old, "MA35"] = 100.0
        df.loc[idx_last, "MA35"] = 90.0
    else:
        df.loc[idx_old, "MA35"] = 90.0
        df.loc[idx_last, "MA35"] = 100.0

    # 固定讓快均線(MA5/MA10)上揚且站上收盤價：避免基礎隨機漫步資料裡MA5
    # 恰好下彎跌破，連帶讓ma_bias意外變成負值，使dead_cross_confirmed
    # （dead_cross and ma_bias<0）跟著被觸發，干擾「只靠kd365_double_
    # check_confirmed/kd365_veto_enabled來否決conclusion」的隔離測試
    # （debug時用print(verdict.reasons)發現：三條件都成立但toggle關掉時，
    # dead_cross_confirmed仍因為MA5意外下彎而獨立成立，讓conclusion還是
    # 「賣出減碼」）。
    close_last = df.loc[idx_last, "close"]
    df.loc[idx_old, "MA5"] = close_last - 5.0
    df.loc[idx_last, "MA5"] = close_last - 2.0
    df.loc[idx_last, "MA10"] = close_last - 1.0

    return df


def _reasons_text(verdict) -> str:
    return "\n".join(verdict.reasons)


def test_all_three_confirmed_forces_sell() -> None:
    df = run_all_indicators(_make_base_df(), "日")
    df = _force_conditions(df, macd_dead_cross=True, kd365_dead_cross=True, key_slope_down=True)
    verdict = evaluate_timeframe(df, "日")
    assert DOUBLE_CHECK_MARKER in _reasons_text(verdict), verdict.reasons
    assert verdict.conclusion == "賣出減碼", verdict.conclusion
    print("OK: 三條件同時成立，conclusion正確變成「賣出減碼」，reasons有Double Check標記")


def test_missing_one_condition_does_not_mark() -> None:
    df = run_all_indicators(_make_base_df(), "日")
    df = _force_conditions(df, macd_dead_cross=True, kd365_dead_cross=True, key_slope_down=False)
    verdict = evaluate_timeframe(df, "日")
    assert DOUBLE_CHECK_MARKER not in _reasons_text(verdict), verdict.reasons
    print("OK: 只有兩項成立（生死線未轉下彎）時，Double Check標記不出現")


def test_toggle_off_disables_veto() -> None:
    df = run_all_indicators(_make_base_df(), "日")
    df = _force_conditions(df, macd_dead_cross=True, kd365_dead_cross=True, key_slope_down=True)

    original = RULES["sell_rules"].get("kd365_veto_enabled", True)
    RULES["sell_rules"]["kd365_veto_enabled"] = False
    try:
        verdict = evaluate_timeframe(df, "日")
    finally:
        RULES["sell_rules"]["kd365_veto_enabled"] = original

    assert verdict.conclusion != "賣出減碼" or DOUBLE_CHECK_MARKER not in _reasons_text(verdict), (
        "kd365_veto_enabled=False時，三條件同時成立不應該單靠這個條件把conclusion判成賣出減碼"
    )
    print("OK: kd365_veto_enabled=False時，三條件同時成立不會強制否決conclusion")


def main() -> None:
    test_all_three_confirmed_forces_sell()
    test_missing_one_condition_does_not_mark()
    test_toggle_off_disables_veto()
    print("\n全部通過")


if __name__ == "__main__":
    main()
