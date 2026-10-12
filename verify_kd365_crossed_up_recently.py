"""
驗證_kd365_crossed_up_recently()：
1. 最近window個交易日內有黃金交叉（前一天K<=D、當天K>D），回傳True。
2. 黃金交叉發生在window窗口之外（太久以前），回傳False。
3. 完全沒有交叉（K365一直在D365之下），回傳False。
4. K365/D365缺資料時安全回傳False，不拋例外。

用合成資料，不需要FinMind token，執行：
    python3 verify_kd365_crossed_up_recently.py
"""
import numpy as np
import pandas as pd

from sop_decision import _kd365_crossed_up_recently


def _make_df(k365_values: list, d365_values: list) -> pd.DataFrame:
    n = len(k365_values)
    return pd.DataFrame({
        "close": np.linspace(100, 110, n),
        "K365": k365_values,
        "D365": d365_values,
    })


def test_cross_within_window() -> None:
    # 今天(最後一筆)K(55)剛穿破D(52)，昨天K(50)<=D(52) -> 剛交叉，在window=3內
    df = _make_df(
        k365_values=[50, 50, 50, 50, 50, 50, 55],
        d365_values=[50, 50, 50, 50, 50, 52, 52],
    )
    assert _kd365_crossed_up_recently(df, window=3) is True
    print("OK: 窗口內有黃金交叉，正確回傳True")


def test_cross_outside_window() -> None:
    # 交叉發生在第1根(index 0->1)，離今天(index 6)已經超過window=3
    df = _make_df(
        k365_values=[40, 45, 50, 50, 50, 50, 50],
        d365_values=[50, 44, 45, 50, 50, 50, 50],
    )
    # index1: prev(40,50)->cur(45,44) 這天K(45)>D(44)且prev K(40)<=D(50)，是交叉點，
    # 但離最後一天(index6)有5天，超過window=3
    assert _kd365_crossed_up_recently(df, window=3) is False
    print("OK: 黃金交叉在窗口之外，正確回傳False")


def test_no_cross_at_all() -> None:
    df = _make_df(
        k365_values=[30, 31, 32, 33, 34, 35, 36],
        d365_values=[50, 50, 50, 50, 50, 50, 50],
    )
    assert _kd365_crossed_up_recently(df, window=3) is False
    print("OK: 完全沒有交叉，正確回傳False")


def test_missing_data_safe() -> None:
    df = pd.DataFrame({"close": [100.0], "K365": [np.nan], "D365": [np.nan]})
    assert _kd365_crossed_up_recently(df, window=3) is False
    print("OK: K365/D365缺資料時安全回傳False，不拋例外")


def main() -> None:
    test_cross_within_window()
    test_cross_outside_window()
    test_no_cross_at_all()
    test_missing_data_safe()
    print("\n全部通過")


if __name__ == "__main__":
    main()
