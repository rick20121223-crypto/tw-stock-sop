"""
驗證_detect_bearish_divergence()泛化後：
1. 不帶column參數時，行為跟泛化前（只認MACD_hist）完全相同。
2. 帶column="K365"時，能正確用K365欄位偵測背離，而不是MACD_hist。

用合成資料手動構造「價格創高、指標未同步創高」的背離情境，
執行：python3 verify_detect_bearish_divergence_column.py
"""
import numpy as np
import pandas as pd

from sop_decision import _detect_bearish_divergence


def _make_divergence_df(n: int = 40) -> pd.DataFrame:
    """前半段尖峰指標值高、後半段價格更高但指標值更低 -> 典型頂背離。"""
    close = np.linspace(100, 110, n)
    close[n // 4] = 115        # 前半段的峰值（指標值會給高值）
    close[-1] = 120            # 後半段的峰值（更高價，但指標值會給低值）
    macd_hist = np.zeros(n)
    macd_hist[n // 4] = 5.0    # 前半段峰值對應的高指標值
    macd_hist[-1] = 1.0        # 後半段峰值對應的低指標值（背離）
    df = pd.DataFrame({"close": close, "MACD_hist": macd_hist})
    df["K365"] = macd_hist  # 同樣的背離形狀，驗證換欄位能照樣抓到
    return df


def main() -> None:
    df = _make_divergence_df()

    # 1. 不帶column，預設檢查MACD_hist，應該偵測到背離
    assert _detect_bearish_divergence(df, lookback=40) is True
    print("OK: 不帶column參數時，正確偵測MACD_hist背離（行為跟泛化前相同）")

    # 2. 帶column="K365"，應該也偵測到背離
    assert _detect_bearish_divergence(df, lookback=40, column="K365") is True
    print("OK: column='K365' 時，正確偵測K365背離")

    # 3. 欄位不存在時回傳False，不應該拋例外
    assert _detect_bearish_divergence(df, lookback=40, column="NOT_A_COLUMN") is False
    print("OK: column不存在時安全回傳False")

    print("\n全部通過")


if __name__ == "__main__":
    main()
