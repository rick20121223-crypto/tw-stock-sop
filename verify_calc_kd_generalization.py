"""
驗證calc_kd()泛化後：
1. 既有呼叫 calc_kd(df) 行為不變（跟泛化前的硬寫公式逐值比對）。
2. 新增(n=36, m1=5, m2=5, k_col="K365", d_col="D365")呼叫能正確算出獨立欄位，
   不影響原本K/D欄位。
3. run_all_indicators()的輸出同時有K/D(9,3,3)跟K365/D365(36,5,5)兩組欄位。

不需要FinMind token，用合成OHLCV資料跑，執行：
    python3 verify_calc_kd_generalization.py
"""
import numpy as np
import pandas as pd

from stock_core import calc_kd, run_all_indicators


def _make_synthetic_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    volume = rng.integers(1000, 5000, n)
    return pd.DataFrame({"date": dates, "open": open_, "max": high,
                          "min": low, "close": close, "volume": volume})


def _old_formula_kd(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    """泛化前calc_kd()的公式，原封不動複製一份，用來跟泛化後的預設呼叫比對。"""
    df = df.copy()
    low_min = df["min"].rolling(window=period).min()
    high_max = df["max"].rolling(window=period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    rsv = ((df["close"] - low_min) / denom * 100).fillna(50.0)
    if len(df) > 0:
        rsv.iloc[0] = 50.0
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    k_seeded = k.copy()
    if len(df) > 0:
        k_seeded.iloc[0] = 50.0
    d = k_seeded.ewm(alpha=1 / 3, adjust=False).mean()
    df["K"] = k
    df["D"] = d
    return df


def main() -> None:
    df = _make_synthetic_df()

    # 1. 預設呼叫跟舊公式逐值比對
    old = _old_formula_kd(df)
    new = calc_kd(df)
    pd.testing.assert_series_equal(old["K"], new["K"], check_names=False)
    pd.testing.assert_series_equal(old["D"], new["D"], check_names=False)
    print("OK: calc_kd(df) 預設值跟泛化前公式逐值相同")

    # 2. 新參數獨立欄位，不污染原本K/D —— 在「已經有K/D」的df上再呼叫一次
    #    自訂參數，比照run_all_indicators()真正的呼叫順序（先算預設KD，
    #    再算KD365），確認K365/D365正確產生、且不覆蓋既有K/D。
    kd365 = calc_kd(new, n=36, m1=5, m2=5, k_col="K365", d_col="D365")
    assert "K365" in kd365.columns and "D365" in kd365.columns
    pd.testing.assert_series_equal(kd365["K"], new["K"], check_names=False)
    pd.testing.assert_series_equal(kd365["D"], new["D"], check_names=False)
    print("OK: calc_kd(df, n=36, m1=5, m2=5, k_col='K365', d_col='D365') 正確輸出獨立欄位，不影響K/D")

    # 3. run_all_indicators() 同時有兩組欄位
    full = run_all_indicators(df, "日")
    assert {"K", "D", "K365", "D365"}.issubset(full.columns)
    assert full["K365"].notna().any() and full["D365"].notna().any()
    print("OK: run_all_indicators() 同時輸出K/D(9,3,3)與K365/D365(36,5,5)")

    print("\n全部通過")


if __name__ == "__main__":
    main()
