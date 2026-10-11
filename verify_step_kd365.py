"""
驗證_step_kd365()三件事：
1. K365/D365出現黃金交叉時，kd365_cross_down=False，reasons有黃金交叉文字。
2. K365/D365出現死亡交叉時，kd365_cross_down=True，reasons有死亡交叉文字。
3. 價格創高但K365未同步創高時，kd365_bearish_divergence=True。

用合成資料手動構造這三種情境，執行：
    python3 verify_step_kd365.py
"""
import numpy as np
import pandas as pd

from sop_decision import _step_kd365


def _base_df(n: int = 40) -> pd.DataFrame:
    close = np.linspace(100, 110, n)
    return pd.DataFrame({"close": close})


def test_golden_cross() -> None:
    df = _base_df()
    df["K365"] = 50.0
    df["D365"] = 50.0
    df.loc[df.index[-2], "K365"] = 40.0
    df.loc[df.index[-2], "D365"] = 45.0
    df.loc[df.index[-1], "K365"] = 50.0
    df.loc[df.index[-1], "D365"] = 45.0
    reasons, caveats = [], []
    bias, cross_down, divergence = _step_kd365(df, "日", df.iloc[-1], reasons, caveats)
    assert cross_down is False
    assert any("黃金交叉" in r for r in reasons), reasons
    print("OK: KD365黃金交叉正確偵測，kd365_cross_down=False")


def test_dead_cross() -> None:
    df = _base_df()
    df["K365"] = 50.0
    df["D365"] = 50.0
    df.loc[df.index[-2], "K365"] = 50.0
    df.loc[df.index[-2], "D365"] = 45.0
    df.loc[df.index[-1], "K365"] = 40.0
    df.loc[df.index[-1], "D365"] = 45.0
    reasons, caveats = [], []
    bias, cross_down, divergence = _step_kd365(df, "日", df.iloc[-1], reasons, caveats)
    assert cross_down is True
    assert any("死亡交叉" in r for r in reasons), reasons
    print("OK: KD365死亡交叉正確偵測，kd365_cross_down=True")


def test_bearish_divergence() -> None:
    # n要跟sop_rules.yaml kd365.divergence_lookback(20)一致：
    # _detect_bearish_divergence()只看df.tail(lookback)這個窗口，n=40配
    # lookback=20會把前半峰值(index 10)切在窗口外，永遠偵測不到背離。
    n = 20
    close = np.linspace(100, 110, n)
    close[n // 4] = 115
    close[-1] = 120
    k365 = np.full(n, 50.0)
    k365[n // 4] = 90.0  # 前半峰值：K值高
    k365[-1] = 60.0       # 後半峰值（更高價）：K值反而較低 -> 背離
    df = pd.DataFrame({"close": close, "K365": k365, "D365": k365})
    reasons, caveats = [], []
    bias, cross_down, divergence = _step_kd365(df, "日", df.iloc[-1], reasons, caveats)
    assert divergence is True
    assert any("背離" in r for r in reasons), reasons
    print("OK: KD365頂背離正確偵測，kd365_bearish_divergence=True")


def test_missing_data_safe() -> None:
    df = pd.DataFrame({"close": [100.0], "K365": [np.nan], "D365": [np.nan]})
    reasons, caveats = [], []
    bias, cross_down, divergence = _step_kd365(df, "日", df.iloc[-1], reasons, caveats)
    assert cross_down is False and divergence is False
    assert any("資料不足" in c for c in caveats), caveats
    print("OK: K365/D365缺資料時安全回傳，不拋例外")


def main() -> None:
    test_golden_cross()
    test_dead_cross()
    test_bearish_divergence()
    test_missing_data_safe()
    print("\n全部通過")


if __name__ == "__main__":
    main()
