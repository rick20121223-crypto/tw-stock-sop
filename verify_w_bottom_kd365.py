"""
驗證_detect_w_bottom()新增kd365_bullish參數後：
1. W底型態偵測到、且均線/MACD/KD365三者都同步轉多時，reasons裡出現
   「確認」版本的文字，且明確提到KD(36,5,5)也同步。
2. 同一個W底型態，但kd365_bullish=False時，reasons裡出現「待確認」
   版本的文字，不會說KD365也同步確認。

用合成資料手工構造一個完整的W底型態（右底比左底高、頸線突破、回測
不破、收紅K），不需要FinMind token，執行：
    python3 verify_w_bottom_kd365.py
"""
import numpy as np
import pandas as pd

from sop_decision import _detect_w_bottom

DOUBLE_CHECK_MARKER = "KD(36,5,5)"


def _build_w_bottom_df() -> pd.DataFrame:
    """手工構造一個完整W底型態：左底約100 -> 頸線約125 -> 右底約112
    （比左底高）-> 突破頸線到132 -> 小拉回128(不破125頸線) -> 收紅K到135。
    每段都塞進sop_rules.yaml pattern_recognition.w_bottom.lookback_bars(60)
    的視窗內，每個轉折點前後都留夠pivot_window(5)根K棒的盤整緩衝，避免
    rolling(window=5, center=True)在線段邊界抓到雜訊轉折點——這組參數已經
    手動驗證過，_detect_w_bottom()在泛化前的版本上會正確觸發偵測。
    """
    close = []
    close += list(np.linspace(150, 100, 10))   # 下跌到左底
    close += [100, 99.5, 100, 100.5, 100]        # 左底盤整
    close += list(np.linspace(100, 125, 10))    # 反彈到頸線
    close += [125, 125.5, 125, 124.5, 125]       # 頸線盤整
    close += list(np.linspace(125, 112, 10))    # 回落到右底(比左底高)
    close += [112, 111.5, 112, 112.5, 112]       # 右底盤整
    close += list(np.linspace(112, 132, 10))    # 突破頸線
    close += list(np.linspace(132, 128, 5))     # 小拉回(不破頸線)
    close += list(np.linspace(128, 135, 4))     # 回升收紅K
    close = np.array(close)
    n = len(close)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    open_[-1] = close[-1] - 2  # 最後一根強制收紅K（close > open）
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"date": dates, "open": open_, "max": high, "min": low,
                          "close": close, "volume": np.full(n, 3000)})


def test_kd365_synced_marks_confirmed() -> None:
    df = _build_w_bottom_df()
    reasons: list = []
    neckline = _detect_w_bottom(df, fast_slope_up=True, macd_bias=1.0,
                                 kd365_bullish=True, reasons=reasons)
    assert neckline is not None, "應該要偵測到W底型態"
    text = "\n".join(reasons)
    assert "轉折訊號確認" in text, reasons
    assert DOUBLE_CHECK_MARKER in text, reasons
    print("OK: 均線/MACD/KD365三者同步，reasons標記「確認」且提到KD(36,5,5)")


def test_kd365_not_synced_stays_pending() -> None:
    df = _build_w_bottom_df()
    reasons: list = []
    neckline = _detect_w_bottom(df, fast_slope_up=True, macd_bias=1.0,
                                 kd365_bullish=False, reasons=reasons)
    assert neckline is not None, "應該要偵測到W底型態（型態本身跟KD365無關）"
    text = "\n".join(reasons)
    assert "型態轉折待確認" in text, reasons
    assert "轉折訊號確認" not in text, reasons
    print("OK: KD365未同步時，reasons維持「待確認」，不會誤標成已確認")


def main() -> None:
    test_kd365_synced_marks_confirmed()
    test_kd365_not_synced_stays_pending()
    print("\n全部通過")


if __name__ == "__main__":
    main()
