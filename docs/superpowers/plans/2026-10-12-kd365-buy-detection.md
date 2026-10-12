# KD365 買進側偵測機制 Implementation Plan（Phase 1 of 3）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在既有的`_detect_w_bottom()`（W底型態偵測）的`synced`判斷裡加入KD(36,5,5)確認，讓偵測到的W底轉折訊號，在均線/MACD/KD365三者同步轉多時才標記「確認」，否則維持「待確認」——只影響`reasons`文字，不影響`conclusion`。

**Architecture:** `_detect_w_bottom()`新增一個`kd365_bullish: bool`參數，`synced = fast_slope_up and macd_bias > 0`改成`synced = fast_slope_up and macd_bias > 0 and kd365_bullish`。唯一呼叫處（`evaluate_timeframe()`）在呼叫前算出`kd365_bullish`（直接比較`latest`的`K365`/`D365`欄位，週線排除，不新增`_step_kd365()`的回傳值）。

**Tech Stack:** Python 3.11、pandas（既有`requirements.txt`，不新增依賴）

**Spec:** [docs/superpowers/specs/2026-10-12-kd365-buy-detection-design.md](../specs/2026-10-12-kd365-buy-detection-design.md)

## Global Constraints

- 只處理課程「起漲打底進場點」——不新增獨立型態偵測器，直接擴充既有`_detect_w_bottom()`的`synced`判斷。
- 不碰`_detect_m_top()`、不碰`conclusion`/`score`/`suggested_exposure`——這輪只改`reasons`文字。
- 不改`_step_kd365()`的回傳介面`(kd365_bias, kd365_cross_down, kd365_bearish_divergence)`——賣出側已經依賴這個簽名，這次直接讀`latest`的`K365`/`D365`欄位做狀態比較。
- `kd365_bullish`排除週線（`tf != "週"`），跟KD365系列既有規則（週線維持原本KD(9,3,3)）一致。
- 本repo沒有pytest，驗證用不需要API token的合成資料`assert`腳本，檔名用`verify_`前綴。

---

## Task 1: `_detect_w_bottom()`新增`kd365_bullish`參數

**Files:**
- Modify: `sop_decision.py`（`_detect_w_bottom()`函式本體、`evaluate_timeframe()`裡的呼叫處）
- Create: `verify_w_bottom_kd365.py`

**Interfaces:**
- Consumes: 無新依賴，`pd`已在`sop_decision.py`匯入
- Produces: `_detect_w_bottom(df, fast_slope_up, macd_bias, kd365_bullish, reasons) -> Optional[float]`——比既有簽名多一個`kd365_bullish: bool`參數（插入在`macd_bias`之後、`reasons`之前）

- [ ] **Step 1: 寫驗證腳本**

建立 `verify_w_bottom_kd365.py`：

```python
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
```

- [ ] **Step 2: 執行驗證腳本，確認現在會失敗**

Run: `python3 verify_w_bottom_kd365.py`
Expected: `TypeError: _detect_w_bottom() got an unexpected keyword argument 'kd365_bullish'`（因為目前`_detect_w_bottom()`還沒有這個參數）

- [ ] **Step 3: 修改`_detect_w_bottom()`簽名與`synced`判斷**

先確認`sop_decision.py`裡目前這段文字還在（這是接下來要改的函式開頭）：

```python
def _detect_w_bottom(df: pd.DataFrame, fast_slope_up: bool, macd_bias: float,
                      reasons: List[str]) -> Optional[float]:
```

改成：

```python
def _detect_w_bottom(df: pd.DataFrame, fast_slope_up: bool, macd_bias: float,
                      kd365_bullish: bool, reasons: List[str]) -> Optional[float]:
```

再找到函式尾端這段文字（`synced`判斷跟兩段`reasons.append`）：

```python
    synced = fast_slope_up and macd_bias > 0
    if synced:
        reasons.append(
            f"📐 偵測到W底型態（左底約{left_low:.2f}／右底約{right_low:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收紅K，且均線斜率與MACD同步轉多，轉折訊號確認"
            "（輔助提醒，結論仍以四關價/均線/MACD/OBV分層否決制為準）"
        )
    else:
        reasons.append(
            f"📐 偵測到W底型態（左底約{left_low:.2f}／右底約{right_low:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收紅K，但均線斜率/MACD尚未同步轉多，標記為「型態轉折待確認」"
            "（僅供觀察，非買進依據）"
        )
    return neckline
```

改成：

```python
    synced = fast_slope_up and macd_bias > 0 and kd365_bullish
    if synced:
        reasons.append(
            f"📐 偵測到W底型態（左底約{left_low:.2f}／右底約{right_low:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收紅K，且均線斜率、MACD、KD(36,5,5)三者同步轉多，轉折訊號確認"
            "（輔助提醒，結論仍以四關價/均線/MACD/OBV分層否決制為準）"
        )
    else:
        reasons.append(
            f"📐 偵測到W底型態（左底約{left_low:.2f}／右底約{right_low:.2f}，頸線約{neckline:.2f}），"
            "回測頸線不破後收紅K，但均線斜率/MACD/KD(36,5,5)尚未同步轉多，標記為「型態轉折待確認」"
            "（僅供觀察，非買進依據）"
        )
    return neckline
```

- [ ] **Step 4: 更新`evaluate_timeframe()`的呼叫處**

先確認這行還在（在`_detect_w_bottom`呼叫前）：

```python
    w_bottom_neckline = _detect_w_bottom(df, fast_slope_up, macd_bias, reasons)
```

改成：

```python
    # KD365目前是否處於多頭排列（K365>D365），用「目前狀態」風格跟
    # macd_bias>0同一個概念，不是「今天剛好黃金交叉」這種事件——W底確認
    # 紅K那天，KD365可能是前幾天就黃金交叉、之後維持多頭排列，不一定
    # 剛好跟確認紅K同一天交叉。排除週線，跟KD365系列既有規則（週線維持
    # 原本KD(9,3,3)）一致。
    kd365_bullish = (
        tf != "週"
        and pd.notna(latest.get("K365")) and pd.notna(latest.get("D365"))
        and latest["K365"] > latest["D365"]
    )
    w_bottom_neckline = _detect_w_bottom(df, fast_slope_up, macd_bias, kd365_bullish, reasons)
```

- [ ] **Step 5: 執行驗證腳本，確認通過**

Run: `python3 verify_w_bottom_kd365.py`
Expected: 兩段`OK:`訊息都印出來，最後印出「全部通過」，exit code 0

- [ ] **Step 6: 跑一次既有的5個驗證腳本，確認沒有回歸**

Run:
```bash
python3 verify_calc_kd_generalization.py
python3 verify_detect_bearish_divergence_column.py
python3 verify_step_kd365.py
python3 verify_suggested_exposure_pipeline.py
python3 verify_kd365_hard_sell.py
```
Expected: 五支腳本都印出「全部通過」

- [ ] **Step 7: Commit**

```bash
git add sop_decision.py verify_w_bottom_kd365.py
git commit -m "$(cat <<'EOF'
KD365買進側Phase1：_detect_w_bottom()的synced判斷加入KD(36,5,5)確認

不新增獨立型態偵測器，直接擴充既有W底偵測的synced條件(均線+MACD+
KD365三者同步才算「轉折訊號確認」，否則維持「待確認」)。kd365_bullish
在呼叫端算出(K365>D365目前狀態，排除週線)，不改_step_kd365()既有的
回傳介面。只影響reasons文字，不影響conclusion/score/suggested_exposure
——對應課程「起漲與打底進場點」，下一輪backtest驗證有沒有效，再決定
要不要讓它影響conclusion。
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage**：spec的「`_detect_w_bottom()`新增`kd365_bullish`參數」「呼叫端算`kd365_bullish`」「不改`_step_kd365()`介面」三個技術決定都在Task1的Step3/4涵蓋；「明確不在這次範圍內」列的四項（M頭、conclusion/score/suggested_exposure、`_step_kd365()`介面、順勢加碼/過濾亂買）這個plan都沒有任務觸碰，範圍一致；「測試/驗證計畫」的兩點分別對應Step1-5（合成資料）、Step6（regression）。
- **Type一致性**：`_detect_w_bottom()`新簽名`(df, fast_slope_up, macd_bias, kd365_bullish, reasons)`在Step3定義、Step4呼叫處使用一致；`kd365_bullish`是純布林值，沒有跨任務的簽名需要對齊。
- **No placeholders**：驗證腳本的合成資料已經實際執行過確認會正確觸發`_detect_w_bottom()`（左底111.70／右底127.70／頸線132.30，這是實際執行的輸出，不是憑空編的數字），不是空想的fixture。
