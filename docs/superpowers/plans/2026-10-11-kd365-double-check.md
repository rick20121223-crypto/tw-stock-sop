# KD(36,5,5) Double Check 與漸進式部位建議 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增KD(36,5,5)指標，跟MACD做Double Check雙重驗證，並讓`evaluate_timeframe()`額外輸出一個不影響現有四選一結論的`suggested_exposure`（建議曝險%）欄位。

**Architecture:** 在`stock_core.py`泛化既有`calc_kd()`支援自訂(n,m1,m2)參數與輸出欄位名，新增一組`K365`/`D365`欄位（不影響既有`K`/`D`欄位與週線規則）。在`sop_decision.py`新增`_step_kd365()`（比照`_step3_macd`的寫法），重用泛化後的`_detect_bearish_divergence()`做背離偵測，只在`tf != "週"`時啟用。`Verdict`新增`suggested_exposure`欄位（純新增、預設`None`），用既有已算出的條件變數做5級映射，不新增否決`conclusion`的硬邏輯。

**Tech Stack:** Python 3.11、pandas、PyYAML（既有`requirements.txt`，這次不新增依賴）

**Spec:** [docs/superpowers/specs/2026-10-11-kd365-double-check-design.md](../specs/2026-10-11-kd365-double-check-design.md)

## Global Constraints

- 範圍只限日線、60分、5分；週線與月線完全不動（`_step2_weekly_kd_golden_cross`沿用既有KD(9,3,3)）。
- 不新增15分/30分週期。
- 這一輪KD365**不否決**`conclusion`（買進/加碼/觀望/賣出減碼四選一邏輯完全不變），只新增`reasons`/`caveats`文字與新欄位`suggested_exposure`。
- `suggested_exposure`是純新增欄位，預設`None`，任何既有程式碼（`notify_email.py`/`home.py`/`pages/*.py`/所有`test_*.py`）都不需要跟著改，也不在這個plan範圍內。
- 本repo沒有pytest（`requirements.txt`沒有這個套件），既有`test_*.py`都是需要FinMind token、抓真實歷史資料的回測報告腳本，不是unit test。這個plan裡的「驗證」步驟一律用**不需要API token的合成資料**、以`assert`寫成可直接`python3 script.py`執行的腳本，檔名用`verify_`前綴跟既有`test_`（回測）命名區分。

---

## Task 1: 泛化 `calc_kd()`，新增KD(36,5,5)欄位

**Files:**
- Modify: `stock_core.py:393-418`（`calc_kd()`函式本體）
- Modify: `stock_core.py:447`（`run_all_indicators()`裡的呼叫處）
- Create: `verify_calc_kd_generalization.py`

**Interfaces:**
- Produces: `calc_kd(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3, k_col: str = "K", d_col: str = "D") -> pd.DataFrame`——既有呼叫`calc_kd(df)`行為完全不變（預設值對應舊的(9,3,3)/`K`/`D`）。`run_all_indicators()`回傳的DataFrame在既有`K`/`D`欄位之外，新增`K365`/`D365`欄位（所有週期都算，包含週線，但週線不會被任何規則讀取）。

- [ ] **Step 1: 寫驗證腳本，先確認現在還沒有K365/D365欄位（預期失敗）**

建立 `verify_calc_kd_generalization.py`：

```python
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

    # 2. 新參數獨立欄位，不污染原本K/D
    kd365 = calc_kd(df, n=36, m1=5, m2=5, k_col="K365", d_col="D365")
    assert "K365" in kd365.columns and "D365" in kd365.columns
    assert "K" in kd365.columns and "D" in kd365.columns
    pd.testing.assert_series_equal(kd365["K"], new["K"], check_names=False)
    print("OK: calc_kd(df, n=36, m1=5, m2=5, k_col='K365', d_col='D365') 正確輸出獨立欄位，不影響K/D")

    # 3. run_all_indicators() 同時有兩組欄位
    full = run_all_indicators(df, "日")
    assert {"K", "D", "K365", "D365"}.issubset(full.columns)
    assert full["K365"].notna().any() and full["D365"].notna().any()
    print("OK: run_all_indicators() 同時輸出K/D(9,3,3)與K365/D365(36,5,5)")

    print("\n全部通過")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 執行驗證腳本，確認現在會失敗（因為K365還沒實作）**

Run: `python3 verify_calc_kd_generalization.py`
Expected: `TypeError: calc_kd() got an unexpected keyword argument 'n'`（因為目前`calc_kd`只有`period`參數，還沒加`n`/`m1`/`m2`/`k_col`/`d_col`）

- [ ] **Step 3: 泛化 `calc_kd()`**

把 `stock_core.py:393-418` 的 `calc_kd` 改成：

```python
def calc_kd(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3,
            k_col: str = "K", d_col: str = "D") -> pd.DataFrame:
    """
    向量化版本（原本是逐列 Python for 迴圈）。
    K[i] = K[i-1]*(1-1/m1) + RSV[i]/m1 這個遞迴式，數學上就是 alpha=1/m1 的
    EWM（指數加權平均，adjust=False）。原本第0筆固定用種子值50（不看
    RSV[0]），所以把序列第0筆換成50再套 EWM，跟原本逐列迴圈的結果
    完全等價，只是不用真的跑 Python for 迴圈。D 同理，用 K 當輸入、alpha=1/m2。

    n：RSV回看窗口（舊稱period）。m1：K值平滑權值（RSVt權值）。m2：D值
    平滑權值（Kt權值）。預設(9,3,3)/K/D欄位是台股軟體慣用的KDJ設定，
    呼叫端不帶參數時行為跟泛化前完全相同。k_col/d_col讓呼叫端可以同時
    並存多組不同參數的KD（例如KD(9,3,3)跟KD(36,5,5)），不互相覆蓋。
    """
    df = df.copy()
    low_min = df["min"].rolling(window=n).min()
    high_max = df["max"].rolling(window=n).max()
    denom = (high_max - low_min).replace(0, np.nan)
    rsv = ((df["close"] - low_min) / denom * 100).fillna(50.0)

    if len(df) > 0:
        rsv.iloc[0] = 50.0
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()

    k_seeded = k.copy()
    if len(df) > 0:
        k_seeded.iloc[0] = 50.0
    d = k_seeded.ewm(alpha=1 / m2, adjust=False).mean()

    df[k_col] = k
    df[d_col] = d
    return df
```

- [ ] **Step 4: 在 `run_all_indicators()` 新增KD(36,5,5)呼叫**

把 `stock_core.py:447` 這一行：

```python
    df = calc_kd(df)
```

改成：

```python
    df = calc_kd(df)   # 既有 KD(9,3,3) -> K/D 欄位，週線規則(_step2_weekly_kd_golden_cross)用這組
    df = calc_kd(df, n=36, m1=5, m2=5, k_col="K365", d_col="D365")  # 新增 KD(36,5,5)，供日/60分/5分的Double Check用
```

- [ ] **Step 5: 執行驗證腳本，確認通過**

Run: `python3 verify_calc_kd_generalization.py`
Expected: 三段`OK:`訊息都印出來，最後印出「全部通過」，exit code 0

- [ ] **Step 6: Commit**

```bash
git add stock_core.py verify_calc_kd_generalization.py
git commit -m "泛化calc_kd()支援自訂KD參數，新增KD(36,5,5)欄位

calc_kd()從只接受period改成接受(n,m1,m2,k_col,d_col)，既有呼叫calc_kd(df)
行為完全不變（預設值對應原本9,3,3）。run_all_indicators()新增一次呼叫
算出K365/D365，供後續KD(36,5,5) Double Check機制使用，週線既有規則
不受影響。"
```

---

## Task 2: 泛化 `_detect_bearish_divergence()` 支援自訂欄位

**Files:**
- Modify: `sop_decision.py:233-252`（`_detect_bearish_divergence()`函式本體）
- Create: `verify_detect_bearish_divergence_column.py`

**Interfaces:**
- Consumes: 無（純函式內部重構）
- Produces: `_detect_bearish_divergence(df: pd.DataFrame, lookback: int, column: str = "MACD_hist") -> bool`——既有呼叫處（`_step3_macd`裡的`_detect_bearish_divergence(df, cfg["divergence_lookback"])`）不用改，因為`column`有預設值。

- [ ] **Step 1: 寫驗證腳本**

建立 `verify_detect_bearish_divergence_column.py`：

```python
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
```

- [ ] **Step 2: 執行驗證腳本，確認現在會失敗**

Run: `python3 verify_detect_bearish_divergence_column.py`
Expected: `TypeError: _detect_bearish_divergence() got an unexpected keyword argument 'column'`

- [ ] **Step 3: 泛化 `_detect_bearish_divergence()`**

把 `sop_decision.py:233-252` 改成：

```python
def _detect_bearish_divergence(df: pd.DataFrame, lookback: int, column: str = "MACD_hist") -> bool:
    if len(df) < lookback or column not in df.columns:
        return False
    window = df.tail(lookback).reset_index(drop=True)
    half = lookback // 2
    prior, recent = window.iloc[:half], window.iloc[half:]
    if prior["close"].isna().all() or recent["close"].isna().all():
        return False

    prior_peak_idx = prior["close"].idxmax()
    recent_peak_idx = recent["close"].idxmax()
    prior_peak_close = prior.loc[prior_peak_idx, "close"]
    recent_peak_close = recent.loc[recent_peak_idx, "close"]
    prior_peak_val = prior.loc[prior_peak_idx, column]
    recent_peak_val = recent.loc[recent_peak_idx, column]

    if pd.isna(prior_peak_val) or pd.isna(recent_peak_val):
        return False

    return recent_peak_close > prior_peak_close and recent_peak_val < prior_peak_val
```

（只是把原本寫死的`"MACD_hist"`改成參數`column`，邏輯完全不變；既有呼叫處`_step3_macd`裡的`_detect_bearish_divergence(df, cfg["divergence_lookback"])`因為`column`有預設值`"MACD_hist"`，不用跟著改。）

- [ ] **Step 4: 執行驗證腳本，確認通過**

Run: `python3 verify_detect_bearish_divergence_column.py`
Expected: 三段`OK:`訊息都印出來，最後印出「全部通過」

- [ ] **Step 5: Commit**

```bash
git add sop_decision.py verify_detect_bearish_divergence_column.py
git commit -m "泛化_detect_bearish_divergence()支援自訂指標欄位

新增column參數(預設MACD_hist，既有呼叫處行為不變)，讓同一套背離偵測
邏輯之後可以重用在KD365(column='K365')，不用重寫第二套背離判斷。"
```

---

## Task 3: 新增 `_step_kd365()` 與 `kd365` 設定區塊

**Files:**
- Modify: `sop_rules.yaml`（在`indicators:`區塊新增`kd365`子區塊）
- Modify: `sop_decision.py`（新增`_step_kd365()`函式，放在`_step3_macd`之後、`_step4_volume`之前）
- Create: `verify_step_kd365.py`

**Interfaces:**
- Consumes: `calc_kd()`產出的`K365`/`D365`欄位（Task 1）、`_detect_bearish_divergence(df, lookback, column=...)`（Task 2）
- Produces: `_step_kd365(df: pd.DataFrame, tf: str, latest: pd.Series, reasons: List[str], caveats: List[str]) -> tuple`，回傳`(kd365_bias: float, kd365_cross_down: bool, kd365_bearish_divergence: bool)`。這一步**還沒**接進`evaluate_timeframe()`，純函式驗證，接線在Task 4做。

- [ ] **Step 1: 在 `sop_rules.yaml` 新增 `kd365` 設定區塊**

在 `sop_rules.yaml` 的 `macd:` 區塊（約70-76行）後面，`volume:` 區塊之前，插入：

```yaml
  kd365:
    # 但丁老師課程新增的第三代指標，跟MACD做Double Check。課程筆記原本
    # 寫「365」，經券商App參數截圖比對確認為「KD-隨機指標(常用)」預設：
    # 期間36／RSVt權值5／Kt權值5，不是單一個period=365的數字。
    # 只適用日/60分/5分，週線與月線維持原本的KD(9,3,3)（weekly_kd_
    # golden_cross規則），不套用這組參數。
    n: 36
    m1: 5
    m2: 5
    golden_cross_bonus: 1       # 待backtest驗證微調的初始值
    dead_cross_penalty: -1      # 待backtest驗證微調的初始值
    blunted_threshold_high: 80  # K值連續N根高於此視為高檔鈍化
    blunted_threshold_low: 20   # K值連續N根低於此視為低檔鈍化
    blunted_min_bars: 3         # 待backtest驗證微調的初始值
    divergence_lookback: 20     # 比照macd.divergence_lookback

```

- [ ] **Step 2: 寫驗證腳本**

建立 `verify_step_kd365.py`：

```python
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
```

- [ ] **Step 3: 執行驗證腳本，確認現在會失敗**

Run: `python3 verify_step_kd365.py`
Expected: `ImportError: cannot import name '_step_kd365' from 'sop_decision'`

- [ ] **Step 4: 實作 `_step_kd365()`**

在 `sop_decision.py` 的 `_step3_macd()` 函式結束之後（約第297行，`_step3_macd`的`return`之後、`# Step 4：量能驗證`註解區塊之前），新增：

```python
# ------------------------------------------------------------------
# Step 3.5：KD(36,5,5) Double Check（但丁老師課程新增的第三代指標）
# 只適用日/60分/5分，呼叫端須在tf != "週"時才呼叫這個函式——週線維持
# 既有_step2_weekly_kd_golden_cross用的KD(9,3,3)，不套這組參數
# （見sop_rules.yaml kd365區塊註解）。這一輪刻意不讓kd365_bias進評分
# 公式、不讓kd365_cross_down/kd365_bearish_divergence否決conclusion，
# 只用於reasons/caveats文字與suggested_exposure的計算，等
# test_kd365_double_check.py回測驗證過才考慮讓它影響conclusion本身。
# ------------------------------------------------------------------
def _step_kd365(df: pd.DataFrame, tf: str, latest: pd.Series,
                 reasons: List[str], caveats: List[str]) -> tuple:
    """回傳 (kd365_bias: float, kd365_cross_down: bool, kd365_bearish_divergence: bool)"""
    cfg = RULES["indicators"]["kd365"]
    kd365_bias = 0.0
    kd365_cross_down = False

    if pd.isna(latest.get("K365")) or pd.isna(latest.get("D365")):
        caveats.append("KD(36,5,5) 資料不足")
        return kd365_bias, False, False

    k, d = latest["K365"], latest["D365"]

    if len(df) >= 2:
        prev = df.iloc[-2]
        if pd.notna(prev.get("K365")) and pd.notna(prev.get("D365")):
            golden = prev["K365"] <= prev["D365"] and k > d
            dead = prev["K365"] >= prev["D365"] and k < d
            if golden:
                kd365_bias += cfg["golden_cross_bonus"]
                reasons.append("KD(36,5,5) 出現黃金交叉")
            elif dead:
                kd365_bias += cfg["dead_cross_penalty"]
                kd365_cross_down = True
                reasons.append("KD(36,5,5) 出現死亡交叉")

    high_th = cfg["blunted_threshold_high"]
    low_th = cfg["blunted_threshold_low"]
    blunted_bars = cfg["blunted_min_bars"]
    if len(df) >= blunted_bars:
        recent_k = df["K365"].tail(blunted_bars)
        if recent_k.notna().all() and (recent_k > high_th).all():
            reasons.append(f"KD(36,5,5) 連續{blunted_bars}根K棒維持高檔鈍化（>{high_th}）")
        elif recent_k.notna().all() and (recent_k < low_th).all():
            reasons.append(f"KD(36,5,5) 連續{blunted_bars}根K棒維持低檔鈍化（<{low_th}）")

    kd365_bearish_divergence = _detect_bearish_divergence(
        df, cfg["divergence_lookback"], column="K365")
    if kd365_bearish_divergence:
        reasons.append("⚠️ KD(36,5,5) 出現頂背離跡象（股價創高、K值未同步創高），先提高警覺")

    return kd365_bias, kd365_cross_down, kd365_bearish_divergence
```

- [ ] **Step 5: 執行驗證腳本，確認通過**

Run: `python3 verify_step_kd365.py`
Expected: 四段`OK:`訊息都印出來，最後印出「全部通過」

- [ ] **Step 6: Commit**

```bash
git add sop_rules.yaml sop_decision.py verify_step_kd365.py
git commit -m "新增_step_kd365()：KD(36,5,5)交叉/鈍化/背離偵測

比照_step3_macd的寫法，新增獨立的KD(36,5,5)判讀函式，重用Task2泛化
後的_detect_bearish_divergence()做背離偵測。只適用日/60分/5分，週線
不套用（維持既有KD(9,3,3)規則）。這一步只是函式本體，還沒接進
evaluate_timeframe()的判讀流程，不影響任何現有行為。"
```

---

## Task 4: 接線進 `evaluate_timeframe()`，新增 `suggested_exposure`

**Files:**
- Modify: `sop_decision.py`（`Verdict` dataclass、`evaluate_timeframe()`、`combine_timeframes()`）
- Modify: `sop_rules.yaml`（新增`exposure_rules`區塊）
- Create: `verify_suggested_exposure_pipeline.py`

**Interfaces:**
- Consumes: `_step_kd365()`（Task 3）
- Produces: `Verdict.suggested_exposure: Optional[int]`（100/70/50/30/0/None）。`combine_timeframes()`回傳的`detail[tf]`字典新增`"建議曝險"`鍵。

- [ ] **Step 1: 在 `sop_rules.yaml` 新增 `exposure_rules` 區塊**

在 `sop_rules.yaml` 檔案最後（`special_notes:`區塊之後、`industry_groups:`區塊之前，或industry_groups之後皆可，這裡放在special_notes後面），新增：

```yaml
# ==================================================================
# 漸進式部位建議（suggested_exposure，不影響conclusion本身）
# ------------------------------------------------------------------
# 但丁老師課程新增的5級部位控管概念：100%(買進/加碼) / 70%(觀望但均線
# 支撐未破) / 50%(單一指標示警) / 30%(MACD+KD365雙重轉弱確認但均線未破)
# / 0%(hard_sell，均線已破位)。這個表格待backtest驗證微調，先建好管線
# 讓資料能跑出來。
# ==================================================================
exposure_rules:
  watch_support_held: 70   # 但丁老師原話「7至8成」，取中間值，可調整
```

- [ ] **Step 2: 寫驗證腳本**

建立 `verify_suggested_exposure_pipeline.py`：

```python
"""
驗證evaluate_timeframe()接上suggested_exposure後：
1. 週線(tf="週")完全不呼叫_step_kd365()，不受影響——用合成資料跑週線，
   確認不會因為沒有K365/D365欄位而出錯（週線的run_all_indicators也會
   算K365/D365，但_step_kd365不應該被呼叫到）。
2. 日線能正常跑出一個suggested_exposure，值落在{0,30,50,70,100,None}之一。
3. combine_timeframes()的detail[tf]字典有"建議曝險"鍵。

用合成資料，不需要FinMind token，執行：
    python3 verify_suggested_exposure_pipeline.py
"""
import numpy as np
import pandas as pd

from sop_decision import combine_timeframes, evaluate_timeframe
from stock_core import run_all_indicators


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


def main() -> None:
    df = _make_synthetic_df()

    # 1. 週線不受影響，不會因為K365邏輯而crash
    df_week = run_all_indicators(df.copy(), "週")
    v_week = evaluate_timeframe(df_week, "週")
    print(f"OK: 週線正常跑完，conclusion={v_week.conclusion}, "
          f"suggested_exposure={v_week.suggested_exposure}（預期None，週線不分類）")

    # 2. 日線能跑出suggested_exposure
    df_day = run_all_indicators(df.copy(), "日")
    v_day = evaluate_timeframe(df_day, "日")
    assert v_day.suggested_exposure in (0, 30, 50, 70, 100, None), v_day.suggested_exposure
    print(f"OK: 日線suggested_exposure={v_day.suggested_exposure}（合法值之一）")

    # 3. combine_timeframes()有「建議曝險」鍵
    detail = combine_timeframes({"日": v_day})
    assert "建議曝險" in detail["各週期明細"]["日"]
    assert detail["各週期明細"]["日"]["建議曝險"] == v_day.suggested_exposure
    print("OK: combine_timeframes()的detail正確帶出「建議曝險」")

    print("\n全部通過")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 執行驗證腳本，確認現在會失敗**

Run: `python3 verify_suggested_exposure_pipeline.py`
Expected: `AttributeError: 'Verdict' object has no attribute 'suggested_exposure'`

- [ ] **Step 4: 擴充 `Verdict` dataclass**

把 `sop_decision.py` 的 `Verdict` dataclass 改成：

```python
@dataclass
class Verdict:
    conclusion: str                 # 買進 / 加碼 / 賣出減碼 / 觀望
    confidence: str                 # 高 / 中 / 低
    score: float
    reasons: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    volume_breakout_confirmed: bool = False  # 三角收斂帶量突破確認（見_detect_triangle_consolidation）
    suggested_exposure: Optional[int] = None  # 100/70/50/30/0，None=無法明確分類（見exposure_rules）
```

- [ ] **Step 5: 在 `evaluate_timeframe()` 呼叫 `_step_kd365()`**

把 `sop_decision.py` 裡這一段（約第1011-1013行）：

```python
    macd_bias, dead_cross, macd_bearish_divergence = _step3_macd(df, latest, reasons, caveats)
    volume_bias, volume_bear_confirm = _step4_volume(
        df, tf, latest, macd_bearish_divergence, reasons, caveats)
```

改成：

```python
    macd_bias, dead_cross, macd_bearish_divergence = _step3_macd(df, latest, reasons, caveats)
    # KD(36,5,5) Double Check：只適用日/60分/5分，週線維持既有KD(9,3,3)
    # 規則不變（見sop_rules.yaml kd365區塊註解）
    if tf != "週":
        kd365_bias, kd365_cross_down, kd365_bearish_divergence = _step_kd365(
            df, tf, latest, reasons, caveats)
    else:
        kd365_bias, kd365_cross_down, kd365_bearish_divergence = 0.0, False, False
    volume_bias, volume_bear_confirm = _step4_volume(
        df, tf, latest, macd_bearish_divergence, reasons, caveats)
```

- [ ] **Step 6: 在 `conclusion` 確定之後，計算 `suggested_exposure`**

把 `sop_decision.py` 裡這一段（約第1161-1168行，confidence計算的地方）：

```python
    abs_score = abs(score)
    if key_ma_down_veto or volume_bear_confirm_for_hard_sell \
            or (ma_bias >= add_cfg["addon_ma_bias"] and volume_bias > 0):
        confidence = "高"
    elif abs_score >= scoring_cfg["confidence_mid_abs_score"]:
        confidence = "中"
    else:
        confidence = "低" if abs_score < scoring_cfg["confidence_low_abs_score"] else "中"
```

後面（在`if not reasons:`之前）新增：

```python
    # ---- 建議曝險（5級，待backtest驗證微調，見sop_rules.yaml exposure_rules）----
    # 這一輪刻意不讓這個欄位影響conclusion——只是把但丁老師課程的部位
    # 控管概念，用既有已經算出來的條件變數做映射，不是重新設計一套邏輯。
    exposure_cfg = RULES.get("exposure_rules", {})
    kd365_macd_double_bear = kd365_cross_down and macd_bearish_divergence
    if hard_sell:
        suggested_exposure = 0
    elif volume_bear_confirm_partial or kd365_macd_double_bear:
        suggested_exposure = 30
    elif caution_only or leading_indicator_exit_alert or kd365_cross_down or kd365_bearish_divergence:
        suggested_exposure = 50
    elif conclusion == "觀望" and not key_ma_down_veto:
        suggested_exposure = exposure_cfg.get("watch_support_held", 70)
    elif conclusion in ("買進", "加碼"):
        suggested_exposure = 100
    else:
        suggested_exposure = None
```

- [ ] **Step 7: 更新 `Verdict` 的 `return`**

把 `sop_decision.py` 裡最後的（約第1173-1175行）：

```python
    return Verdict(conclusion=conclusion, confidence=confidence, score=score,
                    reasons=reasons, caveats=caveats,
                    volume_breakout_confirmed=triangle_breakout_confirmed)
```

改成：

```python
    return Verdict(conclusion=conclusion, confidence=confidence, score=score,
                    reasons=reasons, caveats=caveats,
                    volume_breakout_confirmed=triangle_breakout_confirmed,
                    suggested_exposure=suggested_exposure)
```

- [ ] **Step 8: 更新 `combine_timeframes()` 的 detail 字典**

把 `sop_decision.py` 裡這一段：

```python
    detail = {
        tf: {"結論": v.conclusion, "信心": v.confidence, "理由": v.reasons, "但書": v.caveats,
             "帶量突破": v.volume_breakout_confirmed}
        for tf, v in verdicts.items()
    }
```

改成：

```python
    detail = {
        tf: {"結論": v.conclusion, "信心": v.confidence, "理由": v.reasons, "但書": v.caveats,
             "帶量突破": v.volume_breakout_confirmed, "建議曝險": v.suggested_exposure}
        for tf, v in verdicts.items()
    }
```

- [ ] **Step 9: 檢查 `evaluate_timeframe()` 資料不足的早退路徑**

`sop_decision.py` 裡 `evaluate_timeframe()` 開頭（約第991-992行）的早退：

```python
    if df.empty:
        return Verdict(conclusion="觀望", confidence="低", score=0.0,
                        reasons=["資料為空，無法判讀"], caveats=[])
```

這裡不用改——`suggested_exposure`有預設值`None`，`Verdict`建構子不帶這個參數時自動是`None`，行為正確。

- [ ] **Step 10: 執行驗證腳本，確認通過**

Run: `python3 verify_suggested_exposure_pipeline.py`
Expected: 三段`OK:`訊息都印出來，最後印出「全部通過」

- [ ] **Step 11: 跑一次Task1/2/3的驗證腳本，確認沒有回歸**

Run:
```bash
python3 verify_calc_kd_generalization.py
python3 verify_detect_bearish_divergence_column.py
python3 verify_step_kd365.py
```
Expected: 三支腳本都印出「全部通過」

- [ ] **Step 12: Commit**

```bash
git add sop_decision.py sop_rules.yaml verify_suggested_exposure_pipeline.py
git commit -m "接線KD365 Double Check進evaluate_timeframe()，新增suggested_exposure

Verdict新增suggested_exposure欄位(100/70/50/30/0/None)，用既有已算出
的hard_sell/volume_bear_confirm_partial/caution_only等條件變數加上新的
kd365_cross_down/kd365_bearish_divergence做5級映射，不改變conclusion
判斷邏輯本身。週線(tf=='週')不呼叫_step_kd365()，kd365相關變數維持
False/0.0，行為跟改動前完全相同。combine_timeframes()的detail同步
帶出「建議曝險」鍵。"
```

---

## Task 5: 新增 `test_kd365_double_check.py` 回測驗證腳本

**Files:**
- Create: `test_kd365_double_check.py`

**Interfaces:**
- Consumes: `evaluate_timeframe()`（Task 4，產出`kd365_cross_down`/`kd365_bearish_divergence`相關的`reasons`文字可供比對）、`stock_core.STOCK_NAME_MAP`/`unique_watchlist`/`get_stock_data`/`run_all_indicators`
- Produces: 無新函式，純粹是一支可執行的回測報告腳本，不被其他程式import

這個任務需要**真實歷史資料**（透過FinMind API），執行時需要你自己的FinMind token，沒辦法在這個plan的驗證步驟裡自動跑過。寫法照抄`test_weekly_kd_and_daily_ma35_buy.py`的既有模式（`_build_weekly`/`FORWARD_WINDOWS`/`_forward_returns`這幾個既有輔助函式可以直接重用，不用重寫）。

- [ ] **Step 1: 建立 `test_kd365_double_check.py`**

```python
"""
驗證KD(36,5,5)+MACD的Double Check機制，是不是真的比單看MACD背離更可靠。

比較兩組「訊號後N日報酬／下跌比例」：
    A) 只有MACD頂背離（現有_step3_macd.macd_bearish_divergence單獨成立）
    B) MACD頂背離 + KD365死亡交叉同時成立（Double Check雙重確認）

用意：B組如果比A組保護力更好（後續下跌比例更高、平均報酬更低），
代表Double Check機制真的有篩選效果，才考慮之後讓它影響conclusion。
這支腳本只跑日線（跟test_weekly_kd_and_daily_ma35_buy.py的週線部分
分工，60分/5分的分K資料Fugle API有時間深度限制，之後如果要測短線
版本另外處理）。

重用backtest_signal_log.py既有的_forward_returns(price_df, signal_date)：
吃「完整價格DataFrame + 訊號當天日期字串」，回傳{5: 報酬, 10: 報酬,
20: 報酬}（FORWARD_WINDOWS=(5,10,20)，鍵是整數交易日數、不是字串）。

用法（在專案資料夾內執行，需要FinMind token）：
    python3 test_kd365_double_check.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import _detect_bearish_divergence, _step_kd365
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24


def _collect_events(df: pd.DataFrame) -> dict:
    """逐日重建_step3_macd的背離判斷跟_step_kd365的死叉判斷，標出A組/B組的事件日期字串。"""
    cfg_lookback = 20  # 對應sop_rules.yaml macd.divergence_lookback/kd365.divergence_lookback
    macd_only_dates, double_check_dates = [], []

    for i in range(cfg_lookback, len(df)):
        window = df.iloc[:i + 1]
        latest = window.iloc[-1]

        macd_div = _detect_bearish_divergence(window, cfg_lookback, column="MACD_hist")
        if not macd_div:
            continue

        reasons, caveats = [], []
        _, kd365_cross_down, _ = _step_kd365(window, "日", latest, reasons, caveats)

        signal_date = df.iloc[i]["date"]
        macd_only_dates.append(signal_date)
        if kd365_cross_down:
            double_check_dates.append(signal_date)

    return {"macd_only": macd_only_dates, "double_check": double_check_dates}


def _dates_to_forward_returns(df: pd.DataFrame, dates: list) -> list:
    return [_forward_returns(df, d) for d in dates]


def run(api_token: str, months: int) -> list:
    start = (date.today() - timedelta(days=months * 31 + WARMUP_CALENDAR_DAYS)).isoformat()
    results = []

    def _one(name, code, market):
        try:
            df = get_stock_data(code, market, start, str(date.today()), api_token)
            if df.empty:
                return {"name": name, "code": code, "error": "無資料"}
            df = run_all_indicators(df, "日")
            events = _collect_events(df)
            return {
                "name": name, "code": code,
                "macd_only_events": _dates_to_forward_returns(df, events["macd_only"]),
                "double_check_events": _dates_to_forward_returns(df, events["double_check"]),
            }
        except Exception as exc:  # noqa: BLE001
            return {"name": name, "code": code, "error": str(exc)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_one, name, code, market)
                   for name, code, market in unique_watchlist(STOCK_NAME_MAP)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _print_stats(label: str, events: list) -> None:
    if not events:
        print(f"{label}：無事件")
        return
    n = len(events)
    print(f"{label}（{n}筆事件）：")
    for window in FORWARD_WINDOWS:
        rets = [e[window] for e in events if e.get(window) is not None]
        if not rets:
            continue
        avg = sum(rets) / len(rets)
        down_ratio = sum(1 for r in rets if r < 0) / len(rets)
        print(f"  {window}日後：平均報酬{avg:+.2%}，下跌比例{down_ratio:.1%}")


def main() -> None:
    api_token = sys.argv[1] if len(sys.argv) > 1 else ""
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    all_macd_only, all_double_check = [], []
    for r in results:
        if r.get("error"):
            continue
        all_macd_only.extend(r["macd_only_events"])
        all_double_check.extend(r["double_check_events"])

    print(f"\n=== 回測{months_arg}個月，{len(unique_watchlist(STOCK_NAME_MAP))}檔加總 ===\n")
    _print_stats("A) 只有MACD頂背離", all_macd_only)
    print()
    _print_stats("B) MACD頂背離+KD365死亡交叉（Double Check）", all_double_check)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 語法檢查（不需要API token）**

Run: `python3 -m py_compile test_kd365_double_check.py`
Expected: 無輸出、exit code 0

- [ ] **Step 3: 實際執行回測（需要你自己的FinMind token，在終端機手動跑）**

```bash
python3 test_kd365_double_check.py <你的FinMind_token> 24
```

Expected: 印出A組/B組事件數與各天期的平均報酬/下跌比例。**這一步的結果需要你自己判讀**——如果B組（Double Check）的下跌比例/平均報酬確實比A組（只看MACD）更好，代表機制有效，之後可以考慮讓`kd365_cross_down`+`macd_bearish_divergence`同時成立時升級成影響`conclusion`本身的硬條件（屬於下一輪的改動，不在這個plan範圍內）。

- [ ] **Step 4: Commit**

```bash
git add test_kd365_double_check.py
git commit -m "新增test_kd365_double_check.py：驗證KD365+MACD Double Check保護力

比照test_weekly_kd_and_daily_ma35_buy.py手法，比較「只看MACD頂背離」
vs「MACD頂背離+KD365死亡交叉同時成立」兩組事件後續N日的報酬與下跌
比例，用實際歷史資料驗證Double Check機制是否真的比單一指標更可靠。
結果需要人工判讀，是否讓KD365正式否決conclusion留待下一輪決定。"
```

---

## Self-Review Notes

- **Spec coverage**：spec的四個技術決定（`calc_kd()`泛化／`_step_kd365()`＋重用背離偵測／`Verdict.suggested_exposure`／`sop_rules.yaml`新設定）分別對應Task1-4；spec的「測試/驗證計畫」四點對應Task1-3的`verify_*.py`（單元層級）、Task4的`verify_suggested_exposure_pipeline.py`（映射表管線）、Task5的`test_kd365_double_check.py`（backtest保護力）。spec「明確不在這次範圍內」列的四項（notify_email/home.py顯示、15分/30分、KD365否決conclusion、個股規則）這個plan都沒有任務觸碰，範圍一致。
- **Type一致性**：`_step_kd365()`的回傳型別`(float, bool, bool)`在Task3定義、Task4引用時一致；`Verdict.suggested_exposure: Optional[int]`在Task4定義與使用一致；`_detect_bearish_divergence(df, lookback, column=...)`在Task2定義、Task3/Task5引用時參數名與順序一致。
- **No placeholders**：所有程式碼區塊都是可以直接貼上執行的完整內容，沒有「依此類推」或「同上」。
