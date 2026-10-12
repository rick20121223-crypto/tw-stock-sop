# KD365 Double Check（買進側）偵測機制 — 設計文件（Phase 1 of 3）

日期：2026-10-12
來源：課程錄音《中山北路二段65巷6號.m4a》使用者貼過的買進/加碼條件說明，
對應第1點「抓起漲與打底進場點」。

## 背景

賣出側的KD365 Double Check已經做完三輪（偵測→否決conclusion）並合併進
main。買進側課程內容涵蓋四點（起漲打底、順勢加碼、過濾亂買、均線最高
裁決），範圍比賣出側大，使用者決定拆成三個子輪，這份spec是**第一輪**：
只建立偵測機制，進`reasons`文字，不影響任何結論判斷。第二輪backtest
驗證，第三輪才視結果決定要不要讓「沒雙重確認就降級觀望」正式生效。

## 範圍

**這一輪只處理課程第1點（起漲打底進場點）**，具體做法：探索現有程式碼
後發現`_detect_w_bottom()`（W底型態偵測，`sop_decision.py:563`）已經做了
課程描述的大部分事——抓打底（右底比左底高）、頸線突破、回測不破、確認
紅K，最後有一個`synced`判斷（均線斜率+MACD同步轉多才算「轉折訊號確認」，
否則只是「型態轉折待確認」）。這跟課程「低檔打底+KD365與MACD同步轉強」
的描述高度重疊，**不新增一個獨立的型態偵測器**，而是在既有`synced`判斷
裡加一個KD365條件。

**明確不在這次範圍內**：
- 課程第2點（順勢加碼）、第3點（過濾亂買）——留給之後的輪次（過濾亂買
  是第三輪「否決conclusion」的目標，順勢加碼如果要做，是另一個獨立
  決定，這次不碰）。
- `_detect_m_top()`（M頭偵測）——買進側不需要，不動。
- 不影響`conclusion`、`score`、`suggested_exposure`——這輪只改`reasons`
  文字。
- 不改`_step_kd365()`的回傳介面——賣出側已經依賴這個函式簽名
  `(kd365_bias, kd365_cross_down, kd365_bearish_divergence)`，這次直接
  讀`latest`的`K365`/`D365`欄位做狀態比較，不新增回傳值、不用改任何
  既有呼叫處。

## 設計

### `_detect_w_bottom()`新增`kd365_bullish`參數

函式簽名從：
```python
def _detect_w_bottom(df: pd.DataFrame, fast_slope_up: bool, macd_bias: float,
                      reasons: List[str]) -> Optional[float]:
```
改成：
```python
def _detect_w_bottom(df: pd.DataFrame, fast_slope_up: bool, macd_bias: float,
                      kd365_bullish: bool, reasons: List[str]) -> Optional[float]:
```

`synced`判斷從：
```python
synced = fast_slope_up and macd_bias > 0
```
改成：
```python
synced = fast_slope_up and macd_bias > 0 and kd365_bullish
```

兩段`reasons.append`文字（確認/待確認）都要更新成提到KD(36,5,5)也同步
確認/尚未同步（具體文字見plan）。

### 呼叫端：在`evaluate_timeframe()`算出`kd365_bullish`

`kd365_bullish`代表「KD(36,5,5)目前處於多頭排列（K365>D365）」，跟
`macd_bias > 0`（MACD目前偏多）同一種「目前狀態」風格，不是「今天剛好
黃金交叉」這種「事件」風格——理由：W底確認紅K那天，KD365可能是前幾天
就黃金交叉、之後維持多頭排列，不一定剛好跟確認紅K同一天交叉，用「目前
狀態」比較符合「同步維持多頭」的實際情境。

```python
kd365_bullish = (
    tf != "週"
    and pd.notna(latest.get("K365")) and pd.notna(latest.get("D365"))
    and latest["K365"] > latest["D365"]
)
w_bottom_neckline = _detect_w_bottom(df, fast_slope_up, macd_bias, kd365_bullish, reasons)
```

`tf != "週"`排除週線——跟賣出側、Task4那輪一致的範圍限制（週線維持原本
KD(9,3,3)規則，不套用KD(36,5,5)）。

## 測試／驗證計畫

1. **合成資料正確性**：`verify_w_bottom_kd365.py`（新增）——構造一個完整
   的W底型態（右底高於左底、頸線突破、回測不破、確認紅K），分別測試
   「均線/MACD都同步、KD365也同步」（reasons應包含KD(36,5,5)也同步的
   文字）vs「均線/MACD同步但KD365未同步」（reasons應該是待確認版本）
   兩種情況。
2. **既有regression**：重跑賣出側那輪留下的5支`verify_*.py`，確認這次
   改動沒有破壞之前的行為（`_detect_w_bottom`簽名變了，理論上只有
   `evaluate_timeframe()`這一個呼叫處會被影響，其餘4支不會碰到這個
   函式）。
3. 不需要backtest——這輪刻意不影響`conclusion`，只是把偵測管線做出來，
   backtest驗證是下一輪的事。
