# KD(36,5,5) Double Check 與漸進式部位建議 — 設計文件

日期：2026-10-11
來源：但丁老師課程筆記（逐字稿＋手寫筆記＋券商App參數截圖三方比對確認）

## 背景

但丁老師最新課程對SOP方法論提出兩項調整：

1. 新增一條KD指標，跟現有均線(第一代)/MACD(二代)並列，定位「四大指標之一」，
   跟MACD搭配做「Double Check」雙重驗證，減少誤判。
2. 把現有「買進/加碼/觀望/賣出減碼」四選一結論，進一步細分成5級漸進式部位
   建議（100% / 7-8成 / 5成 / 3成 / 0%）。

課程筆記原文把這條KD參數寫成「365」，經過三輪圖片比對才確認真正數字：

- 第一張截圖（KDJ-隨機指標）顯示現有預設「期間9／RSVt權值3／Kt權值3」，
  確認「933」＝這三個欄位的真正對應。
- 第二張截圖（KD-隨機指標(常用)）顯示但丁老師實際固定在用的參數：
  **期間36／RSVt權值5／Kt權值5**，「365」是課程逐字稿把「36,5,5」誤聽寫
  壓縮成一個數字。
- 最終確認：**KD(n=36, m1=5, m2=5)**，標籤正是「(常用)」。

## 範圍

**這一輪只處理**：日線、60分、5分。

**明確排除**：
- 週線／月線：但丁老師原話「週月線應維持原本參數，因365在大週期時間拉太長」，
  既有`_step2_weekly_kd_golden_cross`規則（用KD(9,3,3)）完全不動。
- 15分／30分：系統目前完全沒有這兩個週期（只有週/日/60分/5分），課程雖提到
  但這是另一件事（新增週期的資料抓取/均線組合管線），這輪不做，列為後續
  獨立的一輪。

## 現狀確認（已讀過的程式碼）

- [stock_core.py:393](../../../stock_core.py) `calc_kd(df, period=9)`：
  `period`只控制RSV回看窗口（對應「期間」），K/D平滑寫死用
  `alpha=1/3`（數學上等價於m1=3,m2=3），**沒有獨立參數可以改K/D平滑**。
- `run_all_indicators()`對所有週期（週/日/60分/5分）都無條件呼叫一次
  `calc_kd(df)`，產出的`K`/`D`欄位目前只有週線的
  `_step2_weekly_kd_golden_cross`會讀，其餘週期算出來但沒人用。
- [sop_decision.py:233](../../../sop_decision.py) `_detect_bearish_divergence(df, lookback)`：
  已經是「價格創高、指標未同步創高」的通用背離偵測邏輯，但**寫死檢查
  `MACD_hist`欄位**，目前只被`_step3_macd`呼叫。
- `Verdict` dataclass（[sop_decision.py:48](../../../sop_decision.py)）目前是
  `conclusion`(四選一字串) + `confidence` + `score` + `reasons`/`caveats`
  + 這個session新增的`volume_breakout_confirmed`(bool)。沒有任何部位
  百分比欄位。現有最接近的先例是`volume_bear_confirm_partial`規則用
  `caveats`文字提示「建議先減碼50%」，不是結構化數字。

## 設計

### 1. `stock_core.py`：泛化 `calc_kd()`

```python
def calc_kd(df, n: int = 9, m1: int = 3, m2: int = 3,
            k_col: str = "K", d_col: str = "D") -> pd.DataFrame:
    ...
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    ...
    d = k_seeded.ewm(alpha=1 / m2, adjust=False).mean()
    df[k_col] = k
    df[d_col] = d
    return df
```

- 既有呼叫 `calc_kd(df)` 不用改，所有預設值等於現在的(9,3,3)/`K`/`D`欄位，
  週線規則行為完全不變。
- `run_all_indicators()` 新增第二次呼叫：
  `df = calc_kd(df, n=36, m1=5, m2=5, k_col="K365", d_col="D365")`，
  對所有週期都算（包含週線也會算出`K365`/`D365`欄位，但不會被使用——
  「週線不適用」這件事由`sop_decision.py`的呼叫端判斷式負責，不是靠
  不算這個欄位來保證，兩層保險但邏輯單純）。

### 2. `sop_decision.py`：新增 `_step_kd365()`

比照現有`_step3_macd`同樣的寫法，只在`tf != "週"`時呼叫：

```python
def _step_kd365(df, tf, latest, reasons, caveats) -> tuple:
    """回傳 (kd365_bias: float, kd365_cross_down: bool, kd365_bearish_divergence: bool)"""
```

三件事：
- **交叉**：比照`_step3_macd`裡黃金/死亡交叉的判斷方式，比較`K365`/`D365`
  前一筆與這一筆的大小關係。
- **鈍化**：`K365`連續N根K棒 > 80（高檔鈍化）或 < 20（低檔鈍化），N與
  門檻放進`sop_rules.yaml`（新增`kd365`設定區塊）。
- **背離**：重用`_detect_bearish_divergence()`，泛化成接受
  `column: str = "MACD_hist"`參數，KD365呼叫時傳`column="K365"`——
  不重寫一套新的背離偵測，既有MACD背離邏輯完全不變。

**Double Check**：不新增額外抽象層，直接在既有`evaluate_timeframe()`
的if/elif鏈裡，把`kd365_cross_down`/`kd365_bearish_divergence`當成
新的輸入變數，跟現有`macd_bearish_divergence`/`dead_cross`並列判斷——
兩者同時成立＝確認轉弱（對應老師說的「雙重鈍化轉下+均線同步向下」）；
只有一個成立＝示警但不到確認等級。

**這一輪KD365不否決`conclusion`**：先只讓它進`reasons`/`caveats`和下面
的`suggested_exposure`，不當作`hard_sell`之類的硬otherwise條件——
照repo慣例，新訊號要先有`test_kd365_double_check.py`backtest驗證過
（參考`test_weekly_kd_and_daily_ma35_buy.py`手法）才正式讓它否決
`conclusion`本身，這個spec只負責把訊號算出來、串好管線。

### 3. `Verdict` 新增 `suggested_exposure` 欄位

```python
suggested_exposure: Optional[int] = None  # 100/70/50/30/0，None=無法明確分類
```

純新增欄位，預設`None`，舊程式碼完全不受影響。`combine_timeframes()`的
`detail[tf]`字典仿照這個session新增的`"帶量突破"`模式，一併加入
`"建議曝險": v.suggested_exposure`。

映射邏輯（在`conclusion`算完之後，重用已經算出來的既有布林變數，
不是重新設計一套新判斷）：

| 條件（沿用既有變數） | suggested_exposure |
|---|---|
| `hard_sell` 成立（均線破位/四關價跌破/死叉確認） | 0 |
| `volume_bear_confirm_partial` 或 KD365+MACD雙重轉弱確認（Double Check兩者皆空頭） | 30 |
| 只有KD365或MACD其中一個轉弱（非雙重確認）、或`caution_only`/`leading_indicator_exit_alert` | 50 |
| `conclusion == "觀望"` 且均線支撐未破（低檔震盪） | 70（但丁老師原話「7至8成」，取中間值70，可設定） |
| `conclusion in ("買進","加碼")` 且無上述任何警示 | 100 |
| 以上皆不符合 | None（不勉強分類） |

這個表格本身**待backtest微調**，先建好管線讓資料能跑出來，具體%數字
之後可能依驗證結果調整——這點會寫進sop_rules.yaml的changelog慣例。

### 4. `sop_rules.yaml` 新增設定

```yaml
kd365:
  n: 36
  m1: 5
  m2: 5
  blunted_threshold_high: 80
  blunted_threshold_low: 20
  blunted_min_bars: 3        # 待backtest驗證的初始值
  divergence_lookback: 20    # 比照macd.divergence_lookback

exposure_rules:
  watch_support_held: 70     # 待backtest微調
```

## 明確不在這次範圍內

- `notify_email.py`/`home.py`/`pages/*.py` 顯示`suggested_exposure`——
  這是獨立的小型Bounded任務，可以沿用這個session已經做過兩次的卡片
  模式（族群聯動提示、帶量突破標記），但不包含在這次spec，之後另開。
- 15分/30分週期支援。
- 讓KD365否決`conclusion`本身（需要先backtest驗證）。
- 個股案例（金像電/聯電/安葆等）——當作之後驗證用的案例，不逐一寫成
  規則。

## 測試／驗證計畫

1. 單元層級：`calc_kd()`泛化後，用合成資料驗證預設呼叫`calc_kd(df)`
   結果跟泛化前逐位元相同（確保週線規則零回歸）。
2. `_step_kd365()`/ `_detect_bearish_divergence(column=...)`：合成資料
   測交叉/鈍化/背離三種情境各自正確觸發。
3. `test_kd365_double_check.py`（新增，backtest）：比照
   `test_weekly_kd_and_daily_ma35_buy.py`手法，拿watchlist歷史資料，
   比較「KD365+MACD雙重確認 vs 只看MACD」在未來N日的保護力/報酬率，
   驗證Double Check機制是否真的比單看MACD更可靠。
4. `suggested_exposure`映射表：用`data/signal_log.csv`歷史資料跑過一輪，
   人工抽查幾檔實際案例（例如課程提到的安葆、聯電）, 確認分類結果跟
   老師上課的判讀邏輯方向一致，不要求完全吻合。
