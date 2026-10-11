# KD(36,5,5) Double Check 否決 conclusion（賣出側）— 設計文件

日期：2026-10-12
來源：使用者明確決定（「紀錄進去，但算入否決」），依據課程錄音
《中山北路二段65巷6號.m4a》的賣出/出場條件說明，以及
`test_kd365_double_check.py`的回測結果。

## 背景

2026-10-11那輪已經把KD(36,5,5)接進`evaluate_timeframe()`，但刻意只讓它
影響`reasons`/`caveats`和新增的`suggested_exposure`欄位，不影響
`conclusion`本身——等有回測證據再決定要不要讓它否決。

這次（2026-10-12）跑了`test_kd365_double_check.py`回測（21檔/24個月），
過程踩過兩次方法論坑才得到可信結果：
1. 第一版誤把「MACD頂背離」當成Double Check的MACD那一半，實際上課程
   講的是MACD自己死叉（DIF穿破訊號線，不管在不在零軸之上）。
2. 改測MACD死叉後，第二版沒有排除「訊號中途已經反轉」的案例，使用者
   親自抓出2026-03-23金居(8358)死叉後約10個交易日KD365就金叉翻多、
   卻被硬算20天後報酬+76.9%當失敗案例的問題。

修正方法論後的結果（三組都已排除訊號中途反轉的案例）：

| 組別 | 事件數 | 5日後 | 10日後 | 20日後 |
|---|---|---|---|---|
| A) 只有MACD死叉 | 406筆 | -1.89%／跌64.9%（265筆） | -4.73%／跌79.3%（169筆） | -10.74%／跌92.9%（70筆） |
| B) Double Check(MACD死叉+KD365死叉) | 167筆 | -2.15%／跌62.6%（115筆） | -4.97%／跌73.5%（68筆） | -11.10%／跌88.0%（25筆） |
| C) B+生死線斜率轉下彎 | 51筆 | -2.66%／跌66.7%（33筆） | -4.33%／跌73.7%（19筆） | -8.50%／跌85.7%（7筆） |

**證據解讀（誠實但書）**：三組在訊號維持期間都呈現清楚下跌傾向，驗證
這些訊號不是雜訊。但A/B/C彼此差異不明顯（B不一定比A好，C在10/20日
樣本數只剩19筆/7筆，太小不可信），所以「KD365雙重確認是否真的比單看
MACD更好」這個核心問題仍待更多樣本驗證。另外這個篩選法本身跟股價走弱
有內建相關性（KD/MACD狀態本來就是從股價算出來的），下跌比例數字不宜
直接當成客觀命中率解讀。

**使用者決定**：儘管證據仍待加強，仍要讓C組條件（Double Check+生死線
轉下彎）正式否決`conclusion`，比照現有`hard_sell`位階處理，但留一個
`sop_rules.yaml`開關，之後證據不支持時可以隨時關掉、不用改程式碼。

## 範圍

- 只影響日線、60分、5分（`_step_kd365()`本來就只在`tf != "週"`時被呼叫，
  週線的`kd365_cross_down`恆為`False`，新條件自然不會在週線觸發，不用
  額外加`tf`判斷）。
- 只新增一個否決條件，不改動`hard_sell`既有四個條件的邏輯與順序。
- 不處理買進/加碼側的Double Check（課程另外講的「抓起漲/加碼/過濾亂買」
  那部分）——使用者已確認分開處理，列為下一輪。

## 設計

### 新增變數：`kd365_double_check_confirmed`

在`sop_decision.py`的`evaluate_timeframe()`裡，`hard_sell = (...)`賦值
之前，新增：

```python
kd365_veto_enabled = sell_cfg.get("kd365_veto_enabled", True)
key_slope_down = ma_slope(df, KEY_MA[tf], lookback=slope_lookback) == "下彎"
kd365_double_check_confirmed = dead_cross and kd365_cross_down and key_slope_down
if kd365_double_check_confirmed:
    reasons.append(
        "⚠️ KD(36,5,5)死叉 + MACD死叉（Double Check）同時成立，且生死線已轉"
        "下彎，課程定義的「必須離場」等級"
    )
```

- `dead_cross`（MACD自己的死叉）、`kd365_cross_down`（KD365死叉）都是
  既有變數，直接重用，不重算。
- `key_slope_down`是新算的一行——**故意不重用`key_ma_down_veto`**，
  因為那個變數除了斜率下彎，還要求「股價已跌破生死線」，比回測C組
  實際測的條件（純粹看斜率方向，不看價格位置）更嚴格，重用它會跟
  已經拿到的回測證據對不上。`KEY_MA`/`ma_slope`在`sop_decision.py`
  已經import過，不用加新import。

### `hard_sell`新增第5個OR條件

```python
hard_sell = (
    (key_ma_down_veto and sell_cfg.get("key_ma_down_veto", True))
    or (four_key_broken and sell_cfg.get("four_key_broken", True))
    or (dead_cross_confirmed and sell_cfg.get("dead_cross_confirmed", True))
    or (volume_bear_confirm_for_hard_sell and sell_cfg.get("volume_bear_confirm", True))
    or (kd365_double_check_confirmed and kd365_veto_enabled)
)
```

### `sop_rules.yaml`新增開關

在`sell_rules:`區塊新增：

```yaml
  kd365_veto_enabled: true
  # 2026-10-12新增：KD(36,5,5)死叉+MACD死叉(Double Check)+生死線斜率轉
  # 下彎，同時成立時否決conclusion、直接賣出減碼。依據課程錄音《中山
  # 北路二段65巷6號.m4a》的賣出/出場條件說明採用，但test_kd365_double_
  # check.py的回測證據還不夠扎實（A/B/C三組差異不明顯、C組長天期樣本數
  # 只有19/7筆），使用者在證據不夠強的情況下仍決定採用，設為false可以
  # 隨時關掉、不用改程式碼（詳見本檔案2026-10-12 changelog）。
```

## 明確不在這次範圍內

- 買進/加碼側的Double Check（起漲打底、順勢加碼、過濾亂買）——下一輪。
- 不改動`suggested_exposure`的5級映射邏輯本身（仍沿用2026-10-11那輪的
  設計，`kd365_cross_down`/`kd365_bearish_divergence`繼續只在那個欄位
  起作用；這次新增的`kd365_double_check_confirmed`只用在`hard_sell`）。
- 不幫`dead_cross_confirmed`（現有條件）加`kd365_cross_down`之類的交叉
  驗證——那是既有條件，這次只新增一個獨立的第5個OR分支，不動既有四個。

## 測試／驗證計畫

因為這次是「已經有回測但證據不夠扎實、使用者仍決定採用」的情況，不是
典型的「先回測驗證過才正式採用」流程，驗證重點放在：

1. **合成資料正確性**：`verify_kd365_hard_sell.py`（新增）——構造MACD
   死叉+KD365死叉+生死線下彎同時成立的合成資料，確認`conclusion`正確
   變成「賣出減碼」；構造只有其中兩項成立（缺一項）的情況，確認
   **不會**誤觸發；確認`kd365_veto_enabled: false`時這個條件完全不影響
   `conclusion`（開關生效）。
2. **既有regression**：重跑`verify_calc_kd_generalization.py`、
   `verify_detect_bearish_divergence_column.py`、`verify_step_kd365.py`、
   `verify_suggested_exposure_pipeline.py`，確認這次改動沒有破壞之前
   4個任務的行為。
3. **人工抽查**（不是重新證明有效性，那個已經做過）：用現有watchlist
   跑一次`multi_timeframe_check.full_check()`，看有沒有股票現在就觸發
   這個新條件，人工看一下觸發的理由是否合理（不要求驗證「準不準」，
   只要求「有沒有邏輯上說得通」）。
