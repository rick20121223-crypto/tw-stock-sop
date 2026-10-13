# KD(36,5,5) Double Check 降級 conclusion（買進側 Phase 3）— 設計文件

日期：2026-10-13
來源：使用者明確決定（「繼續」），延續2026-10-12 W底KD365偵測機制
（Phase 1：偵測/標記、Phase 2：回測驗證），本輪是3階段計畫的最後一輪。

## 背景

- **Phase 1**（2026-10-12）：`_detect_w_bottom()`新增`kd365_bullish`參數，
  內部算出`synced = fast_slope_up and macd_bias > 0 and kd365_bullish`，
  只影響`reasons`文字（「轉折訊號確認」vs「型態轉折待確認」），不影響
  `conclusion`。
- 2026-10-13上午先修正了`kd365_bullish`的定義：原本是「K365目前狀態是否
  >D365」，改成`_kd365_crossed_up_recently(df, window=3)`——「最近3個
  交易日內是否剛黃金交叉」。
- **Phase 2**（回測，21檔/24個月）驗證了這個修正後的定義確實有效，兩種
  操作化方式結果差異明顯：

  | 組別 | 事件數 | 5日後 | 10日後 | 20日後 |
  |---|---|---|---|---|
  | A) W底+均線+MACD確認，KD365非剛交叉 | 1110筆 | +4.70%／漲70.1%（668筆） | +10.51%／漲81.7%（327筆） | +19.93%／漲94.1%（68筆） |
  | B) W底+均線+MACD確認，KD365剛黃金交叉(3日內) | 240筆 | +6.96%／漲81.4%（129筆） | +11.78%／漲85.2%（54筆） | +38.02%／漲100.0%（9筆） |

  B組三個天期都明顯勝出A組（20日樣本數9筆偏小，但方向一致）。

- **Phase 3**（本次）：讓這個已驗證的訊號真正影響`conclusion`——W底型態
  偵測到、但均線/MACD/KD365尚未同步確認時，買進/加碼結論強制降級為觀望。

## 範圍

程式碼裡已經有一個形狀完全一樣的既有規則可以直接參考——
`triangle_force_watch`（三角收斂尚未帶量突破，`sop_decision.py`
約1248-1252行）：

```python
if triangle_force_watch and watch_cfg.get("triangle_consolidation_downgrade", True) \
        and conclusion in ("買進", "加碼"):
    conclusion = "觀望"
```

本次新增的規則跟它同一層級、同一種寫法：

- 只有「這次確實偵測到W底型態」（`w_bottom_neckline is not None`）時才會
  啟動，跟W底無關的日常買進/加碼訊號完全不受影響。
- 只在當下結論「是」買進/加碼時才降級，不凌駕已經判定賣出減碼的硬否決
  （跟三角收斂規則、新規則①②都遵循同一個約定：否決規則只降級買進/加碼，
  不會把賣出減碼升級，也不會把觀望升級成買進）。
- 降級目標是「觀望」（不是軟性的只加caveats），跟賣出側、跟三角收斂規則
  一致。
- 加一個`sop_rules.yaml`開關，比照賣出側`kd365_veto_enabled`的做法，之後
  證據不支持可以直接關掉，不用改程式碼。

明確不在這次範圍內：
- 不改動`_kd365_crossed_up_recently()`、`_detect_w_bottom()`既有的偵測
  邏輯本身（那是今天上午已經做完、backtest驗證過的部分）。
- 不處理課程提到的「順勢加碼」確認（W底之外的加碼情境）——那是另一個
  獨立的題目，之前已經和使用者討論過要分開處理。
- 不影響M頭（`_detect_m_top`）相關邏輯——M頭目前就只是賣出側的輔助提醒，
  本次只處理W底。

## 設計

### `_detect_w_bottom()` 回傳值改成 `(neckline, synced)` tuple

目前`_detect_w_bottom()`回傳型態是`Optional[float]`（只有neckline），
`synced`是函式內部變數，只用來決定`reasons`要寫哪一段文字，呼叫端完全
看不到。但Phase 3需要呼叫端知道「有沒有偵測到型態」以及「這個型態有沒有
同步確認」這兩件事，所以函式簽章要改成回傳
`Optional[Tuple[float, bool]]`：沒偵測到型態時仍然回傳`None`；偵測到時
回傳`(neckline, synced)`。

這樣做而不是在`evaluate_timeframe()`外部重新計算一次`synced`，是因為
`synced`的判斷式已經寫在函式內部（第629行`synced = fast_slope_up and
macd_bias > 0 and kd365_bullish`），外部重算等於維護兩份邏輯，之後改
條件容易漏改一邊。

注意：`sop_decision.py`目前`from typing import Dict, List, Optional`
（第23行）沒有`Tuple`，這次要補上`Tuple`。

```python
def _detect_w_bottom(df: pd.DataFrame, fast_slope_up: bool, macd_bias: float,
                      kd365_bullish: bool, reasons: List[str]) -> Optional[Tuple[float, bool]]:
    ...（中間判斷邏輯不變，所有原本return None的地方不變）...
    synced = fast_slope_up and macd_bias > 0 and kd365_bullish
    if synced:
        reasons.append(...)  # 文字不變
    else:
        reasons.append(...)  # 文字不變
    return neckline, synced
```

### `evaluate_timeframe()` 呼叫端解包，新增降級規則

呼叫端（約1112行）：

```python
w_bottom_result = _detect_w_bottom(df, fast_slope_up, macd_bias, kd365_bullish, reasons)
w_bottom_neckline = w_bottom_result[0] if w_bottom_result else None
w_bottom_pending = bool(w_bottom_result) and not w_bottom_result[1]
```

（`w_bottom_neckline`沿用原名、繼續傳給`_step6_left_side_platform()`當
支撐候選，不受這次改動影響；`w_bottom_pending`是新變數，True代表「有
W底型態、但均線/MACD/KD365還沒同步確認」。）

降級規則放在既有「型態學輔助觸發層」區塊，緊接在`triangle_force_watch`
那條規則之後（約1252行之後）：

```python
# ---- 型態學輔助觸發層：W底尚未經均線/MACD/KD365同步確認 → 買進/加碼
# 強制降級觀望，不凌駕生死線下彎/四關價破底等硬否決產生的「賣出減碼」----
if w_bottom_pending and watch_cfg.get("w_bottom_kd365_downgrade", True) \
        and conclusion in ("買進", "加碼"):
    conclusion = "觀望"
```

### `sop_rules.yaml` 新增開關

在`watch_rules:`區塊，緊接在`triangle_consolidation_downgrade`之後新增：

```yaml
  w_bottom_kd365_downgrade: true  # W底型態偵測到、但均線/MACD/KD(36,5,5)尚未同步確認
                                   # （reasons顯示「型態轉折待確認」）→ 買進/加碼一律降級觀望，
                                   # 不凌駕生死線下彎/四關價破底等硬否決產生的「賣出減碼」。
                                   # 2026-10-13新增，依據test_kd365_w_bottom_buy.py回測（21檔/
                                   # 24個月）：KD365剛黃金交叉(3日內)比非剛交叉在5/10/20日
                                   # 三個天期報酬與上漲比例都明顯更好（詳見kd365.
                                   # w_bottom_fresh_cross_window區塊註解），設為false可隨時
                                   # 關掉、不用改程式碼。
```

## 測試／驗證計畫

1. **合成資料正確性**（新增`verify_w_bottom_kd365_gate.py`）：
   - 構造W底型態已偵測到、但KD365未剛交叉（`synced=False`）的合成資料，
     確認原本的買進/加碼結論被強制降成觀望。
   - 構造W底型態已偵測到、且三者同步確認（`synced=True`）的合成資料，
     確認結論不受影響（維持買進/加碼）。
   - 構造完全沒有W底型態的合成資料（`w_bottom_neckline is None`），確認
     這條新規則不會誤觸發。
   - 確認`w_bottom_kd365_downgrade: false`時這條規則完全不影響
     `conclusion`（開關生效）。
   - 確認已經判定「賣出減碼」的情況不會被這條規則動到（只降級買進/加碼，
     不升級、不覆蓋其他結論）。
2. **既有regression**：重跑現有全部`verify_*.py`腳本（`verify_calc_kd_
   generalization.py`、`verify_detect_bearish_divergence_column.py`、
   `verify_step_kd365.py`、`verify_suggested_exposure_pipeline.py`、
   `verify_kd365_hard_sell.py`、`verify_w_bottom_kd365.py`、`verify_
   kd365_crossed_up_recently.py`），確認這次改動（尤其是`_detect_w_
   bottom()`回傳型態改變）沒有破壞之前任務的行為。
3. **端對端smoke test**：呼叫一次`evaluate_timeframe()`（沿用之前幾輪
   用過的方式），確認`w_bottom_neckline`解包後仍能正常傳給
   `_step6_left_side_platform()`，沒有因為回傳型態改變而出錯。
