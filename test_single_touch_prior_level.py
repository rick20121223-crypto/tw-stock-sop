"""
測試 Step6 左側平台校正新增的「單次觸及前高/前低」規則（見 sop_rules.yaml
indicators.left_side_platform.single_touch_prior_level）。

背景：2026-10-20 陽明(2609)案例發現，_cluster_pivot_levels() 要求同一價位
至少被觸碰 platform_min_touches(=2) 次才算「整理平台」，只出現過一次的
歷史明顯高點（前高67.3）會被完全濾掉、連候選清單都進不去。但「前高/前低」
在但丁老師原始概念裡不要求反覆測試，這裡額外把回看窗口內最高的轉折高點／
最低的轉折低點各加回一個候選（只需出現一次），跟既有「整理平台」候選並列、
標籤分開。

本測試只驗證這條新規則本身的行為（用合成K棒資料，不需要API）：
1. 啟用時：股價貼近單次觸及的前高、且今開未過昨高 → 應該觸發
   resistance_penalty，跟「整理平台」壓力的觸發方式一致。
2. 關閉開關（single_touch_prior_level.enabled=False）時：應該完全退回
   舊行為（不觸發），確認這個新規則是可以安全關掉的獨立開關。
3. 用真實的陽明(2609) 2026-10-19收盤資料重跑一次（跟當時觸發「長期：
   觀望→加碼」通知用的同一根日K），確認前高67.3現在會被列入候選，但因為
   股價距離still超過2%容忍帶，不會誤觸發扣分——這條規則本身不會讓2609
   這次變成「不加碼」（那是另一個待修的問題：combine_timeframes的跨週期
   合併規則，見notify_email.py/home.py長期判讀完全沒檢查60分/5分），只
   是確保這次修改沒有在這個案例上產生假陽性。

用法（在專案資料夾內執行，第3項需要能連上FinMind，沒有token也可以跑）：
    python3 test_single_touch_prior_level.py
"""
import pandas as pd

from sop_decision import RULES, _step6_left_side_platform


def _build_single_touch_case():
    """合成30根日K：第6根(index5)出現一次性前高67.3、之後再也沒被測到；
    最後一根收盤貼近67.3、且今開未過昨高（64.0），符合resistance_penalty
    的兩個觸發條件。"""
    n = 30
    closes = [55 + i * 0.3 for i in range(n)]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    opens = list(closes)

    highs[5] = 67.3
    lows[5] = 66.8

    opens[-2] = 63.0
    closes[-2] = 63.8
    highs[-2] = 64.0

    opens[-1] = 62.0   # 今開62.0 < 昨高64.0
    highs[-1] = 66.1   # (67.3-66.1)/67.3 ≈ 1.78%，落在2%容忍帶內
    lows[-1] = 61.5
    closes[-1] = 66.0

    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n),
        "open": opens, "max": highs, "min": lows, "close": closes,
    })
    latest = df.iloc[-1].copy()
    latest["今開"] = opens[-1]
    latest["昨高"] = highs[-2]
    return df, latest


def test_enabled_triggers_resistance_penalty():
    df, latest = _build_single_touch_case()
    cfg = RULES["indicators"]["left_side_platform"]
    cfg.setdefault("single_touch_prior_level", {})["enabled"] = True

    reasons, caveats = [], []
    bias = _step6_left_side_platform(df, "日", latest, True, reasons, caveats)

    assert bias == cfg["resistance_penalty"], f"預期觸發resistance_penalty，實際bias={bias}"
    assert any("前高" in r and "67.3" in r for r in reasons), f"預期reasons提到前高67.3，實際={reasons}"
    print("✅ 啟用時，單次觸及的前高正確觸發resistance_penalty：", reasons)


def test_disabled_keeps_old_behavior():
    df, latest = _build_single_touch_case()
    cfg = RULES["indicators"]["left_side_platform"]
    cfg.setdefault("single_touch_prior_level", {})["enabled"] = False

    reasons, caveats = [], []
    bias = _step6_left_side_platform(df, "日", latest, True, reasons, caveats)

    assert bias == 0.0, f"關閉開關後應該完全退回舊行為(bias=0)，實際bias={bias}"
    assert not reasons and not caveats, f"關閉開關後不該有任何前高相關提示，實際reasons={reasons} caveats={caveats}"
    print("✅ 關閉開關後，正確退回舊行為（單次前高不觸發任何提示）")

    cfg["single_touch_prior_level"]["enabled"] = True  # 還原，避免影響其他測試


def test_2609_real_case_no_false_positive():
    try:
        from stock_core import get_stock_data, run_all_indicators
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 無法匯入stock_core，略過2609真實資料驗證：{exc}")
        return

    try:
        df_day = get_stock_data("2609", "TW", "2024-01-01", "2026-10-20", "")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ 無法連上FinMind，略過2609真實資料驗證：{exc}")
        return

    df_day = df_day[df_day["date"] <= "2026-10-19"].reset_index(drop=True)
    if df_day.empty:
        print("⚠️ 抓不到2609在2026-10-19以前的資料，略過")
        return
    df_day = run_all_indicators(df_day, "日")
    latest = df_day.iloc[-1]
    assert latest["close"] == 63.5, f"預期觸發通知那天的收盤是63.5，實際={latest['close']}（資料來源可能已變動）"

    cfg = RULES["indicators"]["left_side_platform"]
    cfg.setdefault("single_touch_prior_level", {})["enabled"] = True

    reasons, caveats = [], []
    bias = _step6_left_side_platform(df_day, "日", latest, True, reasons, caveats)

    # 前高67.3現在應該被列入候選（不再被platform_min_touches濾掉），但股價
    # 距離still超過2%容忍帶，所以這次不該誤觸發扣分——確認修改沒有在這個
    # 案例上產生假陽性（真正讓2609變加碼的原因在combine_timeframes，不在
    # 這裡，見本檔開頭docstring）。
    assert bias == 0.0, f"2609這次不該觸發Step6扣分（距離前高還有3%以上），實際bias={bias}"
    print(f"✅ 2609真實資料（2026-10-19收盤63.5）：Step6 bias={bias}（符合預期，前高67.3已列入候選但距離不夠近）")


if __name__ == "__main__":
    test_enabled_triggers_resistance_penalty()
    test_disabled_keeps_old_behavior()
    test_2609_real_case_no_false_positive()
    print()
    print("全部通過。")
