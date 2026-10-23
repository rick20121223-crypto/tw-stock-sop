"""
迴歸測試：combine_timeframes() 在只有60分/5分（沒有日/週）時的行為——
這是 pages/1_短線進場.py 實際呼叫combine_timeframes()的方式（見
analyze_short()，verdicts只會有"60分"/"5分"兩個key，不會有"日"/"週"）。

背景：2026-10-23切換combine_timeframes_mode正式預設為day_60m_primary
當天發現，_combine_timeframes_day_60m_primary()原本看到verdicts裡沒有
"日"就直接回傳「觀望（日資料不足，無法判讀）」，完全不管實際算出來的
60分/5分結果——等於把「短線進場」頁全部股票都打成觀望，是真的production
regression（不是防呆分支誤觸發），不是這個案例本身在考慮範圍內就先天
沒想到。修法：日線不存在時退回「短天期fallback」，60分當主要依據、
5分保留否決力。

用法（在專案資料夾內執行，不需要API）：
    python3 test_combine_short_only_fallback.py
"""
from sop_decision import RULES, Verdict, classify_final, combine_timeframes


def _set_mode(mode: str):
    RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = mode


def test_short_only_both_bullish_is_add():
    _set_mode("day_60m_primary")
    v60 = Verdict(conclusion="加碼", confidence="高", score=5.0)
    v5 = Verdict(conclusion="加碼", confidence="高", score=5.0)
    result = combine_timeframes({"60分": v60, "5分": v5})
    bucket = classify_final(result["最終建議"])
    assert bucket == "加碼", f"短線進場頁情境：60分/5分都加碼時應該判加碼，實際={bucket}（{result['最終建議']}）"
    print(f"✅ 只有60分/5分、都加碼 -> {result['最終建議']}（{bucket}），沒有被日線缺資料蓋成觀望")


def test_short_only_p_short_sell_is_sell():
    _set_mode("day_60m_primary")
    v60_sell = Verdict(conclusion="賣出減碼", confidence="高", score=-5.0)
    v5_buy = Verdict(conclusion="加碼", confidence="高", score=5.0)
    result = combine_timeframes({"60分": v60_sell, "5分": v5_buy})
    bucket = classify_final(result["最終建議"])
    assert bucket == "賣出減碼", f"60分(主要)賣出減碼時應該判賣出減碼，實際={bucket}"
    print(f"✅ 只有60分/5分、60分賣出減碼 -> {result['最終建議']}（{bucket}）")


def test_short_only_s_short_sell_still_vetoes():
    """5分保留否決力：就算60分是加碼，5分賣出減碼一樣要能擋下來。"""
    _set_mode("day_60m_primary")
    v60_buy = Verdict(conclusion="加碼", confidence="高", score=5.0)
    v5_sell = Verdict(conclusion="賣出減碼", confidence="高", score=-5.0)
    result = combine_timeframes({"60分": v60_buy, "5分": v5_sell})
    bucket = classify_final(result["最終建議"])
    assert bucket == "賣出減碼", f"5分(輔助)保留否決力，應該判賣出減碼，實際={bucket}"
    print(f"✅ 60分加碼但5分賣出減碼 -> {result['最終建議']}（{bucket}），5分否決力仍在")


def test_legacy_mode_unaffected():
    """確認legacy模式下的短天期fallback本來就沒問題，這次只是修
    day_60m_primary的回歸，不影響legacy。"""
    _set_mode("legacy")
    v60 = Verdict(conclusion="加碼", confidence="高", score=5.0)
    v5 = Verdict(conclusion="加碼", confidence="高", score=5.0)
    result = combine_timeframes({"60分": v60, "5分": v5})
    bucket = classify_final(result["最終建議"])
    assert bucket == "買進", f"legacy短天期fallback預期是買進桶，實際={bucket}（{result['最終建議']}）"
    print(f"✅ legacy模式：只有60分/5分、都加碼 -> {result['最終建議']}（{bucket}）")
    _set_mode("day_60m_primary")  # 還原，避免影響其他測試


if __name__ == "__main__":
    test_short_only_both_bullish_is_add()
    test_short_only_p_short_sell_is_sell()
    test_short_only_s_short_sell_still_vetoes()
    test_legacy_mode_unaffected()
    print()
    print("全部通過。")
