"""
陽明(2609) 2026-10-20 案例的迴歸測試：驗證「長期」判讀現在會把60分/5分
一起丟進同一次 combine_timeframes()，不再只看週+日（見 notify_email.py
開頭 2026-10-20同日 的說明、_check_long() 的docstring）。

案發經過：2026-10-19收盤後的批次判讀，日線="買進"、週線="加碼"，兩者都在
「買進/加碼」集合裡，照combine_timeframes()既有的all_bullish合併規則，
直接蓋成「加碼（各週期已同步翻多）」寄出通知。但同一時間60分/5分實際上
是偏空的（使用者手動依但丁老師SOP判讀：60分MACD死叉+KD(36,5,5)死叉+
20MA死彎，5分MACD零軸下），系統自己記錄的「短期」桶當時也是「觀望」
——長期判讀完全沒檢查這些，是因為呼叫full_check()時故意把fugle_api_key
傳空字串，等於60分/5分根本沒被納入那次combine_timeframes()。

修法：_check_long()改成把真正的fugle_api_key傳進full_check()，讓
週/日/60分/5分同一次丟進combine_timeframes()，之後只要60分/5分有任一個
不在「買進/加碼」集合裡，既有的combine_timeframes_ambiguous_guard就會
接管，強制改判「觀望」，不准輸出模糊的加碼結論。

這裡只測合併邏輯本身（combine_timeframes/classify_final），不重新驗證
60分自己該不該判「觀望」——那是evaluate_timeframe()既有的SOP判讀，不是
這次要改的範圍，所以60分直接用當時系統實際記錄的結論建構一個Verdict。

用法（在專案資料夾內執行，需要能連上FinMind；沒有token也可以跑）：
    python3 test_2609_long_term_short_sync.py
"""
import pandas as pd

from sop_decision import RULES, Verdict, classify_final, combine_timeframes, evaluate_timeframe


def _load_2609_day_week_verdicts():
    from stock_core import get_stock_data, run_all_indicators

    df_day = get_stock_data("2609", "TW", "2024-01-01", "2026-10-20", "")
    df_day = df_day[df_day["date"] <= "2026-10-19"].reset_index(drop=True)
    if df_day.empty:
        return None
    df_day = run_all_indicators(df_day, "日")
    v_day = evaluate_timeframe(df_day, "日")

    week_src = df_day.copy()
    week_src["date"] = pd.to_datetime(week_src["date"])
    df_week = (
        week_src.set_index("date").resample("W")
        .agg({"open": "first", "max": "max", "min": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"]).reset_index()
    )
    if not df_week.empty and week_src["date"].iloc[-1].weekday() < 4:
        df_week = df_week.iloc[:-1]
    df_week = run_all_indicators(df_week, "週")
    v_week = evaluate_timeframe(df_week, "週")
    return v_day, v_week


def test_old_behavior_without_short_tfs_was_add():
    """先確認問題重現：舊呼叫方式（不含60分/5分）這次確實是「加碼」。
    這個案例本來就是在測legacy模式底下的ambiguous_guard問題（跟後來
    2026-10-23才新增的day_60m_primary權重模式是另一回事），明確鎖定
    mode="legacy"，不要被RULES目前的全域預設（可能已經切成
    day_60m_primary）影響。"""
    loaded = _load_2609_day_week_verdicts()
    if loaded is None:
        print("⚠️ 抓不到2609在2026-10-19以前的資料，略過")
        return
    v_day, v_week = loaded
    assert v_day.conclusion == "買進", f"預期日線=買進，實際={v_day.conclusion}（資料來源可能已變動）"
    assert v_week.conclusion == "加碼", f"預期週線=加碼，實際={v_week.conclusion}（資料來源可能已變動）"

    original_mode = RULES["watch_rules"]["combine_timeframes_mode"]["mode"]
    RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = "legacy"
    try:
        old_result = combine_timeframes({"日": v_day, "週": v_week})
    finally:
        RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = original_mode
    old_bucket = classify_final(old_result["最終建議"])
    assert old_bucket == "加碼", f"預期重現問題：舊行為應該是加碼，實際={old_bucket}"
    print(f"✅ 重現問題：不含60分/5分時，合併結果='{old_result['最終建議']}' -> {old_bucket}")


def test_new_behavior_with_short_tf_watch_becomes_watch():
    """把當時系統實際記錄的60分結論（觀望）一起丟進combine_timeframes()，
    確認ambiguous_guard接管、不再是加碼。同樣明確鎖定mode="legacy"，
    理由同上一個測試。"""
    loaded = _load_2609_day_week_verdicts()
    if loaded is None:
        print("⚠️ 抓不到2609在2026-10-19以前的資料，略過")
        return
    v_day, v_week = loaded

    v_60m_watch = Verdict(conclusion="觀望", confidence="中", score=0.0,
                           reasons=["(測試用) 60分當時實際記錄的結論：觀望"])
    original_mode = RULES["watch_rules"]["combine_timeframes_mode"]["mode"]
    RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = "legacy"
    try:
        new_result = combine_timeframes({"日": v_day, "週": v_week, "60分": v_60m_watch})
    finally:
        RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = original_mode
    new_bucket = classify_final(new_result["最終建議"])

    assert new_bucket != "加碼", f"修法後2609這個案例不該再判加碼，實際={new_bucket}（{new_result['最終建議']}）"
    print(f"✅ 含60分(觀望)後，合併結果='{new_result['最終建議']}' -> {new_bucket}（不再是加碼）")


if __name__ == "__main__":
    test_old_behavior_without_short_tfs_was_add()
    test_new_behavior_with_short_tf_watch_becomes_watch()
    print()
    print("全部通過。")
