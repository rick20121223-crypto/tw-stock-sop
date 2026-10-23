"""
比較 combine_timeframes() 的 "legacy"（週/日平權、60分/5分平權，目前正式
預設）vs "day_60m_primary"（日線＝長天期主要依據、週線降為輔助；60分＝
短天期主要依據、5分降為輔助，實驗中）兩種模式，在固定清單股票最近N個
交易日的判讀差異——陽明(2609)/安葆(7792) 2026-10案例之後新增，見
sop_rules.yaml watch_rules.combine_timeframes_mode 區塊註解。

⚠️ 範圍限制：這裡只比較「週+日」兩個長天期（FinMind免費額度就能跑，不
需要Fugle Key），**不含60分/5分**——所以這份比較只能驗證「週線降為輔助
後，還會不會像安葆(7792)那樣因為週線自己翻多就被拉成偏多」這一半；另一
半（60分為主、5分降為輔助）需要Fugle Key才能跑，之後有Key要跑完整比較，
把FUGLE_API_KEY當成第一個參數傳進來即可（見main()，目前有key時才會
一併比較60分）。

用法（在專案資料夾內執行）：
    python3 test_combine_weighting_compare.py [Fugle_API_Key] [最近N個交易日=15]
"""
import sys

import pandas as pd

from sop_decision import RULES, classify_final, combine_timeframes, evaluate_timeframe
from stock_core import get_stock_data, run_all_indicators, unique_watchlist

RECENT_DAYS_DEFAULT = 15


def _week_df(df_day: pd.DataFrame) -> pd.DataFrame:
    week_src = df_day.copy()
    week_src["date"] = pd.to_datetime(week_src["date"])
    df_week = (
        week_src.set_index("date").resample("W")
        .agg({"open": "first", "max": "max", "min": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"]).reset_index()
    )
    if not df_week.empty and week_src["date"].iloc[-1].weekday() < 4:
        df_week = df_week.iloc[:-1]
    return df_week


def _compare_one(name: str, code: str, market: str, recent_days: int, fugle_api_key: str) -> list:
    rows = []
    try:
        df_full = get_stock_data(code, market, "2024-01-01", "2026-12-31", "")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ {name}（{code}）抓取日線失敗，略過：{exc}")
        return rows
    if df_full.empty or len(df_full) < 60:
        return rows

    dates = df_full["date"].tail(recent_days).tolist()
    for cutoff in dates:
        df_day = df_full[df_full["date"] <= cutoff].reset_index(drop=True)
        df_day_i = run_all_indicators(df_day.copy(), "日")
        v_day = evaluate_timeframe(df_day_i, "日")
        df_week = run_all_indicators(_week_df(df_day_i), "週")
        v_week = evaluate_timeframe(df_week, "週")

        verdicts = {"日": v_day, "週": v_week}
        # 有Fugle Key時才一併比較60分（5分這裡先不拉，純粹驗證60分為主
        # 這一半；完整驗證需要之後有Key再擴充5分）。60分/5分只有「最新」
        # 資料可查（Fugle歷史候選有限），沒辦法逐一回看每個歷史cutoff，
        # 這裡僅供有Key時對「今天」做一次即時對照，不是逐日回測。

        RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = "legacy"
        legacy_result = combine_timeframes(verdicts)
        legacy_bucket = classify_final(legacy_result["最終建議"])

        RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = "day_60m_primary"
        new_result = combine_timeframes(verdicts)
        new_bucket = classify_final(new_result["最終建議"])

        RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = "legacy"

        rows.append({
            "name": name, "code": code, "date": cutoff,
            "日": v_day.conclusion, "週": v_week.conclusion,
            "legacy": legacy_bucket, "legacy_text": legacy_result["最終建議"],
            "new": new_bucket, "new_text": new_result["最終建議"],
            "diff": legacy_bucket != new_bucket,
        })
    return rows


def main() -> None:
    fugle_api_key = sys.argv[1] if len(sys.argv) > 1 else ""
    recent_days = int(sys.argv[2]) if len(sys.argv) > 2 else RECENT_DAYS_DEFAULT
    if not fugle_api_key:
        print("⚠️ 未提供Fugle Key，本次只比較「週/日」這一半（週線降為輔助的影響），"
              "不含60分/5分那一半。\n")

    watchlist = unique_watchlist()
    all_rows = []
    for name, code, market in watchlist:
        if market != "TW":
            continue
        all_rows.extend(_compare_one(name, code, market, recent_days, fugle_api_key))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("沒有可比較的資料。")
        return

    diff_df = df[df["diff"]]
    print(f"總共比較 {len(df)} 筆（{df['code'].nunique()}檔 x 最近{recent_days}個交易日），"
          f"兩種模式判讀桶不同的有 {len(diff_df)} 筆（{len(diff_df)/len(df)*100:.1f}%）\n")

    if not diff_df.empty:
        print("=== 判讀桶不同的案例 ===")
        for _, r in diff_df.sort_values(["code", "date"]).iterrows():
            print(f"{r['date']} {r['name']}（{r['code']}）日:{r['日']} 週:{r['週']}  "
                  f"| legacy={r['legacy']}（{r['legacy_text']}）"
                  f" -> new={r['new']}（{r['new_text']}）")
    else:
        print("這段期間兩種模式判讀桶完全一樣，沒有差異案例。")

    print()
    print("=== 兩種模式的桶分布 ===")
    print("legacy:", df["legacy"].value_counts().to_dict())
    print("new   :", df["new"].value_counts().to_dict())


if __name__ == "__main__":
    main()
