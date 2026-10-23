"""
比較 combine_timeframes() 的 "legacy"（週/日平權、60分/5分平權，目前正式
預設）vs "day_60m_primary"（日線＝長天期主要依據、週線降為輔助；60分＝
短天期主要依據、5分降為輔助，實驗中）兩種模式，在固定清單股票最近N個
交易日的判讀差異——陽明(2609)/安葆(7792) 2026-10案例之後新增，見
sop_rules.yaml watch_rules.combine_timeframes_mode 區塊註解。

分兩部分：
1. 週+日最近N個交易日逐日比較（不含60分/5分，FinMind免費額度就能跑，
   不需要Fugle Key）——驗證「週線降為輔助後，還會不會像安葆(7792)那樣
   因為週線自己翻多就被拉成偏多」這一半。
2. 含60分/5分的「目前」即時快照比較（需要Fugle Key，當成第一個參數
   傳進來）——Fugle的60分/5分歷史candle只給得到「最近一段」，沒辦法
   像週/日那樣逐日回測，只能比較「現在」這一刻。沒給Key時自動略過。

用法（在專案資料夾內執行）：
    python3 test_combine_weighting_compare.py [Fugle_API_Key] [最近N個交易日=15]
"""
import concurrent.futures
import sys

import pandas as pd

from sop_decision import RULES, classify_final, combine_timeframes, evaluate_timeframe
from stock_core import get_intraday_data, get_stock_data, run_all_indicators, unique_watchlist

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


def _compare_one(name: str, code: str, market: str, recent_days: int) -> list:
    """週+日逐日回測比較（不含60分/5分，FinMind免費額度就能跑）。"""
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


def _compare_one_live_with_intraday(name: str, code: str, market: str, fugle_api_key: str):
    """目前即時快照：週+日+60分+5分一起丟進combine_timeframes()比較兩種
    模式——Fugle的60分/5分歷史candle只給得到「最近一段」，沒辦法像週/日
    那樣逐日回測，這裡只能做「現在」這一刻的單筆比較。"""
    try:
        df_full = get_stock_data(code, market, "2024-01-01", "2026-12-31", "")
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "code": code, "error": f"日線抓取失敗：{exc}"}
    if df_full.empty or len(df_full) < 60:
        return {"name": name, "code": code, "error": "日線資料不足"}

    df_day = run_all_indicators(df_full.copy(), "日")
    v_day = evaluate_timeframe(df_day, "日")
    df_week = run_all_indicators(_week_df(df_day), "週")
    v_week = evaluate_timeframe(df_week, "週")

    verdicts = {"日": v_day, "週": v_week}
    try:
        df_60 = get_intraday_data(code, "60", fugle_api_key)
        if not df_60.empty:
            df_60 = run_all_indicators(df_60, "60分")
            verdicts["60分"] = evaluate_timeframe(df_60, "60分")
        df_5 = get_intraday_data(code, "5", fugle_api_key)
        if not df_5.empty:
            df_5 = run_all_indicators(df_5, "5分")
            verdicts["5分"] = evaluate_timeframe(df_5, "5分")
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "code": code, "error": f"60分/5分抓取失敗：{exc}"}

    RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = "legacy"
    legacy_result = combine_timeframes(verdicts)
    legacy_bucket = classify_final(legacy_result["最終建議"])

    RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = "day_60m_primary"
    new_result = combine_timeframes(verdicts)
    new_bucket = classify_final(new_result["最終建議"])

    RULES["watch_rules"]["combine_timeframes_mode"]["mode"] = "legacy"

    return {
        "name": name, "code": code,
        "日": v_day.conclusion, "週": v_week.conclusion,
        "60分": verdicts.get("60分").conclusion if "60分" in verdicts else "—",
        "5分": verdicts.get("5分").conclusion if "5分" in verdicts else "—",
        "legacy": legacy_bucket, "legacy_text": legacy_result["最終建議"],
        "new": new_bucket, "new_text": new_result["最終建議"],
        "diff": legacy_bucket != new_bucket,
    }


def main() -> None:
    fugle_api_key = sys.argv[1] if len(sys.argv) > 1 else ""
    recent_days = int(sys.argv[2]) if len(sys.argv) > 2 else RECENT_DAYS_DEFAULT

    watchlist = [(n, c, m) for n, c, m in unique_watchlist() if m == "TW"]

    print(f"=== 第一部分：週+日 最近{recent_days}個交易日逐日比較（不含60分/5分）===\n")
    all_rows = []
    for name, code, market in watchlist:
        all_rows.extend(_compare_one(name, code, market, recent_days))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("沒有可比較的資料。")
    else:
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

    if not fugle_api_key:
        print("\n⚠️ 未提供Fugle Key，略過第二部分（含60分/5分的即時快照比較）。")
        return

    print(f"\n\n=== 第二部分：目前即時快照，含60分/5分（{len(watchlist)}檔）===\n")
    live_rows = []
    # Fugle的頻率限制比FinMind嚴格，跟其他地方（例如pages/1_短線進場.py）
    # 一樣刻意調低平行度，避免一次撞到429。
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_compare_one_live_with_intraday, name, code, market, fugle_api_key)
            for name, code, market in watchlist
        ]
        for future in concurrent.futures.as_completed(futures):
            live_rows.append(future.result())

    errors = [r for r in live_rows if "error" in r]
    ok_rows = [r for r in live_rows if "error" not in r]
    for r in errors:
        print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    live_df = pd.DataFrame(ok_rows)
    if live_df.empty:
        print("沒有可比較的即時資料。")
        return

    live_diff = live_df[live_df["diff"]]
    print(f"\n總共比較 {len(live_df)} 檔，兩種模式判讀桶不同的有 {len(live_diff)} 檔"
          f"（{len(live_diff)/len(live_df)*100:.1f}%）\n")
    if not live_diff.empty:
        print("=== 判讀桶不同的案例（含60分/5分）===")
        for _, r in live_diff.sort_values("code").iterrows():
            print(f"{r['name']}（{r['code']}）日:{r['日']} 週:{r['週']} 60分:{r['60分']} 5分:{r['5分']}  "
                  f"| legacy={r['legacy']}（{r['legacy_text']}）"
                  f" -> new={r['new']}（{r['new_text']}）")
    else:
        print("目前即時快照兩種模式判讀桶完全一樣，沒有差異案例。")
    print()
    print("=== 兩種模式的桶分布（即時快照）===")
    print("legacy:", live_df["legacy"].value_counts().to_dict())
    print("new   :", live_df["new"].value_counts().to_dict())


if __name__ == "__main__":
    main()
