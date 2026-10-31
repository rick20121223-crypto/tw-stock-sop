"""
比較 scoring.mode 的 "legacy"（現有公式，目前正式預設）vs "weighted_48_20_20"
（均線48%／四關價20%／指標組(MACD+KD+CCI)20%，對照1030錄音「程式單權重
配比」筆記，見sop_rules.yaml indicators.scoring區塊註解）在watchlist真實
股票最近N個交易日的判讀差異——仿照既有test_combine_weighting_compare.py的
比對方式，用來決定要不要把weighted_48_20_20切成正式預設之前先看實際影響
範圍。

分兩部分：
1. 週+日最近N個交易日逐日比較（不含60分/5分，FinMind免費額度就能跑，不需要
   Fugle Key）——週/日的指標組只用MACD+KD兩項平均（CCI不適用），所以這部分
   主要看均線48%／四關價20%／(MACD+KD)20%這組新比例實際造成多少分類差異。
2. 含60分/5分的「目前」即時快照比較（需要Fugle Key，當成第一個參數傳進
   來）——這部分才會真正用到cci_bias（CCI只在60分/5分適用），是這次新增
   指標組納入CCI的實際驗證點。Fugle的60分/5分歷史candle只給得到「最近一
   段」，沒辦法像週/日那樣逐日回測，只能比較「現在」這一刻。沒給Key時自動
   略過。

用法（在專案資料夾內執行）：
    python3 test_scoring_weight_compare.py [Fugle_API_Key] [最近N個交易日=15]
"""
import concurrent.futures
import sys

import pandas as pd

from sop_decision import RULES, evaluate_timeframe
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


def _evaluate_both_modes(df_i: pd.DataFrame, tf: str) -> tuple:
    RULES["indicators"]["scoring"]["mode"] = "legacy"
    v_legacy = evaluate_timeframe(df_i.copy(), tf)
    RULES["indicators"]["scoring"]["mode"] = "weighted_48_20_20"
    v_weighted = evaluate_timeframe(df_i.copy(), tf)
    RULES["indicators"]["scoring"]["mode"] = "legacy"
    return v_legacy, v_weighted


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
        v_day_legacy, v_day_weighted = _evaluate_both_modes(df_day_i, "日")
        df_week = run_all_indicators(_week_df(df_day_i), "週")
        v_week_legacy, v_week_weighted = _evaluate_both_modes(df_week, "週")

        rows.append({
            "name": name, "code": code, "date": cutoff,
            "日_legacy": v_day_legacy.conclusion, "日_legacy_score": v_day_legacy.score,
            "日_weighted": v_day_weighted.conclusion, "日_weighted_score": v_day_weighted.score,
            "日_diff": v_day_legacy.conclusion != v_day_weighted.conclusion,
            "週_legacy": v_week_legacy.conclusion, "週_weighted": v_week_weighted.conclusion,
            "週_diff": v_week_legacy.conclusion != v_week_weighted.conclusion,
        })
    return rows


def _compare_one_live_with_intraday(name: str, code: str, market: str, fugle_api_key: str):
    """目前即時快照：週+日+60分+5分一起比較兩種mode——60分/5分才真正用到
    這次新增的cci_bias（CCI只在60分/5分適用）。Fugle的60分/5分歷史candle
    只給得到「最近一段」，沒辦法像週/日那樣逐日回測，這裡只能做「現在」
    這一刻的單筆比較。"""
    try:
        df_full = get_stock_data(code, market, "2024-01-01", "2026-12-31", "")
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "code": code, "error": f"日線抓取失敗：{exc}"}
    if df_full.empty or len(df_full) < 60:
        return {"name": name, "code": code, "error": "日線資料不足"}

    df_day = run_all_indicators(df_full.copy(), "日")
    v_day_legacy, v_day_weighted = _evaluate_both_modes(df_day, "日")
    df_week = run_all_indicators(_week_df(df_day), "週")
    v_week_legacy, v_week_weighted = _evaluate_both_modes(df_week, "週")

    result = {
        "name": name, "code": code,
        "日_legacy": v_day_legacy.conclusion, "日_weighted": v_day_weighted.conclusion,
        "日_diff": v_day_legacy.conclusion != v_day_weighted.conclusion,
        "週_legacy": v_week_legacy.conclusion, "週_weighted": v_week_weighted.conclusion,
        "週_diff": v_week_legacy.conclusion != v_week_weighted.conclusion,
        "60分_legacy": "—", "60分_weighted": "—", "60分_diff": False,
        "5分_legacy": "—", "5分_weighted": "—", "5分_diff": False,
    }
    try:
        df_60 = get_intraday_data(code, "60", fugle_api_key)
        if not df_60.empty:
            df_60 = run_all_indicators(df_60, "60分")
            v60_legacy, v60_weighted = _evaluate_both_modes(df_60, "60分")
            result["60分_legacy"] = v60_legacy.conclusion
            result["60分_weighted"] = v60_weighted.conclusion
            result["60分_diff"] = v60_legacy.conclusion != v60_weighted.conclusion
        df_5 = get_intraday_data(code, "5", fugle_api_key)
        if not df_5.empty:
            df_5 = run_all_indicators(df_5, "5分")
            v5_legacy, v5_weighted = _evaluate_both_modes(df_5, "5分")
            result["5分_legacy"] = v5_legacy.conclusion
            result["5分_weighted"] = v5_weighted.conclusion
            result["5分_diff"] = v5_legacy.conclusion != v5_weighted.conclusion
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"60分/5分抓取失敗：{exc}"
        return result

    result["diff"] = result["日_diff"] or result["週_diff"] or result["60分_diff"] or result["5分_diff"]
    return result


def main() -> None:
    fugle_api_key = sys.argv[1] if len(sys.argv) > 1 else ""
    recent_days = int(sys.argv[2]) if len(sys.argv) > 2 else RECENT_DAYS_DEFAULT

    watchlist = [(n, c, m) for n, c, m in unique_watchlist() if m == "TW"]

    print(f"=== 第一部分：週+日 最近{recent_days}個交易日逐日比較（不含60分/5分，CCI不適用）===\n")
    all_rows = []
    for name, code, market in watchlist:
        all_rows.extend(_compare_one(name, code, market, recent_days))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("沒有可比較的資料。")
    else:
        day_diff = df[df["日_diff"]]
        week_diff = df[df["週_diff"]]
        print(f"總共比較 {len(df)} 筆（{df['code'].nunique()}檔 x 最近{recent_days}個交易日）")
        print(f"日線判讀不同：{len(day_diff)} 筆（{len(day_diff)/len(df)*100:.1f}%）")
        print(f"週線判讀不同：{len(week_diff)} 筆（{len(week_diff)/len(df)*100:.1f}%）\n")
        if not day_diff.empty:
            print("=== 日線判讀不同的案例 ===")
            for _, r in day_diff.sort_values(["code", "date"]).iterrows():
                print(f"{r['date']} {r['name']}（{r['code']}）"
                      f"legacy={r['日_legacy']}(score={r['日_legacy_score']:.2f}) "
                      f"-> weighted={r['日_weighted']}(score={r['日_weighted_score']:.2f})")
        else:
            print("這段期間日線兩種模式判讀完全一樣，沒有差異案例。")
        print()
        print("=== 兩種模式的日線結論分布 ===")
        print("legacy  :", df["日_legacy"].value_counts().to_dict())
        print("weighted:", df["日_weighted"].value_counts().to_dict())

    if not fugle_api_key:
        print("\n⚠️ 未提供Fugle Key，略過第二部分（含60分/5分即時快照，CCI才真正生效的部分）。")
        return

    print(f"\n\n=== 第二部分：目前即時快照，含60分/5分（{len(watchlist)}檔，CCI在此才生效）===\n")
    live_rows = []
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
    print(f"\n總共比較 {len(live_df)} 檔，任一週期判讀不同的有 {len(live_diff)} 檔"
          f"（{len(live_diff)/len(live_df)*100:.1f}%）\n")
    if not live_diff.empty:
        print("=== 判讀不同的案例（含60分/5分）===")
        for _, r in live_diff.sort_values("code").iterrows():
            print(f"{r['name']}（{r['code']}）"
                  f"日:{r['日_legacy']}->{r['日_weighted']} "
                  f"週:{r['週_legacy']}->{r['週_weighted']} "
                  f"60分:{r['60分_legacy']}->{r['60分_weighted']} "
                  f"5分:{r['5分_legacy']}->{r['5分_weighted']}")
    else:
        print("目前即時快照兩種模式判讀完全一樣，沒有差異案例。")
    print()
    print("=== 60分結論分布（CCI實際生效的週期）===")
    print("legacy  :", live_df["60分_legacy"].value_counts().to_dict())
    print("weighted:", live_df["60分_weighted"].value_counts().to_dict())


if __name__ == "__main__":
    main()
