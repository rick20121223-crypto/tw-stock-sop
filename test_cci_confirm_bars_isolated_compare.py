"""
延伸test_short_term_whipsaw_fix_compare.py：那支同時改cci.confirm_bars跟
left_side_platform.confirm_bars兩個參數，雜訊反轉事件減少多少，分不清楚
是哪一個規則的功勞。這支把left_side_platform.confirm_bars固定在正式
config的值(3，已經獨立驗證過有效)，只單獨改cci.confirm_bars(1/2/3)，
乾淨地回答「CCI的confirm_bars=3這個設定，現在（2026-10-31修好視窗重疊
bug之後）到底有沒有比1或2更好」這個問題。

用法（在專案資料夾內執行）：
    python3 test_cci_confirm_bars_isolated_compare.py [Fugle_API_Key]
"""
import concurrent.futures
import os
import sys

import pandas as pd

from sop_decision import RULES, classify_final, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_intraday_data, run_all_indicators, unique_watchlist

WHIPSAW_WINDOW = 4
WARMUP_BARS = 245
FORWARD_BARS = (4, 8, 16)

CCI_CONFIGS = {
    "cci_confirm_bars=1（不等，看到就講）": 1,
    "cci_confirm_bars=2": 2,
    "cci_confirm_bars=3（目前正式config）": 3,
}
FIXED_PLATFORM_CONFIRM_BARS = 3  # 固定住，跟正式config一致，不是這次要測的變數


def fetch_one(name: str, code: str, market: str, fugle_api_key: str) -> dict:
    if market != "TW":
        return {"name": name, "code": code, "error": "非TW市場，Fugle不提供60分資料", "df": None}
    try:
        df = get_intraday_data(code, "60", fugle_api_key)
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "code": code, "error": str(exc), "df": None}
    if df.empty:
        return {"name": name, "code": code, "error": "60分資料為空", "df": None}
    df = run_all_indicators(df, "60分")
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) <= WARMUP_BARS:
        return {"name": name, "code": code, "error": f"資料只有{len(df)}根，不足暖身{WARMUP_BARS}根", "df": None}
    return {"name": name, "code": code, "error": None, "df": df}


def walk_forward(df: pd.DataFrame) -> dict:
    close = df["close"]
    buckets = []
    cci_fold_count = 0
    for i in range(WARMUP_BARS, len(df)):
        sub = df.iloc[: i + 1]
        verdict = evaluate_timeframe(sub, "60分")
        buckets.append(classify_final(verdict.conclusion))
        if any("折返" in r for r in verdict.reasons) or any("折返" in c for c in verdict.caveats):
            cci_fold_count += 1

    whipsaw_events = 0
    downgrade_fwd_rets = {n: [] for n in FORWARD_BARS}
    for j in range(1, len(buckets)):
        if buckets[j] == buckets[j - 1]:
            continue
        prev_bucket, new_bucket = buckets[j - 1], buckets[j]
        window = buckets[j + 1: j + 1 + WHIPSAW_WINDOW]
        if prev_bucket in window:
            whipsaw_events += 1

        if new_bucket in ("觀望", "賣出減碼") and prev_bucket not in ("觀望", "賣出減碼"):
            entry_idx = WARMUP_BARS + j
            entry_close = close.iloc[entry_idx]
            if pd.notna(entry_close) and entry_close:
                for n in FORWARD_BARS:
                    tgt = entry_idx + n
                    if tgt < len(close) and pd.notna(close.iloc[tgt]):
                        downgrade_fwd_rets[n].append((close.iloc[tgt] - entry_close) / entry_close)

    bucket_counts = pd.Series(buckets).value_counts().to_dict()
    return {
        "whipsaw_events": whipsaw_events,
        "downgrade_fwd_rets": downgrade_fwd_rets,
        "bucket_counts": bucket_counts,
        "cci_fold_count": cci_fold_count,
    }


def run(fugle_api_key: str) -> dict:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    fetched = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_one, name, code, market, fugle_api_key)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            fetched.append(future.result())

    for r in fetched:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    results = {label: {"whipsaw_events": 0, "downgrade_fwd_rets": {n: [] for n in FORWARD_BARS},
                        "bucket_counts": {}, "cci_fold_count": 0} for label in CCI_CONFIGS}

    RULES["indicators"]["left_side_platform"]["confirm_bars"] = FIXED_PLATFORM_CONFIRM_BARS
    for label, cci_cb in CCI_CONFIGS.items():
        RULES["indicators"]["cci"]["confirm_bars"] = cci_cb
        for r in fetched:
            if r.get("error") or r["df"] is None:
                continue
            out = walk_forward(r["df"])
            results[label]["whipsaw_events"] += out["whipsaw_events"]
            results[label]["cci_fold_count"] += out["cci_fold_count"]
            for n in FORWARD_BARS:
                results[label]["downgrade_fwd_rets"][n].extend(out["downgrade_fwd_rets"][n])
            for bucket, cnt in out["bucket_counts"].items():
                results[label]["bucket_counts"][bucket] = results[label]["bucket_counts"].get(bucket, 0) + cnt

    # 用完恢復成預設值
    RULES["indicators"]["cci"]["confirm_bars"] = 3
    RULES["indicators"]["left_side_platform"]["confirm_bars"] = 3

    return results


def _print_stats(label: str, stats: dict) -> None:
    print(f"--- {label} ---")
    print(f"  CCI折返真的觸發次數：{stats['cci_fold_count']}")
    print(f"  雜訊式反轉事件總數：{stats['whipsaw_events']}")
    print(f"  訊號桶分布：{stats['bucket_counts']}")
    for n in FORWARD_BARS:
        vals = stats["downgrade_fwd_rets"][n]
        if vals:
            avg = sum(vals) / len(vals) * 100
            neg_pct = sum(1 for v in vals if v < 0) / len(vals) * 100
            print(f"  降級(觀望/賣出減碼)觸發後{n}根K棒：平均報酬 {avg:.2f}%　"
                  f"下跌比例 {neg_pct:.1f}%　(n={len(vals)})")
        else:
            print(f"  降級觸發後{n}根K棒：無資料")


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    fugle_api_key = token_arg or os.environ.get("FUGLE_API_KEY", "")
    if not fugle_api_key:
        print("⚠️ 未提供 Fugle API Key（參數或 FUGLE_API_KEY 環境變數），無法抓60分資料。")
        sys.exit(1)

    results = run(fugle_api_key)

    print(f"\n=== 只單獨改CCI的confirm_bars，left_side_platform固定在{FIXED_PLATFORM_CONFIRM_BARS}"
          f"（13檔TW個股加總60分K）===\n")
    for label in CCI_CONFIGS:
        _print_stats(label, results[label])
        print()
