"""
比較「加緩衝帶前 vs 後」對短期(60分線)訊號雜訊反轉的實際效果。

根源診斷（diagnose_short_term_whipsaw.py）發現：
    Step6左側平台支撐：49.7%的雜訊反轉事件都有它（最大宗）
    CCI示警：          30.9%（第二大宗，模式跟MTM舊版同一種「單根觸發」）

sop_decision.py 已經幫兩者都加上 confirm_bars 緩衝帶選項（見
_step5b_cci / _step6_left_side_platform，confirm_bars<=1時完全不影響
舊行為）。這支backtest把同一份60分K歷史資料，分別套用：
    baseline：cci.confirm_bars=1, left_side_platform.confirm_bars=1（現行）
    candidate2：兩者都改成2
    candidate3：兩者都改成3
逐根K棒重建evaluate_timeframe()的最終結論桶，比較三種設定下：
    1) 雜訊式反轉事件總數（訊號桶剛變化，WHIPSAW_WINDOW根內又變回去）
    2) 「賣出減碼/觀望」這類降級訊號觸發後，接下來價格有沒有繼續走弱
       （確認緩衝帶沒有把真正有用的示警也一起壓掉、降低保護力）

資料只抓一次（省Fugle額度），三種設定共用同一份已抓好的df，只是重跑
判讀邏輯，不重新打API。

用法（在專案資料夾內執行）：
    python3 test_short_term_whipsaw_fix_compare.py [Fugle_API_Key]
"""
import concurrent.futures
import copy
import os
import sys

import pandas as pd

from sop_decision import RULES, classify_final, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_intraday_data, run_all_indicators, unique_watchlist

WHIPSAW_WINDOW = 4
WARMUP_BARS = 245
FORWARD_BARS = (4, 8, 16)

CONFIGS = {
    "baseline(cb=1/1，現行)": {"cci": 1, "left_side_platform": 1},
    "candidate(cb=2/2)": {"cci": 2, "left_side_platform": 2},
    "candidate(cb=3/3)": {"cci": 3, "left_side_platform": 3},
}


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
    for i in range(WARMUP_BARS, len(df)):
        sub = df.iloc[: i + 1]
        verdict = evaluate_timeframe(sub, "60分")
        buckets.append(classify_final(verdict.conclusion))

    whipsaw_events = 0
    downgrade_fwd_rets = {n: [] for n in FORWARD_BARS}
    for j in range(1, len(buckets)):
        if buckets[j] == buckets[j - 1]:
            continue
        prev_bucket, new_bucket = buckets[j - 1], buckets[j]
        window = buckets[j + 1 : j + 1 + WHIPSAW_WINDOW]
        if prev_bucket in window:
            whipsaw_events += 1

        # 降級事件（變成觀望或賣出減碼）之後的價格變化，用來檢查保護力
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
    }


def run(fugle_api_key: str) -> list:
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
                        "bucket_counts": {}} for label in CONFIGS}

    for label, cb in CONFIGS.items():
        RULES["indicators"]["cci"]["confirm_bars"] = cb["cci"]
        RULES["indicators"]["left_side_platform"]["confirm_bars"] = cb["left_side_platform"]
        for r in fetched:
            if r.get("error") or r["df"] is None:
                continue
            out = walk_forward(r["df"])
            results[label]["whipsaw_events"] += out["whipsaw_events"]
            for n in FORWARD_BARS:
                results[label]["downgrade_fwd_rets"][n].extend(out["downgrade_fwd_rets"][n])
            for bucket, cnt in out["bucket_counts"].items():
                results[label]["bucket_counts"][bucket] = results[label]["bucket_counts"].get(bucket, 0) + cnt

    # 用完恢復成預設值，避免這支腳本跑完後還留著改過的全域RULES
    RULES["indicators"]["cci"]["confirm_bars"] = 1
    RULES["indicators"]["left_side_platform"]["confirm_bars"] = 1

    return results


def _print_stats(label: str, stats: dict) -> None:
    print(f"--- {label} ---")
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

    print(f"\n=== Step6+CCI 緩衝帶 confirm_bars 比較（13檔TW個股加總60分K）===\n")
    for label in CONFIGS:
        _print_stats(label, results[label])
        print()
