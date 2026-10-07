"""
找短期（60分線）訊號「剛變又變回去」的真正根源：之前懷疑是MTM貼零軸雜訊
（test_mtm_confirm_bars_whipsaw.py），但回測顯示MTM本身並不那麼容易雜訊
觸發（confirm_bars=1只有1.1%雜訊觸發），問題應該在別的地方——這支逐根
60分K重建evaluate_timeframe()的完整判讀過程（跟diagnose_sell_lag.py對
長天期做的事一樣，這裡改套在短天期），把「最終結論桶（買進/加碼/觀望/
賣出減碼）剛變化、又在幾根K棒內變回去」的事件抓出來，看當時reasons/
caveats裡出現的是哪個子訊號（均線生死線/MACD死叉/MACD背離/OBV背離/
MTM/CCI/乖離過大/三角收斂...），用關鍵字比對分類、統計各類別在這些
「雜訊式反轉」事件裡出現的次數，找出真正該處理的那一個。

用法（在專案資料夾內執行）：
    python3 diagnose_short_term_whipsaw.py [Fugle_API_Key]
"""
import concurrent.futures
import os
import sys

import pandas as pd

from sop_decision import classify_final, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_intraday_data, run_all_indicators, unique_watchlist

WHIPSAW_WINDOW = 4   # 變化後幾根K棒內又變回原狀，算一次「雜訊式反轉」
WARMUP_BARS = 245    # 60分MA240需要240根暖身，多留5根緩衝

# 跟 diagnose_sell_lag.py 的 MARKERS 同一套邏輯，針對60分線會出現的
# reasons/caveats片段擴充。用「文字是否包含這個片段」比對，不要求精確
# 比對整句（文字裡常帶變數，如均線欄位名稱、價位數字）。
MARKERS = {
    "生死線下彎(均線硬否決)": "且股價已跌破",
    "MACD死亡交叉": "MACD 出現死亡交叉",
    "MACD頂背離": "MACD 出現頂背離",
    "OBV/BBI量價背離": "量價背離",
    "MTM翻空": "MTM 翻空/死叉",
    "MTM連續翻空(新規則)": "MTM 連續",
    "CCI示警": "CCI",
    "乖離過大降級觀望": "乖離過大",
    "三角收斂未突破降級": "三角收斂",
    "左側平台防洗盤": "左側",
    "買進加碼三項同步未確認降級": "未確認",
}


def _matched_markers(text: str) -> set:
    return {label for label, pattern in MARKERS.items() if pattern in text}


def analyze_one(name: str, code: str, market: str, fugle_api_key: str) -> dict:
    if market != "TW":
        return {"name": name, "code": code, "error": "非TW市場，Fugle不提供60分資料"}
    try:
        df = get_intraday_data(code, "60", fugle_api_key)
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "code": code, "error": str(exc)}
    if df.empty:
        return {"name": name, "code": code, "error": "60分資料為空"}

    df = run_all_indicators(df, "60分")
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) <= WARMUP_BARS:
        return {"name": name, "code": code, "error": f"資料只有{len(df)}根，不足暖身{WARMUP_BARS}根"}

    buckets, marker_sets = [], []
    for i in range(WARMUP_BARS, len(df)):
        sub = df.iloc[: i + 1]
        verdict = evaluate_timeframe(sub, "60分")
        buckets.append(classify_final(verdict.conclusion))
        text = " ".join(verdict.reasons + verdict.caveats)
        marker_sets.append(_matched_markers(text))

    # 找「雜訊式反轉」：第j個點的bucket跟前一點不同(剛變化)，
    # 且接下來WHIPSAW_WINDOW個點之內又變回「前一點的bucket」
    whipsaw_marker_counts: dict = {}
    whipsaw_events = 0
    for j in range(1, len(buckets)):
        if buckets[j] == buckets[j - 1]:
            continue
        prev_bucket = buckets[j - 1]
        window = buckets[j + 1 : j + 1 + WHIPSAW_WINDOW]
        if prev_bucket in window:
            whipsaw_events += 1
            for marker in marker_sets[j]:
                whipsaw_marker_counts[marker] = whipsaw_marker_counts.get(marker, 0) + 1

    return {
        "name": name, "code": code, "error": None,
        "total_bars": len(buckets), "whipsaw_events": whipsaw_events,
        "marker_counts": whipsaw_marker_counts,
    }


def run(fugle_api_key: str) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(analyze_one, name, code, market, fugle_api_key)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    fugle_api_key = token_arg or os.environ.get("FUGLE_API_KEY", "")
    if not fugle_api_key:
        print("⚠️ 未提供 Fugle API Key（參數或 FUGLE_API_KEY 環境變數），無法抓60分資料。")
        sys.exit(1)

    results = run(fugle_api_key)

    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    total_whipsaw = sum(r["whipsaw_events"] for r in results if not r.get("error"))
    merged_counts: dict = {}
    for r in results:
        if r.get("error"):
            continue
        for marker, cnt in r["marker_counts"].items():
            merged_counts[marker] = merged_counts.get(marker, 0) + cnt

    print(f"\n=== 短期(60分線)雜訊式反轉事件：共{total_whipsaw}次，13檔加總 ===\n")
    print(f"（反轉定義：訊號桶剛變化，{WHIPSAW_WINDOW}根K棒內又變回前一個桶）\n")
    for marker, cnt in sorted(merged_counts.items(), key=lambda kv: -kv[1]):
        pct = cnt / total_whipsaw * 100 if total_whipsaw else 0
        print(f"{marker}：出現在{cnt}次反轉事件中（{pct:.1f}%）")
