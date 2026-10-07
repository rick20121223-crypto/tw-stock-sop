"""
驗證「領先指標示警」(MTM/CCI，Step5/5b，leading_indicator_exit_alert)本身
到底有沒有保護力——不是測訊號穩不穩（那是whipsaw的問題，已經處理），是測
「這個降級訊號出現之後，價格真的有比較容易繼續走弱嗎」。

跟test_short_term_whipsaw_fix_compare.py不同之處：那支把所有「降級成觀望/
賣出減碼」的事件混在一起看保護力，這支把降級事件拆成四類，分開比較：

    leading_mtm_cci：由 elif leading_indicator_exit_alert 這條路徑降級成
                      觀望（純粹MTM/CCI示警，沒有更硬的均線/MACD死叉證據）
    caution_macd：    由 elif caution_only 這條路徑降級（MACD單獨背離，
                      均線/量能都還沒同步走壞）
    hard_sell：       conclusion直接判定「賣出減碼」（key_ma_down_veto／
                      dead_cross_confirmed／volume_bear_confirm，三者皆
                      比MTM/CCI更接近SOP位階裡的高位階訊號）
    other：           其餘降級到觀望的情況（純score門檻，沒有明確單一指標
                      觸發）

用各自的「剛進入該類別」事件（避免同一段持續狀態被重複計入），比較接下來
4/8/16根K棒的平均報酬與下跌比例，看MTM/CCI這一類是不是真的比其他類別
（尤其是hard_sell）弱很多，還是其實跟其他降級類別差不多。

套用目前sop_rules.yaml裡實際生效中的設定（含已經調整過的confirm_bars=3），
反映系統現在真正在用的邏輯，不另外覆寫RULES。

用法（在專案資料夾內執行）：
    python3 test_leading_indicator_protective_value.py [Fugle_API_Key]
"""
import concurrent.futures
import os
import sys

import pandas as pd

from sop_decision import classify_final, evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_intraday_data, run_all_indicators, unique_watchlist

WARMUP_BARS = 245
FORWARD_BARS = (4, 8, 16)

CATEGORIES = ("leading_mtm_cci", "caution_macd", "hard_sell", "other_觀望", "非降級")


def _classify(conclusion: str, caveats_text: str) -> str:
    bucket = classify_final(conclusion)
    if bucket == "賣出減碼":
        return "hard_sell"
    if bucket != "觀望":
        return "非降級"
    if "已示警" in caveats_text and ("MTM" in caveats_text or "CCI" in caveats_text):
        return "leading_mtm_cci"
    if "MACD出現警戒訊號" in caveats_text:
        return "caution_macd"
    return "other_觀望"


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

    close = df["close"]
    categories, bars_idx = [], []
    for i in range(WARMUP_BARS, len(df)):
        sub = df.iloc[: i + 1]
        verdict = evaluate_timeframe(sub, "60分")
        categories.append(_classify(verdict.conclusion, " ".join(verdict.caveats)))
        bars_idx.append(i)

    fwd_rets = {cat: {n: [] for n in FORWARD_BARS} for cat in CATEGORIES}
    for j in range(1, len(categories)):
        if categories[j] == categories[j - 1]:
            continue  # 只算「剛進入這個類別」的事件，避免同一段狀態重複計入
        cat = categories[j]
        entry_idx = bars_idx[j]
        entry_close = close.iloc[entry_idx]
        if pd.isna(entry_close) or not entry_close:
            continue
        for n in FORWARD_BARS:
            tgt = entry_idx + n
            if tgt < len(close) and pd.notna(close.iloc[tgt]):
                fwd_rets[cat][n].append((close.iloc[tgt] - entry_close) / entry_close)

    # 無條件基準：每一根K棒都算一次進場
    baseline_fwd = {n: [] for n in FORWARD_BARS}
    for i in bars_idx:
        entry_close = close.iloc[i]
        if pd.isna(entry_close) or not entry_close:
            continue
        for n in FORWARD_BARS:
            tgt = i + n
            if tgt < len(close) and pd.notna(close.iloc[tgt]):
                baseline_fwd[n].append((close.iloc[tgt] - entry_close) / entry_close)

    return {"name": name, "code": code, "error": None, "fwd_rets": fwd_rets, "baseline_fwd": baseline_fwd}


def run(fugle_api_key: str) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(analyze_one, name, code, market, fugle_api_key)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _print_stats(label: str, fwd_rets_by_n: dict) -> None:
    n0 = FORWARD_BARS[0]
    count = len(fwd_rets_by_n[n0])
    print(f"--- {label}（共{count}次事件）---")
    for n in FORWARD_BARS:
        vals = fwd_rets_by_n[n]
        if vals:
            avg = sum(vals) / len(vals) * 100
            neg_pct = sum(1 for v in vals if v < 0) / len(vals) * 100
            print(f"  {n}根K棒後：平均報酬 {avg:.2f}%　下跌比例 {neg_pct:.1f}%　(n={len(vals)})")
        else:
            print(f"  {n}根K棒後：無資料")


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

    merged = {cat: {n: [] for n in FORWARD_BARS} for cat in CATEGORIES}
    merged_baseline = {n: [] for n in FORWARD_BARS}
    for r in results:
        if r.get("error"):
            continue
        for cat in CATEGORIES:
            for n in FORWARD_BARS:
                merged[cat][n].extend(r["fwd_rets"][cat][n])
        for n in FORWARD_BARS:
            merged_baseline[n].extend(r["baseline_fwd"][n])

    print(f"\n=== 降級訊號分類保護力驗證（13檔TW個股加總60分K）===\n")
    _print_stats("leading_mtm_cci（純MTM/CCI示警降級觀望）", merged["leading_mtm_cci"])
    print()
    _print_stats("caution_macd（純MACD背離降級觀望）", merged["caution_macd"])
    print()
    _print_stats("hard_sell（生死線下彎/死叉確認/量價背離→賣出減碼）", merged["hard_sell"])
    print()
    _print_stats("other_觀望（純score門檻，無明確單一觸發）", merged["other_觀望"])
    print()
    _print_stats("無條件基準（每根K棒都進場）", merged_baseline)
