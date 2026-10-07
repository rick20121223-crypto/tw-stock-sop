"""
測試 sop_decision.py `_step5_mtm()` 新增的 confirm_bars 選項：現在短期
（60分線）MTM翻空判斷預設(confirm_bars=1)是「單根K棒翻負，或近幾根內
曾死叉」就觸發，沒有任何緩衝——2026-10-07觀察到短期訊號反轉比例單週
近7成，懷疑是MTM貼著零軸雜訊造成（見對頎邦7792/光聖6442的個案討論）。

這支backtest比較 confirm_bars=1（現行）vs 2 vs 3（改成「連續N根都翻負」
才觸發）在真實60分K歷史資料上的差異，看兩件事：

    1) 雜訊是否真的減少：每個訊號「觸發」事件，之後多快又解除
       （解除=MTM翻正，連續翻負的streak被打斷）。streak過短（<=2根）
       視為一次「雜訊式觸發」，比例越高代表越容易來回打架。
    2) 保護效果有沒有被犧牲太多：訊號觸發後接下來4/8/16根K棒的價格
       變化，確認confirm_bars調大之後，還有沒有抓到真正的下跌、還是
       因為等確認而錯過了大部分跌幅。

60分資料來自 Fugle historical.candles（stock_core.get_intraday_data），
預設回看120天（Fugle的60分方案上限，見 stock_core._INTRADAY_LOOKBACK_DAYS），
只對TW市場股票測（跟 notify_intraday.py 的範圍限制同理）。

用法（在專案資料夾內執行）：
    python3 test_mtm_confirm_bars_whipsaw.py [Fugle_API_Key]
"""
import concurrent.futures
import os
import sys

import pandas as pd

from stock_core import STOCK_NAME_MAP, get_intraday_data, run_all_indicators, unique_watchlist

CONFIRM_BARS_VARIANTS = (1, 2, 3)
LOOKBACK_BARS = 3          # 跟 sop_rules.yaml 的 mtm.lookback_bars 一致
WHIPSAW_STREAK_MAX = 2     # 觸發後存活幾根K棒以內算「雜訊式觸發」
FORWARD_BARS = (4, 8, 16)  # 約1/2/3個交易日（60分K，台股盤中約4.5根/天）


def _fired_confirm1(mtm: pd.Series) -> pd.Series:
    """複刻 sop_decision._step5_mtm() confirm_bars<=1（現行預設）的邏輯，
    向量化到整個序列上，而不是只算最後一列。"""
    cross_down_bar = (mtm < 0) & (mtm.shift(1) >= 0)
    cross_down_recent = cross_down_bar.rolling(window=LOOKBACK_BARS + 1, min_periods=1).max().astype(bool)
    return (cross_down_recent | (mtm < 0)).fillna(False)


def _fired_confirm_n(mtm: pd.Series, confirm_bars: int) -> pd.Series:
    neg = (mtm < 0).astype(int)
    return (neg.rolling(window=confirm_bars).sum() == confirm_bars).fillna(False)


def _rising_edges(fired: pd.Series) -> list:
    prev = fired.shift(1, fill_value=False)
    return list(fired.index[fired & ~prev])


def _streak_length(fired: pd.Series, start_idx: int) -> int:
    n = 0
    i = start_idx
    while i < len(fired) and fired.iloc[i]:
        n += 1
        i += 1
    return n


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
    if "MTM" not in df.columns:
        return {"name": name, "code": code, "error": "缺少MTM欄位"}

    close = df["close"]
    out = {"name": name, "code": code, "error": None, "bars": len(df), "variants": {}}

    for cb in CONFIRM_BARS_VARIANTS:
        fired = _fired_confirm1(df["MTM"]) if cb <= 1 else _fired_confirm_n(df["MTM"], cb)
        events = _rising_edges(fired)

        streaks = [_streak_length(fired, i) for i in events]
        whipsaw_count = sum(1 for s in streaks if s <= WHIPSAW_STREAK_MAX)

        fwd_rets = {n: [] for n in FORWARD_BARS}
        for i in events:
            entry = close.iloc[i]
            if pd.isna(entry) or not entry:
                continue
            for n in FORWARD_BARS:
                j = i + n
                if j < len(close) and not pd.isna(close.iloc[j]):
                    fwd_rets[n].append((close.iloc[j] - entry) / entry)

        out["variants"][cb] = {
            "event_count": len(events),
            "whipsaw_count": whipsaw_count,
            "fwd_rets": fwd_rets,
        }
    return out


def run(fugle_api_key: str) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(analyze_one, name, code, market, fugle_api_key)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _print_stats(label: str, event_count: int, whipsaw_count: int, fwd_rets: dict) -> None:
    whipsaw_pct = (whipsaw_count / event_count * 100) if event_count else 0.0
    print(f"--- confirm_bars={label}：共{event_count}次觸發，其中{whipsaw_count}次"
          f"在{WHIPSAW_STREAK_MAX}根K棒內就解除（雜訊式觸發比例 {whipsaw_pct:.1f}%）---")
    for n in FORWARD_BARS:
        vals = [v for v in fwd_rets[n] if v is not None]
        if vals:
            avg = sum(vals) / len(vals) * 100
            neg_pct = sum(1 for v in vals if v < 0) / len(vals) * 100
            print(f"  觸發後{n}根K棒：平均報酬 {avg:.2f}%　下跌比例 {neg_pct:.1f}%　(n={len(vals)})")
        else:
            print(f"  觸發後{n}根K棒：無資料")


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

    print(f"\n=== confirm_bars 1 vs 2 vs 3 比較（13檔TW個股加總60分K）===\n")
    for cb in CONFIRM_BARS_VARIANTS:
        total_events = sum(r["variants"][cb]["event_count"] for r in results if not r.get("error"))
        total_whipsaw = sum(r["variants"][cb]["whipsaw_count"] for r in results if not r.get("error"))
        merged_fwd = {n: [] for n in FORWARD_BARS}
        for r in results:
            if r.get("error"):
                continue
            for n in FORWARD_BARS:
                merged_fwd[n].extend(r["variants"][cb]["fwd_rets"][n])
        _print_stats(cb, total_events, total_whipsaw, merged_fwd)
        print()
