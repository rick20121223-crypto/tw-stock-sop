"""
測試但丁老師買進筆記裡剩下三項、原本以為測不到的60分/5分訊號：

    A) CCI折返（60分線）：CCI跌破-100後回升站上-100，筆記說可以當「早期
       訊號加分買點」。目前sop_decision.py的_step5b_cci()把它定位成
       secondary、不單獨當買進依據——這裡用同一個門檻邏輯(signal_
       threshold=100, lookback_bars=3)獨立測試，看它單獨使用到底有沒有
       超額報酬，驗證現有code裡"不單獨使用"的判斷是否成立。
    B) CCI折返（5分線）：同上，換成5分線資料測一次。
    C) 60分120MA(烏龜線)+240MA：60分MA20站上、MA120(烏龜線)站上且上揚、
       MA240(多空貢獻線)站上且上揚，三線同時確認。注意：production的
       TIMEFRAME_MA_PERIODS["60分"]只算20/240，沒有120——這裡額外加算
       MA120只用於這次測試，不動production設定。
    D) 5分20MA+BBI+MACD：5分MA20站上且上揚、站上BBI(多空指標)、MACD(DIF)
       在零軸之上，三個條件同時成立。

前提：之前用probe_fugle_intraday_depth.py測過，Fugle的60分線至少有360天
資料、5分線至少有300天資料都沒被截斷（受限於API單次請求必須<365天，
這裡用350天/300天取安全值），樣本量足夠做統計意義的驗證。

跟之前的日/週線測試不同，這裡用的事後報酬是「之後N根K棒」而不是「之後
N個交易日」，因為這些本來就是極短線訊號，用日線的天數窗口比不出意義。

用法（需要FUGLE_API_KEY，在能連到Fugle的環境執行）：
    python3 test_intraday_signals_buy.py <FUGLE_API_KEY> [60分lookback=350] [5分lookback=300]
"""
import concurrent.futures
import sys

import pandas as pd

from stock_core import (
    STOCK_NAME_MAP,
    calc_bbi,
    calc_cci,
    calc_ma,
    calc_macd,
    get_intraday_data,
    unique_watchlist,
)

BAR_FORWARD_WINDOWS = (5, 10, 20)
CCI_SIGNAL_THRESHOLD = 100.0
CCI_LOOKBACK_BARS = 3


def _bar_forward_returns(df: pd.DataFrame, idx: int, windows=BAR_FORWARD_WINDOWS) -> dict:
    base = float(df.loc[idx, "close"])
    max_idx = df.index.max()
    out = {}
    for n in windows:
        future_idx = idx + n
        out[n] = float(df.loc[future_idx, "close"]) / base - 1 if future_idx <= max_idx else None
    return out


def _detect_cci_foldback(df: pd.DataFrame, lookback_bars: int = CCI_LOOKBACK_BARS,
                          signal_threshold: float = CCI_SIGNAL_THRESHOLD) -> pd.Series:
    cci = df["CCI"]
    prior_min = cci.shift(1).rolling(window=lookback_bars - 1, min_periods=1).min()
    was_extreme_low = prior_min <= -signal_threshold
    recovering = cci > -signal_threshold
    return (was_extreme_low & recovering).fillna(False)


def _detect_60min_turtle_signal(df: pd.DataFrame) -> pd.Series:
    fast_up = (df["close"] >= df["MA20"]) & (df["MA20"] > df["MA20"].shift(3))
    turtle_up = (df["close"] >= df["MA120"]) & (df["MA120"] > df["MA120"].shift(3))
    key_up = (df["close"] >= df["MA240"]) & (df["MA240"] > df["MA240"].shift(3))
    return (fast_up & turtle_up & key_up).fillna(False)


def _detect_5min_combo_signal(df: pd.DataFrame) -> pd.Series:
    fast_up = (df["close"] >= df["MA20"]) & (df["MA20"] > df["MA20"].shift(3))
    bbi_up = df["close"] >= df["BBI"]
    macd_bullish = df["DIF"] > 0
    return (fast_up & bbi_up & macd_bullish).fillna(False)


def _collect_events(df: pd.DataFrame, mask: pd.Series) -> list:
    out = []
    for idx in df.index[mask]:
        out.append(_bar_forward_returns(df, idx))
    return out


def analyze_one(name: str, code: str, api_key: str, lookback_60min: int, lookback_5min: int) -> dict:
    result = {"name": name, "code": code, "error_60min": None, "error_5min": None}

    try:
        df60 = get_intraday_data(code, "60", api_key, lookback_days=lookback_60min)
    except Exception as exc:  # noqa: BLE001
        result["error_60min"] = str(exc)
        df60 = None
    if df60 is not None and not df60.empty:
        df60 = calc_ma(df60, periods=(20, 120, 240))
        df60 = calc_cci(df60, period=14)
        df60 = df60.reset_index(drop=True)
        cci_signal = _detect_cci_foldback(df60)
        turtle_signal = _detect_60min_turtle_signal(df60)
        result["cci_60min_events"] = _collect_events(df60, cci_signal)
        result["turtle_events"] = _collect_events(df60, turtle_signal)
        result["baseline_60min_events"] = _collect_events(df60, pd.Series(True, index=df60.index))
    elif df60 is not None:
        result["error_60min"] = "60分線資料為空"

    try:
        df5 = get_intraday_data(code, "5", api_key, lookback_days=lookback_5min)
    except Exception as exc:  # noqa: BLE001
        result["error_5min"] = str(exc)
        df5 = None
    if df5 is not None and not df5.empty:
        df5 = calc_ma(df5, periods=(20,))
        df5 = calc_bbi(df5)
        df5 = calc_macd(df5)
        df5 = calc_cci(df5, period=14)
        df5 = df5.reset_index(drop=True)
        cci_signal_5 = _detect_cci_foldback(df5)
        combo_signal = _detect_5min_combo_signal(df5)
        result["cci_5min_events"] = _collect_events(df5, cci_signal_5)
        result["combo_events"] = _collect_events(df5, combo_signal)
        result["baseline_5min_events"] = _collect_events(df5, pd.Series(True, index=df5.index))
    elif df5 is not None:
        result["error_5min"] = "5分線資料為空"

    return result


def run(api_key: str, lookback_60min: int, lookback_5min: int) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    # Fugle 有全域節流鎖(_fugle_throttle)會把實際送出的請求間隔壓在
    # 1.2秒以上，所以這裡的並行數只是讓等待中的執行緒排隊，不會真的
    # 同時衝出去撞429，用適中的並行數就好。
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(analyze_one, name, code, api_key, lookback_60min, lookback_5min)
                   for name, code, _market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _print_stats(label: str, events: list) -> None:
    df = pd.DataFrame(events)
    print(f"--- {label}（共{len(events)}筆）---")
    for n in BAR_FORWARD_WINDOWS:
        valid = df[n].dropna() if (not df.empty and n in df.columns) else pd.Series(dtype=float)
        if len(valid):
            print(f"後{n}根K棒：平均報酬 {valid.mean()*100:.2f}%　正報酬比例 {(valid>0).mean()*100:.1f}%　(n={len(valid)})")
        else:
            print(f"後{n}根K棒：無資料")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python3 test_intraday_signals_buy.py <FUGLE_API_KEY> [60分lookback=350] [5分lookback=300]")
        sys.exit(1)
    api_key = sys.argv[1]
    lookback_60min = int(sys.argv[2]) if len(sys.argv) > 2 else 350
    lookback_5min = int(sys.argv[3]) if len(sys.argv) > 3 else 300

    results = run(api_key, lookback_60min, lookback_5min)

    for r in results:
        if r.get("error_60min"):
            print(f"⚠️ {r['name']}（{r['code']}）60分線：{r['error_60min']}")
        if r.get("error_5min"):
            print(f"⚠️ {r['name']}（{r['code']}）5分線：{r['error_5min']}")

    all_cci_60, all_turtle, all_baseline_60 = [], [], []
    all_cci_5, all_combo, all_baseline_5 = [], [], []
    for r in results:
        all_cci_60.extend(r.get("cci_60min_events", []))
        all_turtle.extend(r.get("turtle_events", []))
        all_baseline_60.extend(r.get("baseline_60min_events", []))
        all_cci_5.extend(r.get("cci_5min_events", []))
        all_combo.extend(r.get("combo_events", []))
        all_baseline_5.extend(r.get("baseline_5min_events", []))

    print(f"\n=== 60分線（lookback={lookback_60min}天），13檔加總 ===\n")
    _print_stats("A) CCI折返(60分，單獨使用)", all_cci_60)
    print()
    _print_stats("C) 60分MA20+MA120烏龜線+MA240三線確認", all_turtle)
    print()
    _print_stats("60分無條件基準（隨便哪根K棒進場）", all_baseline_60)

    print(f"\n=== 5分線（lookback={lookback_5min}天），13檔加總 ===\n")
    _print_stats("B) CCI折返(5分，單獨使用)", all_cci_5)
    print()
    _print_stats("D) 5分MA20+BBI+MACD三條件組合", all_combo)
    print()
    _print_stats("5分無條件基準（隨便哪根K棒進場）", all_baseline_5)
