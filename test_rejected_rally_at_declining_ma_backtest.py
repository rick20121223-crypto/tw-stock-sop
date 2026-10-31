"""
測試「反彈打到下彎均線過不去」這個空方型態：均線本身仍下彎，且最近K棒
的最高點有貼近/碰到均線、但收盤從未真正站上——代表反彈測試均線失敗，
依1030錄音「程式單權重配比」筆記，這種情況搭配CCI12/KD365同步死叉時，
應該觸發「順勢加碼空點」。

只測日線（生死線MA35），跟test_washout_recovery_buy.py同樣的資料限制：
60分版用的是20MA/60MA（跟min60_warning_zone同一組均線，不是60分的
KEY_MA=MA240），但60分/5分只能抓到Fugle最近一段，沒辦法像日線逐日
回測，這次先不測。CCI只在60分/5分適用（sop_rules.yaml indicators.cci.
applicable_timeframes），日線沒有官方CCI角色，所以日線版的「指標同步
確認」只用KD365死叉，不用CCI。

這是空方假設：驗證方式是看之後報酬有沒有比無條件基準明顯更負（不是
看正報酬比例，跟其他買方訊號的驗證方向相反）。

操作化定義：
    均線下彎：MA35[今天] < MA35[slope_lookback天前]（比照ma_slope()同一套
              lookback慣例）
    反彈測不過：過去lookback_bars天內（不含今天），曾經最高價(max)貼近
                MA35（在tolerance%以內）或更高，但這段期間收盤從未真正
                站上MA35；且今天收盤仍在MA35之下（確認真的沒站上去，不是
                已經站上去了才回頭看歷史）
    KD365死叉確認：過去confirm_window天內(含今天)，K365由上往下穿破D365

用法（在專案資料夾內執行）：
    python3 test_rejected_rally_at_declining_ma_backtest.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24
SLOPE_LOOKBACK = 3       # 比照ma_slope()預設lookback
REJECT_LOOKBACK_BARS = 5  # 過去幾天內找「曾經貼近均線但沒站上」的反彈測試
APPROACH_TOLERANCE = 0.02  # 最高價在均線以下2%以內視為「貼近/碰到」
KD365_CONFIRM_WINDOW = 3  # 過去幾天內(含今天)找KD365死叉


def _detect_rejected_rally(df: pd.DataFrame, ma_col: str = "MA35",
                            slope_lookback: int = SLOPE_LOOKBACK,
                            reject_lookback: int = REJECT_LOOKBACK_BARS,
                            tolerance: float = APPROACH_TOLERANCE) -> pd.Series:
    ma = df[ma_col]
    close = df["close"]
    high = df["max"]

    ma_declining = ma < ma.shift(slope_lookback)

    # 過去reject_lookback天內（不含今天），最高價曾經貼近/觸及均線
    approached = (high >= ma * (1 - tolerance)).shift(1).rolling(window=reject_lookback).max() >= 1
    # 同一段期間，收盤從未真正站上均線（嚴格測不過，不是已經站上去了）
    never_closed_above = (close < ma).shift(1).rolling(window=reject_lookback).min() >= 1
    today_still_below = close < ma

    return (ma_declining & approached & never_closed_above & today_still_below).fillna(False)


def _detect_kd365_dead_cross_recent(df: pd.DataFrame, window: int = KD365_CONFIRM_WINDOW) -> pd.Series:
    k, d = df["K365"], df["D365"]
    dead_cross_today = (k.shift(1) >= d.shift(1)) & (k < d)
    return dead_cross_today.rolling(window=window).max().fillna(0) >= 1


def analyze_one(name: str, code: str, market: str, api_token: str, backtest_months: int) -> dict:
    today = date.today()
    backtest_start = today - timedelta(days=backtest_months * 30)
    fetch_start = (backtest_start - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")

    try:
        df_day = get_stock_data(code, market, fetch_start, str(today), api_token)
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "code": code, "error": str(exc)}
    if df_day.empty:
        return {"name": name, "code": code, "error": "日線資料為空"}
    df_day = run_all_indicators(df_day, "日")
    if "MA35" not in df_day.columns or "K365" not in df_day.columns:
        return {"name": name, "code": code, "error": "缺少MA35或K365欄位"}
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    rejected = _detect_rejected_rally(df_day)
    kd365_dead = _detect_kd365_dead_cross_recent(df_day)
    rejected_plus_kd365 = rejected & kd365_dead

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    recent_mask = df_day["date"] >= backtest_start_str

    def _collect(mask):
        out = []
        for idx in df_day.index[recent_mask & mask]:
            d = df_day.loc[idx, "date"]
            fwd = _forward_returns(df_day, d)
            out.append({f"fwd_ret_{n}d": fwd[n] for n in FORWARD_WINDOWS})
        return out

    return {
        "name": name, "code": code, "error": None,
        "rejected_only_events": _collect(rejected),
        "rejected_plus_kd365_events": _collect(rejected_plus_kd365),
        "baseline_events": _collect(pd.Series(True, index=df_day.index)),
    }


def run(api_token: str, backtest_months: int = BACKTEST_MONTHS_DEFAULT) -> list:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(analyze_one, name, code, market, api_token, backtest_months)
                   for name, code, market in watchlist]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results


def _print_stats(label: str, events: list) -> None:
    df = pd.DataFrame(events)
    print(f"--- {label}（共{len(events)}筆）---")
    for n in FORWARD_WINDOWS:
        valid = df[f"fwd_ret_{n}d"].dropna() if not df.empty else pd.Series(dtype=float)
        if len(valid):
            print(f"{n}日：平均報酬 {valid.mean()*100:.2f}%　負報酬比例 {(valid<0).mean()*100:.1f}%　(n={len(valid)})")
        else:
            print(f"{n}日：無資料")


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    buckets = {"rejected_only_events": [], "rejected_plus_kd365_events": [], "baseline_events": []}
    for r in results:
        if r.get("error"):
            continue
        for key in buckets:
            buckets[key].extend(r[key])

    print(f"\n=== 回測{months_arg}個月，13檔加總（空方假設：報酬應比基準更負）===\n")
    _print_stats("反彈測不過MA35單獨", buckets["rejected_only_events"])
    print()
    _print_stats("反彈測不過MA35 + KD365死叉同步確認", buckets["rejected_plus_kd365_events"])
    print()
    _print_stats("無條件基準（隨便哪天進場）", buckets["baseline_events"])
