"""
測試「分級減碼」而非「全有全無」：訊號還不確定時先減一部分曝險，訊號
真正確認時才全部出清，看能不能同時兼顧「不要賣飛」跟「躲崩盤不要太慢」
這兩個互相矛盾的目標。

曝險分級規則（只用日線的四個硬賣條件判斷，週線不納入，簡化模擬）：
    生死線(MA35)下彎+跌破 → 曝險 0%（確認轉空，全部出清）
    其他三個條件(四關價/MACD死叉/OBV背離)裡同時成立 >= 2個 → 曝險 0%
      （多個獨立條件同時爆，雖然生死線沒破，但夠多訊號同時出現，
      當成「確認」處理）
    剛好只有其中1個條件成立 → 曝險 50%（還不確定，先減一半觀察）
    以上條件都沒有 → 曝險 100%（維持滿倉）

用「昨天收盤時的曝險比例」乘上「今天的股價報酬」累積成權益曲線，跟
兩個對照組比較：
    Buy&Hold：全程曝險100%
    現行規則(全有全無)：四個條件任一觸發就曝險0%，否則100%
    分級減碼（這次要測的）：如上述三個曝險等級

比較累積報酬，也比較最大回撤(max drawdown)——分級減碼的意義本來就不是
要賺更多，是要在「不要因為雜訊訊號錯殺」跟「真的崩盤時不要跌太深」之間
找平衡，所以最大回撤才是重點指標，不是單純比總報酬。

用法（在專案資料夾內執行）：
    python3 test_partial_reduce.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from diagnose_sell_lag import _matched_markers
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import evaluate_timeframe
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24
KEY_MA_MARKER = "生死線下彎(均線)"
FAST_MARKERS = {"四關價跌破", "MACD死亡交叉", "OBV量能背離"}


def _exposure_for_markers(markers: set) -> float:
    if KEY_MA_MARKER in markers:
        return 0.0
    fast_count = len(markers & FAST_MARKERS)
    if fast_count >= 2:
        return 0.0
    if fast_count == 1:
        return 0.5
    return 1.0


def _max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    return float(drawdown.min())


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
    df_day = df_day.sort_values("date").reset_index(drop=True)
    df_day["date"] = df_day["date"].astype(str)

    backtest_start_str = backtest_start.strftime("%Y-%m-%d")
    as_of_dates = [d for d in df_day["date"] if d >= backtest_start_str]
    if len(as_of_dates) < 2:
        return {"name": name, "code": code, "error": "回測窗口內資料不足"}

    rows = []
    for d in as_of_dates:
        day_slice = df_day[df_day["date"] <= d]
        verdict_day = evaluate_timeframe(day_slice, "日")
        markers = _matched_markers(verdict_day.reasons)
        close = float(df_day.loc[df_day["date"] == d, "close"].iloc[-1])
        binary_exposure = 0.0 if verdict_day.conclusion == "賣出減碼" else 1.0
        tiered_exposure = _exposure_for_markers(markers)
        rows.append({"date": d, "close": close,
                      "binary_exposure": binary_exposure, "tiered_exposure": tiered_exposure})

    daily = pd.DataFrame(rows)
    daily["stock_ret"] = daily["close"].pct_change()

    # 用「昨天收盤時算出的曝險」乘上「今天的報酬」，避免用到當天才知道的
    # 訊號去決定當天的曝險（look-ahead bias）。
    daily["buyhold_ret"] = daily["stock_ret"]
    daily["binary_ret"] = daily["binary_exposure"].shift(1) * daily["stock_ret"]
    daily["tiered_ret"] = daily["tiered_exposure"].shift(1) * daily["stock_ret"]
    daily = daily.dropna(subset=["stock_ret"]).reset_index(drop=True)

    equity_buyhold = (1 + daily["buyhold_ret"]).cumprod()
    equity_binary = (1 + daily["binary_ret"]).cumprod()
    equity_tiered = (1 + daily["tiered_ret"]).cumprod()

    return {
        "name": name, "code": code, "error": None,
        "final_ret_buyhold": float(equity_buyhold.iloc[-1] - 1),
        "final_ret_binary": float(equity_binary.iloc[-1] - 1),
        "final_ret_tiered": float(equity_tiered.iloc[-1] - 1),
        "mdd_buyhold": _max_drawdown(equity_buyhold),
        "mdd_binary": _max_drawdown(equity_binary),
        "mdd_tiered": _max_drawdown(equity_tiered),
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


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    results = run(api_token, months_arg)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 220)

    print(f"=== 三種策略的累積報酬 vs 最大回撤比較（回測{months_arg}個月）===")
    print(f"{'股票':8s} {'Buy&Hold報酬':>12s} {'現行規則報酬':>12s} {'分級減碼報酬':>12s}  |  "
          f"{'B&H回撤':>8s} {'現行回撤':>8s} {'分級回撤':>8s}")
    valid = [r for r in results if not r.get("error")]
    for r in sorted(valid, key=lambda x: x["name"]):
        print(f"{r['name']:8s} {r['final_ret_buyhold']*100:11.1f}% {r['final_ret_binary']*100:11.1f}% "
              f"{r['final_ret_tiered']*100:11.1f}%  |  "
              f"{r['mdd_buyhold']*100:7.1f}% {r['mdd_binary']*100:7.1f}% {r['mdd_tiered']*100:7.1f}%")
    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    if valid:
        print(f"\n=== 13檔平均 ===")
        for key, label in [("final_ret_buyhold", "Buy&Hold報酬"), ("final_ret_binary", "現行規則報酬"),
                            ("final_ret_tiered", "分級減碼報酬"), ("mdd_buyhold", "B&H回撤"),
                            ("mdd_binary", "現行回撤"), ("mdd_tiered", "分級回撤")]:
            avg = sum(r[key] for r in valid) / len(valid)
            print(f"{label}: {avg*100:.1f}%")
