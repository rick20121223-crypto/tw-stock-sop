"""
測試「分批加碼」買點：筆記說大跌急殺時先分批建倉5-10%，之後每日再加碼
~5%，直到建滿倉。這個跟前面測過的訊號型條件不一樣，它本質是攤平成本
的進場節奏，不是能算勝率的離散訊號，所以測法也不同：

    用每檔股票24個月窗口裡最大的一次崩盤(高點->低點)當場景，比較：
    1. Lump sum：觸發當天(定義：收盤價從近期高點回落超過CRASH_THRESHOLD)
       all-in買滿倉位
    2. 分批加碼：觸發當天先買INITIAL_PCT，之後每個交易日再加DAILY_ADD_PCT，
       直到買滿100%或是遇到窗口結束

比較兩種做法的「平均成本」跟「相對低點的成本價差」，以及之後20個交易日
的報酬率——分批加碼的意義是「如果買早了、股價繼續跌，分批可以攤低成本」，
所以重點看能不能真的攤到比lump sum更低的成本，不是看哪個報酬率更高
（如果lump sum剛好買在次低點，報酬率本來就會贏，这不代表分批加碼沒用，
分批加碼的價值在於「buy早了也不會死很慘」，所以兩個指標都要看）。

用法（在專案資料夾內執行）：
    python3 test_scaled_entry_crash.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from historical_backtest import WARMUP_CALENDAR_DAYS
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24
CRASH_THRESHOLD = -0.15      # 從近期高點回落超過15%才算「大跌急殺」觸發
INITIAL_PCT = 0.10           # 觸發當天先買10%
DAILY_ADD_PCT = 0.05         # 之後每個交易日再加5%
SCALE_WINDOW_DAYS = 20       # 之後20個交易日的事後報酬


def _find_crash_trigger(recent: pd.DataFrame) -> dict:
    """找出窗口內最大的一次崩盤(高點->低點)，回傳觸發日(第一次跌破
    CRASH_THRESHOLD的那天，不是低點本身——分批加碼要在崩盤「進行中」就
    開始買，不是等到低點確認後才買，否則就跟一般抄底訊號沒有差別）。"""
    running_peak = recent["close"].cummax()
    drawdown = (recent["close"] - running_peak) / running_peak
    trough_pos = drawdown.idxmin()
    trough_date = recent.loc[trough_pos, "date"]
    trough_price = float(recent.loc[trough_pos, "close"])
    peak_price = running_peak.loc[trough_pos]
    peak_pos = recent[(recent["date"] <= trough_date) & (recent["close"] == peak_price)].index[-1]
    peak_date = recent.loc[peak_pos, "date"]

    window = recent[(recent.index >= peak_pos) & (recent.index <= trough_pos)]
    triggered = window[(window["close"] / peak_price - 1) <= CRASH_THRESHOLD]
    if triggered.empty:
        return None
    trigger_pos = triggered.index[0]
    return {
        "peak_date": peak_date, "trough_date": trough_date,
        "crash_pct": round((trough_price / peak_price - 1) * 100, 1) if peak_price else None,
        "trigger_pos": trigger_pos, "trough_pos": trough_pos, "trough_price": trough_price,
    }


def _simulate(recent: pd.DataFrame, trigger_pos: int, trough_pos: int) -> dict:
    window = recent[recent.index >= trigger_pos].reset_index(drop=True)
    if len(window) < 2:
        return None

    lump_price = float(window.loc[0, "close"])

    exposure, cost_paid, shares = 0.0, 0.0, 0.0
    weighted_cost_num, weighted_cost_den = 0.0, 0.0
    for i, row in window.iterrows():
        add_pct = INITIAL_PCT if i == 0 else DAILY_ADD_PCT
        add_pct = min(add_pct, 1.0 - exposure)
        if add_pct <= 0:
            continue
        price = float(row["close"])
        weighted_cost_num += add_pct * price
        weighted_cost_den += add_pct
        exposure += add_pct
        if exposure >= 1.0:
            break
    if weighted_cost_den == 0:
        return None
    scaled_avg_cost = weighted_cost_num / weighted_cost_den
    days_to_full = i + 1

    trough_price = float(recent.loc[trough_pos, "close"])
    eval_end_pos = min(trigger_pos + SCALE_WINDOW_DAYS, recent.index.max())
    end_price = float(recent.loc[eval_end_pos, "close"])

    return {
        "lump_price": lump_price,
        "lump_vs_trough_pct": round((lump_price / trough_price - 1) * 100, 1) if trough_price else None,
        "lump_fwd_ret_pct": round((end_price / lump_price - 1) * 100, 1),
        "scaled_avg_cost": scaled_avg_cost,
        "scaled_vs_trough_pct": round((scaled_avg_cost / trough_price - 1) * 100, 1) if trough_price else None,
        "scaled_fwd_ret_pct": round((end_price / scaled_avg_cost - 1) * 100, 1),
        "days_to_full_position": days_to_full,
    }


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
    recent = df_day[df_day["date"] >= backtest_start_str].reset_index(drop=True)
    if recent.empty:
        return {"name": name, "code": code, "error": "回測窗口內資料不足"}

    crash = _find_crash_trigger(recent)
    if crash is None:
        return {"name": name, "code": code, "error": "窗口內沒有超過15%的崩盤"}

    sim = _simulate(recent, crash["trigger_pos"], crash["trough_pos"])
    if sim is None:
        return {"name": name, "code": code, "error": "崩盤後資料不足以模擬分批加碼"}

    return {"name": name, "code": code, "error": None,
            "peak_date": crash["peak_date"], "trough_date": crash["trough_date"],
            "crash_pct": crash["crash_pct"], **sim}


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

    print(f"=== Lump sum vs 分批加碼：13檔最大崩盤情境比較（回測{months_arg}個月，崩盤門檻{CRASH_THRESHOLD*100:.0f}%）===")
    valid = [r for r in results if not r.get("error")]
    for r in sorted(valid, key=lambda x: x["name"]):
        lump_pct = f"{r['lump_vs_trough_pct']:+.1f}%" if r["lump_vs_trough_pct"] is not None else "N/A"
        scaled_pct = f"{r['scaled_vs_trough_pct']:+.1f}%" if r["scaled_vs_trough_pct"] is not None else "N/A"
        print(f"{r['name']:8s} 高點{r['peak_date']} 低點{r['trough_date']}({r['crash_pct']}%) | "
              f"Lump成本比低點{lump_pct},後20日報酬{r['lump_fwd_ret_pct']:+.1f}% | "
              f"分批成本比低點{scaled_pct},後20日報酬{r['scaled_fwd_ret_pct']:+.1f}% "
              f"(花{r['days_to_full_position']}天買滿)")
    for r in results:
        if r.get("error"):
            print(f"⚠️ {r['name']}（{r['code']}）：{r['error']}")

    comparable = [r for r in valid if r["lump_vs_trough_pct"] is not None and r["scaled_vs_trough_pct"] is not None]
    if comparable:
        avg_lump_vs_trough = sum(r["lump_vs_trough_pct"] for r in comparable) / len(comparable)
        avg_scaled_vs_trough = sum(r["scaled_vs_trough_pct"] for r in comparable) / len(comparable)
        avg_lump_fwd = sum(r["lump_fwd_ret_pct"] for r in comparable) / len(comparable)
        avg_scaled_fwd = sum(r["scaled_fwd_ret_pct"] for r in comparable) / len(comparable)
        better_cost_count = sum(1 for r in comparable if r["scaled_vs_trough_pct"] < r["lump_vs_trough_pct"])
        valid = comparable
        print(f"\n=== {len(valid)}檔平均 ===")
        print(f"Lump成本比低點平均: {avg_lump_vs_trough:+.1f}%　後20日平均報酬: {avg_lump_fwd:+.1f}%")
        print(f"分批成本比低點平均: {avg_scaled_vs_trough:+.1f}%　後20日平均報酬: {avg_scaled_fwd:+.1f}%")
        print(f"分批加碼的成本比lump sum更低（更接近低點）的有 {better_cost_count}/{len(valid)} 檔")
