"""
回測「左側平台/前低兩次腳未破」(platform_second_leg) 這條caveat規則的
實際預測力。這條規則目前只是觀察性提示（見sop_decision.py
_detect_platform_second_leg()、sop_rules.yaml indicators.
platform_second_leg），還沒backtest驗證過——這支腳本是為了決定要不要
正式納入評分（例如當作賣出減碼的例外條件）。

讀法：這是「不要砍單」的保護性規則，不是買進訊號，關心的重點是「觸發後
股價有沒有繼續破底」，不是漲跌幅本身多漂亮。所以除了平均報酬/正報酬
勝率，另外算「續跌比例」（N日後收盤 < 觸發當天收盤）當主要指標，跟
「全部歷史日期」(baseline，不分觸發與否)的續跌比例對照——如果觸發後的
續跌比例明顯低於baseline，才算是有保護力的證據。

跟historical_backtest.py同一套「指標只往回看」的效能設計：日線指標在
完整歷史上算一次，每個歷史日期只是切到該日為止重跑偵測，不是每天重新
打API/重新算指標。只回測固定清單（跟platform_second_leg本身不限
timeframe不同，這裡只測日線——週/60分/5分需要週線resample或Fugle分K
歷史，先用日線建立這套量測管線）。

用法（在專案資料夾內執行）：
    python3 test_platform_second_leg_backtest.py [FinMind_Token] [回測月數=24]
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from backtest_signal_log import FORWARD_WINDOWS, _forward_returns
from historical_backtest import WARMUP_CALENDAR_DAYS
from sop_decision import _detect_platform_second_leg, _find_platform_support_resistance
from stock_core import STOCK_NAME_MAP, get_stock_data, run_all_indicators, unique_watchlist

BACKTEST_MONTHS_DEFAULT = 24


def _triggered_as_of(df_day_full: pd.DataFrame, as_of_date: str) -> bool:
    day_slice = df_day_full[df_day_full["date"] <= as_of_date]
    if day_slice.empty:
        return False
    latest = day_slice.iloc[-1]
    levels = _find_platform_support_resistance(day_slice, "日", latest)
    if levels is None:
        return False
    support_levels, _resistance_levels = levels
    level = _detect_platform_second_leg(day_slice, "日", latest, support_levels)
    return level is not None


def _backtest_one(name: str, code: str, market: str, api_token: str, backtest_months: int) -> tuple:
    try:
        today = date.today()
        backtest_start = today - timedelta(days=backtest_months * 30)
        fetch_start = (backtest_start - timedelta(days=WARMUP_CALENDAR_DAYS)).strftime("%Y-%m-%d")

        df_day = get_stock_data(code, market, fetch_start, str(today), api_token)
        if df_day.empty:
            return code, None, "日線資料為空"
        df_day = run_all_indicators(df_day, "日")
        df_day = df_day.sort_values("date").reset_index(drop=True)
        df_day["date"] = df_day["date"].astype(str)

        backtest_start_str = backtest_start.strftime("%Y-%m-%d")
        as_of_dates = [d for d in df_day["date"] if d >= backtest_start_str]

        rows = []
        for as_of in as_of_dates:
            triggered = _triggered_as_of(df_day, as_of)
            close = float(df_day.loc[df_day["date"] == as_of, "close"].iloc[-1])
            row = {"date": as_of, "name": name, "code": code, "triggered": triggered, "close": close}
            fwd = _forward_returns(df_day, as_of)
            for n in FORWARD_WINDOWS:
                row[f"fwd_ret_{n}d"] = fwd[n]
            rows.append(row)
        return code, pd.DataFrame(rows), None
    except Exception as exc:  # noqa: BLE001
        return code, None, str(exc)


def run_backtest(api_token: str, backtest_months: int) -> pd.DataFrame:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    all_frames, errors = [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_backtest_one, name, code, market, api_token, backtest_months)
            for name, code, market in watchlist
        ]
        for future in concurrent.futures.as_completed(futures):
            code, df, err = future.result()
            if err is not None:
                errors[code] = err
            elif df is not None and not df.empty:
                all_frames.append(df)
    for code, err in errors.items():
        print(f"⚠️ {code}：{err}")
    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def _stats(group: pd.DataFrame) -> dict:
    out = {"筆數": len(group), "股票數": group["code"].nunique()}
    for n in FORWARD_WINDOWS:
        col = f"fwd_ret_{n}d"
        valid = group[col].dropna()
        if valid.empty:
            out[f"{n}日_平均報酬%"] = None
            out[f"{n}日_正報酬勝率%"] = None
            out[f"{n}日_續跌比例%"] = None
        else:
            out[f"{n}日_平均報酬%"] = round(valid.mean() * 100, 2)
            out[f"{n}日_正報酬勝率%"] = round((valid > 0).mean() * 100, 1)
            out[f"{n}日_續跌比例%"] = round((valid < 0).mean() * 100, 1)
    return out


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    months_arg = int(sys.argv[2]) if len(sys.argv) > 2 else BACKTEST_MONTHS_DEFAULT

    detail_df = run_backtest(api_token, months_arg)
    if detail_df.empty:
        print("沒有算出任何歷史資料，無法回測。")
        sys.exit(0)

    triggered_df = detail_df[detail_df["triggered"]]
    baseline_df = detail_df  # 全部歷史日期(不分有沒有觸發)當對照組

    print(f"回測窗口：{months_arg}個月，{detail_df['code'].nunique()}檔股票，"
          f"共{len(detail_df)}個歷史交易日")
    print(f"platform_second_leg觸發：{len(triggered_df)}次"
          f"（{triggered_df['code'].nunique()}檔股票，"
          f"觸發率{len(triggered_df)/len(detail_df)*100:.2f}%）\n")

    if triggered_df.empty:
        print("這段期間完全沒有觸發過，樣本數0，無法評估預測力——可能是門檻太嚴格，"
              "或這批股票剛好沒出現過這個型態，建議之後擴大股票池或拉長回測期再看。")
        sys.exit(0)

    trig_stats = _stats(triggered_df)
    base_stats = _stats(baseline_df)

    print("=== 觸發後 vs 全部歷史日期(baseline) 的未來N個交易日表現 ===")
    print(f"{'':10s}{'筆數':>8s}{'股票數':>8s}" + "".join(f"{n}日報酬%".rjust(12) for n in FORWARD_WINDOWS))
    print(f"{'觸發':10s}{trig_stats['筆數']:>8d}{trig_stats['股票數']:>8d}" +
          "".join(f"{trig_stats[f'{n}日_平均報酬%']!s:>12s}" for n in FORWARD_WINDOWS))
    print(f"{'baseline':10s}{base_stats['筆數']:>8d}{base_stats['股票數']:>8d}" +
          "".join(f"{base_stats[f'{n}日_平均報酬%']!s:>12s}" for n in FORWARD_WINDOWS))
    print()
    print(f"{'':10s}" + "".join(f"{n}日續跌比例%".rjust(14) for n in FORWARD_WINDOWS))
    print(f"{'觸發':10s}" + "".join(f"{trig_stats[f'{n}日_續跌比例%']!s:>14s}" for n in FORWARD_WINDOWS))
    print(f"{'baseline':10s}" + "".join(f"{base_stats[f'{n}日_續跌比例%']!s:>14s}" for n in FORWARD_WINDOWS))
    print()
    print("讀法：這是保護性規則，重點看「觸發後的續跌比例」是不是明顯低於baseline"
          "（代表真的比較不會繼續破底）；平均報酬/正報酬勝率當輔助參考。")
