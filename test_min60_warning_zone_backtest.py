"""
回測「60分20MA~60MA多空預警區間」(min60_warning_zone) 這條caveat規則的
實際預測力。這條規則是2026-10-21課程新增，目前只附加caveats提示（見
sop_decision.py _step2_60min_warning_zone()、sop_rules.yaml indicators.
min60_warning_zone），還沒backtest驗證過——這支腳本是為了決定要不要正式
納入評分。

跟test_intraday_signals_buy.py同一套做法：60分指標在完整歷史上算一次
（run_all_indicators），直接在整條序列上用布林條件標出「第一預警」（跌破
20MA）跟「第二預警」（跌破60MA）的K棒，之後N根K棒（不是N個交易日，這裡
是短線訊號，用K棒數窗口才有意義）的報酬變化，跟全部K棒(baseline)對照。

讀法：這是「多空預警」，概念上是偏空訊號（提醒多單減碼/空單進場），所以
理想情況應該看到：觸發後的報酬比baseline更差（下跌機率更高），才算是
有效的示警。

用法（需要FUGLE_API_KEY）：
    python3 test_min60_warning_zone_backtest.py <FUGLE_API_KEY> [lookback天數=350]
"""
import concurrent.futures
import sys

import pandas as pd

from sop_decision import RULES
from stock_core import STOCK_NAME_MAP, get_intraday_data, ma_slope, run_all_indicators, unique_watchlist

BAR_FORWARD_WINDOWS = (5, 10, 20)


def _bar_forward_returns(df: pd.DataFrame, idx: int, windows=BAR_FORWARD_WINDOWS) -> dict:
    base = float(df.loc[idx, "close"])
    max_idx = df.index.max()
    out = {}
    for n in windows:
        future_idx = idx + n
        out[n] = float(df.loc[future_idx, "close"]) / base - 1 if future_idx <= max_idx else None
    return out


def _backtest_one(name: str, code: str, fugle_api_key: str, lookback_days: int) -> tuple:
    try:
        df = get_intraday_data(code, "60", fugle_api_key, lookback_days=lookback_days)
        if df.empty or len(df) < 70:
            return code, None, "60分資料不足"
        df = run_all_indicators(df, "60分").reset_index(drop=True)
        if "MA60" not in df.columns or "MA20" not in df.columns:
            return code, None, "缺MA20/MA60欄位"

        first_warning = df["close"] < df["MA20"]
        second_warning = df["close"] < df["MA60"]

        # MA60斜率分類（下彎 vs 走平/上揚），近似production的ma_slope()邏輯
        # （比較lookback根前的值，門檻用trailing window的平均絕對值*0.001），
        # 驗證第二預警的caveat文字本身有分兩種措辭（MA60下彎=空單加碼，
        # 走平/上揚=防守點）這個區分在數據上站不站得住腳。
        slope_lookback = RULES["indicators"]["ma_slope"]["lookback"]
        ma60 = df["MA60"]
        ma60_diff = ma60 - ma60.shift(slope_lookback)
        scale_window = max(slope_lookback + 1, 20)
        threshold = ma60.abs().rolling(window=scale_window, min_periods=1).mean() * 0.001
        ma60_falling = ma60_diff < -threshold

        rows = []
        for idx in df.index:
            if pd.isna(df.loc[idx, "MA20"]) or pd.isna(df.loc[idx, "MA60"]):
                continue
            fwd = _bar_forward_returns(df, idx)
            row = {
                "name": name, "code": code,
                "first_warning": bool(first_warning.loc[idx]),
                "second_warning": bool(second_warning.loc[idx]),
                "ma60_falling": bool(ma60_falling.loc[idx]) if pd.notna(ma60_diff.loc[idx]) else None,
            }
            for n in BAR_FORWARD_WINDOWS:
                row[f"fwd_ret_{n}b"] = fwd[n]
            rows.append(row)
        return code, pd.DataFrame(rows), None
    except Exception as exc:  # noqa: BLE001
        return code, None, str(exc)


def run_backtest(fugle_api_key: str, lookback_days: int) -> pd.DataFrame:
    watchlist = unique_watchlist(STOCK_NAME_MAP)
    tw_watchlist = [(n, c) for n, c, m in watchlist if m == "TW"]

    all_frames, errors = [], {}
    # Fugle的頻率限制比FinMind嚴格，平行度調低，跟pages/1_短線進場.py同款考量。
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(_backtest_one, name, code, fugle_api_key, lookback_days)
            for name, code in tw_watchlist
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
    for n in BAR_FORWARD_WINDOWS:
        col = f"fwd_ret_{n}b"
        valid = group[col].dropna()
        if valid.empty:
            out[f"{n}根K棒_平均報酬%"] = None
            out[f"{n}根K棒_下跌比例%"] = None
        else:
            out[f"{n}根K棒_平均報酬%"] = round(valid.mean() * 100, 3)
            out[f"{n}根K棒_下跌比例%"] = round((valid < 0).mean() * 100, 1)
    return out


def _print_group(label: str, group: pd.DataFrame) -> None:
    if group.empty:
        print(f"{label}：0筆，無法統計")
        return
    s = _stats(group)
    print(f"{label}：{s['筆數']}筆（{s['股票數']}檔股票）")
    for n in BAR_FORWARD_WINDOWS:
        print(f"  之後{n}根K棒：平均報酬 {s[f'{n}根K棒_平均報酬%']}%　"
              f"下跌比例 {s[f'{n}根K棒_下跌比例%']}%")


if __name__ == "__main__":
    fugle_api_key = sys.argv[1] if len(sys.argv) > 1 else ""
    lookback_days = int(sys.argv[2]) if len(sys.argv) > 2 else 350
    if not fugle_api_key:
        print("需要Fugle API Key才能跑這個回測（60分分K歷史資料）。")
        sys.exit(1)

    detail_df = run_backtest(fugle_api_key, lookback_days)
    if detail_df.empty:
        print("沒有算出任何歷史資料，無法回測。")
        sys.exit(0)

    print(f"回測窗口：{lookback_days}天的60分K，{detail_df['code'].nunique()}檔股票，"
          f"共{len(detail_df)}根K棒\n")

    baseline = detail_df
    first_only = detail_df[detail_df["first_warning"] & ~detail_df["second_warning"]]
    second = detail_df[detail_df["second_warning"]]
    second_falling = second[second["ma60_falling"] == True]  # noqa: E712
    second_flat_up = second[second["ma60_falling"] == False]  # noqa: E712

    _print_group("baseline（全部K棒）", baseline)
    print()
    _print_group("第一預警（跌破20MA，未跌破60MA）", first_only)
    print()
    _print_group("第二預警（跌破60MA，不分MA60斜率）", second)
    print()
    _print_group("第二預警 + MA60同步下彎（caveat文字：空單可加碼）", second_falling)
    print()
    _print_group("第二預警 + MA60走平/上揚（caveat文字：可視為防守點）", second_flat_up)
    print()
    print("讀法：這是偏空的多空預警，理想情況應該看到觸發後報酬比baseline更差、"
          "下跌比例更高，才算是有鑑別力的示警；第二預警理論上該比第一預警更明確偏空，"
          "MA60同步下彎那組應該比走平/上揚那組更偏空（production的caveat文字正是這樣區分的）。")
