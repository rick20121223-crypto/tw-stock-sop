"""
最小回測迴路：把 data/signal_log.csv 裡「長期」訊號接上未來N個交易日的
實際報酬，方便定期回頭檢視這套 SOP 判讀的表現。

目前 signal_log.csv 只累積了約兩週歷史，樣本數還太小，這支腳本的目的
是先把「訊號 → 之後報酬」的量測管線建好，不是要下「這套SOP準不準」的
結論——之後隨著 signal_log.csv 每天累積，隨時重新執行就能看最新統計。

只回測 horizon=="長期"（週+日）的訊號：短期(60分/5分)要另外接 Fugle
分K歷史資料、進出場時間軸對齊也更複雜，暫不在這支腳本範圍內。

用法（在專案資料夾內執行）：
    python3 backtest_signal_log.py [FinMind_Token]

沒帶 Token 時改讀環境變數 FINMIND_TOKEN，兩者都沒有就用免費額度
（較容易撞流量限制，訊號檔股票數多的話可能會有幾檔抓取失敗）。

輸出：
    data/signal_backtest.csv：逐筆訊號 + N個交易日後的報酬率（窗口還沒
        走完的留空，不能拿今天的價格湊數，否則不同訊號的觀察期長短不
        一致，統計會失真）
    標準輸出：依「結論」分組的平均報酬/勝率統計表

讀法提醒：「買進/加碼」想看到正報酬才算對；「賣出減碼」想看到負報酬
（真的躲過下跌）才算對，這支腳本只印原始報酬統計，方向要自己對照看。
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from stock_core import get_stock_data

SIGNAL_LOG_FILE = Path(__file__).parent / "data" / "signal_log.csv"
BACKTEST_OUTPUT_FILE = Path(__file__).parent / "data" / "signal_backtest.csv"

FORWARD_WINDOWS = (5, 10, 20)  # 交易日
# 抓價格時往前多抓一點行事曆天數，確保涵蓋最早那筆訊號的日期本身
# （FinMind 只回傳交易日，這裡是行事曆天數緩衝，不是交易日數）。
_LOOKBACK_BUFFER_DAYS = 30


def _load_long_horizon_signals() -> pd.DataFrame:
    if not SIGNAL_LOG_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(SIGNAL_LOG_FILE, dtype={"code": str})
    df = df[df["horizon"] == "長期"].reset_index(drop=True)
    df["date"] = df["date"].astype(str)
    return df


def _fetch_price_series(code: str, market: str, earliest_date: str, api_token: str) -> pd.DataFrame:
    start = (pd.to_datetime(earliest_date) - timedelta(days=_LOOKBACK_BUFFER_DAYS)).strftime("%Y-%m-%d")
    end = str(date.today())
    df = get_stock_data(code, market, start, end, api_token)
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].astype(str)
    return df


def _forward_returns(price_df: pd.DataFrame, signal_date: str) -> dict:
    """在已依日期排序的 price_df 裡找 signal_date（或其後最近的交易日）
    當進場基準，回傳 {5: 報酬, 10: 報酬, 20: 報酬}；某個窗口還沒走完
    （之後交易日不夠多）就給 None。entry價一律用 price_df 當天實際收盤，
    不用 signal_log 裡記錄的 close，避免兩邊資料來源不一致。"""
    dates = price_df["date"].tolist()
    if signal_date in dates:
        base_idx = dates.index(signal_date)
    else:
        later = [d for d in dates if d >= signal_date]
        if not later:
            return {n: None for n in FORWARD_WINDOWS}
        base_idx = dates.index(later[0])

    entry_close = price_df.iloc[base_idx]["close"]
    result = {}
    for n in FORWARD_WINDOWS:
        target_idx = base_idx + n
        if target_idx >= len(price_df) or pd.isna(entry_close) or not entry_close:
            result[n] = None
            continue
        future_close = price_df.iloc[target_idx]["close"]
        result[n] = None if pd.isna(future_close) else (future_close - entry_close) / entry_close
    return result


def run_backtest(api_token: str) -> pd.DataFrame:
    signals = _load_long_horizon_signals()
    if signals.empty:
        return pd.DataFrame()

    grouped_keys = signals.groupby(["code", "market"])["date"].min()

    def fetch_one(key):
        code, market = key
        try:
            return key, _fetch_price_series(code, market, grouped_keys[key], api_token)
        except Exception as exc:  # noqa: BLE001
            return key, str(exc)

    price_cache, fetch_errors = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_one, key) for key in grouped_keys.index]
        for future in concurrent.futures.as_completed(futures):
            key, result = future.result()
            if isinstance(result, str):
                fetch_errors[key] = result
            else:
                price_cache[key] = result

    rows = []
    for _, row in signals.iterrows():
        key = (row["code"], row["market"])
        record = row.to_dict()
        price_df = price_cache.get(key)
        if price_df is None or price_df.empty:
            record["fetch_error"] = fetch_errors.get(key, "查無價格資料")
            for n in FORWARD_WINDOWS:
                record[f"fwd_ret_{n}d"] = None
            rows.append(record)
            continue
        record["fetch_error"] = None
        fwd = _forward_returns(price_df, row["date"])
        for n in FORWARD_WINDOWS:
            record[f"fwd_ret_{n}d"] = fwd[n]
        rows.append(record)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    for signal, group in df.groupby("signal"):
        row = {"結論": signal, "樣本數(去重股票)": group[["code"]].drop_duplicates().shape[0],
               "訊號筆數": len(group)}
        for n in FORWARD_WINDOWS:
            col = f"fwd_ret_{n}d"
            valid = group[col].dropna()
            row[f"{n}日_有效筆數"] = len(valid)
            row[f"{n}日_平均報酬%"] = round(valid.mean() * 100, 2) if len(valid) else None
            row[f"{n}日_正報酬勝率%"] = round((valid > 0).mean() * 100, 1) if len(valid) else None
        rows.append(row)

    order = {"加碼": 0, "買進": 1, "觀望": 2, "賣出減碼": 3}
    result = pd.DataFrame(rows)
    result["_順序"] = result["結論"].map(order)
    return result.sort_values("_順序").drop(columns="_順序").reset_index(drop=True)


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")

    detail_df = run_backtest(api_token)
    if detail_df.empty:
        print("data/signal_log.csv 裡沒有 horizon==長期 的訊號紀錄，無法回測。")
        sys.exit(0)

    BACKTEST_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(BACKTEST_OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"逐筆回測明細已存到 {BACKTEST_OUTPUT_FILE}（共 {len(detail_df)} 筆）")

    n_errors = detail_df["fetch_error"].notna().sum()
    if n_errors:
        print(f"⚠️ {n_errors} 筆股票價格資料取得失敗，已跳過（可能是Token額度或代碼問題）")

    summary_df = summarize(detail_df)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 200)
    print("\n=== 依結論分組的未來N個交易日報酬統計 ===")
    print("（買進/加碼想看到正報酬才算對；賣出減碼想看到負報酬才算對，方向要自己對照看）")
    print(summary_df.to_string(index=False))
