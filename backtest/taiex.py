"""
加權指數（TAIEX，發行量加權股價指數，價格指數版，不是含息的報酬指數）每日
收盤。改用TWSE官方MI_INDEX bulk端點逐日取「價格指數(臺灣證券交易所)」表裡
的「發行量加權股價指數」收盤值——這個端點之前因為一次性打了太多併發
請求觸發過bot防護(HTTP 428)，這次改成**序列化+固定節流間隔**（不並行、
每次真正打API之間至少間隔SLEEP_SEC秒），比第一輪的併發嘗試慢，但目前
看起來是唯一真正穩定可用的做法。落地快取（backtest/cache/taiex_<date>.csv），
重跑不用重打。
"""

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from backtest.data_fetch import TWSE_MI_INDEX_URL

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_SEC = 0.6  # 節流間隔，上次用併發打爆過這個端點(HTTP 428)，這次刻意序列化放慢


def fetch_taiex_one_day(trade_date: str) -> Optional[float]:
    """trade_date: 'YYYYMMDD'。回傳當天發行量加權股價指數收盤值，非交易日
    回傳None。"""
    cache_file = CACHE_DIR / f"taiex_{trade_date}.csv"
    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        if cached.empty:
            return None
        return float(cached["close"].iloc[0])

    try:
        resp = requests.get(TWSE_MI_INDEX_URL,
                             params={"response": "json", "date": trade_date, "type": "ALLBUT0999"},
                             timeout=20)
        payload = resp.json()
    except Exception:
        return None  # 不落地快取，讓之後可以重試
    if payload.get("stat") != "OK":
        pd.DataFrame(columns=["close"]).to_csv(cache_file, index=False)
        return None

    close = None
    for table in payload.get("tables", []):
        fields = table.get("fields") or []
        if "指數" in fields and "收盤指數" in fields:
            idx_name = fields.index("指數")
            idx_close = fields.index("收盤指數")
            for row in table.get("data", []):
                if row[idx_name] == "發行量加權股價指數":
                    try:
                        close = float(row[idx_close].replace(",", ""))
                    except (ValueError, AttributeError):
                        close = None
                    break
            break

    if close is None:
        pd.DataFrame(columns=["close"]).to_csv(cache_file, index=False)
        return None
    pd.DataFrame({"close": [close]}).to_csv(cache_file, index=False)
    return close


def fetch_taiex_history(start_date: str, end_date: str, verbose: bool = True) -> pd.DataFrame:
    """回傳[date, close]，只含真正的交易日。序列化逐日呼叫＋固定節流，
    不用併發（上次併發打爆TWSE的bot防護，這次刻意避免重蹈覆轍）。"""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    all_dates = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            all_dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)

    results = {}
    for i, ds in enumerate(all_dates):
        was_cached = (CACHE_DIR / f"taiex_{ds}.csv").exists()
        close = fetch_taiex_one_day(ds)
        if close is not None:
            results[ds] = close
        if not was_cached:
            time.sleep(SLEEP_SEC)
        if verbose and (i + 1) % 50 == 0:
            print(f"TAIEX抓取進度 {i + 1}/{len(all_dates)}（目前成功 {len(results)} 天）", flush=True)

    rows = [{"date": f"{ds[:4]}-{ds[4:6]}-{ds[6:]}", "close": c} for ds, c in results.items()]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df
