"""
資料存取層：日線價格（未還原／還原權息）＋流動性排行名單。

- 未還原（接近券商App口徑）：直接沿用 stock_core.get_stock_data()（FinMind
  TaiwanStockPrice），跟主站判讀用的是同一份資料來源/口徑。
- 還原權息：FinMind 的 TaiwanStockPriceAdj（官方還原股價表）官網文件註明
  「只限 Backer/Sponsor 付費方案」，免費額度/一般token呼叫會失敗
  （實測回400：'Your level is free...'）。改用免費可查的
  TaiwanStockDividendResult（除權除息結果表，TWSE/TPEx官方公告的
  before_price/after_price）自行往前還原：對每一次除權息事件，
  ex_date當天及之後的收盤價維持原值，ex_date以前的價格全部乘上
  ratio=after_price/before_price（多次事件連乘），數學上等同標準的
  「以目前股價為基準」還原法。
  限制：TaiwanStockDividendResult只涵蓋除權除息，不含減資
  （capital reduction）——若回測期間個股有減資，減資當天會有一個資料
  本身就存在、但這支程式不會還原的價格斷層，呼叫端用
  detect_price_gaps()抓出來，寫進報告的限制欄位。
- 流動性前100名：FinMind免費額度查不到「不指定data_id」的全市場批次資料
  （實測回400 'Your level is free'），改用本專案institutional_ranking.py
  同款的作法——直接打TWSE／TPEx官方公開API拿全市場當天量值（免金鑰）。

本模組對外部API呼叫一律落地快取成本機CSV（backtest/cache/），重跑分析
不用重打API；快取不進版控（.gitignore: backtest/cache/）。
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from stock_core import fetch_finmind, get_stock_data

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TWSE_MI_INDEX_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
TPEX_OTC_QUOTES_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes"


# ------------------------------------------------------------------
# 未還原日線（券商App口徑，供指標校驗、以及「校驗用」收盤比對）
# ------------------------------------------------------------------
def fetch_raw_daily(code: str, market: str, start_date: str, end_date: str,
                     api_token: str = "") -> pd.DataFrame:
    cache_file = CACHE_DIR / f"raw_{code}_{start_date}_{end_date}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, dtype={"stock_id": str})
    else:
        df = get_stock_data(code, market, start_date, end_date, api_token)
        df.to_csv(cache_file, index=False)
    # FinMind偶爾會出現整列open/max/min/close/volume全是0的壞資料（實測
    # 2317/3491/5347/2449/3665/7792/8042都各有1~2天，比對券商/交易所都是
    # 正常交易日，判斷是資料源那邊的單日異常，不是真的零股價）。這種壞列
    # 會污染MA/EMA/KD等所有指標計算，一律先剔除。
    if "close" in df.columns and not df.empty:
        bad = df["close"] <= 0
        if bad.any():
            df = df[~bad].reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# 除權除息事件（還原用）
# ------------------------------------------------------------------
def fetch_dividend_events(code: str, start_date: str, end_date: str,
                           api_token: str = "") -> pd.DataFrame:
    """回傳 [date, ratio]，ratio = after_price/before_price（<1）。
    date 往前（不含當天）的價格都要乘上這個ratio。"""
    cache_file = CACHE_DIR / f"div_{code}_{start_date}_{end_date}.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        raw = fetch_finmind("TaiwanStockDividendResult", code, start_date, end_date, api_token)
        if raw.empty:
            df = pd.DataFrame(columns=["date", "ratio"])
        else:
            df = pd.DataFrame({
                "date": raw["date"],
                "ratio": raw["after_price"] / raw["before_price"],
            })
        df.to_csv(cache_file, index=False)
    return df


def apply_back_adjustment(df_raw: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """對 open/max/min/close 做「向前還原」：事件日之前（不含當天）的價格
    全部乘上事件 ratio，多個事件連乘。不改 volume（還原股價不等比例還原
    成交量，股數型還原超出本次需求）。"""
    df = df_raw.copy()
    if df.empty:
        return df
    dates = pd.to_datetime(df["date"])
    factor = pd.Series(1.0, index=df.index)
    for _, ev in events.iterrows():
        ev_date = pd.Timestamp(ev["date"])
        mask = dates < ev_date
        factor.loc[mask] *= ev["ratio"]
    for col in ("open", "max", "min", "close"):
        if col in df.columns:
            df[col] = df[col] * factor
    df["adj_factor"] = factor
    return df


def fetch_adjusted_daily(code: str, market: str, start_date: str, end_date: str,
                          api_token: str = "") -> pd.DataFrame:
    """market!="TW" (指數/美股) 沒有除權息事件，直接回傳未還原資料。"""
    df_raw = fetch_raw_daily(code, market, start_date, end_date, api_token)
    if df_raw.empty or market != "TW":
        return df_raw
    events = fetch_dividend_events(code, start_date, end_date, api_token)
    return apply_back_adjustment(df_raw, events)


def detect_price_gaps(df_adj: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
    """粗略抓「還原後仍有單日跳空」的日子（例如減資造成、但
    TaiwanStockDividendResult沒有涵蓋的價格斷層），回傳供報告列限制用，
    不影響回測本身的訊號判斷。"""
    df = df_adj.copy()
    pct = df["close"].pct_change()
    flagged = df.loc[pct.abs() >= threshold, ["date", "close"]].copy()
    flagged["pct_change"] = pct[pct.abs() >= threshold]
    return flagged


# ------------------------------------------------------------------
# 流動性排行名單：TWSE（上市，近5個交易日成交金額均值）＋
# TPEx（上櫃，官方API只回傳最新一個交易日，無歷史查詢參數）
# ------------------------------------------------------------------
def _is_ordinary_stock_code(code: str) -> bool:
    """只留一般個股（4碼純數字、非00開頭的ETF/ETN、非含字母的特別股）。"""
    return code.isdigit() and len(code) == 4 and not code.startswith("00")


def fetch_twse_daily_turnover(trade_date: str) -> Dict[str, Tuple[float, str]]:
    """trade_date: 'YYYYMMDD'。回傳 {code: (成交金額, 證券名稱)}，非交易日回傳空dict。"""
    cache_file = CACHE_DIR / f"twse_turnover_{trade_date}.csv"
    if cache_file.exists():
        cached = pd.read_csv(cache_file, dtype={"code": str, "name": str})
        return {c: (t, n) for c, t, n in zip(cached["code"], cached["turnover"], cached["name"])}

    try:
        resp = requests.get(TWSE_MI_INDEX_URL,
                             params={"response": "json", "date": trade_date, "type": "ALLBUT0999"},
                             timeout=40)
        payload = resp.json()
    except Exception:
        return {}
    if payload.get("stat") != "OK":
        return {}

    result = {}
    for table in payload.get("tables", []):
        fields = table.get("fields") or []
        if "證券代號" in fields and "成交金額" in fields:
            idx_code = fields.index("證券代號")
            idx_name = fields.index("證券名稱")
            idx_amt = fields.index("成交金額")
            for row in table.get("data", []):
                code = row[idx_code]
                if not _is_ordinary_stock_code(code):
                    continue
                try:
                    result[code] = (float(row[idx_amt].replace(",", "")), row[idx_name])
                except (ValueError, AttributeError):
                    continue
            break

    pd.DataFrame([{"code": c, "turnover": t, "name": n} for c, (t, n) in result.items()]) \
        .to_csv(cache_file, index=False)
    return result


def fetch_tpex_latest_turnover() -> Dict[str, Tuple[float, str]]:
    """上櫃：官方open API只提供「最新一個交易日」快照，沒有歷史日期參數
    （跟institutional_ranking.py註記的限制一樣），只能抓到呼叫當下那一天。
    回傳 {code: (成交金額, 證券名稱)}。"""
    cache_file = CACHE_DIR / "tpex_turnover_latest.csv"
    if cache_file.exists():
        cached = pd.read_csv(cache_file, dtype={"code": str, "name": str})
        return {c: (t, n) for c, t, n in zip(cached["code"], cached["turnover"], cached["name"])}

    try:
        resp = requests.get(TPEX_OTC_QUOTES_URL, timeout=40)
        data = resp.json()
    except Exception:
        return {}

    result = {}
    for row in data:
        code = row.get("SecuritiesCompanyCode", "")
        if not _is_ordinary_stock_code(code):
            continue
        try:
            result[code] = (float(row.get("TransactionAmount", "0").replace(",", "")),
                             row.get("CompanyName", ""))
        except (ValueError, AttributeError):
            continue

    pd.DataFrame([{"code": c, "turnover": t, "name": n} for c, (t, n) in result.items()]) \
        .to_csv(cache_file, index=False)
    return result


def _recent_trading_dates(n: int, as_of: Optional[str] = None) -> List[str]:
    """往回找最近n個「可能是交易日」的西曆日期（週一~週五），實際是否真的
    有成交由fetch_twse_daily_turnover()依stat=='OK'過濾。多抓一些緩衝
    （國定假日）。回傳'YYYYMMDD'字串，新到舊。"""
    from datetime import date as date_cls
    from datetime import timedelta
    end = date_cls.fromisoformat(as_of) if as_of else date_cls.today()
    out = []
    d = end
    while len(out) < n + 5 and (end - d).days < 30:
        if d.weekday() < 5:
            out.append(d.strftime("%Y%m%d"))
        d -= timedelta(days=1)
    return out


def build_liquidity_universe(top_n: int = 100, lookback_days: int = 5,
                              as_of: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """回傳流動性前top_n名的 [(code, market, name), ...]，market固定"TW"。
    方法：TWSE抓近lookback_days個交易日的成交金額均值；TPEx只有最新一天的
    快照可用，兩邊各自算完後合併排序取前N（混合口徑的限制見模組docstring／
    報告限制章節）。"""
    candidate_dates = _recent_trading_dates(lookback_days, as_of)
    twse_turnover: Dict[str, float] = {}
    twse_names: Dict[str, str] = {}
    used = 0
    turnover_sums: Dict[str, float] = {}
    turnover_counts: Dict[str, int] = {}
    for d in candidate_dates:
        day_data = fetch_twse_daily_turnover(d)
        if day_data:
            for code, (amt, name) in day_data.items():
                turnover_sums[code] = turnover_sums.get(code, 0.0) + amt
                turnover_counts[code] = turnover_counts.get(code, 0) + 1
                twse_names[code] = name
            used += 1
        if used >= lookback_days:
            break
        time.sleep(0.3)
    twse_avg = pd.Series({c: turnover_sums[c] / turnover_counts[c] for c in turnover_sums})

    tpex_data = fetch_tpex_latest_turnover()
    tpex_turnover = pd.Series({c: v[0] for c, v in tpex_data.items()})
    tpex_names = {c: v[1] for c, v in tpex_data.items()}

    names = {**twse_names, **tpex_names}
    combined = pd.concat([twse_avg, tpex_turnover]).sort_values(ascending=False)
    combined = combined[~combined.index.duplicated(keep="first")]
    top_codes = combined.head(top_n).index.tolist()
    return [(c, "TW", names.get(c, c)) for c in top_codes]
