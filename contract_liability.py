"""
合約負債（先收款）趨勢訊號。

合約負債是IFRS15準則下的科目，白話講就是「客戶已經付款/下訂單，但公司
還沒認列收入」，金額連續上升通常代表在手訂單或預收款增加，可以當作
營收成長的領先指標之一（不是所有產業都適用，例如純代工、客戶付款模式
不同的公司，合約負債佔比可能一直很小、不具參考性）。

資料來源：FinMind TaiwanStockBalanceSheet（跟 stock_core.fetch_finmind()
用同一個FinMind API，不需要另外的金鑰）。財報是季更新（Q1~Q4公布時間
都在季底後1~2個月），FinMind一次查詢就會回傳它保留的完整季度歷史，
不像TDCC股權分散表那樣「只給最新一週」，所以不需要像holding_shares.py
那樣自己每週archive。

FinMind的type欄位固定用 CurrentContractLiabilities，少數大型工程/建案
類公司才會另外報 NoncurrentContractLiabilities，這裡兩者都抓、有的話
就加總。沒有揭露這個科目的公司（沒有先收款商業模式、或本來就沒有這條
科目）會回傳空結果，呼叫端據此判斷「不適用」，不是錯誤。
"""

from typing import Optional

import pandas as pd

from stock_core import fetch_finmind

CONTRACT_LIABILITY_TYPES = ["CurrentContractLiabilities", "NoncurrentContractLiabilities"]


def fetch_contract_liability_trend(stock_id: str, token: str = "") -> pd.DataFrame:
    """
    回傳某檔股票的合約負債季度趨勢，欄位：date（季底日期）、value（合約
    負債金額，流動+非流動加總，單位:元）。FinMind一次查詢就給全部保留
    的歷史，不需要本地累積。抓不到資料（代碼錯誤、沒有這個科目、非台股
    等）就回傳空 DataFrame，呼叫端據此判斷「不適用」即可，不用特別接
    例外。
    """
    df = fetch_finmind(
        "TaiwanStockBalanceSheet", stock_id,
        "2015-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"), token,
    )
    if df.empty or "type" not in df.columns:
        return pd.DataFrame(columns=["date", "value"])

    matched = df[df["type"].isin(CONTRACT_LIABILITY_TYPES)]
    if matched.empty:
        return pd.DataFrame(columns=["date", "value"])

    grouped = matched.groupby("date", as_index=False)["value"].sum()
    grouped["date"] = pd.to_datetime(grouped["date"])
    return grouped.sort_values("date").reset_index(drop=True)


def compute_latest_change(df_trend: pd.DataFrame) -> Optional[dict]:
    """
    回傳最新一季 vs 上一季的合約負債變化：
        {date, value, prev_value, diff, pct_change, direction}
    不足兩季資料（剛上市、或本來就沒幾季揭露這個科目）就回傳 None。
    """
    if len(df_trend) < 2:
        return None
    latest = df_trend.iloc[-1]
    prev = df_trend.iloc[-2]
    diff = latest["value"] - prev["value"]
    pct_change = (diff / prev["value"] * 100) if prev["value"] else None
    direction = "上升" if diff > 0 else ("下降" if diff < 0 else "持平")
    return {
        "date": latest["date"].strftime("%Y-%m-%d"),
        "value": float(latest["value"]),
        "prev_value": float(prev["value"]),
        "diff": float(diff),
        "pct_change": None if pct_change is None else round(float(pct_change), 2),
        "direction": direction,
    }
