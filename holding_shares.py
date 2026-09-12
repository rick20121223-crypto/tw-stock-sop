"""
股權分散表（大股東持股比例）趨勢訊號。

資料來源：TDCC（臺灣集中保管結算所）官方 OpenAPI「集保戶股權分散表」
（https://openapi.tdcc.com.tw/v1/opendata/1-5），免金鑰、完全公開。
跟原本考慮過的 FinMind TaiwanStockHoldingSharesPer 是同一份原始資料，
但 TDCC 直接查是免費的，不用 FinMind 付費 Backer/Sponsor 等級。

**限制**：這個API只回傳「最新一週」全市場快照（一次約4千多檔證券、
9~10MB），沒有查歷史日期的參數。所以歷史資料要靠 archive_tdcc_snapshot()
每週執行一次、自己把「這週」的結果append進本地歷史檔
（data/tdcc_holding_major_holder.csv），累積出可以算「連續N週」的歷史。
剛上線、或某檔股票剛被加進觀察清單時，本地歷史可能是空的或只有
幾週，要等排程多跑幾週才會有完整趨勢，這是正常現象，不是bug。

TDCC「持股分級」欄位是數字代碼（1~17），不是級距文字，代碼對應的股數
區間是官方公開的標準15級距分類（見下面 MAJOR_HOLDER_LEVELS 的說明），
已經用台積電的真實資料驗證過。

三個主要函式：
    archive_tdcc_snapshot()     每週執行：抓最新一週快照、篩大股東級距、
                                 依股票加總、append進本地歷史檔
    fetch_major_holder_trend()  從本地歷史檔讀出某檔股票的週趨勢
    compute_consecutive_signals()  算週對週diff/方向/連續N週同向訊號
    align_to_trading_days()     週頻資料對齊到K線的最近一個交易日
"""

from datetime import date as date_cls
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

TDCC_HOLDING_URL = "https://openapi.tdcc.com.tw/v1/opendata/1-5"

# TDCC官方標準15級距分類（資料本身只給數字代碼，級距文字是官方公開的
# 固定定義，已用台積電真實資料交叉驗證過）：
#   1:1-999  2:1,000-5,000  3:5,001-10,000  4:10,001-15,000
#   5:15,001-20,000  6:20,001-30,000  7:30,001-40,000  8:40,001-50,000
#   9:50,001-100,000  10:100,001-200,000  11:200,001-400,000
#   12:400,001-600,000  13:600,001-800,000  14:800,001-1,000,000
#   15:1,000,001以上
#   16:差異數調整（通常是0，用來讓加總對得上合計列）  17:合計（驗證用）
# 「大股東(>400張)」= >400,000股，第11級距上限剛好400,000股不算，
# 從第12級距（400,001股）開始才算，所以是 12~15 加總。
MAJOR_HOLDER_LEVELS = {12, 13, 14, 15}

DATA_DIR = Path(__file__).parent / "data"
TDCC_ARCHIVE_FILE = DATA_DIR / "tdcc_holding_major_holder.csv"
RETENTION_WEEKS = 60  # 保留約14個月歷史，跟「連續N週」的實用需求打平衡，避免檔案無限long


def _strip_bom(key: str) -> str:
    """TDCC回傳的JSON，第一個欄位鍵名會帶UTF-8 BOM字元（實測是
    "\\ufeff資料日期"），直接用"資料日期"當key會抓不到、丟KeyError，
    要先把每一列的key都normalize掉BOM。"""
    return key.lstrip("﻿")


def fetch_tdcc_snapshot() -> pd.DataFrame:
    """
    抓TDCC集保戶股權分散表「最新一週」全市場快照。
    回傳欄位：date（該週資料日期）、stock_id、level（1~17的級距代碼）、
    people、unit（股數）、percent（占集保庫存比例%）。
    """
    resp = requests.get(TDCC_HOLDING_URL, timeout=60)
    resp.raise_for_status()
    raw = resp.json()
    if not raw:
        return pd.DataFrame(columns=["date", "stock_id", "level", "people", "unit", "percent"])

    rows = []
    for row in raw:
        normalized = {_strip_bom(k): v for k, v in row.items()}
        try:
            rows.append({
                "date": normalized["資料日期"],
                "stock_id": str(normalized["證券代號"]).strip(),
                "level": int(normalized["持股分級"]),
                "people": int(normalized["人數"]),
                "unit": int(normalized["股數"]),
                "percent": float(normalized["占集保庫存數比例%"]),
            })
        except (KeyError, ValueError, TypeError):
            continue  # 個別列格式異常就跳過，不讓整批資料因為一列壞掉而掛掉

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"])
    return df


def _aggregate_major_holders(snapshot: pd.DataFrame) -> pd.DataFrame:
    """把全市場快照篩到>400張的級距(12~15)，依股票代碼加總，回傳
    date/stock_id/percent/people/unit（每檔股票一列）。"""
    if snapshot.empty:
        return pd.DataFrame(columns=["date", "stock_id", "percent", "people", "unit"])
    major = snapshot[snapshot["level"].isin(MAJOR_HOLDER_LEVELS)]
    return (
        major.groupby(["date", "stock_id"], as_index=False)
        .agg(percent=("percent", "sum"), people=("people", "sum"), unit=("unit", "sum"))
    )


def archive_tdcc_snapshot() -> Optional[str]:
    """
    抓最新一週的TDCC股權分散表、篩>400張級距、依股票加總，append進本地
    歷史檔（同一個資料日期只會存一次，重複執行不會重複累加）。
    回傳這次存檔的資料日期字串（YYYY-MM-DD），沒有新資料、抓取失敗、
    或這週已經存過了，都回傳 None。
    """
    snapshot = fetch_tdcc_snapshot()
    if snapshot.empty:
        return None

    aggregated = _aggregate_major_holders(snapshot)
    if aggregated.empty:
        return None

    new_date = aggregated["date"].iloc[0]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TDCC_ARCHIVE_FILE.exists():
        existing = pd.read_csv(TDCC_ARCHIVE_FILE, parse_dates=["date"])
        if (existing["date"] == new_date).any():
            return None  # 這週的資料已經存過了，不重複寫入
        combined = pd.concat([existing, aggregated], ignore_index=True)
    else:
        combined = aggregated

    cutoff = pd.Timestamp(date_cls.today()) - pd.Timedelta(weeks=RETENTION_WEEKS)
    combined = combined[combined["date"] >= cutoff]
    combined = combined.sort_values(["date", "stock_id"]).reset_index(drop=True)
    combined.to_csv(TDCC_ARCHIVE_FILE, index=False)
    return new_date.strftime("%Y-%m-%d")


def fetch_major_holder_trend(stock_id: str) -> pd.DataFrame:
    """
    回傳「大股東（>400張）合計持股比例」週趨勢，從本地歷史檔讀取（不再
    即時呼叫遠端API），欄位：date、percent、people、unit。歷史是靠每週
    排程執行 archive_tdcc_snapshot() 累積的，資料不足（檔案不存在、或
    這檔股票還沒被存過幾週）就回傳空/筆數少的 DataFrame，呼叫端據此
    判斷「還在累積中」即可，不用特別接例外。
    """
    if not TDCC_ARCHIVE_FILE.exists():
        return pd.DataFrame(columns=["date", "percent", "people", "unit"])

    history = pd.read_csv(TDCC_ARCHIVE_FILE, parse_dates=["date"])
    history["stock_id"] = history["stock_id"].astype(str)
    matched = history[history["stock_id"] == str(stock_id).strip()]
    return (
        matched[["date", "percent", "people", "unit"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


def compute_consecutive_signals(df_holding: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    在 df_holding（至少要有 date、percent 欄位）上新增：
        diff        本週 percent 減上週 percent
        direction   "上升"／"下降"／"持平"／"—"（第一週，無上週可比較）
        run_length  目前這個方向已經連續幾週（含當週），方向中斷就歸零
        signal      "" 或 "籌碼連續集中"／"籌碼連續分散"

    訊號判定：連續N週同方向「第一次達到N週」的那一週標記一次（例如
    N=3，連續上升5週，只有第3週會標記，第4、5週不會重複標記同一段
    趨勢），避免同一段連續趨勢被標好幾次。
    """
    df = df_holding.sort_values("date").reset_index(drop=True).copy()
    df["diff"] = df["percent"].diff()

    def _direction(d):
        if pd.isna(d):
            return "—"
        if d > 0:
            return "上升"
        if d < 0:
            return "下降"
        return "持平"

    df["direction"] = df["diff"].apply(_direction)

    run_lengths = []
    current_run = 0
    prev_dir = None
    for d in df["direction"]:
        if d in ("上升", "下降") and d == prev_dir:
            current_run += 1
        elif d in ("上升", "下降"):
            current_run = 1
        else:
            current_run = 0
        run_lengths.append(current_run)
        prev_dir = d if d in ("上升", "下降") else None
    df["run_length"] = run_lengths

    df["signal"] = ""
    df.loc[(df["direction"] == "上升") & (df["run_length"] == n), "signal"] = "籌碼連續集中"
    df.loc[(df["direction"] == "下降") & (df["run_length"] == n), "signal"] = "籌碼連續分散"

    return df


def align_to_trading_days(df_weekly: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    把週頻資料（df_weekly 的 date 欄）對應到 daily_df（K線資料，含 date/
    max/min 欄）「當週最後一個交易日」：用 merge_asof 往回找 <= 週資料
    日期的最近一個交易日，並順便帶出當天的 high/low 供畫圖時決定箭頭要
    標在K棒上方還是下方。太早期、daily_df 還沒開始的週資料會被捨棄
    （merge_asof 找不到符合的交易日，trade_date 會是 NaT）。
    新增欄位：trade_date（對齊後的交易日）、high、low。
    """
    weekly = df_weekly.copy()
    weekly["date"] = pd.to_datetime(weekly["date"])
    weekly = weekly.sort_values("date")

    trading = daily_df[["date", "max", "min"]].copy()
    trading["date"] = pd.to_datetime(trading["date"])
    trading = trading.sort_values("date").rename(columns={"date": "trade_date", "max": "high", "min": "low"})

    merged = pd.merge_asof(weekly, trading, left_on="date", right_on="trade_date", direction="backward")
    merged = merged.dropna(subset=["trade_date"]).reset_index(drop=True)
    return merged


if __name__ == "__main__":
    saved_date = archive_tdcc_snapshot()
    print(f"已存TDCC股權分散表快照：{saved_date}" if saved_date else "沒有新的一週資料（可能這週已經存過，或抓取失敗）")
