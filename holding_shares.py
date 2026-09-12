"""
股權分散表（大股東持股比例）趨勢訊號。

資料來源：FinMind「TaiwanStockHoldingSharesPer」（股權持股分級表），
週頻資料，欄位是 date / stock_id / HoldingSharesLevel / people / percent
/ unit。**這個資料集只有 FinMind Backer/Sponsor 付費會員等級能查**，免費
／一般註冊會員查詢會被 FinMind 擋掉（fetch_finmind 會丟 RuntimeError，
訊息是 FinMind 本身回的「Your level is free...」），呼叫端要自己接住這個
例外，顯示成友善提示，不要讓整頁掛掉。

HoldingSharesLevel 是「股數」區間字串（不是「張」），例如 "1-999"、
"400001-600000"、"1000001以上"，所以「大股東（>400張）」要自己把
>400,000股（下限 > 400,000）的級距挑出來、把 percent 加總，FinMind
不會直接給你一個叫「>400張」的欄位值。

三個主要函式：
    fetch_major_holder_trend()  抓資料＋篩級距＋依週加總 -> 大股東合計比例
    compute_consecutive_signals()  算週對週diff/方向/連續N週同向訊號
    align_to_trading_days()  週頻資料對齊到K線的最近一個交易日
"""

import re
from typing import Optional

import pandas as pd

from stock_core import fetch_finmind

MAJOR_HOLDER_THRESHOLD_SHARES = 400_000  # 大股東門檻：>400張 = >400,000股


def _holding_level_lower_bound(level: str) -> Optional[int]:
    """從 HoldingSharesLevel 級距字串解析下限股數。常見格式："1-999"、
    "400001-600000"、"1000001以上"，開頭都是數字，解析失敗回傳 None
    （該列會被視為無法判斷級距、直接排除，不會誤算進大股東比例）。"""
    if not level:
        return None
    match = re.match(r"\s*(\d+)", str(level).replace(",", ""))
    return int(match.group(1)) if match else None


def fetch_major_holder_trend(
    stock_id: str, start_date: str, end_date: str, token: str,
    threshold_shares: int = MAJOR_HOLDER_THRESHOLD_SHARES,
) -> pd.DataFrame:
    """
    回傳「大股東（門檻以上，預設>400張）合計持股比例」週趨勢，欄位：
    date、percent（合計比例%）、people（合計人數）、unit（合計股數）。
    查詢失敗（含免費會員被FinMind擋掉）會直接讓 RuntimeError 往外拋，
    由呼叫端（UI層）決定怎麼顯示，這裡不吞例外。
    """
    raw = fetch_finmind("TaiwanStockHoldingSharesPer", stock_id, start_date, end_date, token)
    if raw.empty:
        return raw

    raw = raw.copy()
    for col in ("percent", "people", "unit"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw["_lower"] = raw["HoldingSharesLevel"].apply(_holding_level_lower_bound)
    major = raw[raw["_lower"].notna() & (raw["_lower"] > threshold_shares)]
    if major.empty:
        return pd.DataFrame(columns=["date", "percent", "people", "unit"])

    grouped = (
        major.groupby("date", as_index=False)
        .agg(percent=("percent", "sum"), people=("people", "sum"), unit=("unit", "sum"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    return grouped


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
