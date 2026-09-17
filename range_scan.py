"""
平台區間（箱型整理）掃描：抓自選股清單日線，找出目前波動收斂、
在一個固定價格帶內來回震盪的股票。

判斷邏輯（近 window 根日K）：
    區間高／低 = 這段期間的最高價／最低價
    波動幅度%  = (區間高 - 區間低) / 區間低
    觸頂／觸底次數 = 有幾根K棒的高/低價落在區間邊緣 15% 帶內
                    （避免只有一根長影線就誤判成箱型）
    量縮比      = 近window天均量 / 前window天均量（<1代表量縮，籌碼沉澱中）
    是否平台    = 波動幅度% ≤ 門檻，且上下緣都至少被觸碰2次以上

用法（在專案資料夾內執行）：
    python3 range_scan.py [FinMind_Token] [window天數=20] [波動門檻=0.10]

沒有帶 Token 時會改讀環境變數 FINMIND_TOKEN，兩者都沒有就用免費額度
（較容易撞流量限制）。
"""
import concurrent.futures
import os
import sys
from datetime import date, timedelta

import pandas as pd

from stock_core import get_stock_data, unique_watchlist

# 抓寬鬆一點的行事曆天數，確保扣掉假日後仍有 window*2 根以上的日K
# （拿來算「量縮比」需要前後各一個window的資料）。
_LOOKBACK_CALENDAR_DAYS = 150
# 判斷「有沒有真的碰到邊緣」的容忍帶：區間寬度的15%以內都算觸碰。
_TOUCH_BAND_RATIO = 0.15
_MIN_TOUCHES = 2


def detect_platform_range(df: pd.DataFrame, window: int = 20, range_threshold: float = 0.10) -> dict:
    recent = df.tail(window)
    range_high = float(recent["max"].max())
    range_low = float(recent["min"].min())
    range_pct = (range_high - range_low) / range_low if range_low else float("inf")

    band = (range_high - range_low) * _TOUCH_BAND_RATIO
    touches_high = int((recent["max"] >= range_high - band).sum())
    touches_low = int((recent["min"] <= range_low + band).sum())

    latest_close = float(df.iloc[-1]["close"])
    position_pct = (
        (latest_close - range_low) / (range_high - range_low)
        if range_high > range_low else 0.5
    )

    prior = df.iloc[-(window * 2):-window] if len(df) >= window * 2 else pd.DataFrame()
    vol_ratio = None
    if not prior.empty and prior["volume"].notna().any() and recent["volume"].notna().any():
        prior_avg = prior["volume"].mean()
        if prior_avg:
            vol_ratio = recent["volume"].mean() / prior_avg

    is_platform = (
        range_pct <= range_threshold
        and touches_high >= _MIN_TOUCHES
        and touches_low >= _MIN_TOUCHES
    )

    return {
        "收盤": round(latest_close, 2),
        "區間高": round(range_high, 2),
        "區間低": round(range_low, 2),
        "波動幅度%": round(range_pct * 100, 2),
        "目前位置%": round(position_pct * 100, 1),
        "觸頂次數": touches_high,
        "觸底次數": touches_low,
        "量縮比": round(vol_ratio, 2) if vol_ratio is not None else None,
        "是否平台": is_platform,
    }


def scan_watchlist(api_token: str, window: int, range_threshold: float) -> tuple[pd.DataFrame, list]:
    start_date = str(date.today() - timedelta(days=_LOOKBACK_CALENDAR_DAYS))
    end_date = str(date.today())

    def analyze_one(name: str, code: str, market: str) -> dict:
        try:
            df = get_stock_data(code, market, start_date, end_date, api_token)
            if df.empty or len(df) < window + 5:
                return {"名稱": name, "代碼": code, "狀態": "error", "訊息": "資料不足或查無資料"}
            result = detect_platform_range(df, window=window, range_threshold=range_threshold)
            result.update({"名稱": name, "代碼": code, "狀態": "ok"})
            return result
        except Exception as exc:  # noqa: BLE001
            return {"名稱": name, "代碼": code, "狀態": "error", "訊息": str(exc)}

    rows, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(analyze_one, name, code, market)
            for name, code, market in unique_watchlist()
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            (rows if result["狀態"] == "ok" else errors).append(result)

    df_result = pd.DataFrame(rows)
    if not df_result.empty:
        df_result = df_result.sort_values(["是否平台", "波動幅度%"], ascending=[False, True])
        cols = ["名稱", "代碼", "是否平台", "波動幅度%", "目前位置%",
                "收盤", "區間高", "區間低", "觸頂次數", "觸底次數", "量縮比"]
        df_result = df_result[cols].reset_index(drop=True)
    return df_result, errors


if __name__ == "__main__":
    token_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    api_token = token_arg or os.environ.get("FINMIND_TOKEN", "")
    window_days = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10

    df_result, errors = scan_watchlist(api_token, window_days, threshold)

    if df_result.empty:
        print("所有股票都取得失敗，請確認 FinMind Token 是否正確。")
    else:
        pd.set_option("display.unicode.east_asian_width", True)
        pd.set_option("display.width", 160)
        platform_df = df_result[df_result["是否平台"]]
        print(f"\n=== 近{window_days}日符合平台區間（波動幅度 ≤ {threshold * 100:.0f}%，上下緣各觸碰≥{_MIN_TOUCHES}次）===")
        print(platform_df.to_string(index=False) if not platform_df.empty else "（目前沒有股票符合）")
        print(f"\n=== 全部清單（依波動幅度排序，僅供對照）===")
        print(df_result.to_string(index=False))

    if errors:
        print(f"\n⚠️ {len(errors)} 檔資料取得失敗：")
        for e in errors:
            print(f"- {e['名稱']}（{e['代碼']}）：{e.get('訊息', '未知錯誤')}")
