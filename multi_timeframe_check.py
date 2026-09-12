"""
多週期整合判讀：一次抓 日/週/60分/5分，跑完整SOP，套用跨週期整合邏輯。

用法（在你的專案資料夾內執行）：
    python3 multi_timeframe_check.py 2330 TW 你的FinMind_Token 你的Fugle_Key
"""
import sys
from datetime import date
import pandas as pd

from stock_core import get_stock_data, get_intraday_data, run_all_indicators
from sop_decision import evaluate_timeframe, combine_timeframes


def full_check(stock_id: str, market: str, api_token: str, fugle_api_key: str,
               start_date: str = "2024-01-01", include_dataframes: bool = False) -> dict:
    """
    include_dataframes=True 時，回傳結果裡會多一個「原始資料」欄位，帶各
    週期算完指標後的完整 DataFrame（給要畫K線圖的頁面用，例如
    pages/3_多週期整合分析.py）。預設 False，因為 streamlit_app.py／
    notify_email.py 這種一次跑整份清單（40幾檔）的地方只需要文字結論，
    帶著一堆DataFrame在記憶體裡跑很浪費。
    """
    verdicts = {}
    dataframes = {}

    # --- 日線 ---
    df_day = get_stock_data(stock_id, market, start_date, str(date.today()), api_token)
    if df_day.empty:
        raise ValueError(f"日線資料為空，請確認代碼 {stock_id} / 市場 {market} 是否正確")
    df_day = run_all_indicators(df_day, "日")
    verdicts["日"] = evaluate_timeframe(df_day, "日")
    dataframes["日"] = df_day

    # --- 週線（用日線 resample，不額外呼叫API）---
    week_src = df_day.copy()
    week_src["date"] = pd.to_datetime(week_src["date"])
    df_week = (
        week_src.set_index("date")
        .resample("W")
        .agg({"open": "first", "max": "max", "min": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
        .reset_index()
    )
    if market == "INDEX":
        # INDEX 的 volume 全是 NaN，resample後sum會變成0，容易誤導OBV，這裡改回NaN避免假訊號
        df_week["volume"] = pd.NA
    df_week = run_all_indicators(df_week, "週")
    verdicts["週"] = evaluate_timeframe(df_week, "週")
    dataframes["週"] = df_week

    # --- 60分 / 5分：只有台股個股/ETF支援（跟 streamlit_app.py 的限制一致），
    #     且需要 Fugle Key；沒填或抓取失敗都不能讓整個分析掛掉，略過即可。
    if market == "TW" and fugle_api_key:
        try:
            df_60 = get_intraday_data(stock_id, "60", fugle_api_key)
            if not df_60.empty:
                df_60 = run_all_indicators(df_60, "60分")
                verdicts["60分"] = evaluate_timeframe(df_60, "60分")
                dataframes["60分"] = df_60
        except Exception:
            pass  # 60分資料抓取失敗，本次判讀略過該週期

        try:
            df_5 = get_intraday_data(stock_id, "5", fugle_api_key)
            if not df_5.empty:
                df_5 = run_all_indicators(df_5, "5分")
                verdicts["5分"] = evaluate_timeframe(df_5, "5分")
                dataframes["5分"] = df_5
        except Exception:
            pass  # 5分資料抓取失敗，本次判讀略過該週期

    result = combine_timeframes(verdicts)
    latest_close = df_day.iloc[-1]["close"]
    result["收盤"] = float(latest_close) if pd.notna(latest_close) else None
    if include_dataframes:
        result["原始資料"] = dataframes
    return result


def print_result(result: dict) -> None:
    print("=" * 50)
    print("最終建議：", result["最終建議"])
    print("=" * 50)
    for tf, d in result["各週期明細"].items():
        print(f"[{tf}] {d['結論']}（信心：{d['信心']}）")
        for r in d["理由"]:
            print(f"   - {r}")
        for c in d["但書"]:
            print(f"   ⚠️ {c}")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("用法: python3 multi_timeframe_check.py <股票代碼> <市場TW/INDEX/US> <FinMind Token> <Fugle Key>")
        sys.exit(1)
    stock_id, market, api_token, fugle_api_key = sys.argv[1:5]
    result = full_check(stock_id, market, api_token, fugle_api_key)
    print_result(result)
