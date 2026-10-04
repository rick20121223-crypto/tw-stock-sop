"""
探測Fugle歷史分K(60分/5分)實際能回溯多久——這是決定CCI、60分120MA/240MA、
5分20MA+BBI+MACD這些條件能不能背測的關鍵前提，現在stock_core.py裡的
_INTRADAY_LOOKBACK_DAYS={"60":120,"5":20}只是夠用於「即時盤後檢查」的
隨便設值，從沒人測過真正上限在哪。

用法（需要FUGLE_API_KEY，在你自己能連到Fugle的環境執行）：
    python3 probe_fugle_intraday_depth.py <FUGLE_API_KEY> [股票代號=2454]

會依序嘗試60分線 lookback_days = 90, 180, 365, 730 天，和5分線
lookback_days = 30, 60, 90 天（5分資料量大很多，不往更長的測，避免一次
撞到回應過大或rate limit），印出每次「實際回傳的K棒數量」「最早日期」
「最晚日期」——如果某個lookback_days回傳的最早日期明顯比「今天-
lookback_days」晚很多（代表資料被截斷了），那就是真正的深度上限；如果
最早日期跟請求的起始日期幾乎吻合，代表這個長度還沒碰到上限，可以再往
更長測。
"""
import sys
import time

from stock_core import get_intraday_data

TEST_60MIN_LOOKBACKS = [90, 180, 365, 730]
TEST_5MIN_LOOKBACKS = [30, 60, 90]


def probe(symbol: str, timeframe: str, lookback_days: int, api_key: str) -> None:
    try:
        df = get_intraday_data(symbol, timeframe, api_key, lookback_days=lookback_days)
    except Exception as exc:  # noqa: BLE001
        print(f"  lookback_days={lookback_days:4d} | 失敗：{exc}")
        return
    if df is None or df.empty:
        print(f"  lookback_days={lookback_days:4d} | 回傳空資料")
        return
    date_col = "date" if "date" in df.columns else df.columns[0]
    earliest = str(df[date_col].min())
    latest = str(df[date_col].max())
    print(f"  lookback_days={lookback_days:4d} | 共{len(df):5d}根K棒 | 最早:{earliest} 最晚:{latest}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python3 probe_fugle_intraday_depth.py <FUGLE_API_KEY> [股票代號=2454]")
        sys.exit(1)
    api_key = sys.argv[1]
    symbol = sys.argv[2] if len(sys.argv) > 2 else "2454"

    print(f"=== 探測 {symbol} 60分線的歷史深度 ===")
    for lb in TEST_60MIN_LOOKBACKS:
        probe(symbol, "60", lb, api_key)
        time.sleep(1)

    print(f"\n=== 探測 {symbol} 5分線的歷史深度 ===")
    for lb in TEST_5MIN_LOOKBACKS:
        probe(symbol, "5", lb, api_key)
        time.sleep(1)
