"""
探測Fugle歷史分K(60分/5分)實際能回溯多久——這是決定CCI、60分120MA/240MA、
5分20MA+BBI+MACD這些條件能不能背測的關鍵前提，現在stock_core.py裡的
_INTRADAY_LOOKBACK_DAYS={"60":120,"5":20}只是夠用於「即時盤後檢查」的
隨便設值，從沒人測過真正上限在哪。

用法（需要FUGLE_API_KEY，在你自己能連到Fugle的環境執行）：
    python3 probe_fugle_intraday_depth.py <FUGLE_API_KEY> [股票代號=2454]

第一輪測試（90/180/365/730天）已經發現：Fugle的歷史分K端點對「單次
請求」的日期區間有硬性限制（from~to必須小於365天，超過會直接回400錯誤
"Date range must be less than one year"），跟資料本身實際保留多久是
兩件事——180天內60分線、90天內5分線都完全沒有被截斷的跡象。這一輪改
測300/350/360天（60分）跟150/200/250/300天（5分），刻意避開365天那條
API硬線，才能看出資料本身真正存多深。印出的「最早日期」如果跟「今天-
lookback_days」幾乎吻合，代表這個長度還沒碰到資料本身的上限；如果卡在
某個更早的日期不再往前，那天就是資料實際保留的起點。
"""
import sys
import time

from stock_core import get_intraday_data

TEST_60MIN_LOOKBACKS = [300, 350, 360]
TEST_5MIN_LOOKBACKS = [150, 200, 250, 300]


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
