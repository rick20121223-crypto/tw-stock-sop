"""
台股/大盤/美股技術指標分析程式
抓取 FinMind 股價資料，計算：四關價、均線(MA)、MTM、MACD、OBV、KD
輸出格式設計成方便直接套用「但丁老師四關價/均線/MACD/OBV-BBI」多週期協同分析 SOP。

安裝套件：
    pip3 install requests pandas fugle-marketdata --break-system-packages
    (或加 --user，視你的環境而定)

資料來源說明：
    - 日線：FinMind（免費/註冊帳號皆可，需填 API_TOKEN）
    - 60分線／5分線：富果(Fugle) 行情 API（需先開通玉山證券富果帳戶，
      申請行情 API Key，填入 FUGLE_API_KEY），目前只支援台股個股/ETF，
      加權指數與美股(SLS)分K暫不支援。

FinMind 免費額度：未註冊約 300 次/小時，註冊 API Token 後可提高到 600~1500 次/小時
註冊網址：https://finmindtrade.com/analysis/#/login
"""

import requests
import pandas as pd

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


# ------------------------------------------------------------------
# 0-1. 富果(Fugle)行情 API：抓取盤中分K資料（60分線、5分線）
#      需要安裝: pip3 install fugle-marketdata --break-system-packages
# ------------------------------------------------------------------
def get_intraday_data(symbol: str, timeframe: str, fugle_api_key: str) -> pd.DataFrame:
    """
    symbol:    股票代碼，例如 '2330'
    timeframe: 分K週期，例如 '60'（60分線）、'5'（5分線）、'1'（1分線）
    """
    from fugle_marketdata import RestClient

    client = RestClient(api_key=fugle_api_key)
    resp = client.stock.intraday.candles(symbol=symbol, timeframe=timeframe)
    data = resp.get("data", [])

    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Fugle 回傳欄位通常是 date, open, high, low, close, volume -> 統一改成跟日線一致的欄名
    df = df.rename(columns={"high": "max", "low": "min"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# 0. 股票代碼對照表：名稱 -> (代碼, 市場類型)
#    市場類型: "TW" = 台股個股/ETF, "INDEX" = 大盤指數, "US" = 美股
# ------------------------------------------------------------------
STOCK_NAME_MAP = {
    "台積電": ("2330", "TW"),
    "聯發科": ("2454", "TW"),
    "聯電": ("2303", "TW"),
    "台達電": ("2308", "TW"),
    "京元電子": ("2449", "TW"),
    "群聯": ("8299", "TW"),
    "華邦電": ("2344", "TW"),
    "南亞科": ("2408", "TW"),
    "國巨": ("2327", "TW"),
    "嘉澤": ("3533", "TW"),
    "環球晶": ("6488", "TW"),
    "台光電": ("2383", "TW"),
    "金像電": ("2368", "TW"),
    "高力": ("8996", "TW"),
    "0050": ("0050", "TW"),
    "元大台灣50": ("0050", "TW"),
    "加權指數": ("TAIEX", "INDEX"),
    "大盤": ("TAIEX", "INDEX"),
    "SLS": ("SLS", "US"),
}


# ------------------------------------------------------------------
# 1. 抓取資料（依市場類型呼叫不同的 FinMind 資料集）
# ------------------------------------------------------------------
def fetch_finmind(dataset: str, data_id: str, start_date: str, end_date: str, token: str) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    params = {
        "dataset": dataset,
        "data_id": data_id,
        "start_date": start_date,
        "end_date": end_date,
    }
    resp = requests.get(FINMIND_URL, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    return pd.DataFrame(payload.get("data", []))


def get_stock_data(code: str, market: str, start_date: str, end_date: str, token: str = "") -> pd.DataFrame:
    """
    code:   股票代碼 / 指數代碼 / 美股代碼
    market: "TW"（台股個股或ETF）、"INDEX"（大盤指數）、"US"（美股）
    統一輸出欄位：date, open, max, min, close, volume（INDEX 無 volume，會是 NaN）
    """
    if market == "TW":
        df = fetch_finmind("TaiwanStockPrice", code, start_date, end_date, token)
        # 欄位: date, stock_id, Trading_Volume, Trading_money, open, max, min, close, spread, Trading_turnover
        if not df.empty:
            df = df.rename(columns={"Trading_Volume": "volume"})

    elif market == "INDEX":
        df = fetch_finmind("TaiwanStockTotalReturnIndex", code, start_date, end_date, token)
        # 欄位: date, stock_id, price  -> 指數只有單一價格，沒有開高低量
        if not df.empty:
            df = df.rename(columns={"price": "close"})
            df["open"] = df["close"]
            df["max"] = df["close"]
            df["min"] = df["close"]
            df["volume"] = pd.NA  # 指數沒有成交量，OBV 無法計算

    elif market == "US":
        df = fetch_finmind("USStockPrice", code, start_date, end_date, token)
        # 欄位: date, stock_id, Adj_Close, Close, High, Low, Open, Volume -> 統一改成小寫欄位
        if not df.empty:
            df = df.rename(columns={
                "Close": "close", "High": "max", "Low": "min",
                "Open": "open", "Volume": "volume",
            })

    else:
        raise ValueError(f"未知的市場類型: {market}")

    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# 2. 四關價：昨高、昨低、昨收、今開
#    SOP 裡最高位階指標，決定當天基本格局
# ------------------------------------------------------------------
def calc_four_key_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["昨高"] = df["max"].shift(1)
    df["昨低"] = df["min"].shift(1)
    df["昨收"] = df["close"].shift(1)
    df["今開"] = df["open"]
    return df


# ------------------------------------------------------------------
# 3. 均線 MA（第一代指標，斜率比價位本身更重要）
# ------------------------------------------------------------------
def calc_ma(df: pd.DataFrame, periods=(5, 10, 20, 35)) -> pd.DataFrame:
    df = df.copy()
    for p in periods:
        df[f"MA{p}"] = df["close"].rolling(window=p).mean()
    return df


def ma_slope(df: pd.DataFrame, ma_col: str, lookback: int = 3) -> str:
    """判斷均線最近 lookback 天的斜率方向：上揚 / 下彎 / 走平"""
    if ma_col not in df.columns or len(df) < lookback + 1:
        return "資料不足"
    recent = df[ma_col].dropna()
    if len(recent) < lookback + 1:
        return "資料不足"
    diff = recent.iloc[-1] - recent.iloc[-1 - lookback]
    if pd.isna(diff):
        return "資料不足"
    threshold = abs(recent.iloc[-1]) * 0.001  # 極小變動視為走平
    if diff > threshold:
        return "上揚"
    elif diff < -threshold:
        return "下彎"
    else:
        return "走平"


# ------------------------------------------------------------------
# 4. 計算 MTM 動量指標
# ------------------------------------------------------------------
def calc_mtm(df: pd.DataFrame, period: int = 10, ma_period: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["MTM"] = df["close"] - df["close"].shift(period)
    df["MTM_MA"] = df["MTM"].rolling(window=ma_period).mean()
    return df


# ------------------------------------------------------------------
# 5. 計算 MACD
# ------------------------------------------------------------------
def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["MACD_signal"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = (df["DIF"] - df["MACD_signal"]) * 2
    return df


# ------------------------------------------------------------------
# 6. 計算 OBV（量能與股價同步驗證用；指數無成交量資料，無法計算）
# ------------------------------------------------------------------
def calc_obv(df: pd.DataFrame, ma_period: int = 20) -> pd.DataFrame:
    df = df.copy()
    if df["volume"].isna().all():
        df["OBV"] = pd.NA
        df["OBV_MA"] = pd.NA
        return df

    obv_values = [0]
    for i in range(1, len(df)):
        vol = df["volume"].iloc[i] if pd.notna(df["volume"].iloc[i]) else 0
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv_values.append(obv_values[-1] + vol)
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv_values.append(obv_values[-1] - vol)
        else:
            obv_values.append(obv_values[-1])
    df["OBV"] = obv_values
    df["OBV_MA"] = df["OBV"].rolling(window=ma_period).mean()
    return df

# ------------------------------------------------------------------
# 7. 計算 CCI（順序型指標，配合 MTM 作為三代領先指標）
# ------------------------------------------------------------------
def calc_cci(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    typical_price = (df["max"] + df["min"] + df["close"]) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: (x - x.mean()).abs().mean()
    )
    df["CCI"] = (typical_price - sma) / (0.015 * mean_deviation)
    return df

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 這兩個 token/key 只要填一次，之後不用再改
    API_TOKEN = "你的FinMind token填在這裡"
    FUGLE_API_KEY = "你的Fugle行情API Key填在這裡"

    START_DATE = "2025-01-01"
    END_DATE = "2026-09-06"

    def analyze_one(label: str, stock_id: str, market: str, timeframe: str = "日") -> None:
        print(f"\n{'=' * 70}")
        print(f"{label}（{stock_id} / {market}）— {timeframe}線 SOP 資料")
        print("=" * 70)

        if timeframe == "日":
            df = get_stock_data(stock_id, market, START_DATE, END_DATE, API_TOKEN)
        else:
            if market != "TW":
                print(f"目前分K資料只支援台股個股/ETF，{market} 類型暫不支援，SOP此步驟略過。")
                return
            df = get_intraday_data(stock_id, timeframe, FUGLE_API_KEY)

        if df.empty:
            print(f"查無資料，請確認代碼 {stock_id} 或日期區間是否正確。")
            return

        df = calc_four_key_prices(df)
        df = calc_ma(df)
        df = calc_mtm(df, period=10, ma_period=10)
        df = calc_macd(df)
        df = calc_obv(df)
        df = calc_kd(df)

        latest = df.iloc[-1]

        # --- Step 1：四關價（此概念是「昨日 vs 今日」，僅日線適用；分K資料此步驟略過） ---
        if timeframe == "日":
            print("\n【四關價】")
            print(f"  今開 {latest['今開']:.2f} / 昨高 {latest['昨高']:.2f} / "
                  f"昨低 {latest['昨低']:.2f} / 昨收 {latest['昨收']:.2f}")
            if latest["今開"] < latest["昨低"]:
                print("  → 今開跌破昨低，格局偏弱")
            elif latest["今開"] > latest["昨高"]:
                print("  → 今開站上昨高，格局偏強")
            else:
                print("  → 今開落在昨日高低區間內，格局中性")
        else:
            print("\n【四關價】")
            print(f"  （四關價是「昨日 vs 今日」的日線概念，{timeframe}分線暫不適用，SOP此步驟略過）")

        # --- Step 2：均線與斜率 ---
        print(f"\n【均線 MA（{timeframe}線）】")
        for p in (5, 10, 20, 35):
            col = f"MA{p}"
            if col in df.columns and pd.notna(latest.get(col)):
                slope = ma_slope(df, col)
                above = "站上" if latest["close"] >= latest[col] else "跌破"
                print(f"  MA{p} = {latest[col]:.2f}｜股價{above}｜斜率：{slope}")

        # --- Step 3：MACD ---
        print("\n【MACD】")
        print(f"  DIF={latest['DIF']:.3f}, 訊號線={latest['MACD_signal']:.3f}, "
              f"柱狀圖={latest['MACD_hist']:.3f}")
        zone = "零軸之上" if latest["DIF"] > 0 else "零軸之下"
        print(f"  → DIF 位於{zone}")

        # --- Step 4：OBV ---
        print("\n【OBV（量價驗證）】")
        if pd.isna(latest.get("OBV")):
            print("  此標的無成交量資料（如加權指數），OBV 無法計算，SOP 此步驟略過")
        else:
            obv_slope = ma_slope(df, "OBV_MA")
            print(f"  OBV={latest['OBV']:.0f}, OBV_MA={latest['OBV_MA']:.0f}｜近期趨勢：{obv_slope}")

        # --- Step 5：MTM ---
        print(f"\n【MTM（{timeframe}線）】")
        print(f"  MTM={latest['MTM']:.2f}, MTM_MA={latest['MTM_MA']:.2f}")

        df["MTM_signal"] = "-"
        cross_up = (df["MTM"] > 0) & (df["MTM"].shift(1) <= 0)
        cross_down = (df["MTM"] < 0) & (df["MTM"].shift(1) >= 0)
        df.loc[cross_up, "MTM_signal"] = "翻多"
        df.loc[cross_down, "MTM_signal"] = "翻空"
        recent_signals = df[df["MTM_signal"] != "-"][["date", "close", "MTM", "MTM_signal"]].tail(3)
        if not recent_signals.empty:
            print("  近期翻多/翻空紀錄：")
            print("  " + recent_signals.to_string(index=False).replace("\n", "\n  "))

        # --- KD 參考 ---
        print("\n【KD（參考用）】")
        if market == "INDEX":
            print("  （加權指數無開高低資料，K/D 僅供參考，不代表真實盤中高低點）")
        print(f"  K={latest['K']:.1f}, D={latest['D']:.1f}")

        print("\n（以上資料可直接複製貼給但丁老師 SOP 技能，請它依此判讀買賣/加碼/觀望建議）")

    timeframe_input = input(
        "請選擇週期：日線請按 Enter，60分線輸入 60，5分線輸入 5：\n"
    ).strip()
    TIMEFRAME_MAP = {"": "日", "60": "60", "5": "5"}
    timeframe = TIMEFRAME_MAP.get(timeframe_input, "日")

    user_input = input(
        "請輸入股票代碼或名稱（例如 2330、台積電、加權指數、0050、SLS），"
        "或輸入 all（或直接按 Enter）查詢清單內所有股票：\n"
    ).strip()

    if user_input == "" or user_input.lower() == "all":
        for name, (stock_id, market) in STOCK_NAME_MAP.items():
            analyze_one(name, stock_id, market, timeframe)
    elif user_input in STOCK_NAME_MAP:
        stock_id, market = STOCK_NAME_MAP[user_input]
        analyze_one(user_input, stock_id, market, timeframe)
    else:
        analyze_one(user_input, user_input, "TW", timeframe)
