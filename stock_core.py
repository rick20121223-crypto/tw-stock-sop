"""
股票技術指標核心計算模組
被 streamlit_app.py 匯入使用，本身不會直接執行。
"""

import requests
import pandas as pd

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


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
# 1. 抓取日線資料（FinMind）
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
    # FinMind 對無效 token／超額度等錯誤，HTTP 狀態碼仍是 200，
    # 但 payload 裡的 status 不是 200，且沒有 data，要另外檢查並丟出來，
    # 不然上層只會看到「查無資料」，看不出真正原因。
    if payload.get("status") != 200 and not payload.get("data"):
        raise RuntimeError(f"FinMind API 錯誤（status={payload.get('status')}）：{payload.get('msg', '未知錯誤')}")
    return pd.DataFrame(payload.get("data", []))


def get_stock_data(code: str, market: str, start_date: str, end_date: str, token: str = "") -> pd.DataFrame:
    """
    統一輸出欄位：date, open, max, min, close, volume（INDEX 無 volume，會是 NaN）
    """
    if market == "TW":
        df = fetch_finmind("TaiwanStockPrice", code, start_date, end_date, token)
        if not df.empty:
            df = df.rename(columns={"Trading_Volume": "volume"})

    elif market == "INDEX":
        df = fetch_finmind("TaiwanStockTotalReturnIndex", code, start_date, end_date, token)
        if not df.empty:
            df = df.rename(columns={"price": "close"})
            df["open"] = df["close"]
            df["max"] = df["close"]
            df["min"] = df["close"]
            df["volume"] = pd.NA

    elif market == "US":
        df = fetch_finmind("USStockPrice", code, start_date, end_date, token)
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
# 1-1. 抓取分K資料（富果 Fugle 行情 API，60分線/5分線）
# ------------------------------------------------------------------
def get_intraday_data(symbol: str, timeframe: str, fugle_api_key: str) -> pd.DataFrame:
    from fugle_marketdata import RestClient

    client = RestClient(api_key=fugle_api_key)
    resp = client.stock.intraday.candles(symbol=symbol, timeframe=timeframe)
    data = resp.get("data", [])

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = df.rename(columns={"high": "max", "low": "min"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ------------------------------------------------------------------
# 2. 四關價
# ------------------------------------------------------------------
def calc_four_key_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["昨高"] = df["max"].shift(1)
    df["昨低"] = df["min"].shift(1)
    df["昨收"] = df["close"].shift(1)
    df["今開"] = df["open"]
    return df


# ------------------------------------------------------------------
# 3. 均線 MA + 斜率判斷
# ------------------------------------------------------------------
def calc_ma(df: pd.DataFrame, periods=(5, 10, 20, 35)) -> pd.DataFrame:
    df = df.copy()
    for p in periods:
        df[f"MA{p}"] = df["close"].rolling(window=p).mean()
    return df


def ma_slope(df: pd.DataFrame, ma_col: str, lookback: int = 3) -> str:
    if ma_col not in df.columns or len(df) < lookback + 1:
        return "資料不足"
    recent = df[ma_col].dropna()
    if len(recent) < lookback + 1:
        return "資料不足"
    diff = recent.iloc[-1] - recent.iloc[-1 - lookback]
    if pd.isna(diff):
        return "資料不足"
    threshold = abs(recent.iloc[-1]) * 0.001
    if diff > threshold:
        return "上揚"
    elif diff < -threshold:
        return "下彎"
    else:
        return "走平"


# ------------------------------------------------------------------
# 4. MTM
# ------------------------------------------------------------------
def calc_mtm(df: pd.DataFrame, period: int = 10, ma_period: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["MTM"] = df["close"] - df["close"].shift(period)
    df["MTM_MA"] = df["MTM"].rolling(window=ma_period).mean()
    return df


# ------------------------------------------------------------------
# 5. MACD
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
# 6. OBV
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
# 8. KD
# ------------------------------------------------------------------
def calc_kd(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    df = df.copy()
    low_min = df["min"].rolling(window=period).min()
    high_max = df["max"].rolling(window=period).max()
    denom = (high_max - low_min).replace(0, pd.NA)
    rsv = (df["close"] - low_min) / denom * 100

    k_values = [50.0]
    d_values = [50.0]
    for i in range(1, len(df)):
        rsv_val = rsv.iloc[i] if pd.notna(rsv.iloc[i]) else 50.0
        k_values.append(k_values[-1] * 2 / 3 + rsv_val / 3)
        d_values.append(d_values[-1] * 2 / 3 + k_values[-1] / 3)

    df["K"] = k_values
    df["D"] = d_values
    return df


def run_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次跑完全部指標計算，回傳完整 DataFrame"""
    df = calc_four_key_prices(df)
    df = calc_ma(df)
    df = calc_mtm(df)
    df = calc_macd(df)
    df = calc_obv(df)
    df = calc_cci(df)   # 新增：CCI，配合MTM作為三代領先指標
    df = calc_kd(df)
    return df
