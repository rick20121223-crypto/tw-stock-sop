"""
步驟1：指標校驗（gate）。拿券商App顯示的校驗值，跟本模組算出來的
DIF/MACD(訊號線)/36K/36D/BBI比對，誤差在1%內才算通過。校驗用「未還原」
日線（跟券商App口徑一致，也是FinMind TaiwanStockPrice預設回傳的口徑），
不是回測要用的還原權息價格——這是使用者規格明確要求的兩種口徑分開用法。

用法：python3 -m backtest.validate_indicators [FinMind token]
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.data_fetch import fetch_raw_daily
from backtest.indicators import run_indicators

OUTPUT_FILE = Path(__file__).parent / "output" / "step1_calibration.csv"

FETCH_START = "2023-01-01"  # 往前抓夠長的暖身歷史（KD365/MACD都需要），見下方收斂測試

# (code, market, check_date, {欄位: 券商App顯示值})
CHECKPOINTS = [
    ("6830", "TW", "2026-05-06", {"DIF": 96.37, "MACD": 96.51, "36K": 82.17, "36D": 81.28, "BBI": 840.43}),
    ("6830", "TW", "2026-04-23", {"DIF": 129.13, "MACD": 118.30, "36K": 88.80, "BBI": 783.56}),
    ("2308", "TW", "2026-10-13", {"DIF": 37.05, "MACD": 40.54, "36K": 66.69, "36D": 70.69}),
]

COL_MAP = {"DIF": "DIF", "MACD": "MACD_signal", "36K": "K365", "36D": "D365", "BBI": "BBI"}


def _convergence_bars(code: str, market: str, check_date: str, api_token: str,
                       col_internal: str, full_value: float) -> int:
    """拿「完整暖身歷史算出來的值」當基準，逐步縮短起算日，看資料筆數少到
    多少根K棒時，算出來的值就跟基準誤差超過0.1%——回報「至少需要多少根
    K棒才收斂」。回傳None代表資料不足以測試（例如整段歷史都不夠長）。"""
    df_full = fetch_raw_daily(code, market, FETCH_START, check_date, api_token)
    full_len = len(df_full)
    for bars_back in range(20, full_len, 10):
        trimmed = df_full.tail(bars_back).reset_index(drop=True)
        trimmed = run_indicators(trimmed)
        row = trimmed[trimmed["date"] == check_date]
        if row.empty:
            continue
        val = row[col_internal].iloc[0]
        if pd.isna(val):
            continue
        err = abs(val - full_value) / abs(full_value) if full_value else abs(val - full_value)
        if err <= 0.001:
            return bars_back
    return None


def run_validation(api_token: str = "") -> pd.DataFrame:
    rows = []
    for code, market, check_date, expected in CHECKPOINTS:
        df = fetch_raw_daily(code, market, FETCH_START, check_date, api_token)
        if df.empty:
            rows.append({"code": code, "date": check_date, "欄位": "(全部)", "狀態": "無資料",
                         "券商值": None, "計算值": None, "誤差%": None})
            continue
        df = run_indicators(df)
        row = df[df["date"] == check_date]
        if row.empty:
            rows.append({"code": code, "date": check_date, "欄位": "(全部)", "狀態": "查無當日資料",
                         "券商值": None, "計算值": None, "誤差%": None})
            continue
        row = row.iloc[0]

        for label, expected_val in expected.items():
            internal_col = COL_MAP[label]
            calc_val = row[internal_col]
            if pd.isna(calc_val):
                rows.append({"code": code, "date": check_date, "欄位": label, "狀態": "計算值為NaN",
                             "券商值": expected_val, "計算值": None, "誤差%": None})
                continue
            err_pct = abs(calc_val - expected_val) / abs(expected_val) * 100
            status = "通過" if err_pct <= 1.0 else "未通過"
            conv_bars = None
            if status == "未通過":
                conv_bars = _convergence_bars(code, market, check_date, api_token,
                                               internal_col, calc_val)
            rows.append({"code": code, "date": check_date, "欄位": label, "狀態": status,
                         "券商值": expected_val, "計算值": round(float(calc_val), 2),
                         "誤差%": round(err_pct, 3), "收斂所需K棒數": conv_bars})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FINMIND_TOKEN", "")
    result = run_validation(token)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(result.to_string(index=False))
    n_fail = (result["狀態"] == "未通過").sum()
    n_other = result["狀態"].isin(["無資料", "查無當日資料", "計算值為NaN"]).sum()
    print(f"\n未通過: {n_fail} 筆，異常: {n_other} 筆，共 {len(result)} 筆。")
