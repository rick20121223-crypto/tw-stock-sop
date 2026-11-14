"""
紙上交易績效計算模組。純函式 + dataclass，不 import streamlit，輸入是
成交(fill)清單、輸出是統計結果，方便日後被 backtest/ 跟第3個月的自動
交易引擎重用。

限制：
- 不處理除權息調整，歷史交易若遇到除權息，損益計算不會還原股價。
- 只處理現股多頭（先買後賣），不支援融資融券/放空。
- 假設所有當沖都是現股當沖（非資券），資券當沖會被誤判為一般交易
  （稅率套用stock_rate/etf_rate而非day_trade_rate，低估稅額）。
- 手續費以「原始委託」為單位各算一次（不依配對後的子批次重複計算），
  證交稅則依配對後的子批次分別判斷是否當沖（跟手續費不是同一個計費
  單位——這是刻意的設計，見下方 _finalize_round_trip 說明）。
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_COSTS_PATH = Path(__file__).with_name("trading_costs.yaml")


def _load_costs() -> Dict:
    with open(_COSTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


COSTS = _load_costs()


def calc_fee(price: float, shares: int, costs: Dict = COSTS) -> float:
    """買進或賣出的券商手續費，四捨五入到整數元，低於最低手續費時以
    最低手續費計。"""
    fee_cfg = costs["brokerage_fee"]
    raw = price * shares * fee_cfg["legal_rate"] * fee_cfg["discount"]
    return max(round(raw), fee_cfg["min_fee"])


def calc_sell_tax(price: float, shares: int, is_etf: bool, is_day_trade: bool,
                   costs: Dict = COSTS) -> float:
    """賣出時的證交稅。判斷順序：is_etf優先（固定etf_rate，當沖也不減半）
    → 否則看is_day_trade（stock的現股當沖減半）→ 否則一般股票稅率。"""
    tax_cfg = costs["securities_tax"]
    if is_etf:
        rate = tax_cfg["etf_rate"]
    elif is_day_trade:
        rate = tax_cfg["day_trade_rate"]
    else:
        rate = tax_cfg["stock_rate"]
    return round(price * shares * rate)
