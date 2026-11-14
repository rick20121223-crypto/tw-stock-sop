"""
紙上交易績效計算模組。純函式 + dataclass，不 import streamlit，輸入是
成交(fill)清單、輸出是統計結果，方便日後被 backtest/ 跟第3個月的自動
交易引擎重用。

限制：
- 不處理除權息調整，歷史交易若遇到除權息，損益計算不會還原股價。
- 只處理現股多頭（先買後賣），不支援融資融券/放空。
- 假設所有當沖都是現股當沖（非資券），資券當沖會被誤判為一般交易
  （稅率套用stock_rate/etf_rate而非day_trade_rate，低估稅額）。
- `Fill`的`fee`/`tax`是成交當下就決定好的固定值（由建立這筆成交的人/
  程式——例如`ledger_store.py`——呼叫`calc_fee()`/`calc_sell_tax()`算
  好存進去，就跟真實券商對帳單一樣一次確定不會再變），`match_fifo()`/
  `compute_positions()`只負責FIFO配對與加總，不會重新計算手續費或稅額，
  也不需要知道`is_etf`/`is_day_trade`這兩個只在成交當下才需要的屬性。
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


@dataclass
class Fill:
    account_id: str
    code: str
    side: str  # "買" 或 "賣"
    price: float
    shares: int
    fill_date: date
    fee: float
    tax: float = 0.0
    is_day_trade: bool = False  # 由建立這筆成交的人/程式決定，match_fifo不重新判斷
    source: str = "system"  # "system" 或 "manual"
    planned_stop_loss: Optional[float] = None  # 從originating order複製過來，只有買方有意義，R倍數計算用
    exit_reason: str = ""
    id: str = ""
    order_id: str = ""
    created_at: Optional[datetime] = None


@dataclass
class ClosedTrade:
    account_id: str
    code: str
    entry_date: date
    exit_date: date
    shares: int
    avg_entry_price: float
    avg_exit_price: float
    buy_cost: float
    sell_cost: float
    gross_pnl: float
    net_pnl: float
    r_multiple: Optional[float]
    is_day_trade: bool
    source: str  # "system" 或 "manual"
    holding_days: int


@dataclass
class Position:
    account_id: str
    code: str
    shares: int
    avg_cost: float


@dataclass
class _Chunk:
    side: str
    trade: Fill
    shares: int
    matched_entry: Optional[Fill] = None  # 賣出chunk才有值：對應的買入批次原始Fill


@dataclass
class _Lot:
    entry_trade: Fill
    remaining_shares: int


def _walk_account_code(ordered: List[Fill]) -> Tuple[List[List[_Chunk]], List[_Lot]]:
    """對已經按(fill_date,id)排序好的同一個(account_id,code)成交清單，跑一次
    FIFO配對。回傳 (每一輪已平倉的chunk清單所組成的list、目前還沒平倉的
    open_lots)——match_fifo()只要已平倉的部分，compute_positions()只要
    還沒平倉的open_lots，兩者共用這同一套走法，避免邏輯重複一次。"""
    open_lots: List[_Lot] = []
    chunk_fills: List[_Chunk] = []
    closed_groups: List[List[_Chunk]] = []

    for f in ordered:
        if f.side == "買":
            if sum(l.remaining_shares for l in open_lots) == 0:
                chunk_fills = []  # 前一輪已經平倉，開始新的一輪
            open_lots.append(_Lot(entry_trade=f, remaining_shares=f.shares))
            chunk_fills.append(_Chunk(side="買", trade=f, shares=f.shares))
        elif f.side == "賣":
            remaining = f.shares
            while remaining > 0:
                if not open_lots:
                    raise ValueError("賣出股數超過目前持倉，資料可能有誤")
                lot = open_lots[0]
                matched = min(lot.remaining_shares, remaining)
                chunk_fills.append(_Chunk(side="賣", trade=f, shares=matched,
                                           matched_entry=lot.entry_trade))
                lot.remaining_shares -= matched
                remaining -= matched
                if lot.remaining_shares == 0:
                    open_lots.pop(0)
            if sum(l.remaining_shares for l in open_lots) == 0:
                closed_groups.append(chunk_fills)
                chunk_fills = []
        else:
            raise ValueError(f"未知的買賣方向: {f.side!r}")
    return closed_groups, open_lots


def _group_by_account_code(fills: List[Fill]) -> Dict[Tuple[str, str], List[Fill]]:
    by_key: Dict[Tuple[str, str], List[Fill]] = {}
    for f in fills:
        by_key.setdefault((f.account_id, f.code), []).append(f)
    return by_key


def match_fifo(fills: List[Fill]) -> List[ClosedTrade]:
    """買賣以先進先出配對成一筆完整交易，支援分批進出，先按
    (account_id, code) 分組再各自配對（不同帳戶的持倉互相獨立）。只處理
    現股多頭（先買後賣），賣出股數超過目前持倉會直接丟 ValueError（資料
    顯然有誤，不靜默吞掉）。"""
    closed: List[ClosedTrade] = []
    for (account_id, code), group in _group_by_account_code(fills).items():
        ordered = sorted(group, key=lambda f: (f.fill_date, f.id))
        try:
            closed_groups, _ = _walk_account_code(ordered)
        except ValueError as e:
            raise ValueError(f"{account_id}/{code}: {e}")
        for chunk_group in closed_groups:
            closed.append(_finalize_round_trip(account_id, code, chunk_group))
    return closed


def compute_positions(fills: List[Fill]) -> List[Position]:
    """算出目前未平倉的持倉快照（FIFO加權平均成本）。完全平倉或從未交易
    過的(account_id, code)不會出現在結果裡（shares=0的狀態不產生任何列，
    不是列出來顯示0）。"""
    positions: List[Position] = []
    for (account_id, code), group in _group_by_account_code(fills).items():
        ordered = sorted(group, key=lambda f: (f.fill_date, f.id))
        try:
            _, open_lots = _walk_account_code(ordered)
        except ValueError as e:
            raise ValueError(f"{account_id}/{code}: {e}")
        total_shares = sum(l.remaining_shares for l in open_lots)
        if total_shares > 0:
            avg_cost = sum(l.remaining_shares * l.entry_trade.price for l in open_lots) / total_shares
            positions.append(Position(account_id=account_id, code=code,
                                       shares=total_shares, avg_cost=avg_cost))
    return positions


def _finalize_round_trip(account_id: str, code: str, fills: List[_Chunk]) -> ClosedTrade:
    """把一輪倉位的所有成交片段彙總成一筆ClosedTrade。fee/tax已經是每筆
    Fill建立時就固定好的值，這裡只依「原始成交」為單位各算一次（不管它
    被拆成幾個chunk），避免重複加總。"""
    buy_fills = [f for f in fills if f.side == "買"]
    sell_fills = [f for f in fills if f.side == "賣"]
    total_shares = sum(f.shares for f in sell_fills)

    sources = {f.trade.source for f in buy_fills + sell_fills}
    if len(sources) > 1:
        raise ValueError(f"{account_id}/{code}: 同一輪倉位裡system與manual來源混在一起，資料可能有誤")

    avg_entry_price = sum(f.shares * f.trade.price for f in buy_fills) / total_shares
    avg_exit_price = sum(f.shares * f.trade.price for f in sell_fills) / total_shares

    unique_buys = {f.trade.id: f.trade for f in buy_fills}
    unique_sells = {f.trade.id: f.trade for f in sell_fills}
    buy_cost = sum(t.fee for t in unique_buys.values())
    sell_cost = sum(t.fee + t.tax for t in unique_sells.values())

    gross_pnl = (sum(f.shares * f.trade.price for f in sell_fills)
                 - sum(f.shares * f.trade.price for f in buy_fills))
    net_pnl = gross_pnl - buy_cost - sell_cost

    stop_losses = [f.matched_entry.planned_stop_loss for f in sell_fills]
    if all(sl is not None for sl in stop_losses):
        risk_amount = sum(
            f.shares * (f.matched_entry.price - f.matched_entry.planned_stop_loss)
            for f in sell_fills
        )
        r_multiple = (net_pnl / risk_amount) if risk_amount else None
    else:
        r_multiple = None

    entry_date = min(t.fill_date for t in unique_buys.values())
    exit_date = max(t.fill_date for t in unique_sells.values())
    is_day_trade_whole = all(t.is_day_trade for t in unique_sells.values())

    return ClosedTrade(
        account_id=account_id,
        code=code,
        entry_date=entry_date,
        exit_date=exit_date,
        shares=total_shares,
        avg_entry_price=avg_entry_price,
        avg_exit_price=avg_exit_price,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        r_multiple=r_multiple,
        is_day_trade=is_day_trade_whole,
        source=next(iter(sources)),
        holding_days=(exit_date - entry_date).days,
    )
