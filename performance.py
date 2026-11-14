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
from typing import Callable, Dict, List, Optional, Tuple

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
        # 只排fill_date，刻意不加f.id當次要排序鍵：同一天兩筆成交（例如
        # 當沖買賣同一天）的相對先後順序，靠Python排序穩定(stable sort)
        # 保留輸入清單原本的順序（list_fills()回傳的是CSV寫入順序=實際
        # 發生順序）。曾經用(f.fill_date, f.id)排序，id是建立時隨機產生
        # 的uuid，同一天兩筆成交的先後變成看隨機字串比大小，單元測試因此
        # 間歇性失敗（約一半機率把賣出排到買入前面，誤判成「賣出超過持倉」）。
        ordered = sorted(group, key=lambda f: f.fill_date)
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
        # 只排fill_date，刻意不加f.id當次要排序鍵：同一天兩筆成交（例如
        # 當沖買賣同一天）的相對先後順序，靠Python排序穩定(stable sort)
        # 保留輸入清單原本的順序（list_fills()回傳的是CSV寫入順序=實際
        # 發生順序）。曾經用(f.fill_date, f.id)排序，id是建立時隨機產生
        # 的uuid，同一天兩筆成交的先後變成看隨機字串比大小，單元測試因此
        # 間歇性失敗（約一半機率把賣出排到買入前面，誤判成「賣出超過持倉」）。
        ordered = sorted(group, key=lambda f: f.fill_date)
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


@dataclass
class PerformanceStats:
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: Optional[float]
    expectancy: float
    avg_r_multiple: Optional[float]
    trades_with_r: int
    equity_curve: List[Tuple[date, float]]
    max_drawdown: float
    avg_holding_days: float


def compute_stats(closed_trades: List[ClosedTrade], source: Optional[str] = None) -> PerformanceStats:
    """win = net_pnl > 0；loss = net_pnl <= 0（打平算虧損，偏保守）。
    equity_curve 依exit_date排序累加；max_drawdown是累積損益曲線上任一
    高點到之後低點的最大跌幅（正數）。source給"system"或"manual"時只納入
    該來源的交易（例如只看系統自動單的績效，排除人工介入單），留None
    (預設)則不篩選、兩種來源都算。多帳戶比較請由呼叫端先依account_id
    篩好closed_trades再各自呼叫這個函式，這裡不內建跨帳戶分組。"""
    if source is not None:
        closed_trades = [t for t in closed_trades if t.source == source]

    total = len(closed_trades)
    wins = [t.net_pnl for t in closed_trades if t.net_pnl > 0]
    losses = [t.net_pnl for t in closed_trades if t.net_pnl <= 0]
    win_rate = len(wins) / total if total else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    profit_factor = (avg_win / abs(avg_loss)) if avg_loss else None
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    r_values = [t.r_multiple for t in closed_trades if t.r_multiple is not None]
    avg_r_multiple = sum(r_values) / len(r_values) if r_values else None

    ordered = sorted(closed_trades, key=lambda t: t.exit_date)
    equity_curve: List[Tuple[date, float]] = []
    cumulative = 0.0
    for t in ordered:
        cumulative += t.net_pnl
        equity_curve.append((t.exit_date, cumulative))

    max_drawdown = 0.0
    peak = float("-inf")
    for _, cum in equity_curve:
        peak = max(peak, cum)
        max_drawdown = max(max_drawdown, peak - cum)

    avg_holding_days = sum(t.holding_days for t in closed_trades) / total if total else 0.0

    return PerformanceStats(
        total_trades=total,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        avg_r_multiple=avg_r_multiple,
        trades_with_r=len(r_values),
        equity_curve=equity_curve,
        max_drawdown=max_drawdown,
        avg_holding_days=avg_holding_days,
    )


def compare_to_benchmark(closed_trades: List[ClosedTrade],
                          benchmark_prices: List[Tuple[date, float]]) -> Dict:
    """benchmark_prices：依日期升冪排列的(日期,收盤價)序列，由呼叫端自行
    用stock_core.get_stock_data()取得後轉換傳入——本函式不自己打API，
    保持純函式、好測試。

    回傳的portfolio_return是用「總進場成本」當分母的近似報酬率，不是
    嚴謹的時間加權報酬率(TWR)，重疊資金效應未處理，僅供個人粗略比較。
    """
    if not closed_trades:
        return {"portfolio_return": None, "benchmark_return": None,
                "period_start": None, "period_end": None}

    start = min(t.entry_date for t in closed_trades)
    end = max(t.exit_date for t in closed_trades)
    start_price = _price_on_or_before(benchmark_prices, start)
    end_price = _price_on_or_before(benchmark_prices, end)
    if start_price is None or end_price is None:
        return {"portfolio_return": None, "benchmark_return": None,
                "period_start": start, "period_end": end}

    benchmark_return = (end_price / start_price) - 1
    total_cost_basis = sum(t.avg_entry_price * t.shares for t in closed_trades)
    total_net_pnl = sum(t.net_pnl for t in closed_trades)
    portfolio_return = (total_net_pnl / total_cost_basis) if total_cost_basis else None

    return {
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "period_start": start,
        "period_end": end,
    }


def _price_on_or_before(prices: List[Tuple[date, float]], target: date) -> Optional[float]:
    candidates = [p for d, p in prices if d <= target]
    return candidates[-1] if candidates else None


@dataclass
class DailyEquity:
    account_id: str
    equity_date: date
    cash: float
    positions_value: float
    total_equity: float
    cumulative_pnl: float


# compute_daily_equity() 的 price_lookup(code, date) -> Optional[float]（當天
# 收盤價，查無資料回None），跟 fill_rules.PriceLookup（回傳dict含prev_close）
# 是不同的簽名——兩邊需求不同，不要搞混。
DailyPriceLookup = Callable[[str, date], Optional[float]]


def compute_daily_equity(account_id: str, initial_capital: float, cash: float,
                          positions: List[Position], price_lookup: DailyPriceLookup,
                          as_of_date: date) -> DailyEquity:
    """給定某天的現金餘額、目前未平倉的positions（由compute_positions()
    算出）、跟一個「查某代號某天收盤價」的函式，算出那一天的淨值快照。
    查無收盤價的持倉會用avg_cost代替做mark-to-market（假設當天沒有變動），
    不是精確數字，只是避免整列算不出來。

    這個函式只算「單一一天」的快照，不會自己往前回補歷史、也不會自己
    判斷「今天是不是交易日」——這些留給之後的自動引擎或你手動呼叫時決定。
    """
    positions_value = 0.0
    for p in positions:
        price = price_lookup(p.code, as_of_date)
        mark_price = price if price is not None else p.avg_cost
        positions_value += p.shares * mark_price
    total_equity = cash + positions_value
    cumulative_pnl = total_equity - initial_capital
    return DailyEquity(
        account_id=account_id,
        equity_date=as_of_date,
        cash=cash,
        positions_value=positions_value,
        total_equity=total_equity,
        cumulative_pnl=cumulative_pnl,
    )
