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

import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml

_COSTS_PATH = Path(__file__).with_name("trading_costs.yaml")
_RULES_PATH = Path(__file__).with_name("trading_rules.yaml")


def _load_costs() -> Dict:
    with open(_COSTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_trading_rules() -> Dict:
    with open(_RULES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


COSTS = _load_costs()
TRADING_RULES = _load_trading_rules()


def calc_fee(price: float, shares: int, costs: Dict = COSTS) -> float:
    """買進或賣出的券商手續費，無條件捨去到整數元（元大官網〔3101股票
    庫存欄位說明〕：「手續費率...皆無條件捨去至整數位」，不是四捨五入），
    低於最低手續費時以最低手續費計。股數是1000的倍數視為整股（套用
    min_fee_round_lot），否則視為零股（套用min_fee_odd_lot）。"""
    fee_cfg = costs["brokerage_fee"]
    raw = price * shares * fee_cfg["legal_rate"] * fee_cfg["discount"]
    min_fee = fee_cfg["min_fee_round_lot"] if shares % 1000 == 0 else fee_cfg["min_fee_odd_lot"]
    return max(math.floor(raw), min_fee)


def calc_sell_tax(price: float, shares: int, is_etf: bool, is_day_trade: bool,
                   costs: Dict = COSTS, trade_date: Optional[date] = None) -> float:
    """賣出時的證交稅，無條件捨去到整數元（同calc_fee的捨去方式）。
    判斷順序：is_etf優先（固定etf_rate，當沖也不減半）→ 否則看
    is_day_trade且trade_date未超過day_trade_discount_expires（stock的現股
    當沖減半，落日後退回一般稅率）→ 否則一般股票稅率。trade_date未提供時
    預設today()（沿用呼叫當下的日期判斷減半優惠是否仍有效）。"""
    tax_cfg = costs["securities_tax"]
    if trade_date is None:
        trade_date = date.today()
    expires = tax_cfg.get("day_trade_discount_expires")
    day_trade_discount_active = not expires or trade_date <= date.fromisoformat(expires)
    if is_etf:
        rate = tax_cfg["etf_rate"]
    elif is_day_trade and day_trade_discount_active:
        rate = tax_cfg["day_trade_rate"]
    else:
        rate = tax_cfg["stock_rate"]
    return math.floor(price * shares * rate)


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
    universe_version: str = ""  # 建立這筆成交當時使用的股票池版本（見trading_rules.yaml的stock_pool.version_tag）。放最後面是遵守ledger_store.py docstring的欄位順序規則


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
class CorporateAction:
    """除權息事件（最低限度處理）。cash_per_share>0表示現金股利（每股配發
    金額），stock_dividend_ratio>0表示股票股利（例如0.1表示每1股配發0.1股，
    持有1000股的部位會變成1100股，總成本不變、每股成本等比例下降）。兩者
    可以同時不為0（股票同時配發現金+股票股利）。這個月只定義資料結構+
    純函式（見compute_positions的corporate_actions參數、
    compute_pending_dividend_cash()），實際把現金寫進帳戶現金是第3個月
    自動引擎的工作。

    已知限制（第3個月接引擎時要處理）：股票股利目前只影響compute_positions
    的持倉快照，match_fifo()的FIFO配對只看fills本身——如果實際賣出的股數
    包含配股配來的股數（例如1000股配0.1變1100股後賣出1100股），match_fifo
    會誤判「賣出股數超過目前持倉」。屆時引擎產生賣單時股數要以含配股後的
    持倉為上限，且賣出配股股數的FIFO配對方式要先設計（例如把配股記成一筆
    0元成本的買入fill），不能直接沿用現狀。"""
    account_id: str
    code: str
    ex_date: date
    cash_per_share: float = 0.0
    stock_dividend_ratio: float = 0.0
    source: str = "system"
    id: str = ""
    created_at: Optional[datetime] = None


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
    cost_per_share: float = 0.0  # 0.0時(_walk_account_code建立時)視為尚未套用股票股利調整，用entry_trade.price

    def __post_init__(self):
        if not self.cost_per_share:
            self.cost_per_share = self.entry_trade.price


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


def compute_positions(fills: List[Fill],
                       corporate_actions: Optional[List[CorporateAction]] = None) -> List[Position]:
    """算出目前未平倉的持倉快照（FIFO加權平均成本）。完全平倉或從未交易
    過的(account_id, code)不會出現在結果裡（shares=0的狀態不產生任何列，
    不是列出來顯示0）。

    corporate_actions給的話，股票股利（stock_dividend_ratio!=0）會套用在
    「當時已經存在、且到現在還沒平倉」的lot上：依ex_date由舊到新依序
    對每個符合的lot把remaining_shares乘上(1+ratio)、cost_per_share除以
    (1+ratio)（總成本不變）。不影響match_fifo()的已實現損益——那邊用的
    是買賣當下的實際成交價，是獨立的一條計算，不需要知道股票股利。
    現金股利（cash_per_share）在這裡不處理，見compute_pending_dividend_cash()。
    """
    actions_by_key: Dict[Tuple[str, str], List[CorporateAction]] = {}
    for a in (corporate_actions or []):
        if a.stock_dividend_ratio:
            actions_by_key.setdefault((a.account_id, a.code), []).append(a)
    for acts in actions_by_key.values():
        acts.sort(key=lambda a: a.ex_date)

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

        for action in actions_by_key.get((account_id, code), []):
            for lot in open_lots:
                if lot.entry_trade.fill_date <= action.ex_date:
                    ratio = action.stock_dividend_ratio
                    lot.remaining_shares = round(lot.remaining_shares * (1 + ratio))
                    lot.cost_per_share = lot.cost_per_share / (1 + ratio)

        total_shares = sum(l.remaining_shares for l in open_lots)
        if total_shares > 0:
            avg_cost = sum(l.remaining_shares * l.cost_per_share for l in open_lots) / total_shares
            positions.append(Position(account_id=account_id, code=code,
                                       shares=total_shares, avg_cost=avg_cost))
    return positions


def compute_pending_dividend_cash(fills: List[Fill],
                                   corporate_actions: List[CorporateAction]) -> List[Dict]:
    """對每一筆corporate_actions裡cash_per_share>0的現金股利事件，算出
    「除息日當天實際持有股數」（用除息日之前——不含當天——的成交去算
    compute_positions()，同時套用更早的股票股利調整，確保股數基礎正確）
    乘上cash_per_share，回傳待入帳清單。這個月只做這個純計算，不實際寫進
    任何帳本——第3個月自動引擎跑到對應日期時，應該呼叫這個函式、把結果
    存成一筆新的corporate_actions帳本紀錄（見ledger_store.py）。

    回傳 [{"account_id", "code", "ex_date", "held_shares", "cash_amount"}, ...]，
    held_shares為0的事件（事件當天沒有持股）不會出現在結果裡。
    """
    stock_actions = [a for a in corporate_actions if a.stock_dividend_ratio]
    results: List[Dict] = []
    for action in corporate_actions:
        if not action.cash_per_share:
            continue
        fills_before = [f for f in fills
                        if f.account_id == action.account_id and f.code == action.code
                        and f.fill_date < action.ex_date]
        prior_stock_actions = [a for a in stock_actions
                                if a.account_id == action.account_id and a.code == action.code
                                and a.ex_date < action.ex_date]
        positions = compute_positions(fills_before, corporate_actions=prior_stock_actions)
        held = next((p.shares for p in positions
                     if p.account_id == action.account_id and p.code == action.code), 0)
        if held > 0:
            results.append({
                "account_id": action.account_id, "code": action.code,
                "ex_date": action.ex_date, "held_shares": held,
                "cash_amount": held * action.cash_per_share,
            })
    return results


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


@dataclass
class GateResult:
    passed: bool
    status: str  # "達標"/"未達標"/"資料不足"——交易筆數未達下限時固定是"資料不足"，蓋過其他條件
    beats_benchmark: Optional[bool]  # None=還沒有足夠資料能比較
    max_drawdown_ok: bool
    max_drawdown_pct: Optional[float]  # None=base_equity無效，算不出百分比
    trade_count_ok: bool
    forward_vs_backtest_note: str = "尚未串接"  # 占位欄位，前向結果與回測是否相近，之後再接


def check_live_trading_gate(stats: PerformanceStats, base_equity: float,
                             portfolio_return: Optional[float],
                             benchmark_return: Optional[float],
                             rules: Dict = TRADING_RULES) -> GateResult:
    """「放行實單標準」：扣成本後報酬優於0050、最大回撤不超過門檻、已平倉
    交易筆數達到下限，三者同時成立才算放行。`base_equity`通常傳帳戶的
    `initial_capital`，用來把`stats.max_drawdown`（絕對金額）換算成占淨值
    的百分比。portfolio_return/benchmark_return任一為None（還沒有足夠資料
    算出報酬率對照）時，視為「無法確認優於0050」，`require_beat_benchmark`
    開著的話這條件就不算過（寧可保守，不猜）。"""
    gate_cfg = rules["live_trading_gate"]

    trade_count_ok = stats.total_trades >= gate_cfg["min_trade_count"]

    max_drawdown_pct = (stats.max_drawdown / base_equity) if base_equity else None
    max_drawdown_ok = max_drawdown_pct is not None and max_drawdown_pct <= gate_cfg["max_drawdown_pct"]

    if portfolio_return is None or benchmark_return is None:
        beats_benchmark: Optional[bool] = None
    else:
        beats_benchmark = portfolio_return > benchmark_return

    require_beat = gate_cfg.get("require_beat_benchmark", True)
    benchmark_ok = (beats_benchmark is True) if require_beat else True

    passed = trade_count_ok and max_drawdown_ok and benchmark_ok
    if not trade_count_ok:
        status = "資料不足"
    elif passed:
        status = "達標"
    else:
        status = "未達標"

    return GateResult(
        passed=passed,
        status=status,
        beats_benchmark=beats_benchmark,
        max_drawdown_ok=max_drawdown_ok,
        max_drawdown_pct=max_drawdown_pct,
        trade_count_ok=trade_count_ok,
    )
