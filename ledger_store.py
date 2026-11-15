"""
紙上交易記帳持久化層。LedgerStore 是對外介面，現在只有 CSVLedgerStore
這個實作（本機CSV，開發用；部署到Streamlit Community Cloud後重啟會
清空，請搭配匯出/匯入自行備份，或等私有GitHub repo後端做好）。之後若
改用私有GitHub repo，只要新增一個同樣實作LedgerStore介面的類別，
pages/5_交易日誌.py、pages/6_績效統計.py完全不用改。

accounts/orders/fills/daily_equity/corporate_actions各自是獨立讀寫的表；
positions不是獨立儲存的表，是呼叫performance.compute_positions()對fills
（套用corporate_actions的股票股利調整）算出來的派生快照（list_positions()
只是方便呼叫的包裝，不接受直接寫入）。daily_equity/corporate_actions這個
月不會自動寫入（那是第3個月自動引擎的工作），add_daily_equity()/
add_corporate_action()給你手動呼叫驗證流程。

【欄位順序規則】這裡的Account/Order與performance.py的Fill/DailyEquity/
CorporateAction這幾個dataclass，之後要新增欄位時**一律加在最後面**，不要
插在中間：CSV/Sheets的表頭順序直接來自dataclass欄位順序，Google Sheets
的既有工作表只會被修正表頭、不會搬移資料列——欄位插在中間會讓已有資料
整排錯位（google_sheets_store._ensure_header()遇到這種情況會直接報錯拒絕
覆寫，要求人工遷移；加在最後面則是安全的，舊資料列缺的尾端欄位會被讀成
空字串、套預設值）。
"""

import csv
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import streamlit as st

from performance import TRADING_RULES, CorporateAction, DailyEquity, Fill, Position, compute_positions

DATA_DIR = Path(__file__).parent / "data" / "paper_trading"

# 帳戶風險偏好參數的預設值唯一來源：trading_rules.yaml的account_defaults
# 區塊。Account的dataclass欄位預設值、pages/5_交易日誌.py的表單預設值都
# 從這裡讀，不要各自寫死（見trading_rules.yaml該區塊的註解）。
_ACCOUNT_DEFAULTS = TRADING_RULES.get("account_defaults", {})


@dataclass
class Account:
    account_id: str
    label: str
    rule_version_tag: str = ""
    initial_capital: float = 1_000_000.0
    max_position_pct: float = _ACCOUNT_DEFAULTS.get("max_position_pct", 0.10)  # 單檔部位上限，占「目前帳戶淨值」的比例
    max_holdings_count: int = _ACCOUNT_DEFAULTS.get("max_holdings_count", 8)
    max_loss_pct: float = _ACCOUNT_DEFAULTS.get("max_loss_pct", 0.01)  # 單筆最大虧損，占「目前帳戶淨值」的比例，position_sizing.py用來反推股數
    holding_period_type: str = "短線"  # "短線" 或 "波段"
    created_date: Optional[date] = None
    is_active: bool = True
    go_live_date: Optional[date] = None  # 正式開始算6個月放行觀察期的起點；留空=尚未上線（引擎上線時才填，不自動補值）。放最後面是遵守模組docstring的欄位順序規則


@dataclass
class Order:
    account_id: str
    code: str
    side: str  # "買" 或 "賣"
    source: str = "system"  # "system" 或 "manual"
    signal_date: Optional[date] = None
    entry_reason: Optional[str] = None  # Verdict.conclusion穩定字串，如"買進"
    entry_reason_detail: str = ""  # reasons/caveats文字快照，供人工回顧，非機器可查詢ID
    planned_stop_loss: Optional[float] = None
    planned_target: Optional[float] = None
    requested_shares: int = 0
    status: str = "pending"  # pending/filled/deferred/cancelled
    priority_rank: Optional[int] = None  # 同日訊號數超過空位/現金時，position_sizing.prioritize_candidates()排出的名次
    priority_reason: str = ""  # 排序依據的文字說明，供事後稽核「為什麼這張單被排擠掉」
    id: str = ""
    created_at: Optional[datetime] = None
    universe_version: str = ""  # 產生這張委託當時使用的股票池版本（見trading_rules.yaml的stock_pool.version_tag）。放最後面是遵守模組docstring的欄位順序規則


def get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def require_pin() -> bool:
    """在頁面最上方呼叫。PIN正確才回傳True；False時呼叫端應立刻讓頁面
    stop()，不要往下執行任何交易資料的讀寫/顯示。沒設定JOURNAL_PIN時
    fail closed——直接鎖住整頁（回傳False），不是放行，避免忘記設定PIN
    時交易資料對任何打開這個網址的人都公開可寫/可讀。"""
    configured_pin = get_secret("JOURNAL_PIN")
    if not configured_pin:
        st.error("尚未設定 JOURNAL_PIN，此頁已鎖定。請在 secrets 設定 JOURNAL_PIN 後才能使用。")
        return False
    if st.session_state.get("journal_unlocked"):
        return True
    entered = st.text_input("請輸入存取 PIN", type="password", key="journal_pin_input")
    if entered == configured_pin:
        st.session_state["journal_unlocked"] = True
        return True
    if entered:
        st.error("PIN 錯誤")
    return False


def _s(value) -> str:
    """強制轉成字串。CSV讀出來本來就是str，這裡是no-op；但
    GoogleSheetsLedgerStore的gspread.get_all_records()會把「看起來像純
    數字」的儲存格值自動讀成int/float（即使寫入時是字串、且用了
    value_input_option="RAW"——這是gspread讀取端自己的型別推斷，不受
    寫入選項影響），純數字股票代號（如"2330"）會因此被讀成int、但
    有前導零的代號（如"0050"）不會，同一個欄位在不同值上型別不一致，
    會讓下游用(account_id, code)當key分組的邏輯(performance.py的
    _group_by_account_code)把同一支股票誤判成不同股票。所有從row dict
    取出、dataclass宣告是str的欄位，一律套用這個函式再放進dataclass。"""
    return "" if value is None else str(value)


def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------
# accounts
# ------------------------------------------------------------------
_ACCOUNT_FIELDS = [f.name for f in fields(Account)]


def _account_to_row(a: Account) -> dict:
    row = asdict(a)
    row["created_date"] = a.created_date.isoformat() if a.created_date else ""
    row["go_live_date"] = a.go_live_date.isoformat() if a.go_live_date else ""
    return row


def _row_to_account(row: dict) -> Account:
    return Account(
        account_id=_s(row["account_id"]),
        label=_s(row["label"]),
        rule_version_tag=_s(row.get("rule_version_tag", "")),
        initial_capital=float(row["initial_capital"]),
        max_position_pct=float(row["max_position_pct"]),
        max_holdings_count=int(row["max_holdings_count"]),
        max_loss_pct=float(row["max_loss_pct"]) if row.get("max_loss_pct") else 0.01,
        holding_period_type=_s(row.get("holding_period_type")) or "短線",
        created_date=date.fromisoformat(_s(row["created_date"])) if row.get("created_date") else None,
        go_live_date=date.fromisoformat(_s(row["go_live_date"])) if row.get("go_live_date") else None,
        is_active=_s(row["is_active"]) in ("True", "true", "1"),
    )


# ------------------------------------------------------------------
# orders
# ------------------------------------------------------------------
_ORDER_FIELDS = [f.name for f in fields(Order)]


def _order_to_row(o: Order) -> dict:
    row = asdict(o)
    row["signal_date"] = o.signal_date.isoformat() if o.signal_date else ""
    row["created_at"] = o.created_at.isoformat() if o.created_at else ""
    row["planned_stop_loss"] = "" if o.planned_stop_loss is None else o.planned_stop_loss
    row["planned_target"] = "" if o.planned_target is None else o.planned_target
    row["entry_reason"] = o.entry_reason or ""
    row["priority_rank"] = "" if o.priority_rank is None else o.priority_rank
    return row


def _row_to_order(row: dict) -> Order:
    return Order(
        id=_s(row["id"]),
        account_id=_s(row["account_id"]),
        code=_s(row["code"]),
        side=_s(row["side"]),
        source=_s(row.get("source", "system")),
        signal_date=date.fromisoformat(_s(row["signal_date"])) if row.get("signal_date") else None,
        entry_reason=_s(row["entry_reason"]) or None,
        entry_reason_detail=_s(row.get("entry_reason_detail", "")),
        planned_stop_loss=float(row["planned_stop_loss"]) if row.get("planned_stop_loss") else None,
        planned_target=float(row["planned_target"]) if row.get("planned_target") else None,
        requested_shares=int(row["requested_shares"]),
        status=_s(row.get("status", "pending")),
        priority_rank=int(row["priority_rank"]) if row.get("priority_rank") else None,
        priority_reason=_s(row.get("priority_reason", "")),
        universe_version=_s(row.get("universe_version", "")),
        created_at=datetime.fromisoformat(_s(row["created_at"])) if row.get("created_at") else None,
    )


# ------------------------------------------------------------------
# fills（Fill dataclass本身定義在performance.py，這裡只負責CSV序列化）
# ------------------------------------------------------------------
_FILL_FIELDS = [f.name for f in fields(Fill)]


def _fill_to_row(f: Fill) -> dict:
    row = asdict(f)
    row["fill_date"] = f.fill_date.isoformat()
    row["created_at"] = f.created_at.isoformat() if f.created_at else ""
    row["planned_stop_loss"] = "" if f.planned_stop_loss is None else f.planned_stop_loss
    return row


def _row_to_fill(row: dict) -> Fill:
    return Fill(
        id=_s(row["id"]),
        account_id=_s(row["account_id"]),
        code=_s(row["code"]),
        side=_s(row["side"]),
        price=float(row["price"]),
        shares=int(row["shares"]),
        fill_date=date.fromisoformat(_s(row["fill_date"])),
        fee=float(row["fee"]),
        tax=float(row["tax"]) if row.get("tax") else 0.0,
        is_day_trade=_s(row["is_day_trade"]) in ("True", "true", "1"),
        source=_s(row.get("source", "system")),
        planned_stop_loss=float(row["planned_stop_loss"]) if row.get("planned_stop_loss") else None,
        exit_reason=_s(row.get("exit_reason", "")),
        universe_version=_s(row.get("universe_version", "")),
        order_id=_s(row.get("order_id", "")),
        created_at=datetime.fromisoformat(_s(row["created_at"])) if row.get("created_at") else None,
    )


# ------------------------------------------------------------------
# daily_equity（DailyEquity dataclass定義在performance.py）
# ------------------------------------------------------------------
_DAILY_EQUITY_FIELDS = [f.name for f in fields(DailyEquity)]


def _daily_equity_to_row(d: DailyEquity) -> dict:
    row = asdict(d)
    row["equity_date"] = d.equity_date.isoformat()
    return row


def _row_to_daily_equity(row: dict) -> DailyEquity:
    return DailyEquity(
        account_id=_s(row["account_id"]),
        equity_date=date.fromisoformat(_s(row["equity_date"])),
        cash=float(row["cash"]),
        positions_value=float(row["positions_value"]),
        total_equity=float(row["total_equity"]),
        cumulative_pnl=float(row["cumulative_pnl"]),
    )


# ------------------------------------------------------------------
# corporate_actions（CorporateAction dataclass定義在performance.py，這裡
# 只負責CSV序列化，跟fills/daily_equity同一套模式）
# ------------------------------------------------------------------
_CORPORATE_ACTION_FIELDS = [f.name for f in fields(CorporateAction)]


def _corporate_action_to_row(a: CorporateAction) -> dict:
    row = asdict(a)
    row["ex_date"] = a.ex_date.isoformat()
    row["created_at"] = a.created_at.isoformat() if a.created_at else ""
    return row


def _row_to_corporate_action(row: dict) -> CorporateAction:
    return CorporateAction(
        id=_s(row["id"]),
        account_id=_s(row["account_id"]),
        code=_s(row["code"]),
        ex_date=date.fromisoformat(_s(row["ex_date"])),
        cash_per_share=float(row["cash_per_share"]) if row.get("cash_per_share") else 0.0,
        stock_dividend_ratio=float(row["stock_dividend_ratio"]) if row.get("stock_dividend_ratio") else 0.0,
        source=_s(row.get("source", "system")),
        created_at=datetime.fromisoformat(_s(row["created_at"])) if row.get("created_at") else None,
    )


class LedgerStore(ABC):
    @abstractmethod
    def list_accounts(self) -> List[Account]: ...

    @abstractmethod
    def add_account(self, account: Account) -> str: ...

    @abstractmethod
    def list_orders(self, account_id: Optional[str] = None) -> List[Order]: ...

    @abstractmethod
    def add_order(self, order: Order) -> str: ...

    @abstractmethod
    def update_order_status(self, order_id: str, status: str) -> None: ...

    @abstractmethod
    def list_fills(self, account_id: Optional[str] = None) -> List[Fill]: ...

    @abstractmethod
    def add_fill(self, fill: Fill) -> str: ...

    @abstractmethod
    def list_daily_equity(self, account_id: Optional[str] = None) -> List[DailyEquity]: ...

    @abstractmethod
    def add_daily_equity(self, record: DailyEquity) -> None: ...

    @abstractmethod
    def list_corporate_actions(self, account_id: Optional[str] = None) -> List[CorporateAction]: ...

    @abstractmethod
    def add_corporate_action(self, action: CorporateAction) -> str: ...

    def list_positions(self, account_id: Optional[str] = None) -> List[Position]:
        """派生快照，不是獨立儲存——直接對fills跑compute_positions()，順便
        套用已登記的股票股利調整（corporate_actions）。"""
        return compute_positions(
            self.list_fills(account_id=account_id),
            corporate_actions=self.list_corporate_actions(account_id=account_id),
        )


class CSVLedgerStore(LedgerStore):
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.accounts_path = data_dir / "accounts.csv"
        self.orders_path = data_dir / "orders.csv"
        self.fills_path = data_dir / "fills.csv"
        self.daily_equity_path = data_dir / "daily_equity.csv"
        self.corporate_actions_path = data_dir / "corporate_actions.csv"

    # accounts ------------------------------------------------------
    def list_accounts(self) -> List[Account]:
        return [_row_to_account(r) for r in _read_csv(self.accounts_path)]

    def add_account(self, account: Account) -> str:
        accounts = self.list_accounts()
        if any(a.account_id == account.account_id for a in accounts):
            raise ValueError(f"account_id={account.account_id!r} 已經存在")
        accounts.append(account)
        _write_csv(self.accounts_path, _ACCOUNT_FIELDS, [_account_to_row(a) for a in accounts])
        return account.account_id

    # orders ----------------------------------------------------------
    def list_orders(self, account_id: Optional[str] = None) -> List[Order]:
        orders = [_row_to_order(r) for r in _read_csv(self.orders_path)]
        if account_id is not None:
            orders = [o for o in orders if o.account_id == account_id]
        return orders

    def add_order(self, order: Order) -> str:
        orders = [_row_to_order(r) for r in _read_csv(self.orders_path)]
        if not order.id:
            order.id = str(uuid4())
        order.created_at = order.created_at or datetime.now()
        orders.append(order)
        _write_csv(self.orders_path, _ORDER_FIELDS, [_order_to_row(o) for o in orders])
        return order.id

    def update_order_status(self, order_id: str, status: str) -> None:
        orders = [_row_to_order(r) for r in _read_csv(self.orders_path)]
        for o in orders:
            if o.id == order_id:
                o.status = status
                _write_csv(self.orders_path, _ORDER_FIELDS, [_order_to_row(x) for x in orders])
                return
        raise ValueError(f"找不到 order_id={order_id!r}")

    # fills -----------------------------------------------------------
    def list_fills(self, account_id: Optional[str] = None) -> List[Fill]:
        fills = [_row_to_fill(r) for r in _read_csv(self.fills_path)]
        if account_id is not None:
            fills = [f for f in fills if f.account_id == account_id]
        return fills

    def add_fill(self, fill: Fill) -> str:
        fills = [_row_to_fill(r) for r in _read_csv(self.fills_path)]
        if not fill.id:
            fill.id = str(uuid4())
        fill.created_at = fill.created_at or datetime.now()
        fills.append(fill)
        _write_csv(self.fills_path, _FILL_FIELDS, [_fill_to_row(f) for f in fills])
        return fill.id

    # daily_equity ------------------------------------------------------
    def list_daily_equity(self, account_id: Optional[str] = None) -> List[DailyEquity]:
        records = [_row_to_daily_equity(r) for r in _read_csv(self.daily_equity_path)]
        if account_id is not None:
            records = [r for r in records if r.account_id == account_id]
        return records

    def add_daily_equity(self, record: DailyEquity) -> None:
        """同一個(account_id, equity_date)重複呼叫會取代舊紀錄（upsert），
        方便重算某一天時不用先手動刪除。"""
        records = [_row_to_daily_equity(r) for r in _read_csv(self.daily_equity_path)]
        records = [
            r for r in records
            if not (r.account_id == record.account_id and r.equity_date == record.equity_date)
        ]
        records.append(record)
        _write_csv(self.daily_equity_path, _DAILY_EQUITY_FIELDS,
                    [_daily_equity_to_row(r) for r in records])

    # corporate_actions -------------------------------------------------
    def list_corporate_actions(self, account_id: Optional[str] = None) -> List[CorporateAction]:
        actions = [_row_to_corporate_action(r) for r in _read_csv(self.corporate_actions_path)]
        if account_id is not None:
            actions = [a for a in actions if a.account_id == account_id]
        return actions

    def add_corporate_action(self, action: CorporateAction) -> str:
        """同一個(account_id, code, ex_date)重複呼叫視為已處理過，直接回傳
        既有紀錄的id不重複寫入——同一檔股票同一天不會有兩次除權息，這是
        天然鍵。沒有這層冪等的話，第3個月自動引擎補跑/重跑同一天會把
        現金股利重複入帳。"""
        actions = [_row_to_corporate_action(r) for r in _read_csv(self.corporate_actions_path)]
        existing = next(
            (a for a in actions
             if a.account_id == action.account_id and a.code == action.code
             and a.ex_date == action.ex_date),
            None,
        )
        if existing:
            return existing.id
        if not action.id:
            action.id = str(uuid4())
        action.created_at = action.created_at or datetime.now()
        actions.append(action)
        _write_csv(self.corporate_actions_path, _CORPORATE_ACTION_FIELDS,
                    [_corporate_action_to_row(a) for a in actions])
        return action.id


def get_ledger_store() -> LedgerStore:
    """`GOOGLE_SHEETS_SPREADSHEET_ID`這個secret有設定就回傳
    `GoogleSheetsLedgerStore`，否則回傳`CSVLedgerStore`（本機開發預設）。
    兩個Streamlit頁面都應該呼叫這個函式拿store，不要直接寫死
    `CSVLedgerStore()`——之後想切換後端只要改secrets，不用改頁面程式碼。

    `google_sheets_store`刻意延遲到函式內部才import：那個模組會
    `from ledger_store import ...`，如果在這個檔案的模組最上層就
    import google_sheets_store，兩個檔案會互相匯入造成循環依賴。"""
    if get_secret("GOOGLE_SHEETS_SPREADSHEET_ID"):
        from google_sheets_store import GoogleSheetsLedgerStore
        return GoogleSheetsLedgerStore()
    return CSVLedgerStore()
