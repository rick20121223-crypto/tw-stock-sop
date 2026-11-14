"""
紙上交易記帳持久化層。LedgerStore 是對外介面，現在只有 CSVLedgerStore
這個實作（本機CSV，開發用；部署到Streamlit Community Cloud後重啟會
清空，請搭配匯出/匯入自行備份，或等私有GitHub repo後端做好）。之後若
改用私有GitHub repo，只要新增一個同樣實作LedgerStore介面的類別，
pages/5_交易日誌.py、pages/6_績效統計.py完全不用改。

accounts/orders/fills/daily_equity各自是獨立讀寫的表；positions不是
獨立儲存的表，是呼叫performance.compute_positions()對fills算出來的
派生快照（list_positions()只是方便呼叫的包裝，不接受直接寫入）。
daily_equity這個月不會自動寫入（那是第3個月自動引擎的工作），
add_daily_equity()給你手動呼叫performance.compute_daily_equity()算出
一列後存起來驗證流程。
"""

import csv
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import streamlit as st

from performance import DailyEquity, Fill, Position, compute_positions

DATA_DIR = Path(__file__).parent / "data" / "paper_trading"


@dataclass
class Account:
    account_id: str
    label: str
    rule_version_tag: str = ""
    initial_capital: float = 1_000_000.0
    max_position_pct: float = 0.2  # 單筆部位上限，占「目前帳戶淨值」的比例
    max_holdings_count: int = 10
    created_date: Optional[date] = None
    is_active: bool = True


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
    id: str = ""
    created_at: Optional[datetime] = None


def get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def require_pin() -> bool:
    """在頁面最上方呼叫。PIN正確才回傳True；False時呼叫端應立刻讓頁面
    stop()，不要往下執行任何交易資料的讀寫/顯示。沒設定JOURNAL_PIN時
    視為尚未啟用保護，直接放行（但會在側邊欄警告）。"""
    configured_pin = get_secret("JOURNAL_PIN")
    if not configured_pin:
        st.sidebar.warning("尚未設定 JOURNAL_PIN，此頁目前無存取保護。")
        return True
    if st.session_state.get("journal_unlocked"):
        return True
    entered = st.text_input("請輸入存取 PIN", type="password", key="journal_pin_input")
    if entered == configured_pin:
        st.session_state["journal_unlocked"] = True
        return True
    if entered:
        st.error("PIN 錯誤")
    return False


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
    return row


def _row_to_account(row: dict) -> Account:
    return Account(
        account_id=row["account_id"],
        label=row["label"],
        rule_version_tag=row.get("rule_version_tag", ""),
        initial_capital=float(row["initial_capital"]),
        max_position_pct=float(row["max_position_pct"]),
        max_holdings_count=int(row["max_holdings_count"]),
        created_date=date.fromisoformat(row["created_date"]) if row.get("created_date") else None,
        is_active=row["is_active"] in ("True", "true", "1"),
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
    return row


def _row_to_order(row: dict) -> Order:
    return Order(
        id=row["id"],
        account_id=row["account_id"],
        code=row["code"],
        side=row["side"],
        source=row.get("source", "system"),
        signal_date=date.fromisoformat(row["signal_date"]) if row.get("signal_date") else None,
        entry_reason=row["entry_reason"] or None,
        entry_reason_detail=row.get("entry_reason_detail", ""),
        planned_stop_loss=float(row["planned_stop_loss"]) if row.get("planned_stop_loss") else None,
        planned_target=float(row["planned_target"]) if row.get("planned_target") else None,
        requested_shares=int(row["requested_shares"]),
        status=row.get("status", "pending"),
        created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else None,
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
        id=row["id"],
        account_id=row["account_id"],
        code=row["code"],
        side=row["side"],
        price=float(row["price"]),
        shares=int(row["shares"]),
        fill_date=date.fromisoformat(row["fill_date"]),
        fee=float(row["fee"]),
        tax=float(row["tax"]) if row.get("tax") else 0.0,
        is_day_trade=row["is_day_trade"] in ("True", "true", "1"),
        source=row.get("source", "system"),
        planned_stop_loss=float(row["planned_stop_loss"]) if row.get("planned_stop_loss") else None,
        exit_reason=row.get("exit_reason", ""),
        order_id=row.get("order_id", ""),
        created_at=datetime.fromisoformat(row["created_at"]) if row.get("created_at") else None,
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
        account_id=row["account_id"],
        equity_date=date.fromisoformat(row["equity_date"]),
        cash=float(row["cash"]),
        positions_value=float(row["positions_value"]),
        total_equity=float(row["total_equity"]),
        cumulative_pnl=float(row["cumulative_pnl"]),
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

    def list_positions(self, account_id: Optional[str] = None) -> List[Position]:
        """派生快照，不是獨立儲存——直接對fills跑compute_positions()。"""
        return compute_positions(self.list_fills(account_id=account_id))


class CSVLedgerStore(LedgerStore):
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.accounts_path = data_dir / "accounts.csv"
        self.orders_path = data_dir / "orders.csv"
        self.fills_path = data_dir / "fills.csv"
        self.daily_equity_path = data_dir / "daily_equity.csv"

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
