"""
紙上交易記帳持久化層的第二個實作：Google Sheets 後端。跟
`ledger_store.CSVLedgerStore`實作同一個`LedgerStore`介面，呼叫端
（Streamlit頁面）透過`ledger_store.get_ledger_store()`拿到哪一個實作，
完全不用改呼叫方式。

認證只認兩個secrets（程式碼與repo裡都不會出現真實值）：
- `GOOGLE_SERVICE_ACCOUNT_JSON`：服務帳戶金鑰JSON的完整內容（字串）
- `GOOGLE_SHEETS_SPREADSHEET_ID`：試算表ID

`positions`維持derived（不開獨立工作表）——`LedgerStore.list_positions()`
這個共用方法在ABC基底類別裡已經做好，這裡開5個工作表：accounts/
orders/fills/daily_equity/corporate_actions。

`gspread`/`google-auth`只在真的要連線（沒有注入假client）時才會
import——開發機沒裝這兩個套件、只用本機CSV時，這個模組本身仍然可以被
匯入（`ledger_store.get_ledger_store()`在CSV模式下根本不會碰到這個檔案，
但萬一有人直接import這個模組，不應該因為缺兩個選用套件就整個壞掉）。

測試時透過建構子的`client`參數注入假的client（不打真實Google API，見
`test_google_sheets_store.py`的`_FakeWorksheet`/`_FakeSpreadsheet`/
`_FakeClient`），完全不需要安裝`gspread`/`google-auth`就能測。
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from ledger_store import (
    _ACCOUNT_FIELDS,
    _CORPORATE_ACTION_FIELDS,
    _DAILY_EQUITY_FIELDS,
    _FILL_FIELDS,
    _ORDER_FIELDS,
    Account,
    LedgerStore,
    Order,
    _account_to_row,
    _corporate_action_to_row,
    _daily_equity_to_row,
    _fill_to_row,
    _order_to_row,
    _row_to_account,
    _row_to_corporate_action,
    _row_to_daily_equity,
    _row_to_fill,
    _row_to_order,
    get_secret,
)
from performance import CorporateAction, DailyEquity, Fill

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# 讀取快取：key是"{spreadsheet_id}:{worksheet_name}"，value是
# (到期時間, 讀到的rows)。模組層級的全域dict，故意不是實例屬性——
# Streamlit每次重跑腳本都會透過get_ledger_store()重新建立一個新的
# GoogleSheetsLedgerStore，實例層級的快取活不過一次rerun，放模組層級
# 才能真的在重跑之間省下API呼叫。
_READ_CACHE: Dict[str, Dict] = {}
_CACHE_TTL_SECONDS = 30


class LedgerReadError(Exception):
    """從Google Sheets讀取資料失敗（重試用盡後）。"""


class LedgerWriteError(Exception):
    """寫入資料到Google Sheets失敗（重試用盡後）。"""


def _with_retry(func, error_cls, operation_desc: str, max_attempts: int = 3,
                 backoff_base: float = 1.0, sleep_func=time.sleep):
    """執行func()，失敗（網路錯誤/429限流/暫時性5xx等任何例外）時用指數
    退避重試最多max_attempts次（等待時間backoff_base、backoff_base*2、
    backoff_base*4...），全部失敗才包成error_cls往上丟——不吞掉，呼叫端
    （目前是Streamlit頁面）會看到明確的錯誤訊息，知道這筆沒有真的寫入/
    讀到最新資料。"""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - 刻意攔截所有例外，重試用盡後統一轉型
            last_exc = exc
            if attempt < max_attempts - 1:
                sleep_func(backoff_base * (2 ** attempt))
    raise error_cls(f"{operation_desc}失敗，已重試{max_attempts}次：{last_exc}") from last_exc


def _build_client():
    """用`GOOGLE_SERVICE_ACCOUNT_JSON`secret建立真正的gspread client。
    只在沒有注入假client時才會被呼叫（見GoogleSheetsLedgerStore.__init__），
    `gspread`/`google-auth`刻意延遲到這裡才import。"""
    import gspread
    from google.oauth2.service_account import Credentials

    raw_json = get_secret("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw_json:
        raise LedgerReadError("缺少 GOOGLE_SERVICE_ACCOUNT_JSON secret，無法連線 Google Sheets")
    info = json.loads(raw_json)
    credentials = Credentials.from_service_account_info(info, scopes=_SCOPES)
    return gspread.authorize(credentials)


def _ensure_header(worksheet, expected_fields: List[str], backoff_base: float = 1.0) -> None:
    """表頭不存在或跟預期不符時（欄位缺漏/順序不對）覆寫第一列，資料列
    不受影響。"""
    current = _with_retry(lambda: worksheet.row_values(1), LedgerReadError, "讀取表頭",
                           backoff_base=backoff_base)
    if current != expected_fields:
        _with_retry(
            lambda: worksheet.update("A1", [expected_fields], value_input_option="RAW"),
            LedgerWriteError, "寫入表頭", backoff_base=backoff_base,
        )


def _get_or_create_worksheet(spreadsheet, name: str, expected_fields: List[str],
                              backoff_base: float = 1.0):
    """工作表不存在就建立一個新的，不管是不是新建都會確保表頭正確。用
    `spreadsheet.worksheets()`列出現有工作表名稱判斷存不存在，不依賴
    gspread特定的例外類別——這樣假client測試double完全不需要認識gspread
    這個套件本身。"""
    worksheets = _with_retry(spreadsheet.worksheets, LedgerReadError, "列出工作表",
                              backoff_base=backoff_base)
    existing = {ws.title: ws for ws in worksheets}
    worksheet = existing.get(name)
    if worksheet is None:
        worksheet = _with_retry(
            lambda: spreadsheet.add_worksheet(title=name, rows=1000, cols=max(len(expected_fields), 1)),
            LedgerWriteError, f"建立工作表{name}", backoff_base=backoff_base,
        )
    _ensure_header(worksheet, expected_fields, backoff_base=backoff_base)
    return worksheet


def _cached_read_records(worksheet, cache_key: str, backoff_base: float = 1.0) -> List[Dict]:
    """30秒內同一個cache_key的重複讀取直接吃快取，不重打API。"""
    cached = _READ_CACHE.get(cache_key)
    now = time.time()
    if cached is not None and cached["expires_at"] > now:
        return cached["rows"]
    rows = _with_retry(worksheet.get_all_records, LedgerReadError, "讀取資料列",
                        backoff_base=backoff_base)
    _READ_CACHE[cache_key] = {"expires_at": now + _CACHE_TTL_SECONDS, "rows": rows}
    return rows


def _invalidate_cache(cache_key: str) -> None:
    _READ_CACHE.pop(cache_key, None)


def _to_sheet_row(row: Dict, fields: List[str]) -> List[str]:
    """把row dict的每個值都轉成字串才送進Sheets——Google Sheets的儲存格
    有自己的型別系統（數字、布林值），`get_all_records()`預設會把它們讀
    回Python的int/float/bool而不是字串，但`_row_to_account()`等既有的
    轉換函式是照CSV的字串格式寫的（例如`row["is_active"] in ("True",
    "true","1")`）。這裡統一轉成字串送出、搭配`value_input_option="RAW"`
    （不讓Sheets自作主張解讀成數字/布林值），讀回來就一定是字串，兩種
    LedgerStore實作的row轉換邏輯才能共用同一套，不用分別為CSV跟Sheets各
    寫一套解析規則。"""
    return [str(row[f]) for f in fields]


class GoogleSheetsLedgerStore(LedgerStore):
    """跟`ledger_store.CSVLedgerStore`同一個`LedgerStore`介面的Google
    Sheets實作。`positions`沿用基底類別的`list_positions()`（對`fills`
    跑`compute_positions()`），不開獨立工作表。

    已知限制：每次新增/更新都會重新檢查一次工作表是否存在、表頭對不對
    （`_get_or_create_worksheet`/`_ensure_header`各打一次API），對這個
    「一人、一天幾筆手動成交」的量級無所謂，但如果之後用量變大，這裡
    會是第一個需要優化（例如整個store生命週期內快取「工作表已確認過」
    這件事）的地方。
    """

    def __init__(self, client=None, spreadsheet_id: str = None, retry_backoff_base: float = 1.0):
        self._backoff_base = retry_backoff_base
        self._client = client or _build_client()
        self._spreadsheet_id = spreadsheet_id or get_secret("GOOGLE_SHEETS_SPREADSHEET_ID")
        if not self._spreadsheet_id:
            raise LedgerReadError("缺少 GOOGLE_SHEETS_SPREADSHEET_ID secret，無法連線 Google Sheets")
        self._spreadsheet = _with_retry(
            lambda: self._client.open_by_key(self._spreadsheet_id),
            LedgerReadError, "開啟試算表", backoff_base=self._backoff_base,
        )

    def _worksheet(self, name: str, expected_fields: List[str]):
        return _get_or_create_worksheet(self._spreadsheet, name, expected_fields,
                                         backoff_base=self._backoff_base)

    def _cache_key(self, name: str) -> str:
        return f"{self._spreadsheet_id}:{name}"

    def _read_rows(self, name: str, expected_fields: List[str]) -> List[Dict]:
        worksheet = self._worksheet(name, expected_fields)
        return _cached_read_records(worksheet, self._cache_key(name), backoff_base=self._backoff_base)

    def _invalidate(self, name: str) -> None:
        _invalidate_cache(self._cache_key(name))

    def _retry(self, func, error_cls, operation_desc: str):
        return _with_retry(func, error_cls, operation_desc, backoff_base=self._backoff_base)

    # accounts ----------------------------------------------------------
    def list_accounts(self) -> List[Account]:
        return [_row_to_account(r) for r in self._read_rows("accounts", _ACCOUNT_FIELDS)]

    def add_account(self, account: Account) -> str:
        if any(a.account_id == account.account_id for a in self.list_accounts()):
            raise ValueError(f"account_id={account.account_id!r} 已經存在")
        worksheet = self._worksheet("accounts", _ACCOUNT_FIELDS)
        row = _account_to_row(account)
        self._retry(
            lambda: worksheet.append_row(_to_sheet_row(row, _ACCOUNT_FIELDS), value_input_option="RAW"),
            LedgerWriteError, "新增帳戶",
        )
        self._invalidate("accounts")
        return account.account_id

    # orders --------------------------------------------------------------
    def list_orders(self, account_id: Optional[str] = None) -> List[Order]:
        orders = [_row_to_order(r) for r in self._read_rows("orders", _ORDER_FIELDS)]
        if account_id is not None:
            orders = [o for o in orders if o.account_id == account_id]
        return orders

    def add_order(self, order: Order) -> str:
        if not order.id:
            order.id = str(uuid4())
        order.created_at = order.created_at or datetime.now()
        worksheet = self._worksheet("orders", _ORDER_FIELDS)
        row = _order_to_row(order)
        self._retry(
            lambda: worksheet.append_row(_to_sheet_row(row, _ORDER_FIELDS), value_input_option="RAW"),
            LedgerWriteError, "新增委託",
        )
        self._invalidate("orders")
        return order.id

    def update_order_status(self, order_id: str, status: str) -> None:
        """直接讀最新的(不吃快取)，才能確保行號正確。"""
        worksheet = self._worksheet("orders", _ORDER_FIELDS)
        records = self._retry(worksheet.get_all_records, LedgerReadError, "讀取委託")
        for i, row in enumerate(records):
            if row.get("id") == order_id:
                order = _row_to_order(row)
                order.status = status
                updated_row = _order_to_row(order)
                self._retry(
                    lambda: worksheet.update(f"A{i + 2}", [_to_sheet_row(updated_row, _ORDER_FIELDS)],
                                              value_input_option="RAW"),
                    LedgerWriteError, "更新委託狀態",
                )
                self._invalidate("orders")
                return
        raise ValueError(f"找不到 order_id={order_id!r}")

    # fills ------------------------------------------------------------------
    def list_fills(self, account_id: Optional[str] = None) -> List[Fill]:
        fills = [_row_to_fill(r) for r in self._read_rows("fills", _FILL_FIELDS)]
        if account_id is not None:
            fills = [f for f in fills if f.account_id == account_id]
        return fills

    def add_fill(self, fill: Fill) -> str:
        """同一個order_id重複呼叫視為已處理過，直接跳過不重複寫入——
        冪等的判斷基準是order_id，不是「帳戶+日期+代號+方向」，因為同一
        天同一股票同方向分兩批成交(例如分批進場)是合理情境，用字面規則
        會誤擋。沒有關聯委託(手動輸入)的成交不受這個去重規則影響，跟
        CSVLedgerStore一致。"""
        if fill.order_id:
            existing = [f for f in self.list_fills() if f.order_id == fill.order_id]
            if existing:
                return existing[0].id
        if not fill.id:
            fill.id = str(uuid4())
        fill.created_at = fill.created_at or datetime.now()
        worksheet = self._worksheet("fills", _FILL_FIELDS)
        row = _fill_to_row(fill)
        self._retry(
            lambda: worksheet.append_row(_to_sheet_row(row, _FILL_FIELDS), value_input_option="RAW"),
            LedgerWriteError, "新增成交",
        )
        self._invalidate("fills")
        return fill.id

    def add_fills_batch(self, fills: List[Fill]) -> List[str]:
        """給第3個月的自動引擎用：收盤後一次把當天所有成交寫進去，一次
        API呼叫(`append_rows`)寫多列，而不是每筆呼叫一次`add_fill()`。
        跟`add_fill()`套用同一套order_id冪等規則，逐筆過濾掉已存在的，
        剩下的才一次batch寫入。"""
        existing_order_ids = {f.order_id for f in self.list_fills() if f.order_id}
        to_write = []
        ids = []
        seen_in_batch = set()
        for fill in fills:
            if fill.order_id and (fill.order_id in existing_order_ids or fill.order_id in seen_in_batch):
                existing_match = next(
                    (f for f in self.list_fills() if f.order_id == fill.order_id), None
                )
                ids.append(existing_match.id if existing_match else fill.id)
                continue
            if not fill.id:
                fill.id = str(uuid4())
            fill.created_at = fill.created_at or datetime.now()
            row = _fill_to_row(fill)
            to_write.append(_to_sheet_row(row, _FILL_FIELDS))
            ids.append(fill.id)
            if fill.order_id:
                seen_in_batch.add(fill.order_id)
        if to_write:
            worksheet = self._worksheet("fills", _FILL_FIELDS)
            self._retry(
                lambda: worksheet.append_rows(to_write, value_input_option="RAW"),
                LedgerWriteError, "批次新增成交",
            )
            self._invalidate("fills")
        return ids

    # daily_equity --------------------------------------------------------
    def list_daily_equity(self, account_id: Optional[str] = None) -> List[DailyEquity]:
        records = [_row_to_daily_equity(r) for r in self._read_rows("daily_equity", _DAILY_EQUITY_FIELDS)]
        if account_id is not None:
            records = [r for r in records if r.account_id == account_id]
        return records

    def add_daily_equity(self, record: DailyEquity) -> None:
        """同一個(account_id, equity_date)重複呼叫會取代舊紀錄(upsert)，
        跟CSVLedgerStore行為一致。"""
        worksheet = self._worksheet("daily_equity", _DAILY_EQUITY_FIELDS)
        records_raw = self._retry(worksheet.get_all_records, LedgerReadError, "讀取每日淨值")
        for i, row in enumerate(records_raw):
            existing = _row_to_daily_equity(row)
            if existing.account_id == record.account_id and existing.equity_date == record.equity_date:
                new_row = _daily_equity_to_row(record)
                self._retry(
                    lambda: worksheet.update(f"A{i + 2}", [_to_sheet_row(new_row, _DAILY_EQUITY_FIELDS)],
                                              value_input_option="RAW"),
                    LedgerWriteError, "更新每日淨值",
                )
                self._invalidate("daily_equity")
                return
        new_row = _daily_equity_to_row(record)
        self._retry(
            lambda: worksheet.append_row(_to_sheet_row(new_row, _DAILY_EQUITY_FIELDS), value_input_option="RAW"),
            LedgerWriteError, "新增每日淨值",
        )
        self._invalidate("daily_equity")

    # corporate_actions ---------------------------------------------------
    def list_corporate_actions(self, account_id: Optional[str] = None) -> List[CorporateAction]:
        actions = [_row_to_corporate_action(r)
                   for r in self._read_rows("corporate_actions", _CORPORATE_ACTION_FIELDS)]
        if account_id is not None:
            actions = [a for a in actions if a.account_id == account_id]
        return actions

    def add_corporate_action(self, action: CorporateAction) -> str:
        if not action.id:
            action.id = str(uuid4())
        action.created_at = action.created_at or datetime.now()
        worksheet = self._worksheet("corporate_actions", _CORPORATE_ACTION_FIELDS)
        row = _corporate_action_to_row(action)
        self._retry(
            lambda: worksheet.append_row(_to_sheet_row(row, _CORPORATE_ACTION_FIELDS), value_input_option="RAW"),
            LedgerWriteError, "新增除權息事件",
        )
        self._invalidate("corporate_actions")
        return action.id
