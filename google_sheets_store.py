"""
紙上交易記帳持久化層的第二個實作：Google Sheets 後端。跟
`ledger_store.CSVLedgerStore`實作同一個`LedgerStore`介面，呼叫端
（Streamlit頁面）透過`ledger_store.get_ledger_store()`拿到哪一個實作，
完全不用改呼叫方式。

認證只認兩個secrets（程式碼與repo裡都不會出現真實值）：
- `GOOGLE_SERVICE_ACCOUNT_JSON`：服務帳戶金鑰JSON的完整內容（字串）
- `GOOGLE_SHEETS_SPREADSHEET_ID`：試算表ID

`positions`維持derived（不開獨立工作表）——`LedgerStore.list_positions()`
這個共用方法在ABC基底類別裡已經做好，這裡只開4個工作表：accounts/
orders/fills/daily_equity。

`gspread`/`google-auth`只在真的要連線（沒有注入假client）時才會
import——開發機沒裝這兩個套件、只用本機CSV時，這個模組本身仍然可以被
匯入（`ledger_store.get_ledger_store()`在CSV模式下根本不會碰到這個檔案，
但萬一有人直接import這個模組，不應該因為缺兩個選用套件就整個壞掉）。

測試時透過建構子的`client`參數注入假的client（不打真實Google API，見
`test_google_sheets_store.py`的`_FakeWorksheet`/`_FakeSpreadsheet`/
`_FakeClient`），完全不需要安裝`gspread`/`google-auth`就能測。
"""

import json
from typing import List

from ledger_store import get_secret

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


class LedgerReadError(Exception):
    """從Google Sheets讀取資料失敗（重試用盡後）。"""


class LedgerWriteError(Exception):
    """寫入資料到Google Sheets失敗（重試用盡後）。"""


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


def _ensure_header(worksheet, expected_fields: List[str]) -> None:
    """表頭不存在或跟預期不符時（欄位缺漏/順序不對）覆寫第一列，資料列
    不受影響。"""
    current = worksheet.row_values(1)
    if current != expected_fields:
        worksheet.update("A1", [expected_fields])


def _get_or_create_worksheet(spreadsheet, name: str, expected_fields: List[str]):
    """工作表不存在就建立一個新的，不管是不是新建都會確保表頭正確。用
    `spreadsheet.worksheets()`列出現有工作表名稱判斷存不存在，不依賴
    gspread特定的例外類別——這樣假client測試double完全不需要認識gspread
    這個套件本身。"""
    existing = {ws.title: ws for ws in spreadsheet.worksheets()}
    worksheet = existing.get(name)
    if worksheet is None:
        worksheet = spreadsheet.add_worksheet(title=name, rows=1000, cols=max(len(expected_fields), 1))
    _ensure_header(worksheet, expected_fields)
    return worksheet


class GoogleSheetsLedgerStore:
    """跟`ledger_store.LedgerStore`同一套介面的Google Sheets實作。繼承
    關係留到完整CRUD補上後再正式`(LedgerStore)`化——這一步先把認證跟
    表頭管理的骨架搭好。"""

    def __init__(self, client=None, spreadsheet_id: str = None):
        self._client = client or _build_client()
        self._spreadsheet_id = spreadsheet_id or get_secret("GOOGLE_SHEETS_SPREADSHEET_ID")
        if not self._spreadsheet_id:
            raise LedgerReadError("缺少 GOOGLE_SHEETS_SPREADSHEET_ID secret，無法連線 Google Sheets")
        self._spreadsheet = self._client.open_by_key(self._spreadsheet_id)
