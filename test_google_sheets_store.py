import unittest
from datetime import date

from ledger_store import Account, Order
from performance import DailyEquity, Fill
from google_sheets_store import (
    GoogleSheetsLedgerStore,
    LedgerReadError,
    _get_or_create_worksheet,
    _invalidate_cache,
)


class _FakeWorksheet:
    def __init__(self, title):
        self.title = title
        self.rows = []  # rows[0]是表頭(若有)，之後都是資料列，每列都是List[str]
        self.append_row_calls = 0
        self.append_rows_calls = 0

    def row_values(self, row_number):
        if row_number == 1 and self.rows:
            return self.rows[0]
        return []

    def update(self, range_name, values, value_input_option=None):
        row_number = int(range_name[1:])  # "A5" -> 5（這裡只用A欄當起點，跟程式碼一致）
        while len(self.rows) < row_number:
            self.rows.append([])
        self.rows[row_number - 1] = list(values[0])

    def get_all_records(self):
        if not self.rows:
            return []
        header = self.rows[0]
        return [dict(zip(header, row)) for row in self.rows[1:]]

    def append_row(self, values, value_input_option=None):
        self.append_row_calls += 1
        self.rows.append(list(values))

    def append_rows(self, values_list, value_input_option=None):
        self.append_rows_calls += 1
        for values in values_list:
            self.rows.append(list(values))


class _FakeSpreadsheet:
    def __init__(self):
        self._worksheets = {}

    def worksheets(self):
        return list(self._worksheets.values())

    def worksheet(self, name):
        return self._worksheets[name]

    def add_worksheet(self, title, rows, cols):
        ws = _FakeWorksheet(title)
        self._worksheets[title] = ws
        return ws


class _FakeClient:
    def __init__(self, spreadsheet):
        self._spreadsheet = spreadsheet

    def open_by_key(self, spreadsheet_id):
        return self._spreadsheet


class TestGetOrCreateWorksheet(unittest.TestCase):
    def test_creates_worksheet_with_header_when_missing(self):
        spreadsheet = _FakeSpreadsheet()
        ws = _get_or_create_worksheet(spreadsheet, "accounts", ["account_id", "label"])
        self.assertEqual(ws.title, "accounts")
        self.assertEqual(ws.rows, [["account_id", "label"]])
        self.assertIn("accounts", [w.title for w in spreadsheet.worksheets()])

    def test_fixes_mismatched_header_without_touching_data_rows(self):
        spreadsheet = _FakeSpreadsheet()
        existing = spreadsheet.add_worksheet(title="accounts", rows=10, cols=2)
        existing.rows = [["舊欄位1", "舊欄位2"], ["v1", "規則版本1"]]

        ws = _get_or_create_worksheet(spreadsheet, "accounts", ["account_id", "label"])

        self.assertEqual(ws.rows[0], ["account_id", "label"])
        self.assertEqual(ws.rows[1], ["v1", "規則版本1"])  # 資料列不受影響

    def test_header_already_correct_is_left_alone(self):
        spreadsheet = _FakeSpreadsheet()
        existing = spreadsheet.add_worksheet(title="accounts", rows=10, cols=2)
        existing.rows = [["account_id", "label"], ["v1", "規則版本1"]]

        ws = _get_or_create_worksheet(spreadsheet, "accounts", ["account_id", "label"])

        self.assertEqual(ws.rows, [["account_id", "label"], ["v1", "規則版本1"]])

    def test_reuses_existing_worksheet_instead_of_creating_duplicate(self):
        spreadsheet = _FakeSpreadsheet()
        spreadsheet.add_worksheet(title="accounts", rows=10, cols=2)

        _get_or_create_worksheet(spreadsheet, "accounts", ["account_id", "label"])

        self.assertEqual(len(spreadsheet.worksheets()), 1)


class _GoogleSheetsStoreTestCase(unittest.TestCase):
    """_READ_CACHE是模組層級的全域dict（故意的，見google_sheets_store.py
    的註解），每個測試開始前都要清空，不然不同測試之間會透過這個全域
    快取互相汙染（即使每個測試自己建立新的假spreadsheet，只要
    spreadsheet_id重複就會吃到上一個測試留下的快取資料）。"""

    def setUp(self):
        _invalidate_cache_all()

    def _make_store(self, spreadsheet=None, spreadsheet_id="fake-id"):
        spreadsheet = spreadsheet or _FakeSpreadsheet()
        return GoogleSheetsLedgerStore(client=_FakeClient(spreadsheet), spreadsheet_id=spreadsheet_id)


def _invalidate_cache_all():
    from google_sheets_store import _READ_CACHE
    _READ_CACHE.clear()


class TestGoogleSheetsLedgerStoreInit(unittest.TestCase):
    def test_init_with_injected_client_does_not_require_gspread(self):
        spreadsheet = _FakeSpreadsheet()
        store = GoogleSheetsLedgerStore(client=_FakeClient(spreadsheet), spreadsheet_id="fake-id")
        self.assertIs(store._spreadsheet, spreadsheet)

    def test_init_raises_clear_error_when_spreadsheet_id_missing(self):
        spreadsheet = _FakeSpreadsheet()
        with self.assertRaises(LedgerReadError):
            GoogleSheetsLedgerStore(client=_FakeClient(spreadsheet), spreadsheet_id="")


class TestAccountsCrud(_GoogleSheetsStoreTestCase):
    def test_add_and_list_accounts(self):
        store = self._make_store()
        store.add_account(Account(account_id="v1", label="短線v1", initial_capital=1_000_000.0,
                                   created_date=date(2026, 11, 14)))
        accounts = store.list_accounts()
        self.assertEqual(len(accounts), 1)
        self.assertEqual(accounts[0].account_id, "v1")
        self.assertEqual(accounts[0].initial_capital, 1_000_000.0)
        self.assertEqual(accounts[0].created_date, date(2026, 11, 14))

    def test_duplicate_account_id_raises(self):
        store = self._make_store()
        store.add_account(Account(account_id="v1", label="短線v1"))
        with self.assertRaises(ValueError):
            store.add_account(Account(account_id="v1", label="重複"))

    def test_values_sent_to_sheet_are_all_strings(self):
        """is_active是bool、initial_capital是float——寫進Sheets前都要轉成
        字串，不然get_all_records()讀回來的型別會跟CSV版本的解析邏輯對
        不上（見_to_sheet_row()的說明）。"""
        store = self._make_store()
        store.add_account(Account(account_id="v1", label="短線v1", is_active=True))
        worksheet = store._spreadsheet.worksheet("accounts")
        for value in worksheet.rows[1]:
            self.assertIsInstance(value, str)


class TestOrdersCrud(_GoogleSheetsStoreTestCase):
    def _order(self, account_id="v1", code="2330"):
        return Order(account_id=account_id, code=code, side="買", signal_date=date(2026, 11, 10),
                     entry_reason="買進", planned_stop_loss=95.0, requested_shares=1000)

    def test_add_and_list_orders(self):
        store = self._make_store()
        order_id = store.add_order(self._order())
        orders = store.list_orders()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].id, order_id)
        self.assertEqual(orders[0].planned_stop_loss, 95.0)
        self.assertEqual(orders[0].status, "pending")

    def test_list_orders_filtered_by_account(self):
        store = self._make_store()
        store.add_order(self._order(account_id="v1"))
        store.add_order(self._order(account_id="v2"))
        self.assertEqual(len(store.list_orders()), 2)
        self.assertEqual(len(store.list_orders(account_id="v1")), 1)

    def test_update_order_status(self):
        store = self._make_store()
        order_id = store.add_order(self._order())
        store.update_order_status(order_id, "filled")
        self.assertEqual(store.list_orders()[0].status, "filled")

    def test_update_order_status_missing_id_raises(self):
        store = self._make_store()
        with self.assertRaises(ValueError):
            store.update_order_status("not-an-id", "filled")


class TestFillsCrud(_GoogleSheetsStoreTestCase):
    def _fill(self, account_id="v1", code="2330", order_id=""):
        return Fill(account_id=account_id, code=code, side="買", price=100.0, shares=1000,
                    fill_date=date(2026, 11, 14), fee=142.0, order_id=order_id)

    def test_add_and_list_fills(self):
        store = self._make_store()
        fill_id = store.add_fill(self._fill())
        fills = store.list_fills()
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].id, fill_id)
        self.assertEqual(fills[0].price, 100.0)

    def test_list_positions_derived_from_fills(self):
        """positions沒有獨立工作表，是base class對fills跑compute_positions()
        算出來的——這裡確認GoogleSheetsLedgerStore自動繼承這個行為。"""
        store = self._make_store()
        store.add_fill(self._fill(code="2330"))
        [pos] = store.list_positions()
        self.assertEqual(pos.code, "2330")
        self.assertEqual(pos.shares, 1000)


class TestFillIdempotency(_GoogleSheetsStoreTestCase):
    def _fill(self, order_id):
        return Fill(account_id="v1", code="2330", side="買", price=100.0, shares=1000,
                    fill_date=date(2026, 11, 14), fee=142.0, order_id=order_id)

    def test_duplicate_order_id_is_not_written_twice(self):
        store = self._make_store()
        first_id = store.add_fill(self._fill(order_id="order-1"))
        second_id = store.add_fill(self._fill(order_id="order-1"))
        self.assertEqual(first_id, second_id)
        self.assertEqual(len(store.list_fills()), 1)

    def test_manual_fill_without_order_id_is_not_deduped(self):
        """沒有關聯委託(手動輸入)的成交不受去重規則影響，跟CSVLedgerStore
        行為一致——同樣內容的手動成交重複呼叫就是會寫兩筆。"""
        store = self._make_store()
        store.add_fill(self._fill(order_id=""))
        store.add_fill(self._fill(order_id=""))
        self.assertEqual(len(store.list_fills()), 2)

    def test_batch_write_dedupes_against_existing_and_within_batch(self):
        store = self._make_store()
        store.add_fill(self._fill(order_id="order-1"))  # 已經存在

        batch = [
            self._fill(order_id="order-1"),  # 跟既有的重複，應該被跳過
            self._fill(order_id="order-2"),  # 新的，應該寫入
            self._fill(order_id="order-2"),  # 同一個batch裡重複，只寫一次
        ]
        ids = store.add_fills_batch(batch)

        self.assertEqual(len(ids), 3)  # 每個輸入都回傳一個id(含被跳過的)
        self.assertEqual(len(store.list_fills()), 2)  # 但實際只有2筆成交存在

    def test_batch_write_uses_single_append_rows_call(self):
        """批次寫入要用一次append_rows() API呼叫寫多列，不是迴圈呼叫
        append_row()——這是quota管理的重點。"""
        store = self._make_store()
        batch = [self._fill(order_id="order-1"), self._fill(order_id="order-2")]
        store.add_fills_batch(batch)
        fills_ws = store._spreadsheet.worksheet("fills")
        self.assertEqual(fills_ws.append_rows_calls, 1)
        self.assertEqual(fills_ws.append_row_calls, 0)


class TestDailyEquityCrud(_GoogleSheetsStoreTestCase):
    def _record(self, equity_date=date(2026, 11, 14), total_equity=1_010_000.0):
        return DailyEquity(account_id="v1", equity_date=equity_date, cash=900_000.0,
                            positions_value=110_000.0, total_equity=total_equity,
                            cumulative_pnl=total_equity - 1_000_000.0)

    def test_add_and_list_daily_equity(self):
        store = self._make_store()
        store.add_daily_equity(self._record())
        records = store.list_daily_equity()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].total_equity, 1_010_000.0)

    def test_add_daily_equity_upserts_same_day(self):
        store = self._make_store()
        store.add_daily_equity(self._record(total_equity=1_010_000.0))
        store.add_daily_equity(self._record(total_equity=1_020_000.0))
        records = store.list_daily_equity()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].total_equity, 1_020_000.0)


class TestReadCache(_GoogleSheetsStoreTestCase):
    def test_repeated_read_within_ttl_does_not_hit_worksheet_again(self):
        store = self._make_store()
        store.add_account(Account(account_id="v1", label="短線v1"))
        worksheet = store._spreadsheet.worksheet("accounts")
        store.list_accounts()  # 這一次會建立快取

        # 用一個計數包裝get_all_records，確認之後的list_accounts()不會真的再呼叫一次
        original = worksheet.get_all_records
        call_count = {"n": 0}

        def _counting_get_all_records():
            call_count["n"] += 1
            return original()

        worksheet.get_all_records = _counting_get_all_records
        store.list_accounts()
        store.list_accounts()
        self.assertEqual(call_count["n"], 0)  # 兩次都吃快取，完全沒再呼叫get_all_records

    def test_write_invalidates_cache_for_that_table(self):
        store = self._make_store()
        store.add_account(Account(account_id="v1", label="短線v1"))
        store.list_accounts()  # 建立快取

        store.add_account(Account(account_id="v2", label="波段v1"))  # 寫入應該讓快取失效
        accounts = store.list_accounts()
        self.assertEqual(len(accounts), 2)  # 讀到最新的2筆，不是快取住的舊的1筆


if __name__ == "__main__":
    unittest.main()
