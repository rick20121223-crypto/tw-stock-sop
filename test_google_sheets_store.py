import unittest
from datetime import date
from unittest.mock import patch

from ledger_store import Account, Order
from performance import DailyEquity, Fill
from google_sheets_store import (
    GoogleSheetsLedgerStore,
    LedgerReadError,
    LedgerWriteError,
    _get_or_create_worksheet,
    _with_retry,
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

    def _make_store(self, spreadsheet=None, spreadsheet_id="fake-id", retry_backoff_base=0):
        spreadsheet = spreadsheet or _FakeSpreadsheet()
        return GoogleSheetsLedgerStore(client=_FakeClient(spreadsheet), spreadsheet_id=spreadsheet_id,
                                        retry_backoff_base=retry_backoff_base)


def _invalidate_cache_all():
    from google_sheets_store import _READ_CACHE
    _READ_CACHE.clear()


class TestGoogleSheetsLedgerStoreInit(unittest.TestCase):
    def test_init_with_injected_client_does_not_require_gspread(self):
        spreadsheet = _FakeSpreadsheet()
        store = GoogleSheetsLedgerStore(client=_FakeClient(spreadsheet), spreadsheet_id="fake-id")
        self.assertIs(store._spreadsheet, spreadsheet)

    def test_init_raises_clear_error_when_spreadsheet_id_missing(self):
        """明確mock掉get_secret，不依賴「這台機器沒有設定真實secrets」這個
        環境假設——本機開發用的.streamlit/secrets.toml存在時，這個測試
        不應該因此被real secret蓋過去而失敗。"""
        spreadsheet = _FakeSpreadsheet()
        with patch("google_sheets_store.get_secret", return_value=""):
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


class _FlakyCounter:
    """前`fail_times`次呼叫都丟例外，之後呼叫真正的`real_func`。"""

    def __init__(self, fail_times, real_func=None, result=None):
        self.fail_times = fail_times
        self.real_func = real_func
        self.result = result
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("模擬暫時性錯誤（網路/429限流）")
        return self.real_func(*args, **kwargs) if self.real_func else self.result


class TestWithRetry(unittest.TestCase):
    def test_succeeds_after_transient_failures(self):
        flaky = _FlakyCounter(fail_times=2, result="ok")
        result = _with_retry(flaky, LedgerWriteError, "測試操作", max_attempts=3,
                              backoff_base=0, sleep_func=lambda s: None)
        self.assertEqual(result, "ok")
        self.assertEqual(flaky.calls, 3)

    def test_raises_after_exhausting_retries_not_silently(self):
        flaky = _FlakyCounter(fail_times=10, result="ok")
        with self.assertRaises(LedgerWriteError):
            _with_retry(flaky, LedgerWriteError, "測試操作", max_attempts=3,
                        backoff_base=0, sleep_func=lambda s: None)
        self.assertEqual(flaky.calls, 3)  # 剛好試滿max_attempts次就停止，不會無限重試

    def test_uses_given_error_class_for_reads(self):
        flaky = _FlakyCounter(fail_times=10)
        with self.assertRaises(LedgerReadError):
            _with_retry(flaky, LedgerReadError, "測試讀取", max_attempts=2,
                        backoff_base=0, sleep_func=lambda s: None)

    def test_error_message_includes_operation_desc_and_original_exception(self):
        flaky = _FlakyCounter(fail_times=10)
        try:
            _with_retry(flaky, LedgerWriteError, "新增成交", max_attempts=2,
                        backoff_base=0, sleep_func=lambda s: None)
            self.fail("應該要丟出LedgerWriteError")
        except LedgerWriteError as exc:
            self.assertIn("新增成交", str(exc))
            self.assertIn("模擬暫時性錯誤", str(exc))


class TestRetryIntegration(_GoogleSheetsStoreTestCase):
    """確認重試機制真的接到實際的CRUD方法上，不是只有_with_retry本身
    的單元測試過而已。"""

    def test_add_account_retries_transient_failure_then_succeeds(self):
        spreadsheet = _FakeSpreadsheet()
        store = self._make_store(spreadsheet=spreadsheet)
        store.add_account(Account(account_id="v0", label="暖身"))  # 先把工作表/表頭建好
        worksheet = store._spreadsheet.worksheet("accounts")
        flaky = _FlakyCounter(fail_times=2, real_func=worksheet.append_row)
        worksheet.append_row = flaky

        store.add_account(Account(account_id="v1", label="短線v1"))

        self.assertEqual(flaky.calls, 3)
        self.assertEqual(len(store.list_accounts()), 2)

    def test_add_account_raises_ledger_write_error_after_exhausting_retries(self):
        spreadsheet = _FakeSpreadsheet()
        store = self._make_store(spreadsheet=spreadsheet)
        store.add_account(Account(account_id="v0", label="暖身"))
        worksheet = store._spreadsheet.worksheet("accounts")
        worksheet.append_row = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("永久性錯誤"))

        with self.assertRaises(LedgerWriteError):
            store.add_account(Account(account_id="v1", label="短線v1"))

        self.assertEqual(len(store.list_accounts()), 1)  # v1確實沒寫進去，不是悄悄失敗又假裝成功

    def test_list_accounts_retries_transient_read_failure(self):
        spreadsheet = _FakeSpreadsheet()
        store = self._make_store(spreadsheet=spreadsheet)
        store.add_account(Account(account_id="v1", label="短線v1"))
        worksheet = store._spreadsheet.worksheet("accounts")
        flaky = _FlakyCounter(fail_times=1, real_func=worksheet.get_all_records)
        worksheet.get_all_records = flaky

        accounts = store.list_accounts()

        self.assertEqual(len(accounts), 1)
        self.assertEqual(flaky.calls, 2)


if __name__ == "__main__":
    unittest.main()
