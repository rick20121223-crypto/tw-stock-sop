import unittest

from google_sheets_store import (
    GoogleSheetsLedgerStore,
    LedgerReadError,
    _get_or_create_worksheet,
)


class _FakeWorksheet:
    def __init__(self, title):
        self.title = title
        self.rows = []  # rows[0]是表頭(若有)，之後都是資料列，每列都是List[str]

    def row_values(self, row_number):
        if row_number == 1 and self.rows:
            return self.rows[0]
        return []

    def update(self, range_name, values):
        if not self.rows:
            self.rows.append(list(values[0]))
        else:
            self.rows[0] = list(values[0])

    def get_all_records(self):
        if not self.rows:
            return []
        header = self.rows[0]
        return [dict(zip(header, row)) for row in self.rows[1:]]

    def append_row(self, values):
        self.rows.append(list(values))

    def append_rows(self, values_list):
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


class TestGoogleSheetsLedgerStoreInit(unittest.TestCase):
    def test_init_with_injected_client_does_not_require_gspread(self):
        spreadsheet = _FakeSpreadsheet()
        store = GoogleSheetsLedgerStore(client=_FakeClient(spreadsheet), spreadsheet_id="fake-id")
        self.assertIs(store._spreadsheet, spreadsheet)

    def test_init_raises_clear_error_when_spreadsheet_id_missing(self):
        spreadsheet = _FakeSpreadsheet()
        with self.assertRaises(LedgerReadError):
            GoogleSheetsLedgerStore(client=_FakeClient(spreadsheet), spreadsheet_id="")


if __name__ == "__main__":
    unittest.main()
