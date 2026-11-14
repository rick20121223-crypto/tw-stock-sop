import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from ledger_store import Account, CSVLedgerStore, Order, get_ledger_store
from performance import CorporateAction, DailyEquity, Fill


class TestCSVLedgerStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = CSVLedgerStore(data_dir=Path(self.tmpdir.name))

    def tearDown(self):
        self.tmpdir.cleanup()

    def _account(self, account_id="v1"):
        return Account(account_id=account_id, label="規則版本1", rule_version_tag="sop_rules v1.3",
                       initial_capital=1_000_000.0, max_position_pct=0.2, max_holdings_count=10,
                       max_loss_pct=0.01, holding_period_type="短線",
                       created_date=date(2026, 11, 14))

    def test_add_and_list_accounts(self):
        self.store.add_account(self._account())
        accounts = self.store.list_accounts()
        self.assertEqual(len(accounts), 1)
        self.assertEqual(accounts[0].account_id, "v1")
        self.assertEqual(accounts[0].initial_capital, 1_000_000.0)
        self.assertEqual(accounts[0].max_loss_pct, 0.01)
        self.assertEqual(accounts[0].holding_period_type, "短線")
        self.assertEqual(accounts[0].created_date, date(2026, 11, 14))
        self.assertTrue(accounts[0].is_active)

    def test_duplicate_account_id_raises(self):
        self.store.add_account(self._account())
        with self.assertRaises(ValueError):
            self.store.add_account(self._account())

    def test_go_live_date_defaults_to_created_date_when_not_set(self):
        account = Account(account_id="v2", label="測試", created_date=date(2026, 11, 14))
        self.assertEqual(account.go_live_date, date(2026, 11, 14))

    def test_go_live_date_can_differ_from_created_date(self):
        account = Account(account_id="v3", label="測試", created_date=date(2026, 11, 1),
                           go_live_date=date(2026, 11, 14))
        self.assertEqual(account.go_live_date, date(2026, 11, 14))
        self.assertEqual(account.created_date, date(2026, 11, 1))

    def test_go_live_date_roundtrips_through_csv(self):
        self.store.add_account(self._account())
        [account] = self.store.list_accounts()
        self.assertEqual(account.go_live_date, date(2026, 11, 14))

    def _order(self, account_id="v1", code="2330", side="買"):
        return Order(account_id=account_id, code=code, side=side, source="system",
                     signal_date=date(2026, 11, 10), entry_reason="買進",
                     entry_reason_detail="四關價：今開突破昨高", planned_stop_loss=95.0,
                     planned_target=120.0, requested_shares=1000)

    def test_add_and_list_orders(self):
        order_id = self.store.add_order(self._order())
        orders = self.store.list_orders()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].id, order_id)
        self.assertEqual(orders[0].entry_reason, "買進")
        self.assertEqual(orders[0].planned_stop_loss, 95.0)
        self.assertEqual(orders[0].status, "pending")
        self.assertIsNone(orders[0].priority_rank)
        self.assertIsNotNone(orders[0].created_at)

    def test_order_priority_rank_roundtrips(self):
        order = self._order()
        order.priority_rank = 2
        order.priority_reason = "20日均成交值排名第2，空位足夠配置"
        self.store.add_order(order)
        orders = self.store.list_orders()
        self.assertEqual(orders[0].priority_rank, 2)
        self.assertEqual(orders[0].priority_reason, "20日均成交值排名第2，空位足夠配置")

    def test_order_universe_version_roundtrips(self):
        order = self._order()
        order.universe_version = "v1-固定清單-核心持股"
        self.store.add_order(order)
        orders = self.store.list_orders()
        self.assertEqual(orders[0].universe_version, "v1-固定清單-核心持股")

    def test_list_orders_filtered_by_account(self):
        self.store.add_order(self._order(account_id="v1"))
        self.store.add_order(self._order(account_id="v2"))
        self.assertEqual(len(self.store.list_orders()), 2)
        self.assertEqual(len(self.store.list_orders(account_id="v1")), 1)

    def test_update_order_status(self):
        order_id = self.store.add_order(self._order())
        self.store.update_order_status(order_id, "filled")
        orders = self.store.list_orders()
        self.assertEqual(orders[0].status, "filled")

    def test_update_order_status_missing_id_raises(self):
        with self.assertRaises(ValueError):
            self.store.update_order_status("not-an-id", "filled")

    def _fill(self, account_id="v1", code="2330", side="買", price=100.0, shares=1000):
        return Fill(account_id=account_id, code=code, side=side, price=price, shares=shares,
                    fill_date=date(2026, 11, 10), fee=142.0, planned_stop_loss=95.0)

    def test_add_and_list_fills(self):
        fill_id = self.store.add_fill(self._fill())
        fills = self.store.list_fills()
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].id, fill_id)
        self.assertEqual(fills[0].price, 100.0)
        self.assertEqual(fills[0].planned_stop_loss, 95.0)

    def test_fill_universe_version_roundtrips(self):
        fill = self._fill()
        fill.universe_version = "v1-固定清單-核心持股"
        self.store.add_fill(fill)
        fills = self.store.list_fills()
        self.assertEqual(fills[0].universe_version, "v1-固定清單-核心持股")

    def test_list_positions_derived_from_fills(self):
        self.store.add_fill(self._fill(side="買", shares=1000))
        self.store.add_fill(self._fill(side="賣", shares=400))
        [pos] = self.store.list_positions()
        self.assertEqual(pos.account_id, "v1")
        self.assertEqual(pos.shares, 600)

    def _daily_equity(self, account_id="v1", equity_date=date(2026, 11, 10), total_equity=1_010_000.0):
        return DailyEquity(account_id=account_id, equity_date=equity_date, cash=900_000.0,
                            positions_value=110_000.0, total_equity=total_equity,
                            cumulative_pnl=total_equity - 1_000_000.0)

    def test_add_and_list_daily_equity(self):
        self.store.add_daily_equity(self._daily_equity())
        records = self.store.list_daily_equity()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].total_equity, 1_010_000.0)

    def test_add_daily_equity_upserts_same_day(self):
        self.store.add_daily_equity(self._daily_equity(total_equity=1_010_000.0))
        self.store.add_daily_equity(self._daily_equity(total_equity=1_020_000.0))
        records = self.store.list_daily_equity()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].total_equity, 1_020_000.0)

    def _corporate_action(self, account_id="v1", code="2330"):
        return CorporateAction(account_id=account_id, code=code, ex_date=date(2026, 11, 15),
                                cash_per_share=2.0, stock_dividend_ratio=0.1)

    def test_add_and_list_corporate_actions(self):
        action_id = self.store.add_corporate_action(self._corporate_action())
        actions = self.store.list_corporate_actions()
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].id, action_id)
        self.assertEqual(actions[0].ex_date, date(2026, 11, 15))
        self.assertEqual(actions[0].cash_per_share, 2.0)
        self.assertEqual(actions[0].stock_dividend_ratio, 0.1)
        self.assertIsNotNone(actions[0].created_at)

    def test_list_corporate_actions_filtered_by_account(self):
        self.store.add_corporate_action(self._corporate_action(account_id="v1"))
        self.store.add_corporate_action(self._corporate_action(account_id="v2"))
        self.assertEqual(len(self.store.list_corporate_actions()), 2)
        self.assertEqual(len(self.store.list_corporate_actions(account_id="v1")), 1)

    def test_list_positions_applies_registered_corporate_actions(self):
        self.store.add_fill(self._fill(side="買", shares=1000))
        self.store.add_corporate_action(self._corporate_action())
        [pos] = self.store.list_positions()
        self.assertEqual(pos.shares, 1100)


class TestGetLedgerStore(unittest.TestCase):
    def test_returns_csv_store_when_spreadsheet_id_secret_not_set(self):
        with patch("ledger_store.get_secret", return_value=""):
            store = get_ledger_store()
        self.assertIsInstance(store, CSVLedgerStore)

    def test_returns_google_sheets_store_when_spreadsheet_id_secret_set(self):
        sentinel = object()
        with patch("ledger_store.get_secret", return_value="fake-spreadsheet-id"), \
             patch("google_sheets_store.GoogleSheetsLedgerStore", return_value=sentinel) as mock_cls:
            store = get_ledger_store()
        self.assertIs(store, sentinel)
        mock_cls.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
