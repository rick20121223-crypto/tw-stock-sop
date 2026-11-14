import unittest
from datetime import date

from fill_rules import resolve_fill


def _lookup_from(prices: dict):
    """prices: {date: {"close":.., "prev_close":..}} -> price_lookup函式"""
    def _lookup(code, d):
        return prices.get(d)
    return _lookup


class TestResolveFill(unittest.TestCase):
    def test_normal_fill_on_signal_date(self):
        d0 = date(2026, 1, 5)
        prices = {d0: {"close": 105.0, "prev_close": 100.0}}
        outcome = resolve_fill("2330", "買", d0, [d0], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d0)
        self.assertEqual(outcome.fill_price, 105.0)
        self.assertEqual(outcome.status, "filled")

    def test_buy_deferred_when_limit_up_locked(self):
        d0, d1 = date(2026, 1, 5), date(2026, 1, 6)
        prices = {
            d0: {"close": 110.0, "prev_close": 100.0},  # 收在+10%，買單鎖住
            d1: {"close": 108.0, "prev_close": 110.0},  # 沒鎖住
        }
        outcome = resolve_fill("2330", "買", d0, [d0, d1], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d1)
        self.assertEqual(outcome.fill_price, 108.0)

    def test_sell_deferred_when_limit_down_locked(self):
        d0, d1 = date(2026, 2, 3), date(2026, 2, 4)
        prices = {
            d0: {"close": 90.0, "prev_close": 100.0},  # 收在-10%，賣單鎖住
            d1: {"close": 92.0, "prev_close": 90.0},  # 沒鎖住
        }
        outcome = resolve_fill("2330", "賣", d0, [d0, d1], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d1)
        self.assertEqual(outcome.fill_price, 92.0)

    def test_two_consecutive_locked_days_then_fill_on_third(self):
        d0, d1, d2 = date(2026, 3, 2), date(2026, 3, 3), date(2026, 3, 4)
        prices = {
            d0: {"close": 110.0, "prev_close": 100.0},  # 鎖住
            d1: {"close": 121.0, "prev_close": 110.0},  # 鎖住（連續漲停）
            d2: {"close": 125.0, "prev_close": 121.0},  # 沒鎖住
        }
        outcome = resolve_fill("2330", "買", d0, [d0, d1, d2], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d2)
        self.assertEqual(outcome.fill_price, 125.0)

    def test_missing_price_data_returns_deferred_not_filled(self):
        d0 = date(2026, 4, 1)
        outcome = resolve_fill("2330", "買", d0, [d0], _lookup_from({}))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "deferred")
        self.assertIsNone(outcome.fill_date)

    def test_runs_out_of_trading_dates_returns_deferred(self):
        d0 = date(2026, 5, 1)
        prices = {d0: {"close": 110.0, "prev_close": 100.0}}  # 鎖住，且沒有下一天可試
        outcome = resolve_fill("2330", "買", d0, [d0], _lookup_from(prices))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "deferred")


if __name__ == "__main__":
    unittest.main()
