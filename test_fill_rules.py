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

    def test_buy_cancelled_after_exceeding_max_defer_days(self):
        """買單鎖住超過max_buy_defer_days(預設3個交易日)就取消，不是繼續
        deferred——訊號當天+3天共4次嘗試機會都鎖住才取消。"""
        dates = [date(2026, 6, 1 + i) for i in range(5)]  # d0~d4，共5個交易日可用
        prices = {}
        prev = 100.0
        for d in dates:
            close = round(prev * 1.1, 2)  # 每天都鎖漲停
            prices[d] = {"close": close, "prev_close": prev}
            prev = close
        outcome = resolve_fill("2330", "買", dates[0], dates, _lookup_from(prices))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "cancelled")

    def test_buy_fills_on_exactly_the_fourth_attempt(self):
        """第4次嘗試（訊號當天+3天）剛好解鎖，應該成交，不應該被誤判取消。"""
        d0, d1, d2, d3 = [date(2026, 7, 1 + i) for i in range(4)]
        prices = {
            d0: {"close": 110.0, "prev_close": 100.0},
            d1: {"close": 121.0, "prev_close": 110.0},
            d2: {"close": 133.1, "prev_close": 121.0},
            d3: {"close": 130.0, "prev_close": 133.1},  # 第4次嘗試，沒鎖住
        }
        outcome = resolve_fill("2330", "買", d0, [d0, d1, d2, d3], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d3)
        self.assertEqual(outcome.status, "filled")

    def test_sell_not_cancelled_even_after_exceeding_buy_defer_cap(self):
        """賣單預設不設順延上限，鎖住天數超過買單的3天上限也不會取消，
        持續嘗試到資料用盡為止，狀態仍是deferred不是cancelled。"""
        dates = [date(2026, 8, 1 + i) for i in range(6)]  # 6天，比買單上限(4次嘗試)多
        prices = {}
        prev = 100.0
        for d in dates:
            close = round(prev * 0.9, 2)  # 每天都鎖跌停
            prices[d] = {"close": close, "prev_close": prev}
            prev = close
        outcome = resolve_fill("2330", "賣", dates[0], dates, _lookup_from(prices))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "deferred")

    def test_sell_fills_after_unlocking_on_day_beyond_buy_cap(self):
        """賣單在第5個交易日才解鎖（超過買單的4次嘗試上限），仍然要能成交，
        證明賣單確實不受買單的上限約束。"""
        dates = [date(2026, 9, 1 + i) for i in range(5)]
        prices = {
            dates[0]: {"close": 90.0, "prev_close": 100.0},
            dates[1]: {"close": 81.0, "prev_close": 90.0},
            dates[2]: {"close": 72.9, "prev_close": 81.0},
            dates[3]: {"close": 65.61, "prev_close": 72.9},
            dates[4]: {"close": 70.0, "prev_close": 65.61},  # 第5次嘗試才解鎖
        }
        outcome = resolve_fill("2330", "賣", dates[0], dates, _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, dates[4])
        self.assertEqual(outcome.status, "filled")


if __name__ == "__main__":
    unittest.main()
