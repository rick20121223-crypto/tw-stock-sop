import unittest
from datetime import date

from fill_rules import resolve_fill


def _lookup_from(prices: dict):
    """prices: {date: {"open":.., "close":.., "prev_close":..}} -> price_lookup函式"""
    def _lookup(code, d):
        return prices.get(d)
    return _lookup


class TestResolveFill(unittest.TestCase):
    def test_fills_on_next_trading_day_using_open_price(self):
        """訊號日d0收盤後確認，成交應該發生在下一個交易日d1，且用d1的
        開盤價，不是收盤價——同一天不可能先看到收盤資料又用同一天的
        價格成交。"""
        d0, d1 = date(2026, 1, 5), date(2026, 1, 6)
        prices = {d1: {"open": 102.0, "close": 105.0, "prev_close": 100.0}}
        outcome = resolve_fill("2330", "買", d0, [d0, d1], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d1)
        self.assertEqual(outcome.fill_price, 102.0)
        self.assertEqual(outcome.status, "filled")

    def test_no_trading_date_after_signal_returns_deferred(self):
        """trading_dates只到signal_date當天（呼叫端還沒提供之後的交易日），
        不會硬是用signal_date當天的價格成交，而是回報deferred。"""
        d0 = date(2026, 1, 5)
        prices = {d0: {"open": 101.0, "close": 105.0, "prev_close": 100.0}}
        outcome = resolve_fill("2330", "買", d0, [d0], _lookup_from(prices))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "deferred")
        self.assertIsNone(outcome.fill_date)

    def test_buy_deferred_when_limit_up_locked(self):
        d0, d1, d2 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)
        prices = {
            d1: {"open": 101.0, "close": 110.0, "prev_close": 100.0},  # 收在+10%，買單鎖住
            d2: {"open": 109.0, "close": 108.0, "prev_close": 110.0},  # 沒鎖住
        }
        outcome = resolve_fill("2330", "買", d0, [d0, d1, d2], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d2)
        self.assertEqual(outcome.fill_price, 109.0)

    def test_sell_deferred_when_limit_down_locked(self):
        d0, d1, d2 = date(2026, 2, 3), date(2026, 2, 4), date(2026, 2, 5)
        prices = {
            d1: {"open": 91.0, "close": 90.0, "prev_close": 100.0},  # 收在-10%，賣單鎖住
            d2: {"open": 93.0, "close": 92.0, "prev_close": 90.0},  # 沒鎖住
        }
        outcome = resolve_fill("2330", "賣", d0, [d0, d1, d2], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d2)
        self.assertEqual(outcome.fill_price, 93.0)

    def test_two_consecutive_locked_days_then_fill_on_third(self):
        d0, d1, d2, d3 = (date(2026, 3, 2), date(2026, 3, 3),
                           date(2026, 3, 4), date(2026, 3, 5))
        prices = {
            d1: {"open": 101.0, "close": 110.0, "prev_close": 100.0},  # 鎖住
            d2: {"open": 111.0, "close": 121.0, "prev_close": 110.0},  # 鎖住（連續漲停）
            d3: {"open": 122.0, "close": 125.0, "prev_close": 121.0},  # 沒鎖住
        }
        outcome = resolve_fill("2330", "買", d0, [d0, d1, d2, d3], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d3)
        self.assertEqual(outcome.fill_price, 122.0)

    def test_missing_price_data_returns_deferred_not_filled(self):
        d0, d1 = date(2026, 4, 1), date(2026, 4, 2)
        outcome = resolve_fill("2330", "買", d0, [d0, d1], _lookup_from({}))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "deferred")
        self.assertIsNone(outcome.fill_date)

    def test_runs_out_of_trading_dates_while_still_locked_returns_deferred(self):
        d0, d1 = date(2026, 5, 1), date(2026, 5, 2)
        prices = {d1: {"open": 109.0, "close": 110.0, "prev_close": 100.0}}  # 鎖住，且沒有下一天可試
        outcome = resolve_fill("2330", "買", d0, [d0, d1], _lookup_from(prices))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "deferred")

    def test_buy_cancelled_after_exceeding_max_defer_days(self):
        """買單鎖住超過max_buy_defer_days(預設3個交易日)就取消，不是繼續
        deferred——signal_date之後的第1天+3天共4次嘗試機會都鎖住才取消。"""
        dates = [date(2026, 6, 1 + i) for i in range(5)]  # d0(訊號)~d4，共4個候選交易日(d1~d4)
        prices = {}
        prev = 100.0
        for d in dates[1:]:
            close = round(prev * 1.1, 2)  # 每天都鎖漲停
            prices[d] = {"open": round(prev * 1.005, 2), "close": close, "prev_close": prev}
            prev = close
        outcome = resolve_fill("2330", "買", dates[0], dates, _lookup_from(prices))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "cancelled")

    def test_buy_fills_on_exactly_the_fourth_attempt(self):
        """第4次嘗試（signal_date之後第1天+3天）剛好解鎖，應該成交，不應該
        被誤判取消。"""
        d0, d1, d2, d3, d4 = [date(2026, 7, 1 + i) for i in range(5)]
        prices = {
            d1: {"open": 101.0, "close": 110.0, "prev_close": 100.0},
            d2: {"open": 111.0, "close": 121.0, "prev_close": 110.0},
            d3: {"open": 122.0, "close": 133.1, "prev_close": 121.0},
            d4: {"open": 134.0, "close": 130.0, "prev_close": 133.1},  # 第4次嘗試，沒鎖住
        }
        outcome = resolve_fill("2330", "買", d0, [d0, d1, d2, d3, d4], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d4)
        self.assertEqual(outcome.fill_price, 134.0)
        self.assertEqual(outcome.status, "filled")

    def test_sell_not_cancelled_even_after_exceeding_buy_defer_cap(self):
        """賣單預設不設順延上限，鎖住天數超過買單的4次嘗試上限也不會取消，
        持續嘗試到資料用盡為止，狀態仍是deferred不是cancelled。"""
        dates = [date(2026, 8, 1 + i) for i in range(7)]  # d0(訊號)+6個候選交易日，比買單上限(4次)多
        prices = {}
        prev = 100.0
        for d in dates[1:]:
            close = round(prev * 0.9, 2)  # 每天都鎖跌停
            prices[d] = {"open": round(prev * 0.995, 2), "close": close, "prev_close": prev}
            prev = close
        outcome = resolve_fill("2330", "賣", dates[0], dates, _lookup_from(prices))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "deferred")

    def test_sell_deferred_when_halted_without_checking_price(self):
        """處置/暫停交易/下市不是用價格判斷的——is_halted回True的那天根本
        不查price_lookup，直接當賣不掉，defer_reason要標記原因。"""
        d0, d1, d2 = date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3)
        prices = {d2: {"open": 50.0, "close": 51.0, "prev_close": 50.0}}

        def is_halted(code, d):
            return d == d1

        outcome = resolve_fill("2330", "賣", d0, [d0, d1, d2], _lookup_from(prices),
                                is_halted=is_halted)
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, d2)
        self.assertEqual(outcome.fill_price, 50.0)

    def test_sell_stays_deferred_while_halted_with_no_cap(self):
        d0, d1, d2 = date(2026, 10, 10), date(2026, 10, 11), date(2026, 10, 12)

        def is_halted(code, d):
            return True

        outcome = resolve_fill("2330", "賣", d0, [d0, d1, d2], _lookup_from({}),
                                is_halted=is_halted)
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "deferred")
        self.assertEqual(outcome.defer_reason, "處置或暫停交易或下市")

    def test_defer_reason_marks_limit_lock_cause(self):
        d0, d1 = date(2026, 10, 20), date(2026, 10, 21)
        prices = {d1: {"open": 91.0, "close": 90.0, "prev_close": 100.0}}  # 跌停鎖住
        outcome = resolve_fill("2330", "賣", d0, [d0, d1], _lookup_from(prices))
        self.assertFalse(outcome.filled)
        self.assertEqual(outcome.status, "deferred")
        self.assertEqual(outcome.defer_reason, "跌停")

    def test_filled_outcome_has_no_defer_reason(self):
        d0, d1 = date(2026, 10, 25), date(2026, 10, 26)
        prices = {d1: {"open": 102.0, "close": 105.0, "prev_close": 100.0}}
        outcome = resolve_fill("2330", "買", d0, [d0, d1], _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertIsNone(outcome.defer_reason)

    def test_sell_fills_after_unlocking_on_day_beyond_buy_cap(self):
        """賣單在第5個候選交易日才解鎖（超過買單的4次嘗試上限），仍然要能
        成交，證明賣單確實不受買單的上限約束。"""
        dates = [date(2026, 9, 1 + i) for i in range(6)]  # d0(訊號)+5個候選交易日
        prices = {
            dates[1]: {"open": 91.0, "close": 90.0, "prev_close": 100.0},
            dates[2]: {"open": 82.0, "close": 81.0, "prev_close": 90.0},
            dates[3]: {"open": 73.0, "close": 72.9, "prev_close": 81.0},
            dates[4]: {"open": 66.0, "close": 65.61, "prev_close": 72.9},
            dates[5]: {"open": 71.0, "close": 70.0, "prev_close": 65.61},  # 第5次嘗試才解鎖
        }
        outcome = resolve_fill("2330", "賣", dates[0], dates, _lookup_from(prices))
        self.assertTrue(outcome.filled)
        self.assertEqual(outcome.fill_date, dates[5])
        self.assertEqual(outcome.fill_price, 71.0)
        self.assertEqual(outcome.status, "filled")


if __name__ == "__main__":
    unittest.main()
