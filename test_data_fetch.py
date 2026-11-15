import unittest

import pandas as pd

from backtest.data_fetch import apply_total_return_adjustment


class TestApplyTotalReturnAdjustment(unittest.TestCase):
    def test_dates_before_event_are_unaffected(self):
        df_raw = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "open": [100.0, 100.0, 95.0],
            "max": [100.0, 100.0, 95.0],
            "min": [100.0, 100.0, 95.0],
            "close": [100.0, 100.0, 95.0],
        })
        events = pd.DataFrame({"date": ["2026-01-03"], "ratio": [0.95]})
        result = apply_total_return_adjustment(df_raw, events)
        self.assertAlmostEqual(result.loc[0, "close"], 100.0)
        self.assertAlmostEqual(result.loc[1, "close"], 100.0)

    def test_dates_on_and_after_event_are_grossed_up(self):
        """跟apply_back_adjustment()方向相反：事件日(含)之後的價格要往上乘
        1/ratio，模擬股利重新投入。"""
        df_raw = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02"],
            "open": [100.0, 95.0], "max": [100.0, 95.0],
            "min": [100.0, 95.0], "close": [100.0, 95.0],
        })
        events = pd.DataFrame({"date": ["2026-01-02"], "ratio": [0.95]})
        result = apply_total_return_adjustment(df_raw, events)
        self.assertAlmostEqual(result.loc[1, "close"], 95.0 / 0.95)

    def test_total_return_exceeds_price_only_return_across_dividend_event(self):
        """除息當天價格下跌95→100純粹是股利造成的價格缺口，含息(總報酬)
        序列應該把這段缺口補回來，報酬率因此比純價格序列的報酬率高、
        約等於0（股利剛好補回價格缺口）。"""
        df_raw = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02"],
            "open": [100.0, 95.0], "max": [100.0, 95.0],
            "min": [100.0, 95.0], "close": [100.0, 95.0],
        })
        events = pd.DataFrame({"date": ["2026-01-02"], "ratio": [0.95]})
        result = apply_total_return_adjustment(df_raw, events)

        price_only_return = 95.0 / 100.0 - 1
        total_return = result.loc[1, "close"] / result.loc[0, "close"] - 1
        self.assertGreater(total_return, price_only_return)
        self.assertAlmostEqual(total_return, 0.0, places=6)

    def test_multiple_events_compound(self):
        df_raw = pd.DataFrame({
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "open": [100.0, 90.0, 85.0], "max": [100.0, 90.0, 85.0],
            "min": [100.0, 90.0, 85.0], "close": [100.0, 90.0, 85.0],
        })
        events = pd.DataFrame({"date": ["2026-01-02", "2026-01-03"], "ratio": [0.9, 0.9]})
        result = apply_total_return_adjustment(df_raw, events)
        self.assertAlmostEqual(result.loc[0, "close"], 100.0)
        self.assertAlmostEqual(result.loc[1, "close"], 90.0 / 0.9)
        self.assertAlmostEqual(result.loc[2, "close"], 85.0 / 0.9 / 0.9)

    def test_empty_dataframe_does_not_crash(self):
        df_raw = pd.DataFrame(columns=["date", "open", "max", "min", "close"])
        events = pd.DataFrame({"date": [], "ratio": []})
        result = apply_total_return_adjustment(df_raw, events)
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
