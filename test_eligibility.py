import unittest
from datetime import date

from eligibility import is_eligible


def _rules(exclude_disposition=True, exclude_full_delivery=True, exclude_new_listing=True,
           new_listing_min_trading_days=60, min_avg_trade_value_20d=30_000_000):
    return {
        "liquidity": {"min_avg_trade_value_20d": min_avg_trade_value_20d},
        "eligibility": {
            "exclude_disposition": exclude_disposition,
            "exclude_full_delivery": exclude_full_delivery,
            "exclude_new_listing": exclude_new_listing,
            "new_listing_min_trading_days": new_listing_min_trading_days,
        },
    }


class TestIsEligible(unittest.TestCase):
    def test_all_checks_pass(self):
        result = is_eligible(
            code="2330", as_of_date=date(2026, 11, 14), listing_date=date(2000, 1, 1),
            avg_trade_value_20d=100_000_000.0, is_disposition=False, is_full_delivery=False,
            rules=_rules(),
        )
        self.assertTrue(result.eligible)
        self.assertEqual(result.reasons, [])

    def test_disposition_stock_is_ineligible(self):
        result = is_eligible(
            code="2330", as_of_date=date(2026, 11, 14), listing_date=date(2000, 1, 1),
            avg_trade_value_20d=100_000_000.0, is_disposition=True, is_full_delivery=False,
            rules=_rules(),
        )
        self.assertFalse(result.eligible)
        self.assertIn("處置股", result.reasons[0])

    def test_full_delivery_stock_is_ineligible(self):
        result = is_eligible(
            code="2330", as_of_date=date(2026, 11, 14), listing_date=date(2000, 1, 1),
            avg_trade_value_20d=100_000_000.0, is_disposition=False, is_full_delivery=True,
            rules=_rules(),
        )
        self.assertFalse(result.eligible)
        self.assertIn("全額交割股", result.reasons[0])

    def test_new_listing_is_ineligible(self):
        result = is_eligible(
            code="9999", as_of_date=date(2026, 11, 14), listing_date=date(2026, 11, 1),
            avg_trade_value_20d=100_000_000.0, is_disposition=False, is_full_delivery=False,
            rules=_rules(new_listing_min_trading_days=60),
        )
        self.assertFalse(result.eligible)
        self.assertIn("上市僅", result.reasons[0])

    def test_insufficient_liquidity_is_ineligible(self):
        result = is_eligible(
            code="2330", as_of_date=date(2026, 11, 14), listing_date=date(2000, 1, 1),
            avg_trade_value_20d=1_000_000.0, is_disposition=False, is_full_delivery=False,
            rules=_rules(min_avg_trade_value_20d=30_000_000),
        )
        self.assertFalse(result.eligible)
        self.assertIn("20日均成交值", result.reasons[0])

    def test_missing_listing_date_fails_closed(self):
        result = is_eligible(
            code="2330", as_of_date=date(2026, 11, 14), listing_date=None,
            avg_trade_value_20d=100_000_000.0, is_disposition=False, is_full_delivery=False,
            rules=_rules(),
        )
        self.assertFalse(result.eligible)
        self.assertIn("查無上市日期", result.reasons[0])

    def test_missing_liquidity_data_fails_closed(self):
        result = is_eligible(
            code="2330", as_of_date=date(2026, 11, 14), listing_date=date(2000, 1, 1),
            avg_trade_value_20d=None, is_disposition=False, is_full_delivery=False,
            rules=_rules(),
        )
        self.assertFalse(result.eligible)
        self.assertIn("查無20日均成交值", result.reasons[0])

    def test_multiple_reasons_combine(self):
        result = is_eligible(
            code="2330", as_of_date=date(2026, 11, 14), listing_date=date(2000, 1, 1),
            avg_trade_value_20d=1_000_000.0, is_disposition=True, is_full_delivery=False,
            rules=_rules(),
        )
        self.assertFalse(result.eligible)
        self.assertEqual(len(result.reasons), 2)

    def test_disabled_check_is_skipped(self):
        result = is_eligible(
            code="2330", as_of_date=date(2026, 11, 14), listing_date=date(2000, 1, 1),
            avg_trade_value_20d=100_000_000.0, is_disposition=True, is_full_delivery=False,
            rules=_rules(exclude_disposition=False),
        )
        self.assertTrue(result.eligible)


if __name__ == "__main__":
    unittest.main()
