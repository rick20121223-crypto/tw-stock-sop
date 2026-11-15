import unittest
from dataclasses import dataclass

from position_sizing import calc_position_size, prioritize_candidates


@dataclass
class _FakeAccount:
    max_loss_pct: float = 0.01
    max_position_pct: float = 0.2
    max_holdings_count: int = 8


class TestCalcPositionSize(unittest.TestCase):
    def test_bound_by_position_cap(self):
        account = _FakeAccount()
        result = calc_position_size(
            equity=1_000_000.0, entry_price=100.0, stop_loss=99.0,
            cash_available=500_000.0, current_holdings_count=0, account=account,
        )
        # risk: floor(1_000_000*0.01/1) = 10000；position_cap: floor(200000/100) = 2000；
        # cash: floor(500000/100) = 5000 -> 最小值是position_cap
        self.assertTrue(result.approved)
        self.assertEqual(result.binding_constraint, "position_cap")
        self.assertEqual(result.shares, 2000)

    def test_bound_by_cash(self):
        account = _FakeAccount()
        result = calc_position_size(
            equity=1_000_000.0, entry_price=100.0, stop_loss=90.0,
            cash_available=50_000.0, current_holdings_count=0, account=account,
        )
        # risk: floor(10000/10) = 1000；position_cap: floor(200000/100) = 2000；
        # cash: floor(50000/100) = 500 -> 最小值是cash
        self.assertTrue(result.approved)
        self.assertEqual(result.binding_constraint, "cash")
        self.assertEqual(result.shares, 500)

    def test_bound_by_risk(self):
        account = _FakeAccount()
        result = calc_position_size(
            equity=1_000_000.0, entry_price=100.0, stop_loss=50.0,
            cash_available=5_000_000.0, current_holdings_count=0, account=account,
        )
        # risk: floor(10000/50) = 200；position_cap: floor(200000/100) = 2000；
        # cash: floor(5000000/100) = 50000 -> 最小值是risk
        self.assertTrue(result.approved)
        self.assertEqual(result.binding_constraint, "risk")
        self.assertEqual(result.shares, 200)

    def test_rejected_when_holdings_count_at_cap(self):
        account = _FakeAccount(max_holdings_count=8)
        result = calc_position_size(
            equity=1_000_000.0, entry_price=100.0, stop_loss=95.0,
            cash_available=500_000.0, current_holdings_count=8, account=account,
        )
        self.assertFalse(result.approved)
        self.assertEqual(result.binding_constraint, "holdings_count")
        self.assertEqual(result.shares, 0)

    def test_rejected_when_no_stop_loss(self):
        account = _FakeAccount()
        result = calc_position_size(
            equity=1_000_000.0, entry_price=100.0, stop_loss=None,
            cash_available=500_000.0, current_holdings_count=0, account=account,
        )
        self.assertFalse(result.approved)
        self.assertEqual(result.binding_constraint, "invalid_stop_loss")

    def test_rejected_when_stop_loss_not_below_entry(self):
        account = _FakeAccount()
        result = calc_position_size(
            equity=1_000_000.0, entry_price=100.0, stop_loss=105.0,
            cash_available=500_000.0, current_holdings_count=0, account=account,
        )
        self.assertFalse(result.approved)
        self.assertEqual(result.binding_constraint, "invalid_stop_loss")

    def test_rejected_when_cash_insufficient_for_even_one_share(self):
        account = _FakeAccount()
        result = calc_position_size(
            equity=1_000_000.0, entry_price=100.0, stop_loss=95.0,
            cash_available=50.0, current_holdings_count=0, account=account,
        )
        self.assertFalse(result.approved)
        self.assertEqual(result.binding_constraint, "cash")
        self.assertEqual(result.shares, 0)


class TestPrioritizeCandidates(unittest.TestCase):
    def test_sorts_descending_by_default(self):
        candidates = [
            {"code": "2330", "avg_trade_value_20d": 5e8},
            {"code": "2303", "avg_trade_value_20d": 2e9},
            {"code": "2454", "avg_trade_value_20d": 1e9},
        ]
        ranked = prioritize_candidates(candidates)
        self.assertEqual([c["code"] for c in ranked], ["2303", "2454", "2330"])
        self.assertEqual([c["priority_rank"] for c in ranked], [1, 2, 3])
        self.assertIn("由高到低", ranked[0]["priority_reason"])

    def test_sorts_ascending_when_requested(self):
        candidates = [
            {"code": "2330", "avg_trade_value_20d": 5e8},
            {"code": "2303", "avg_trade_value_20d": 2e9},
        ]
        ranked = prioritize_candidates(candidates, descending=False)
        self.assertEqual([c["code"] for c in ranked], ["2330", "2303"])

    def test_does_not_mutate_original_dicts(self):
        candidates = [{"code": "2330", "avg_trade_value_20d": 5e8}]
        prioritize_candidates(candidates)
        self.assertNotIn("priority_rank", candidates[0])


if __name__ == "__main__":
    unittest.main()
