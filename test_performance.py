import unittest

from performance import calc_fee, calc_sell_tax


def _costs(discount: float = 1.0, min_fee: float = 0) -> dict:
    return {
        "brokerage_fee": {"legal_rate": 0.001425, "discount": discount, "min_fee": min_fee},
        "securities_tax": {
            "stock_rate": 0.003,
            "etf_rate": 0.001,
            "day_trade_rate": 0.0015,
            "day_trade_discount_expires": "2027-12-31",
        },
    }


class TestFees(unittest.TestCase):
    def test_calc_fee_no_discount(self):
        fee = calc_fee(price=37, shares=1000, costs=_costs())
        self.assertEqual(fee, round(37 * 1000 * 0.001425 * 1.0))

    def test_calc_fee_with_discount(self):
        fee = calc_fee(price=37, shares=1000, costs=_costs(discount=0.6))
        self.assertEqual(fee, round(37 * 1000 * 0.001425 * 0.6))

    def test_calc_fee_hits_min_fee_floor(self):
        costs = _costs(discount=0.6, min_fee=20)
        fee = calc_fee(price=5, shares=100, costs=costs)
        raw_rounded = round(5 * 100 * 0.001425 * 0.6)
        self.assertEqual(raw_rounded, 0)  # 沒有min_fee時原本會是0
        self.assertEqual(fee, 20)  # 被min_fee墊高

    def test_calc_sell_tax_stock_not_day_trade(self):
        tax = calc_sell_tax(price=37, shares=1000, is_etf=False, is_day_trade=False, costs=_costs())
        self.assertEqual(tax, round(37 * 1000 * 0.003))

    def test_calc_sell_tax_stock_day_trade_is_halved(self):
        tax = calc_sell_tax(price=38, shares=1000, is_etf=False, is_day_trade=True, costs=_costs())
        self.assertEqual(tax, round(38 * 1000 * 0.0015))

    def test_calc_sell_tax_etf_not_day_trade(self):
        tax = calc_sell_tax(price=37, shares=1000, is_etf=True, is_day_trade=False, costs=_costs())
        self.assertEqual(tax, round(37 * 1000 * 0.001))

    def test_calc_sell_tax_etf_day_trade_is_not_halved(self):
        """ETF當沖不適用證交稅減半，稅率應跟非當沖的ETF一樣是0.1%。"""
        costs = _costs()
        tax_day_trade = calc_sell_tax(price=37, shares=1000, is_etf=True, is_day_trade=True, costs=costs)
        tax_not_day_trade = calc_sell_tax(price=37, shares=1000, is_etf=True, is_day_trade=False, costs=costs)
        self.assertEqual(tax_day_trade, tax_not_day_trade)
        self.assertEqual(tax_day_trade, round(37 * 1000 * 0.001))


if __name__ == "__main__":
    unittest.main()
