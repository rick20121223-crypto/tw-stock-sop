import unittest
from datetime import date

from performance import Fill, calc_fee, calc_sell_tax, compute_positions, match_fifo


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


class TestMatchFifo(unittest.TestCase):
    def test_single_winning_trade(self):
        costs = _costs()
        buy = Fill(id="b1", account_id="v1", code="2330", side="買", price=101, shares=1000,
                   fill_date=date(2026, 1, 5), fee=calc_fee(101, 1000, costs),
                   planned_stop_loss=95)
        sell = Fill(id="s1", account_id="v1", code="2330", side="賣", price=110, shares=1000,
                    fill_date=date(2026, 1, 20), fee=calc_fee(110, 1000, costs),
                    tax=calc_sell_tax(110, 1000, is_etf=False, is_day_trade=False, costs=costs))
        [ct] = match_fifo([buy, sell])

        expected_net_pnl = (110 * 1000 - 101 * 1000) - buy.fee - sell.fee - sell.tax
        expected_r = expected_net_pnl / ((101 - 95) * 1000)

        self.assertEqual(ct.account_id, "v1")
        self.assertEqual(ct.net_pnl, expected_net_pnl)
        self.assertAlmostEqual(ct.r_multiple, expected_r, places=9)
        self.assertEqual(ct.holding_days, 15)
        self.assertFalse(ct.is_day_trade)
        self.assertEqual(ct.source, "system")
        self.assertEqual(ct.shares, 1000)

    def test_single_losing_trade(self):
        costs = _costs()
        buy = Fill(id="b2", account_id="v1", code="2330", side="買", price=101, shares=1000,
                   fill_date=date(2026, 2, 3), fee=calc_fee(101, 1000, costs),
                   planned_stop_loss=95)
        sell = Fill(id="s2", account_id="v1", code="2330", side="賣", price=90, shares=1000,
                    fill_date=date(2026, 2, 10), fee=calc_fee(90, 1000, costs),
                    tax=calc_sell_tax(90, 1000, is_etf=False, is_day_trade=False, costs=costs))
        [ct] = match_fifo([buy, sell])

        expected_net_pnl = (90 * 1000 - 101 * 1000) - buy.fee - sell.fee - sell.tax

        self.assertEqual(ct.net_pnl, expected_net_pnl)
        self.assertLess(ct.net_pnl, 0)
        self.assertLess(ct.r_multiple, 0)

    def test_batched_entry_and_exit(self):
        costs = _costs()
        buy1 = Fill(id="b3", account_id="v1", code="2330", side="買", price=100, shares=500,
                    fill_date=date(2026, 3, 2), fee=calc_fee(100, 500, costs),
                    planned_stop_loss=95)
        buy2 = Fill(id="b4", account_id="v1", code="2330", side="買", price=102, shares=500,
                    fill_date=date(2026, 3, 3), fee=calc_fee(102, 500, costs),
                    planned_stop_loss=97)
        sell1 = Fill(id="s3", account_id="v1", code="2330", side="賣", price=110, shares=700,
                     fill_date=date(2026, 3, 10), fee=calc_fee(110, 700, costs),
                     tax=calc_sell_tax(110, 700, is_etf=False, is_day_trade=False, costs=costs))
        sell2 = Fill(id="s4", account_id="v1", code="2330", side="賣", price=112, shares=300,
                     fill_date=date(2026, 3, 15), fee=calc_fee(112, 300, costs),
                     tax=calc_sell_tax(112, 300, is_etf=False, is_day_trade=False, costs=costs))
        [ct] = match_fifo([buy1, buy2, sell1, sell2])

        expected_avg_entry = (500 * 100 + 500 * 102) / 1000
        expected_avg_exit = (700 * 110 + 300 * 112) / 1000
        expected_gross_pnl = (700 * 110 + 300 * 112) - (500 * 100 + 500 * 102)
        expected_buy_cost = buy1.fee + buy2.fee
        expected_sell_cost = (sell1.fee + sell1.tax) + (sell2.fee + sell2.tax)
        expected_net_pnl = expected_gross_pnl - expected_buy_cost - expected_sell_cost
        # 風險金額依「配對後子批次」各自的進場價/停損價加權（sell1拆成
        # 500股配對buy1 + 200股配對buy2，sell2的300股配對buy2剩餘部分）
        expected_risk = 500 * (100 - 95) + 200 * (102 - 97) + 300 * (102 - 97)
        expected_r = expected_net_pnl / expected_risk

        self.assertAlmostEqual(ct.avg_entry_price, expected_avg_entry)
        self.assertAlmostEqual(ct.avg_exit_price, expected_avg_exit)
        self.assertEqual(ct.net_pnl, expected_net_pnl)
        self.assertAlmostEqual(ct.r_multiple, expected_r, places=9)
        self.assertEqual(ct.holding_days, 13)
        self.assertEqual(ct.shares, 1000)

    def test_missing_stop_loss_excluded_from_r_but_counted_in_pnl(self):
        costs = _costs()
        buy = Fill(id="b5", account_id="v1", code="2330", side="買", price=101, shares=1000,
                   fill_date=date(2026, 4, 1), fee=calc_fee(101, 1000, costs))  # 沒填planned_stop_loss
        sell = Fill(id="s5", account_id="v1", code="2330", side="賣", price=110, shares=1000,
                    fill_date=date(2026, 4, 20), fee=calc_fee(110, 1000, costs),
                    tax=calc_sell_tax(110, 1000, is_etf=False, is_day_trade=False, costs=costs))
        [ct] = match_fifo([buy, sell])
        self.assertIsNone(ct.r_multiple)
        self.assertGreater(ct.net_pnl, 0)

    def test_etf_day_trade_fee_tax_already_fixed_on_fill(self):
        """fee/tax是成交當下就算好固定在Fill上的，match_fifo只負責加總，
        不重新判斷is_etf/is_day_trade——這裡確認即使是ETF當沖，match_fifo
        也只是原封不動地加總Fill上已經算好的tax(0.1%)，不會自己另外套用
        day_trade_rate。"""
        costs = _costs()
        correct_tax = calc_sell_tax(145, 1000, is_etf=True, is_day_trade=True, costs=costs)
        buy = Fill(id="b6", account_id="v1", code="0050", side="買", price=141, shares=1000,
                   fill_date=date(2026, 4, 1), fee=calc_fee(141, 1000, costs),
                   planned_stop_loss=135)
        sell = Fill(id="s6", account_id="v1", code="0050", side="賣", price=145, shares=1000,
                    fill_date=date(2026, 4, 1), fee=calc_fee(145, 1000, costs),
                    tax=correct_tax, is_day_trade=True)
        [ct] = match_fifo([buy, sell])

        expected_net_pnl = (145 * 1000 - 141 * 1000) - buy.fee - sell.fee - sell.tax
        self.assertTrue(ct.is_day_trade)
        self.assertEqual(ct.net_pnl, expected_net_pnl)
        self.assertEqual(sell.tax, round(145 * 1000 * 0.001))  # etf_rate，不是day_trade_rate

    def test_sell_more_than_held_raises(self):
        costs = _costs()
        buy = Fill(id="b7", account_id="v1", code="2330", side="買", price=100, shares=100,
                   fill_date=date(2026, 5, 1), fee=calc_fee(100, 100, costs))
        sell = Fill(id="s7", account_id="v1", code="2330", side="賣", price=110, shares=200,
                    fill_date=date(2026, 5, 2), fee=calc_fee(110, 200, costs),
                    tax=calc_sell_tax(110, 200, is_etf=False, is_day_trade=False, costs=costs))
        with self.assertRaises(ValueError):
            match_fifo([buy, sell])

    def test_different_accounts_are_independent(self):
        """同一股票代號，不同帳戶各自獨立配對，不會互相影響。"""
        costs = _costs()
        buy_v1 = Fill(id="b8", account_id="v1", code="2330", side="買", price=100, shares=100,
                      fill_date=date(2026, 6, 1), fee=calc_fee(100, 100, costs))
        buy_v2 = Fill(id="b9", account_id="v2", code="2330", side="買", price=200, shares=100,
                      fill_date=date(2026, 6, 1), fee=calc_fee(200, 100, costs))
        sell_v1 = Fill(id="s8", account_id="v1", code="2330", side="賣", price=110, shares=100,
                       fill_date=date(2026, 6, 10), fee=calc_fee(110, 100, costs),
                       tax=calc_sell_tax(110, 100, is_etf=False, is_day_trade=False, costs=costs))
        closed = match_fifo([buy_v1, buy_v2, sell_v1])
        self.assertEqual(len(closed), 1)  # v2的買入還沒被賣出配對，不會出現在已平倉清單
        self.assertEqual(closed[0].account_id, "v1")


class TestComputePositions(unittest.TestCase):
    def test_open_position_after_partial_fill(self):
        costs = _costs()
        buy = Fill(id="b10", account_id="v1", code="2330", side="買", price=100, shares=1000,
                   fill_date=date(2026, 7, 1), fee=calc_fee(100, 1000, costs))
        sell = Fill(id="s9", account_id="v1", code="2330", side="賣", price=110, shares=400,
                    fill_date=date(2026, 7, 10), fee=calc_fee(110, 400, costs),
                    tax=calc_sell_tax(110, 400, is_etf=False, is_day_trade=False, costs=costs))
        [pos] = compute_positions([buy, sell])
        self.assertEqual(pos.account_id, "v1")
        self.assertEqual(pos.code, "2330")
        self.assertEqual(pos.shares, 600)
        self.assertAlmostEqual(pos.avg_cost, 100.0)

    def test_fully_closed_position_produces_no_row(self):
        costs = _costs()
        buy = Fill(id="b11", account_id="v1", code="2330", side="買", price=100, shares=500,
                   fill_date=date(2026, 8, 1), fee=calc_fee(100, 500, costs))
        sell = Fill(id="s10", account_id="v1", code="2330", side="賣", price=110, shares=500,
                    fill_date=date(2026, 8, 10), fee=calc_fee(110, 500, costs),
                    tax=calc_sell_tax(110, 500, is_etf=False, is_day_trade=False, costs=costs))
        positions = compute_positions([buy, sell])
        self.assertEqual(positions, [])

    def test_no_fills_at_all_produces_no_rows(self):
        self.assertEqual(compute_positions([]), [])


if __name__ == "__main__":
    unittest.main()
