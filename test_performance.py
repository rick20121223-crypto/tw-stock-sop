import math
import unittest
from datetime import date

from performance import (
    ClosedTrade,
    CorporateAction,
    Fill,
    PerformanceStats,
    Position,
    calc_fee,
    calc_sell_tax,
    check_live_trading_gate,
    compare_to_benchmark,
    compute_daily_equity,
    compute_pending_dividend_cash,
    compute_positions,
    compute_stats,
    match_fifo,
)


def _gate_rules(min_trade_count=30, max_drawdown_pct=0.15, require_beat_benchmark=True):
    return {
        "live_trading_gate": {
            "min_trade_count": min_trade_count,
            "max_drawdown_pct": max_drawdown_pct,
            "require_beat_benchmark": require_beat_benchmark,
        }
    }


def _stats(total_trades=30, max_drawdown=100_000.0):
    return PerformanceStats(
        total_trades=total_trades, win_rate=0.6, avg_win=10_000.0, avg_loss=-5_000.0,
        profit_factor=2.0, expectancy=3_000.0, avg_r_multiple=1.2, trades_with_r=total_trades,
        equity_curve=[], max_drawdown=max_drawdown, avg_holding_days=10.0,
    )


def _costs(discount: float = 1.0, min_fee_round_lot: float = 0, min_fee_odd_lot: float = 0) -> dict:
    return {
        "brokerage_fee": {
            "legal_rate": 0.001425,
            "discount": discount,
            "min_fee_round_lot": min_fee_round_lot,
            "min_fee_odd_lot": min_fee_odd_lot,
        },
        "securities_tax": {
            "stock_rate": 0.003,
            "etf_rate": 0.001,
            "day_trade_rate": 0.0015,
            "day_trade_discount_expires": "2027-12-31",
        },
    }


class TestFees(unittest.TestCase):
    def test_calc_fee_no_discount_truncates_not_rounds(self):
        """37*1000*0.001425=52.725——無條件捨去（元大官網的算法）應該是52，
        不是四捨五入的53，用小數部分>=0.5的數字驗證真的是floor不是round。"""
        fee = calc_fee(price=37, shares=1000, costs=_costs())
        self.assertEqual(fee, 52)
        self.assertEqual(fee, math.floor(37 * 1000 * 0.001425 * 1.0))

    def test_calc_fee_with_discount_truncates(self):
        fee = calc_fee(price=37, shares=1000, costs=_costs(discount=0.6))
        self.assertEqual(fee, 31)
        self.assertEqual(fee, math.floor(37 * 1000 * 0.001425 * 0.6))

    def test_calc_fee_hits_round_lot_min_fee_floor(self):
        costs = _costs(discount=0.6, min_fee_round_lot=20, min_fee_odd_lot=1)
        fee = calc_fee(price=5, shares=1000, costs=costs)  # 整股（1000的倍數）
        raw_floor = math.floor(5 * 1000 * 0.001425 * 0.6)
        self.assertEqual(raw_floor, 4)  # 沒有min_fee時原本只有4
        self.assertEqual(fee, 20)  # 被min_fee_round_lot墊高

    def test_calc_fee_hits_odd_lot_min_fee_floor(self):
        costs = _costs(discount=0.6, min_fee_round_lot=20, min_fee_odd_lot=1)
        fee = calc_fee(price=5, shares=100, costs=costs)  # 零股（不是1000的倍數）
        raw_floor = math.floor(5 * 100 * 0.001425 * 0.6)
        self.assertEqual(raw_floor, 0)
        self.assertEqual(fee, 1)  # 被min_fee_odd_lot墊高（不是round_lot的20）

    def test_calc_sell_tax_stock_not_day_trade(self):
        tax = calc_sell_tax(price=37, shares=1000, is_etf=False, is_day_trade=False, costs=_costs())
        self.assertEqual(tax, math.floor(37 * 1000 * 0.003))

    def test_calc_sell_tax_stock_day_trade_is_halved(self):
        tax = calc_sell_tax(price=38, shares=1000, is_etf=False, is_day_trade=True, costs=_costs(),
                             trade_date=date(2027, 1, 1))
        self.assertEqual(tax, math.floor(38 * 1000 * 0.0015))

    def test_calc_sell_tax_stock_day_trade_after_expiry_not_halved(self):
        """超過day_trade_discount_expires後，現股當沖減半優惠自動失效，
        退回一般股票稅率（不是day_trade_rate）。"""
        tax = calc_sell_tax(price=38, shares=1000, is_etf=False, is_day_trade=True, costs=_costs(),
                             trade_date=date(2028, 1, 1))
        self.assertEqual(tax, math.floor(38 * 1000 * 0.003))

    def test_calc_sell_tax_defaults_trade_date_to_today(self):
        """沒傳trade_date時用today()判斷減半優惠是否仍有效——這裡只驗證
        「不會因為沒傳就整個壞掉」，不特別驗證今天在落日日期前後哪一邊。"""
        costs = _costs()
        tax = calc_sell_tax(price=38, shares=1000, is_etf=False, is_day_trade=True, costs=costs)
        self.assertIn(tax, (math.floor(38 * 1000 * 0.0015), math.floor(38 * 1000 * 0.003)))

    def test_calc_sell_tax_etf_not_day_trade(self):
        tax = calc_sell_tax(price=37, shares=1000, is_etf=True, is_day_trade=False, costs=_costs())
        self.assertEqual(tax, math.floor(37 * 1000 * 0.001))

    def test_calc_sell_tax_etf_day_trade_is_not_halved(self):
        """ETF當沖不適用證交稅減半，稅率應跟非當沖的ETF一樣是0.1%。"""
        costs = _costs()
        tax_day_trade = calc_sell_tax(price=37, shares=1000, is_etf=True, is_day_trade=True, costs=costs)
        tax_not_day_trade = calc_sell_tax(price=37, shares=1000, is_etf=True, is_day_trade=False, costs=costs)
        self.assertEqual(tax_day_trade, tax_not_day_trade)
        self.assertEqual(tax_day_trade, math.floor(37 * 1000 * 0.001))

    def test_calc_sell_tax_truncates_not_rounds(self):
        """333.5*1000*0.001=333.5剛好卡在無條件捨去跟四捨五入(銀行家捨入法
        會進位成334)會給出不同結果的臨界點，驗證真的是floor。"""
        tax = calc_sell_tax(price=333.5, shares=1000, is_etf=True, is_day_trade=False, costs=_costs())
        self.assertEqual(tax, 333)


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

    def test_stock_dividend_increases_shares_and_lowers_cost_per_share(self):
        """持有1000股、每股成本100，除息日配發股票股利ratio=0.1（每股配
        發0.1股）後應變成1100股、每股成本約90.91（總成本100,000不變）。"""
        costs = _costs()
        buy = Fill(id="b12", account_id="v1", code="2330", side="買", price=100, shares=1000,
                   fill_date=date(2026, 9, 1), fee=calc_fee(100, 1000, costs))
        action = CorporateAction(account_id="v1", code="2330", ex_date=date(2026, 9, 15),
                                  stock_dividend_ratio=0.1)
        [pos] = compute_positions([buy], corporate_actions=[action])
        self.assertEqual(pos.shares, 1100)
        self.assertAlmostEqual(pos.avg_cost, 100_000 / 1100)

    def test_stock_dividend_before_lot_existed_does_not_apply(self):
        """除息日在這筆買進之前成立的股票股利，不該套用到之後才買進的
        這批股數上（這批股數當時根本還沒持有，不該收到股利）。"""
        costs = _costs()
        buy = Fill(id="b13", account_id="v1", code="2330", side="買", price=100, shares=1000,
                   fill_date=date(2026, 9, 20), fee=calc_fee(100, 1000, costs))
        action = CorporateAction(account_id="v1", code="2330", ex_date=date(2026, 9, 15),
                                  stock_dividend_ratio=0.1)
        [pos] = compute_positions([buy], corporate_actions=[action])
        self.assertEqual(pos.shares, 1000)
        self.assertAlmostEqual(pos.avg_cost, 100.0)

    def test_corporate_actions_none_keeps_old_behavior(self):
        costs = _costs()
        buy = Fill(id="b14", account_id="v1", code="2330", side="買", price=100, shares=1000,
                   fill_date=date(2026, 9, 1), fee=calc_fee(100, 1000, costs))
        [pos] = compute_positions([buy])
        self.assertEqual(pos.shares, 1000)
        self.assertAlmostEqual(pos.avg_cost, 100.0)


class TestComputePendingDividendCash(unittest.TestCase):
    def test_cash_dividend_credited_for_shares_held_before_ex_date(self):
        costs = _costs()
        buy = Fill(id="b15", account_id="v1", code="2330", side="買", price=100, shares=1000,
                   fill_date=date(2026, 9, 1), fee=calc_fee(100, 1000, costs))
        action = CorporateAction(account_id="v1", code="2330", ex_date=date(2026, 9, 15),
                                  cash_per_share=2.0)
        [credit] = compute_pending_dividend_cash([buy], [action])
        self.assertEqual(credit["held_shares"], 1000)
        self.assertAlmostEqual(credit["cash_amount"], 2000.0)

    def test_shares_bought_after_ex_date_not_credited(self):
        costs = _costs()
        buy = Fill(id="b16", account_id="v1", code="2330", side="買", price=100, shares=1000,
                   fill_date=date(2026, 9, 20), fee=calc_fee(100, 1000, costs))
        action = CorporateAction(account_id="v1", code="2330", ex_date=date(2026, 9, 15),
                                  cash_per_share=2.0)
        self.assertEqual(compute_pending_dividend_cash([buy], [action]), [])

    def test_shares_fully_sold_before_ex_date_not_credited(self):
        costs = _costs()
        buy = Fill(id="b17", account_id="v1", code="2330", side="買", price=100, shares=1000,
                   fill_date=date(2026, 9, 1), fee=calc_fee(100, 1000, costs))
        sell = Fill(id="s11", account_id="v1", code="2330", side="賣", price=110, shares=1000,
                    fill_date=date(2026, 9, 10), fee=calc_fee(110, 1000, costs),
                    tax=calc_sell_tax(110, 1000, is_etf=False, is_day_trade=False, costs=costs,
                                       trade_date=date(2026, 9, 10)))
        action = CorporateAction(account_id="v1", code="2330", ex_date=date(2026, 9, 15),
                                  cash_per_share=2.0)
        self.assertEqual(compute_pending_dividend_cash([buy, sell], [action]), [])


class TestComputeStats(unittest.TestCase):
    def _closed_trade(self, net_pnl, r_multiple, exit_date, holding_days, source="system"):
        return ClosedTrade(
            account_id="v1", code="2330", entry_date=exit_date, exit_date=exit_date, shares=1000,
            avg_entry_price=100.0, avg_exit_price=100.0, buy_cost=0.0, sell_cost=0.0,
            gross_pnl=net_pnl, net_pnl=net_pnl, r_multiple=r_multiple,
            is_day_trade=False, source=source, holding_days=holding_days,
        )

    def test_stats_over_three_trades(self):
        ct1 = self._closed_trade(8369, 1.3948333333333334, date(2026, 1, 20), 15)
        ct2 = self._closed_trade(-11542, -1.9236666666666666, date(2026, 2, 10), 7)
        ct3 = self._closed_trade(8966, None, date(2026, 3, 15), 13)
        stats = compute_stats([ct1, ct2, ct3])

        self.assertEqual(stats.total_trades, 3)
        self.assertAlmostEqual(stats.win_rate, 2 / 3)
        self.assertAlmostEqual(stats.avg_win, (8369 + 8966) / 2)
        self.assertAlmostEqual(stats.avg_loss, -11542)
        self.assertAlmostEqual(stats.profit_factor, 8667.5 / 11542)
        self.assertAlmostEqual(stats.expectancy, (2 / 3) * 8667.5 + (1 / 3) * (-11542))
        self.assertEqual(stats.trades_with_r, 2)
        self.assertAlmostEqual(stats.avg_r_multiple, (1.3948333333333334 - 1.9236666666666666) / 2)
        self.assertEqual(
            stats.equity_curve,
            [
                (date(2026, 1, 20), 8369),
                (date(2026, 2, 10), 8369 - 11542),
                (date(2026, 3, 15), 8369 - 11542 + 8966),
            ],
        )
        self.assertAlmostEqual(stats.max_drawdown, 11542)
        self.assertAlmostEqual(stats.avg_holding_days, (15 + 7 + 13) / 3)

    def test_empty_list_does_not_crash(self):
        stats = compute_stats([])
        self.assertEqual(stats.total_trades, 0)
        self.assertEqual(stats.win_rate, 0.0)
        self.assertIsNone(stats.profit_factor)
        self.assertIsNone(stats.avg_r_multiple)
        self.assertEqual(stats.equity_curve, [])

    def test_source_filter_excludes_manual(self):
        ct_system = self._closed_trade(8369, 1.0, date(2026, 1, 20), 15, source="system")
        ct_manual = self._closed_trade(-500, -0.5, date(2026, 1, 25), 5, source="manual")
        stats_all = compute_stats([ct_system, ct_manual])
        stats_system_only = compute_stats([ct_system, ct_manual], source="system")

        self.assertEqual(stats_all.total_trades, 2)
        self.assertEqual(stats_system_only.total_trades, 1)
        self.assertEqual(stats_system_only.avg_win, 8369)


class TestCompareToBenchmark(unittest.TestCase):
    def _closed_trade(self, entry_date, exit_date, avg_entry_price, shares, net_pnl):
        return ClosedTrade(
            account_id="v1", code="2330", entry_date=entry_date, exit_date=exit_date, shares=shares,
            avg_entry_price=avg_entry_price, avg_exit_price=avg_entry_price, buy_cost=0.0,
            sell_cost=0.0, gross_pnl=net_pnl, net_pnl=net_pnl, r_multiple=None,
            is_day_trade=False, source="system", holding_days=1,
        )

    def test_compare_to_0050(self):
        ct_a = self._closed_trade(date(2026, 1, 5), date(2026, 1, 20), 100.0, 1000, 5000.0)
        ct_b = self._closed_trade(date(2026, 3, 2), date(2026, 3, 15), 200.0, 500, -2000.0)
        benchmark_prices = [
            (date(2026, 1, 1), 130.0),
            (date(2026, 1, 20), 135.0),
            (date(2026, 3, 15), 140.0),
        ]
        result = compare_to_benchmark([ct_a, ct_b], benchmark_prices)

        expected_portfolio_return = (5000.0 - 2000.0) / (100.0 * 1000 + 200.0 * 500)
        expected_benchmark_return = 140.0 / 130.0 - 1

        self.assertAlmostEqual(result["portfolio_return"], expected_portfolio_return)
        self.assertAlmostEqual(result["benchmark_return"], expected_benchmark_return)
        self.assertEqual(result["period_start"], date(2026, 1, 5))
        self.assertEqual(result["period_end"], date(2026, 3, 15))

    def test_empty_closed_trades_returns_none(self):
        result = compare_to_benchmark([], [(date(2026, 1, 1), 100.0)])
        self.assertIsNone(result["portfolio_return"])
        self.assertIsNone(result["benchmark_return"])


class TestComputeDailyEquity(unittest.TestCase):
    def test_marks_position_to_latest_close(self):
        positions = [Position(account_id="v1", code="2330", shares=100, avg_cost=100.0)]
        result = compute_daily_equity(
            account_id="v1", initial_capital=1_000_000.0, cash=900_000.0,
            positions=positions, price_lookup=lambda code, d: 110.0,
            as_of_date=date(2026, 6, 1),
        )
        self.assertEqual(result.positions_value, 100 * 110.0)
        self.assertEqual(result.total_equity, 900_000.0 + 100 * 110.0)
        self.assertEqual(result.cumulative_pnl, result.total_equity - 1_000_000.0)

    def test_missing_price_falls_back_to_avg_cost(self):
        positions = [Position(account_id="v1", code="2330", shares=100, avg_cost=100.0)]
        result = compute_daily_equity(
            account_id="v1", initial_capital=1_000_000.0, cash=900_000.0,
            positions=positions, price_lookup=lambda code, d: None,
            as_of_date=date(2026, 6, 1),
        )
        self.assertEqual(result.positions_value, 100 * 100.0)

    def test_no_positions_equity_is_just_cash(self):
        result = compute_daily_equity(
            account_id="v1", initial_capital=1_000_000.0, cash=1_000_000.0,
            positions=[], price_lookup=lambda code, d: None, as_of_date=date(2026, 6, 1),
        )
        self.assertEqual(result.positions_value, 0.0)
        self.assertEqual(result.total_equity, 1_000_000.0)
        self.assertEqual(result.cumulative_pnl, 0.0)


class TestCheckLiveTradingGate(unittest.TestCase):
    def test_all_three_conditions_pass(self):
        stats = _stats(total_trades=30, max_drawdown=100_000.0)  # 10% of 1,000,000 base_equity
        result = check_live_trading_gate(
            stats, base_equity=1_000_000.0, portfolio_return=0.12, benchmark_return=0.05,
            rules=_gate_rules(),
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "達標")
        self.assertTrue(result.trade_count_ok)
        self.assertTrue(result.max_drawdown_ok)
        self.assertAlmostEqual(result.max_drawdown_pct, 0.10)
        self.assertTrue(result.beats_benchmark)

    def test_fails_on_trade_count(self):
        stats = _stats(total_trades=10, max_drawdown=100_000.0)
        result = check_live_trading_gate(
            stats, base_equity=1_000_000.0, portfolio_return=0.12, benchmark_return=0.05,
            rules=_gate_rules(),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "資料不足")
        self.assertFalse(result.trade_count_ok)
        self.assertTrue(result.max_drawdown_ok)

    def test_trade_count_short_is_insufficient_data_even_if_other_conditions_fail_too(self):
        """交易筆數不足時，狀態固定是「資料不足」，不受其他條件（這裡
        連回撤也超標）影響，不會被誤判成「未達標」。"""
        stats = _stats(total_trades=5, max_drawdown=200_000.0)
        result = check_live_trading_gate(
            stats, base_equity=1_000_000.0, portfolio_return=0.12, benchmark_return=0.05,
            rules=_gate_rules(),
        )
        self.assertEqual(result.status, "資料不足")

    def test_fails_on_drawdown(self):
        stats = _stats(total_trades=30, max_drawdown=200_000.0)  # 20% > 15%門檻
        result = check_live_trading_gate(
            stats, base_equity=1_000_000.0, portfolio_return=0.12, benchmark_return=0.05,
            rules=_gate_rules(),
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "未達標")
        self.assertFalse(result.max_drawdown_ok)
        self.assertAlmostEqual(result.max_drawdown_pct, 0.20)

    def test_fails_when_not_beating_benchmark(self):
        stats = _stats(total_trades=30, max_drawdown=100_000.0)
        result = check_live_trading_gate(
            stats, base_equity=1_000_000.0, portfolio_return=0.03, benchmark_return=0.05,
            rules=_gate_rules(),
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.beats_benchmark)

    def test_missing_benchmark_data_fails_closed_when_required(self):
        stats = _stats(total_trades=30, max_drawdown=100_000.0)
        result = check_live_trading_gate(
            stats, base_equity=1_000_000.0, portfolio_return=None, benchmark_return=None,
            rules=_gate_rules(),
        )
        self.assertFalse(result.passed)
        self.assertIsNone(result.beats_benchmark)

    def test_benchmark_not_required_can_still_pass(self):
        stats = _stats(total_trades=30, max_drawdown=100_000.0)
        result = check_live_trading_gate(
            stats, base_equity=1_000_000.0, portfolio_return=0.03, benchmark_return=0.05,
            rules=_gate_rules(require_beat_benchmark=False),
        )
        self.assertTrue(result.passed)
        self.assertFalse(result.beats_benchmark)  # 數值仍然如實回報，只是不列入passed判斷


if __name__ == "__main__":
    unittest.main()
