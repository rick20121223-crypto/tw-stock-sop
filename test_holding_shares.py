import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import holding_shares
from holding_shares import (
    HUNDRED_LOT_LEVELS,
    MAJOR_HOLDER_LEVELS,
    THOUSAND_LOT_LEVELS,
    _aggregate_by_levels,
    archive_hundred_lot_snapshot,
    archive_thousand_lot_snapshot,
    archive_tdcc_snapshot,
    fetch_hundred_lot_trend,
    fetch_thousand_lot_trend,
)


def _snapshot(date_str="2026-11-13"):
    """模擬TDCC單週快照：2330在level 10~15都有人持股，用來驗證不同
    級距集合的加總結果各自正確。"""
    rows = [
        {"date": pd.Timestamp(date_str), "stock_id": "2330", "level": level,
         "people": 10, "unit": 100_000 * level, "percent": 1.0}
        for level in range(1, 16)
    ]
    return pd.DataFrame(rows)


class TestLevelConstants(unittest.TestCase):
    def test_hundred_lot_levels_is_superset_of_thousand_and_major(self):
        self.assertTrue(THOUSAND_LOT_LEVELS.issubset(HUNDRED_LOT_LEVELS))
        self.assertTrue(MAJOR_HOLDER_LEVELS.issubset(HUNDRED_LOT_LEVELS))
        self.assertEqual(THOUSAND_LOT_LEVELS, {15})


class TestAggregateByLevels(unittest.TestCase):
    def test_sums_only_matching_levels(self):
        result = _aggregate_by_levels(_snapshot(), {12, 13})
        row = result[result["stock_id"] == "2330"].iloc[0]
        self.assertEqual(row["unit"], 100_000 * 12 + 100_000 * 13)

    def test_empty_snapshot_returns_empty_frame(self):
        result = _aggregate_by_levels(pd.DataFrame(), {15})
        self.assertTrue(result.empty)
        self.assertEqual(list(result.columns), ["date", "stock_id", "percent", "people", "unit"])


class TestArchiveAndFetchRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self.tmpdir.name)
        self.patches = [
            patch.object(holding_shares, "TDCC_ARCHIVE_FILE", tmp_path / "major.csv"),
            patch.object(holding_shares, "TDCC_HUNDRED_LOT_FILE", tmp_path / "hundred.csv"),
            patch.object(holding_shares, "TDCC_THOUSAND_LOT_FILE", tmp_path / "thousand.csv"),
            patch.object(holding_shares, "DATA_DIR", tmp_path),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmpdir.cleanup()

    def test_thousand_lot_only_counts_level_15(self):
        archive_thousand_lot_snapshot(_snapshot())
        trend = fetch_thousand_lot_trend("2330")
        self.assertEqual(len(trend), 1)
        self.assertEqual(trend.iloc[0]["unit"], 100_000 * 15)

    def test_hundred_lot_counts_levels_10_through_15(self):
        archive_hundred_lot_snapshot(_snapshot())
        trend = fetch_hundred_lot_trend("2330")
        expected = sum(100_000 * level for level in range(10, 16))
        self.assertEqual(trend.iloc[0]["unit"], expected)

    def test_major_holder_archive_unaffected_by_new_buckets(self):
        archive_tdcc_snapshot(_snapshot())
        trend = holding_shares.fetch_major_holder_trend("2330")
        expected = sum(100_000 * level for level in range(12, 16))
        self.assertEqual(trend.iloc[0]["unit"], expected)

    def test_duplicate_week_is_not_double_counted(self):
        archive_hundred_lot_snapshot(_snapshot())
        archive_hundred_lot_snapshot(_snapshot())
        trend = fetch_hundred_lot_trend("2330")
        self.assertEqual(len(trend), 1)


if __name__ == "__main__":
    unittest.main()
