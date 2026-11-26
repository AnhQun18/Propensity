"""Unit tests enforcing zero OOT leakage, temporal embargo, and fold isolation for H=3."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from h3_pipeline.protocol_h3 import (
    assert_fold_isolation,
    assert_prior_null_at_cold_month,
    load_h3_spec,
    validate_h3_schedule,
)


class TestH3LeakageGuards(unittest.TestCase):
    def test_h3_spec_passes_zero_oot_leakage(self) -> None:
        spec = load_h3_spec(REPO_ROOT / "configs" / "h3_propensity_spec.json")
        self.assertEqual(spec["horizon"], 3)
        quarantine = set(spec["quarantined_months"])
        self.assertTrue({"2022-09", "2022-10", "2022-11", "2022-12", "2023-01"}.issubset(quarantine))
        self.assertEqual(len(spec["top50_features"]), 50)
        self.assertEqual(len(set(spec["top50_features"])), 50)

    def test_rejects_oot_or_embargo_leakage(self) -> None:
        spec = load_h3_spec(REPO_ROOT / "configs" / "h3_propensity_spec.json")

        bad_oot = {**spec, "val_schedule": [{"eval_month": "2022-09", "train_months": ["2022-02"]}]}
        with self.assertRaisesRegex(AssertionError, "OOT LEAKAGE"):
            validate_h3_schedule(bad_oot)

        bad_train = {**spec, "oot_test_schedule": [{"eval_month": "2022-11", "train_months": ["2022-09"]}]}
        with self.assertRaisesRegex(AssertionError, "OOT LEAKAGE"):
            validate_h3_schedule(bad_train)

        bad_gap = {**spec, "val_schedule": [{"eval_month": "2022-07", "train_months": ["2022-05"]}]}
        with self.assertRaisesRegex(AssertionError, "EMBARGO LEAKAGE"):
            validate_h3_schedule(bad_gap)

    def test_fold_isolation_catches_overlap(self) -> None:
        assert_fold_isolation(np.array(["c1", "c2"]), np.array(["c3", "c4"]), "clean")
        with self.assertRaisesRegex(AssertionError, "CLIENT OVERLAP"):
            assert_fold_isolation(np.array(["c1", "c2"]), np.array(["c2", "c3"]), "leaky")

    def test_cold_month_prior_is_nan(self) -> None:
        good = pd.DataFrame({
            "month": ["2022-02", "2022-03"],
            "prior_months_observed": [0, 1],
            "prior_1m_bought_any": [np.nan, 0.0],
            "prior_3m_buy_months": [np.nan, 0.0],
            "prior_all_bought_any": [np.nan, 0.0],
        })
        assert_prior_null_at_cold_month(good)

        bad = good.copy()
        bad.loc[0, "prior_1m_bought_any"] = 0.0
        with self.assertRaises(AssertionError):
            assert_prior_null_at_cold_month(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
