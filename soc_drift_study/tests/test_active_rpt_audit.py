"""Synthetic invariant tests for active_rpt_update_audit (no raw CMU data)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from active_rpt_update_audit import (
    current_support_eligible,
    fixed_periodic_events,
    strictly_previous_index,
    update_prediction,
)


class TestUpdateAudit(unittest.TestCase):
    def test_freeze_is_identity(self):
        base = np.array([0.1, 0.5, 0.9])
        np.testing.assert_allclose(update_prediction(base, 1.4, -0.1, "full", 0), base)

    def test_full_strength_and_shrink(self):
        base = np.array([0.2, 0.6])
        np.testing.assert_allclose(
            update_prediction(base, 1.2, 0.10, "full", 1),
            np.array([0.34, 0.82]),
        )
        np.testing.assert_allclose(
            update_prediction(base, 1.2, 0.10, "full", 0.5),
            np.array([0.27, 0.71]),
        )

    def test_bias_only(self):
        base = np.array([0.2, 0.6])
        np.testing.assert_allclose(
            update_prediction(base, 1.2, 0.10, "bias", 0.25),
            np.array([0.225, 0.625]),
        )

    def test_gate_blocks_unsupported_current(self):
        train = pd.DataFrame({"I_mean_A": [-16, -15, -14, -5, -4, -4]})
        bad_ref = pd.DataFrame({"I_mean_A": [-0.5, -0.6, -0.7]})
        good_ref = pd.DataFrame({"I_mean_A": [-15, -14, -0.6]})
        self.assertFalse(current_support_eligible(train, bad_ref))
        self.assertTrue(current_support_eligible(train, good_ref))
        base = np.array([0.25])
        np.testing.assert_allclose(
            update_prediction(base, 1.5, 0.15, "gated", 1, eligible=False),
            base,
        )

    def test_rpt_only_affects_later_missions(self):
        events = [25, 50, 75]
        self.assertIsNone(strictly_previous_index(events, 24))
        self.assertIsNone(strictly_previous_index(events, 25))
        self.assertEqual(strictly_previous_index(events, 26), 25)
        self.assertEqual(strictly_previous_index(events, 50), 25)
        self.assertEqual(strictly_previous_index(events, 76), 75)

    def test_fixed_event_schedule_never_exceeds_budget(self):
        refs = {i: {} for i in [10, 20, 30, 40, 50, 60]}
        chosen = fixed_periodic_events(refs, 15, budget=4)
        self.assertEqual(len(chosen), 4)
        self.assertTrue(all(x > 15 for x in chosen))
        self.assertEqual(sorted(chosen), chosen)

    def test_predictions_clipped(self):
        result = update_prediction(np.array([0.0, 1.0]), 1.5, 0.15, "full", 1)
        self.assertTrue(np.all((result >= 0.0) & (result <= 1.0)))


if __name__ == "__main__":
    unittest.main()
