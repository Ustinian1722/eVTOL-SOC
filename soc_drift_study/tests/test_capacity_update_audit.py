"""Noncircular capacity-state update timing and simple-policy tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from capacity_state_update_audit import evaluate, observed_capacity, selected_rpt_indices


class TestCapacityUpdate(unittest.TestCase):
    def test_strict_past_no_current_target(self):
        refs = [(0, 3.0), (50, 2.8)]
        self.assertEqual(observed_capacity(refs, 50, "last_reference"), 3.0)
        self.assertAlmostEqual(observed_capacity(refs, 51, "last_reference"), 2.8)
        self.assertAlmostEqual(observed_capacity(refs, 100, "freeze_first"), 3.0)

    def test_decline_forecast_extrapolates_only_past(self):
        refs = [(0, 3.0), (50, 2.9)]
        self.assertAlmostEqual(observed_capacity(refs, 100, "two_reference_rate_aware"), 2.8)
        self.assertEqual(observed_capacity([(0, 3.0)], 100, "two_reference_rate_aware"), 3.0)

    def test_offline_schedule_obeys_budget_and_excludes_endpoints(self):
        ids = [0, 50, 100, 150, 200, 250, 300]
        chosen = selected_rpt_indices(ids, budget=4)
        self.assertEqual(len(chosen), 4)
        self.assertNotIn(ids[0], chosen)
        self.assertNotIn(ids[-1], chosen)

    def test_selected_measurement_released_after_prediction(self):
        toy = pd.DataFrame({
            "cell_id": ["TEST"] * 7,
            "charge_cycle_index": [0, 50, 100, 150, 200, 250, 300],
            "QDischarge_max_Ah": [3.0, 2.98, 2.95, 2.91, 2.89, 2.85, 2.81],
        })
        preds, refs = evaluate(toy)
        # At the first selected RPT, only the initialization is visible.
        first_selected = int(refs.iloc[0]["selected_rpt_cycles"].split(";")[0])
        scored = preds[(preds.target_rpt_index == first_selected) & (preds.policy == "last_reference")]
        self.assertEqual(int(scored.observed_refs_before_prediction.iloc[0]), 1)
        self.assertEqual(float(scored.prediction_capacity_Ah.iloc[0]), 3.0)
        after = preds[(preds.target_rpt_index == 300) & (preds.policy == "last_reference")]
        self.assertLess(float(after.prediction_capacity_Ah.iloc[0]), 3.0)


if __name__ == "__main__":
    unittest.main()
