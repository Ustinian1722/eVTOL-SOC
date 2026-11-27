"""Regression tests for nonuniform quote, causal release and budget queue."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from test_stateful_marginal_roi_maintenance import toy_cell
from variable_cost_queue_nested import quote, simulate


class ConstantPredictor:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class VariableCostQueueTests(unittest.TestCase):
    def setUp(self):
        self.cell = toy_cell(n=210)
        self.models = (ConstantPredictor(0.9), ConstantPredictor(0.9))

    def test_quote_uses_only_completed_mission_rows_and_duration(self):
        mission = self.cell.test[24]
        expected = 0.1 * (0.5 + 0.25 * mission.X.shape[0] / 100
                          + 0.25 * mission.duration / 960)
        self.assertAlmostEqual(quote(mission), expected, places=14)

    def test_release_is_exactly_purchase_plus_lag_plus_one(self):
        for lag in (0, 5, 25):
            with self.subTest(lag=lag):
                result = simulate(self.cell, self.models, "delay175_12",
                                  cap=0.08, lag=lag, variable=True)
                self.assertEqual(result["events"], "24")
                activation = 24 + lag + 1
                expected = self.cell.frozen_loss.copy()
                expected[activation:] = self.cell.loss[24][activation:]
                self.assertAlmostEqual(result["MAE_pp"],
                                       float(expected.mean()), places=12)
                self.assertLessEqual(result["cost"], 0.08 + 1e-12)
                self.assertEqual(result["purchases"], 1)

    def test_pending_reference_blocks_overlapping_purchase(self):
        result = simulate(self.cell, self.models, "delay75_12",
                          cap=0.20, lag=25, variable=True)
        self.assertEqual(result["events"], "24;99;174")
        self.assertEqual(result["purchases"], 3)
        self.assertLessEqual(result["cost"], 0.20 + 1e-12)

    def test_cost_aware_threshold_changes_decision_but_not_quote(self):
        models = (ConstantPredictor(0.55), ConstantPredictor(0.9))
        variable = simulate(self.cell, models, "delay175_12",
                            cap=0.08, lag=0, variable=True)
        flat = simulate(self.cell, models, "delay175_12",
                        cap=0.08, lag=0, variable=False)
        self.assertEqual(variable["events"], "24")
        self.assertEqual(flat["events"], "")
        self.assertAlmostEqual(variable["cost"], quote(self.cell.test[24]), places=14)
        self.assertEqual(flat["cost"], 0.0)

    def test_unpurchased_future_labels_do_not_change_online_purchase(self):
        altered = copy.deepcopy(self.cell)
        for mission in altered.test[25:]:
            mission.y[:] = 0.0
        a = simulate(self.cell, self.models, "delay175_12",
                     cap=0.08, lag=0, variable=True)
        b = simulate(altered, self.models, "delay175_12",
                     cap=0.08, lag=0, variable=True)
        self.assertEqual(a["events"], b["events"])

    def test_hard_budget_and_purchase_count_are_respected(self):
        for cap in (0.2, 0.4, 0.6):
            result = simulate(self.cell, self.models, "delay75_12",
                              cap=cap, lag=25, variable=True)
            self.assertLessEqual(result["cost"], cap + 1e-12)
            self.assertLessEqual(result["purchases"], 4)


if __name__ == "__main__":
    unittest.main()
