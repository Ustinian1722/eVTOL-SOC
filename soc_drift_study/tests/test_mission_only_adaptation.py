"""Synthetic tests for mission-only adaptation and causal replay."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from mission_only_active_adaptation import (
    ARMS, Detector, _fit_source, relative_rows, replay_cell,
    ridge_theta, safe_relative_update,
)


def synthetic_cell(n=29):
    rng = np.random.default_rng(13)
    rows = []
    for cycle in range(1, n + 1):
        for ns in [4, 5, 6]:
            q = (ns - 3) * 0.20
            rows.append({
                "cell_id": "TEST", "charge_cycle_index": cycle,
                "phase_Ns": ns, "duration_s": 100 * ns,
                "I_mean_A": -4.0 * ns, "I_peak_A": 4.5 * ns,
                "Vmin_V": 4.1 - q * .7 - cycle * .0005 + rng.normal(0,.002),
                "Vmax_V": 4.2 - q * .6 - cycle * .0005,
                "Tmean_C": 23 + .3*ns, "Tmax_C": 24 + .3*ns,
                "QDischarge_end_Ah": q, "SOC_proxy": 1-q/3.0,
                "SOC_nominal_3Ah": 1-q/3.0,
            })
    return pd.DataFrame(rows)


class MissionOnlyTest(unittest.TestCase):
    def test_relative_rows_no_absolute_q(self):
        z = np.array([[0.,1.],[.1,1.2],[.5,1.5]])
        a,b = relative_rows(z,np.array([.1,.3,.6]))
        a2,b2 = relative_rows(z,np.array([2.1,2.3,2.6]))
        np.testing.assert_allclose(a,a2)
        np.testing.assert_allclose(b,b2)

    def test_bad_nonmonotone_charge_mask(self):
        z = np.ones((3,2))
        a,b=relative_rows(z,np.array([0.1,0.4,0.2]))
        self.assertEqual(len(b),0)

    def test_theta_prior_without_information(self):
        X=np.array([[1.,0.],[1.,1.]])
        t=np.array([.4,.6])
        theta=ridge_theta(X,t,np.array([1.,1.]),np.array([.5,.1]))
        self.assertEqual(len(theta),2)
        self.assertTrue(np.all(np.isfinite(theta)))

    def test_guard_without_buffer_is_noop(self):
        theta=np.zeros(9)
        output,ok,status=safe_relative_update(theta,np.zeros((2,9)),np.zeros(2),[],"guarded",theta)
        self.assertFalse(ok)
        np.testing.assert_allclose(output,theta)

    def test_detector_uses_only_provided_sensor_scores(self):
        d=Detector(0,1,0,1)
        fired=None
        for _ in range(3):
            fired=d.step(5,5)
        self.assertTrue(fired["voltage"])
        self.assertTrue(fired["disagreement"])
        d.reset_alarm()
        self.assertEqual(d.ph,0)

    def test_source_fit_and_source_only_frozen(self):
        cell=synthetic_cell()
        first=cell[cell.charge_cycle_index.le(20)]
        source,ys,theta,v,model=_fit_source(first,"SOC_nominal_3Ah")
        self.assertEqual(source.shape[1],9)
        out,_=replay_cell(cell,"SOC_nominal_3Ah",("frozen","none","none"))
        self.assertEqual(out["test_missions"],9)
        self.assertEqual(out["attempted_updates"],0)

    def test_causality_completed_mission_only(self):
        cell=synthetic_cell()
        a,_=replay_cell(cell,"SOC_nominal_3Ah",("frozen","none","none"))
        b,_=replay_cell(cell,"SOC_nominal_3Ah",("periodic_physics","physics","periodic"))
        self.assertAlmostEqual(a["proxy_MAE_pp"],b["proxy_MAE_pp"],places=10)
        self.assertEqual(b["attempted_updates"],0)

    def test_all_predeclared_arms_are_unique(self):
        names=[x[0] for x in ARMS]
        self.assertEqual(len(names),len(set(names)))


if __name__ == "__main__":
    unittest.main()
