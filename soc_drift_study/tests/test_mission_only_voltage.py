"""Unit checks for regression-based mission-only voltage adaptation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from mission_only_online_voltage import (
    _prepare, one_run, replay_update, rls_step, weighted_ridge,
)


def toy(n=23):
    rows=[]
    for k in range(1,n+1):
        for ns in [4,5,6]:
            q=(ns-3)*0.1
            rows.append({
                "cell_id":"TEST","charge_cycle_index":k,"phase_Ns":ns,
                "duration_s":40*ns, "I_mean_A":-ns*3,"I_peak_A":ns*4,
                "Vmin_V":4.1-.04*ns-.0008*k,
                "Vmax_V":4.2-.03*ns,
                "Tmean_C":25+.1*ns,"Tmax_C":26+.2*ns,
                "SOC_proxy":1-q/3,"SOC_nominal_3Ah":1-q/3,
            })
    return pd.DataFrame(rows)


class TestVoltageUpdate(unittest.TestCase):
    def test_rls_online_single_sensor_measurement(self):
        theta=np.array([0.,1.])
        P=np.eye(2)
        newer,newP=rls_step(theta,P,np.array([1.,1.]),4.,.99)
        self.assertEqual(len(newer),2)
        self.assertTrue(np.isfinite(newP).all())
        self.assertGreater(float(np.sum(newer)),1.0)

    def test_weighted_ridge_shapes(self):
        X=np.array([[1.,0.],[1.,1.],[1.,2.]])
        t=np.array([1.,2.,3.])
        theta=weighted_ridge(X,t,np.ones(3),np.zeros(2),lam=1)
        self.assertEqual(theta.shape,(2,))

    def test_guard_no_past_data(self):
        zeros=np.zeros(8)
        theta,accepted=replay_update(zeros,zeros,np.ones((10,8)),
                                     np.zeros(10),[],guarded=True)
        self.assertFalse(accepted)
        np.testing.assert_allclose(theta,zeros)

    def test_source_design_and_causal_no_update_toy(self):
        d=toy()
        info=_prepare(d,"SOC_nominal_3Ah")
        self.assertEqual(len(info[-1]),3)
        a=one_run(info,"SOC_nominal_3Ah",("frozen","none","none"),0.0,"TEST")
        b=one_run(info,"SOC_nominal_3Ah",("periodic_replay","replay","periodic"),.5,"TEST")
        self.assertEqual(a["attempts"],0)
        self.assertEqual(b["attempts"],0)
        self.assertAlmostEqual(a["measured_Vmin_MAE_mV"],b["measured_Vmin_MAE_mV"],places=8)
        self.assertAlmostEqual(a["SOC_proxy_MAE_pp"],b["SOC_proxy_MAE_pp"],places=8)


if __name__=="__main__":
    unittest.main()
