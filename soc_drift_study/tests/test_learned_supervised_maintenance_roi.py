"""Reproducible leak/causality tests for supervised-label purchase ROI meta-model."""
from __future__ import annotations
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from learned_supervised_maintenance_roi import (
    FEATURE_NAMES, visible_features, frozen_partition,
    counterfactual_roi, learned_schedule, train_select_development,
)
from supervised_maintenance_when import (
    make_cell,scheduled_opportunities,event_models,predictor_for_events,
)


def fake(n=260,cell="VAH77"):
    out=[]
    for i in range(n):
        for ph in (4,5,6):
            q=.16*(ph-3)
            out.append({
                "cell_id":cell,"charge_cycle_index":i+1,"phase_Ns":ph,
                "duration_s":ph*60,"I_mean_A":-3.0*ph,
                "I_peak_A":4.0*ph,
                "Vmin_V":4.1-.12*q-.0004*i,
                "Vmax_V":4.2-.08*q-.0002*i,
                "Tmean_C":25.0+ph*.08,"Tmax_C":26.0+ph*.08,
                "SOC_nominal_3Ah":1-q/3.,
                "SOC_proxy":1-q/2.85,
            })
    return pd.DataFrame(out)


class RoiPolicyTests(unittest.TestCase):
    def test_future_soc_labels_do_not_affect_online_features(self):
        cell=make_cell(fake())
        mod=copy.deepcopy(cell)
        mod.label_broker[:]=.01
        for t in [24,49,74,99,149]:
            self.assertEqual(visible_features(cell,t),visible_features(mod,t))

    def test_future_voltage_cannot_change_past_features(self):
        cell=make_cell(fake())
        mod=copy.deepcopy(cell)
        mod.voltage_residual[150:]+=10.
        mod.encoded_test_features[150:]+=100.
        for t in [24,49,74,99]:
            original=visible_features(cell,t)
            altered=visible_features(mod,t)
            for name in FEATURE_NAMES:
                self.assertAlmostEqual(original[name],altered[name],places=10)

    def test_future_SOC_affects_only_offline_roi(self):
        cell=make_cell(fake())
        indices=scheduled_opportunities(cell)
        bank=event_models(cell,indices)
        roi=counterfactual_roi(cell,bank,indices[0])
        mod=copy.deepcopy(cell)
        mod.label_broker[indices[0]+1:indices[0]+50]=.1
        self.assertNotAlmostEqual(roi,counterfactual_roi(mod,bank,indices[0]))

    def test_model_policy_blind_to_hidden_soc_labels(self):
        cell=make_cell(fake())
        mod=copy.deepcopy(cell)
        mod.label_broker[:]=.35
        class AlwaysPositive:
            def predict(self,X):
                assert list(X.columns)==FEATURE_NAMES
                return np.array([1.0]*len(X))
        p=scheduled_opportunities(cell)
        orig,trace=learned_schedule(cell,p,AlwaysPositive(),4,threshold=0.)
        other,trace2=learned_schedule(mod,p,AlwaysPositive(),4,threshold=0.)
        self.assertEqual(orig,other)
        self.assertEqual(trace,trace2)
        self.assertLessEqual(len(orig),4)
        self.assertTrue(all(b-a>=75 for a,b in zip(orig,orig[1:])))

    def test_dev_partition_stable_no_random_holdout(self):
        ids=[f"VAH{i:02d}" for i in range(1,23)]
        a=frozen_partition(ids)
        b=frozen_partition(ids[::-1])
        self.assertEqual(a,b)
        self.assertEqual(list(a.values()).count("development"),14)

    def test_meta_selection_uses_cell_block_groups(self):
        # Exercises method without requiring real target labels.
        rng=np.random.default_rng(1)
        rows=[]
        for k in range(10):
            for i in range(12):
                x={name:float(rng.normal()) for name in FEATURE_NAMES}
                rows.append({"cell_id":f"C{k}","future_100_SoCproxy_gain_pp":.15*x["last_voltage_z"]+.01,**x})
        frame=pd.DataFrame(rows)
        model,cv,name,oof=train_select_development(frame)
        self.assertEqual(len(oof),len(frame))
        self.assertEqual(cv.dev_cells.min(),10)
        self.assertIn(name,{"ridge","huber","constant"})
        self.assertTrue(np.isfinite(model.predict(frame[FEATURE_NAMES])).all())


if __name__=="__main__":
    unittest.main()
