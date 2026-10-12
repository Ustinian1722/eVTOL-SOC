"""Synthetic tests isolate learned WHEN decisions from target scoring labels."""
from __future__ import annotations
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from learned_mission_update_value import (
    SIGNALS, ARMS, SEED, FUTURE, proposal_features,
    make_classifier, future_roi, rollout,
)
from mission_update_utility import prepare


def toy(n=215,cell="TEST"):
    rows=[]
    for i in range(1,n+1):
        for ph in (4,5,6):
            q=(ph-3)*.19
            rows.append({
                "cell_id":cell,"charge_cycle_index":i,"phase_Ns":ph,
                "duration_s":80*ph,"I_mean_A":-3.1*ph,
                "I_peak_A":4.1*ph,"Vmin_V":4.1-.06*ph-.001*i,
                "Vmax_V":4.2-.055*ph-.0008*i,
                "Tmean_C":24+.1*ph,"Tmax_C":25+.2*ph,
                "QDischarge_end_Ah":q,"SOC_nominal_3Ah":1-q/3,
                "SOC_proxy":1-q/2.8,
            })
    return pd.DataFrame(rows)


class DummyClassifier:
    def predict_proba(self,frame):
        assert list(frame.columns)==SIGNALS
        return np.array([[.20,.80]]*len(frame),dtype=float)


class LearnedUtilityTests(unittest.TestCase):
    def test_no_future_outcome_in_observable_features(self):
        a=prepare(toy())
        b=copy.deepcopy(a)
        for m in b.missions:
            m["score_after_replay"]=np.full(len(m["score_after_replay"]),.1)
        def work(src):
            hist=[{k:m[k] for k in ["cycle","Z","dZ","dq","voltage_abs_resid_mV"]}
                  for m in src.missions[:35]]
            return proposal_features(src,hist,src.frozen)
        ca,da,fa=work(a)
        cb,db,fb=work(b)
        np.testing.assert_allclose(ca,cb)
        self.assertEqual(fa,fb)

    def test_meta_target_requires_later_score_and_is_separate(self):
        x=prepare(toy())
        cand=x.frozen.copy();cand[1:]+=0.05
        one=future_roi(x,40,x.frozen,cand)
        mutated=copy.deepcopy(x)
        for m in mutated.missions[40:]:
            m["score_after_replay"]=np.zeros_like(m["score_after_replay"])
        two=future_roi(mutated,40,x.frozen,cand)
        self.assertNotAlmostEqual(one,two)

    def test_target_label_mutation_cannot_change_active_schedule(self):
        x=prepare(toy())
        other=copy.deepcopy(x)
        for m in other.missions:
            m["score_after_replay"]=1-m["score_after_replay"]
        for a in ARMS:
            first,events=rollout(x,"TEST",a,DummyClassifier())
            second,events2=rollout(other,"TEST",a,DummyClassifier())
            self.assertEqual(events,events2)
            self.assertEqual(first["attempts"],second["attempts"])
            self.assertEqual(first["accepted"],second["accepted"])

    def test_schedule_never_exceeds_b4(self):
        x=prepare(toy())
        for policy in ARMS:
            out,_=rollout(x,"TEST",policy,DummyClassifier())
            self.assertLessEqual(out["attempts"],4)
            self.assertLessEqual(out["accepted"],out["attempts"])

    def test_classifier_fixed_architecture(self):
        m=make_classifier()
        self.assertTrue(hasattr(m,"fit"))
        self.assertEqual(len(SIGNALS),13)


if __name__=="__main__":
    unittest.main()
