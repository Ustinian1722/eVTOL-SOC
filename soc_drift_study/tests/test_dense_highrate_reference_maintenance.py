"""Causal and exact-dynamic-programming unit tests for dense reference purchases."""
import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from dense_highrate_reference_maintenance import (
    Mission,select_reference_rows,fit_calibration,choose_when,
    optimal_hindsight,replay,opportunities_for,sensor_risks,
)


def task(n=107):
    q=np.linspace(0,.8,n)
    ns=np.r_[np.full(15,4),np.full(n-30,5),np.full(15,6)]
    X=np.column_stack([4.2-.5*q,np.full(n,-5.),np.full(n,25.),
                      1-q/3.,np.arange(n)/60,
                      (ns==4).astype(float),(ns==5).astype(float),(ns==6).astype(float)])
    return Mission(10,X,1-q/2.8,1-q/3,ns,100.,2.8,n,
                   predicted=np.clip(1-q/3.,0,1),voltage_resid=np.full(n,.005))


class DenseReferenceWhenTests(unittest.TestCase):
    def test_reference_density_is_native_and_contains_phase_endpoints(self):
        m=task()
        x=select_reference_rows(m,"endpoints3")
        self.assertEqual(len(x),3)
        self.assertListEqual(m.ns[x].tolist(),[4,5,6])
        self.assertEqual(len(select_reference_rows(m,"uniform16")),16)
        self.assertEqual(len(select_reference_rows(m,"native_all")),m.n)

    def test_calibrator_uses_only_unlocked_reference(self):
        m=task()
        ix=select_reference_rows(m,"native_all")
        a,b=fit_calibration(m.predicted[ix],m.y[ix])
        self.assertTrue(np.isfinite([a,b]).all())
        self.assertGreaterEqual(a,.7);self.assertLessEqual(a,1.3)
        self.assertGreaterEqual(b,-.15);self.assertLessEqual(b,.15)

    def test_stage_balanced_curve_preserves_equal_phase_information(self):
        # When values are constant within each phase, three phase endpoints
        # and a phase-balanced full sampled curve must estimate same head.
        p=np.repeat([.89,.59,.29],[6,60,12])
        y=np.repeat([.86,.54,.25],[6,60,12])
        w=np.repeat([1/18,1/180,1/36],[6,60,12])
        a,b=fit_calibration(p,y,weights=w)
        ac,bc=fit_calibration(np.array([.89,.59,.29]),
                              np.array([.86,.54,.25]))
        self.assertAlmostEqual(a,ac,places=11)
        self.assertAlmostEqual(b,bc,places=11)

    def test_when_is_label_blind(self):
        n=175
        sensors={"disagreement_z":np.r_[np.zeros(49),np.full(n-49,5.)],
                 "voltage_z":np.r_[np.zeros(24),np.full(n-24,4.)],
                 "mission_duration":np.full(n,100.)}
        options=opportunities_for(n)
        for name in ("frozen","periodic_planned","periodic150","random",
                     "disagreement3","voltage3","joint3"):
            selected=choose_when(sensors,options,n,4,name,"TOY")
            self.assertLessEqual(len(selected),4)
            self.assertTrue(all(e in options for e in selected))

    def test_oracle_exhaustive_all_prices_and_budgets(self):
        rng=np.random.default_rng(42)
        n=21;events=[3,7,11,16]
        frozen=rng.uniform(1.,4.,n)
        candidates={e:rng.uniform(.5,3.5,n) for e in events}
        def loss(indices,price):
            return replay(frozen,candidates,indices).mean()+price*len(indices)
        for budget in (0,1,2,3):
            for price in (0.,.2,3.):
                expected=min(loss(choices,price)
                    for k in range(budget+1)
                    for choices in itertools.combinations(events,k))
                chosen=optimal_hindsight(frozen,candidates,events,budget,price)
                self.assertAlmostEqual(loss(chosen,price),expected,places=8)

    def test_replay_prediction_unchanged_before_purchase(self):
        baseline=np.full(45,2.)
        after=np.full(45,.5)
        after2=np.full(45,1.)
        times=[7,25]
        out=replay(baseline,{7:after,25:after2},times)
        self.assertTrue(np.all(out[:8]==2.))
        self.assertTrue(np.all(out[8:26]==.5))
        self.assertTrue(np.all(out[26:]==1.))


if __name__=="__main__":
    unittest.main()
