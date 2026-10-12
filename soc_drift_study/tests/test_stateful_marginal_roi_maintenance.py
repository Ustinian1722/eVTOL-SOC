"""Regression contracts for online stateful marginal-ROI decision.

No target SOC labels may be consumed by a pre-purchase feature. Previously
purchased label can inform deployed affine params and the next decision.
"""
from __future__ import annotations
import copy
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from dense_highrate_reference_maintenance import Mission
from stateful_marginal_roi_maintenance import (
    Cell,FIELDS,cell_split,learned_events,meta_target,meta_examples,
    train_development_oof_policy,POLICY_GRID,
)


def toy_cell(cid="TOY",n=210):
    missions=[]
    for t in range(n):
        ns=np.r_[np.full(8,4),np.full(32,5),np.full(10,6)]
        q=np.linspace(.02,.85,50)
        temp=np.full(50,25.+.01*t)
        volts=4.2-.55*q-.0001*t
        x=np.column_stack([volts,np.full(50,-6.),temp,1-q/3.,
            np.arange(50)/60.,(ns==4).astype(float),
            (ns==5).astype(float),(ns==6).astype(float)])
        p=np.clip(1-q/3. +.001*np.sin(t/20.),0,1)
        missions.append(Mission(t,x,1-q/(3.-.0005*t),1-q/3.,ns,
             100.,3.-.0005*t,50,predicted=p,voltage_resid=np.abs(volts-4.1)))
    ops=list(range(24,n,25))
    bank={}
    for e in ops:
        bank[e]=((1.,float(e*.00001)),50)
    losses={e:np.array([np.abs(m.predicted-m.y).mean()*100+.0001*e for m in missions])
            for e in ops}
    frozen=np.array([np.abs(m.predicted-m.y).mean()*100 for m in missions])
    return Cell(
        cid,missions[:20],missions,bank,losses,frozen,
        {"disagreement_z":np.zeros(n),"voltage_z":np.zeros(n)},
        np.array([100*np.mean(np.abs(m.predicted-m.qnom)) for m in missions]),
        np.array([np.mean(m.voltage_resid) for m in missions]),
        np.array([np.mean(np.abs(m.X[:,1])) for m in missions]),
        np.array([np.mean(m.X[:,2]) for m in missions]),
        np.array([np.max(m.X[:,2]) for m in missions]),
        np.array([np.mean(m.X[m.ns==6,0])-np.mean(m.X[m.ns==4,0])
                  for m in missions]),
        np.array([np.mean(m.X[m.ns==6,2])-np.mean(m.X[m.ns==4,2])
                  for m in missions]),
        ops,{"raw_rows":50*n,"bad_missions":0,"full_normal_missions":n,
             "test_missions":n}
    )


class Model:
    def predict(self,frame):
        assert list(frame.columns)==list(FIELDS)
        return np.array([float(frame.since_last_reference.iloc[0])/120.-.5])


class StatefulWhenTests(unittest.TestCase):
    def test_split_stable_by_cell_id(self):
        s=cell_split()
        self.assertEqual(len(s),22)
        self.assertEqual(sum(x=="development" for x in s.values()),14)
        self.assertEqual(s,cell_split(tuple(reversed(list(s)))))

    def test_prepurchase_features_blind_to_future_and_unbought_soc(self):
        c=toy_cell()
        altered=copy.deepcopy(c)
        for m in altered.test:
            m.y[:]=np.linspace(.01,.99,m.n) # all target proxy labels hidden
        for t in c.opportunities:
            self.assertEqual(c.feature(t,-1,0),altered.feature(t,-1,0))
        for t in c.opportunities:
            self.assertEqual(c.feature(t,24,1),altered.feature(t,24,1))
            if t==24:continue

    def test_future_sensor_can_not_change_past_state(self):
        c=toy_cell()
        changed=copy.deepcopy(c)
        for m in changed.test[100:]:
            m.X[:,0]+=2.
            m.predicted[:]+=.3
            m.voltage_resid[:]+=10.
        for t in (24,49,74,99):
            self.assertEqual(c.feature(t,-1,0),changed.feature(t,-1,0))

    def test_previous_purchased_label_is_legal(self):
        c=toy_cell()
        altered=copy.deepcopy(c)
        altered.test[24].y[:]=0.
        old=c.feature(74,24,1)
        new=altered.feature(74,24,1)
        self.assertNotEqual(old["last_reference_fit_MAE_pp"],
                            new["last_reference_fit_MAE_pp"])

    def test_hidden_labels_do_not_change_first_online_purchases(self):
        c=toy_cell()
        altered=copy.deepcopy(c)
        for m in altered.test:
            m.y[:]=0.
        a,tr=learned_events(c,Model(),4,.1,125,.1,trace=True)
        b,tr2=learned_events(altered,Model(),4,.1,125,.1,trace=True)
        self.assertEqual(a,b)
        self.assertEqual(tr,tr2)
        self.assertLessEqual(len(a),4)
        self.assertTrue(all(v-u>=125 for u,v in zip(a,a[1:])))

    def test_source_counterfactual_uses_deployed_head_not_frozen(self):
        c=toy_cell()
        t=99;last=49
        ans=meta_target(c,t,last)
        old=c.loss[last];new=c.loss[t]
        desired=float(np.mean(old[t+1:min(t+101,c.n)]-new[t+1:min(t+101,c.n)]))
        self.assertAlmostEqual(ans,desired)
        rows=meta_examples(c)
        self.assertGreater(len(rows),2)
        self.assertTrue(any(row["prev_purchased_idx"]>=0 for row in rows))
        self.assertTrue(all(np.isfinite(row["incremental_future100_ROI_pp"]) for row in rows))

    def test_candidate_at_end_is_not_censored_using_future_lifespan(self):
        c=toy_cell(n=200)
        self.assertEqual(c.opportunities[-1],199)
        self.assertIn(199,c.opportunities)
        self.assertLess(len(POLICY_GRID),100)


if __name__=="__main__":
    unittest.main()
