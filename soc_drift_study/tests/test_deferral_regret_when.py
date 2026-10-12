"""Source-only deferral-regret teacher and causal online buying regression tests."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from test_stateful_marginal_roi_maintenance import toy_cell
from deferral_regret_when import (
    HOLD,GRID,counterfactual_delay,source_examples,decide,replay_policy,
    paired_scores,
)
from stateful_marginal_roi_maintenance import FIELDS


class ConstantPredictor:
    def __init__(self,value):self.value=value
    def predict(self,x):
        assert list(x.columns)==list(FIELDS)
        return np.full(len(x),self.value)


class DeferralTests(unittest.TestCase):
    def test_source_teacher_compares_two_paid_actions(self):
        c=toy_cell(n=210)
        t=74;prev=24
        n=t+HOLD
        end=min(c.n,t+101)
        a=np.r_[c.loss[prev][t+1:n+1],c.loss[n][n+1:end]]
        b=c.loss[t][t+1:end]
        self.assertAlmostEqual(counterfactual_delay(c,t,prev),
                               np.mean(a-b),places=12)
        self.assertTrue(np.isnan(counterfactual_delay(c,199,174)))
        rows=source_examples(c)
        self.assertGreater(len(rows),5)
        self.assertTrue(all(np.isfinite(x["delay25_regret_pp"]) for x in rows))

    def test_online_defer_strategy_is_hidden_label_blind_until_buy(self):
        c=toy_cell(n=210)
        altered=copy.deepcopy(c)
        for m in altered.test:m.y[:]=0.
        models=(ConstantPredictor(.9),ConstantPredictor(.3))
        config=("delay",125,.12)
        a,events=decide(c,models,config,3,.1,trace=True)
        b,altered_events=decide(altered,models,config,3,.1,trace=True)
        self.assertEqual(a,b)
        self.assertEqual(events,altered_events)
        self.assertTrue(all(y-x>=125 for x,y in zip(a,a[1:])))
        self.assertLessEqual(len(a),3)

    def test_budget_reservation_gate_cannot_decrease_after_purchase(self):
        c=toy_cell(n=210)
        models=(ConstantPredictor(.9),ConstantPredictor(.3))
        conf=("reserve",75,0.)
        selected,logs=decide(c,models,conf,4,.1,trace=True)
        self.assertTrue(all(log["reserve"]>=0 for log in logs))
        res=[log["reserve"] for log in logs]
        self.assertTrue(all(b>=a for a,b in zip(res,res[1:])))
        self.assertTrue(all(t in c.opportunities for t in selected))

    def test_online_candidate_not_censored_by_future_stop(self):
        c=toy_cell(n=200)
        self.assertEqual(c.opportunities[-1],199)
        selected,log=decide(c,(ConstantPredictor(.9),ConstantPredictor(.5)),
                            ("delay_only",75,0.),4,.1,trace=True)
        self.assertLessEqual(len(selected),4)
        self.assertTrue(all(t in c.opportunities for t in selected))

    def test_pairing_uses_physical_cells(self):
        rows=[]
        for cid in ("X","Y","Z"):
            for policy,j in (("source_chosen",1.),("periodic150",1.4)):
                rows.append({"cell_id":cid,"price":.1,"budget":4,
                             "policy":policy,"J":j,"MAE_pp":j,
                             "purchases":2})
        x=paired_scores(pd.DataFrame(rows))
        y=x[x.policy=="source_chosen"].iloc[0]
        self.assertAlmostEqual(y.deltaJ_fixed150,-.4)
        self.assertEqual(y.n_cells,3)
        self.assertEqual(y.n_better_than150,3)


if __name__=="__main__":unittest.main()
