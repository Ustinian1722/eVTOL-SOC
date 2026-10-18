"""Synthetic causal queue invariants: delayed release, rejection and legacy replay."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import causal_queue_core as q

class ConstantModel:
    def __init__(self,value): self.value=value
    def predict(self,frame): return np.array([self.value])

class FakeCell:
    def __init__(self):
        self.n=10
        self.test=[SimpleNamespace(predicted=np.array([.9,.7,.5])) for _ in range(self.n)]
        self.opportunities=[2,4,6,8]
        self.bank={e:((.8,0.),3) for e in self.opportunities}
        self.loss={e:np.full(self.n,1.) for e in self.opportunities}
        self.frozen_loss=np.full(self.n,3.)
        self.calls=[]
    def deployed_params(self,last):
        return (1.,0.) if last<0 else self.bank[last][0]
    def feature(self,t,last,n):
        self.calls.append((t,last,n))
        if t<=last: raise AssertionError("noncausal")
        return {"seen_missions":float(t+1)}

class TestQueue(unittest.TestCase):
    def test_delayed_label_changes_next_when_only_after_arrival(self):
        c=FakeCell()
        with patch.object(q,"FIELDS",("seen_missions",)),patch.dict(q.ACTIVE,{"test":("delay",1,0.)}):
            z=q.queue_replay(c,(ConstantModel(2.),ConstantModel(2.)),
                             "test",2,.1,2)
        self.assertEqual(z["events"],"2;6")
        self.assertEqual(z["accepted_events"],"2;6")
        self.assertIn((2,-1,0),c.calls)
        self.assertIn((6,2,1),c.calls)
        self.assertAlmostEqual(z["MAE_pp"],2.)
    def test_rejected_paid_label_does_not_deploy(self):
        c=FakeCell()
        z=q.queue_replay(c,(None,None),"periodic3",2,.1,0,gate=0.)
        self.assertEqual(z["purchases"],2)
        self.assertEqual(z["accepted"],0)
        self.assertEqual(z["rejected"],2)
        self.assertAlmostEqual(z["MAE_pp"],3.)
        self.assertAlmostEqual(z["J"],3.2)
    def test_zero_delay_uses_next_mission(self):
        c=FakeCell()
        z=q.queue_replay(c,(None,None),"periodic3",2,.1,0)
        self.assertEqual(z["events"],"2;8")
        self.assertEqual(z["accepted"],2)
        self.assertAlmostEqual(z["MAE_pp"],1.6)
    def test_unarrived_label_never_updates(self):
        c=FakeCell()
        z=q.queue_replay(c,(None,None),"periodic3",2,.1,20)
        self.assertEqual(z["purchases"],1)
        self.assertEqual(z["accepted"],0)
        self.assertEqual(z["pending_end"],1)
        self.assertAlmostEqual(z["MAE_pp"],3.)
        self.assertAlmostEqual(z["J"],3.1)

if __name__=="__main__": unittest.main()
