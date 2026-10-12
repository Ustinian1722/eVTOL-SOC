"""Tests for label-blind strong-GRU mission-only scalar adaptation."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from strong_gru_mission_adapter import (
    candidate_slope, physics_loss, replay, try_update, OUTPUT_METHODS, BUDGET,
)


def missions(n=200):
    out=[]
    for cycle in range(1,n+1):
        q=np.array([.16,.35,.62])
        p=np.array([.94,.88,.78])-.0001*cycle
        out.append({
            "cycle":cycle,"p":p,
            "dP":p[1:]-p[0],"dQ":q[1:]-q[0],
            "score_only_soc_proxy":np.array([.93,.87,.75]),
        })
    return out


class StrongGRUAdapterTests(unittest.TestCase):
    def test_candidate_uses_physics_only(self):
        src=missions(35)
        fit=candidate_slope(src,3.,123)
        self.assertGreater(fit,0)
        self.assertLess(fit,1.4)

    def test_physics_loss_depends_on_slope_and_capacity(self):
        src=missions(10)
        a=physics_loss(1.,src,3.)
        b=physics_loss(1.2,src,3.)
        self.assertGreaterEqual(a,0)
        self.assertNotEqual(a,b)

    def test_no_SoC_proxy_online_leak(self):
        src=missions()
        mutated=copy.deepcopy(src)
        for item in mutated:
            item["score_only_soc_proxy"]=1.-item["score_only_soc_proxy"]
        for arm in OUTPUT_METHODS:
            ra,ea,pa=replay(src,arm,"TEST",17)
            rb,eb,pb=replay(mutated,arm,"TEST",17)
            np.testing.assert_allclose(pa,pb)
            self.assertEqual(ea,eb)
        self.assertNotEqual(replay(src,"frozen","TEST",17)[0]["SOC_proxy_MAE_pp"],
                            replay(mutated,"frozen","TEST",17)[0]["SOC_proxy_MAE_pp"])

    def test_no_future_online_leak(self):
        src=missions(300)
        mutated=copy.deepcopy(src)
        for item in mutated[151:]:
            item["dQ"]=item["dQ"]*2.0
        ra,ea,pa=replay(src,"periodic_C3","TEST",17)
        rb,eb,pb=replay(mutated,"periodic_C3","TEST",17)
        np.testing.assert_allclose(pa[:150*3],pb[:150*3])
        self.assertEqual([x for x in ea if x["event_index"]<=150],
                         [x for x in eb if x["event_index"]<=150])

    def test_budgets_fixed_except_stream(self):
        src=missions(200)
        for arm in OUTPUT_METHODS:
            row,_,_=replay(src,arm,"TEST",17)
            if arm!="stream_C3":
                self.assertLessEqual(row["attempts"],BUDGET)
                self.assertLessEqual(row["accepted"],BUDGET)


if __name__=="__main__":
    unittest.main()
