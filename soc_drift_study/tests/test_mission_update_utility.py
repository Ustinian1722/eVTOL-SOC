"""Tests: no RPT target can influence online decisions; absolute Q offset cancels."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from mission_update_utility import (
    ARMS, BUDGET, candidate_diagnostics, evaluate_one, fit_and_gate,
    mission_differences, physics_residual, prepare, selected_cell_protocol,
)


def toy_data(n=220):
    rows = []
    for cycle in range(1,n+1):
        for phase in [4,5,6]:
            q = .17*(phase-3)
            rows.append({
                "cell_id":"TEST",
                "charge_cycle_index":cycle,"phase_Ns":phase,
                "duration_s":phase*78,
                "I_mean_A":-4*phase+cycle*.00001,
                "I_peak_A":4.5*phase,
                "Vmin_V":4.0-.12*q-.0008*cycle,
                "Vmax_V":4.12-.10*q-.0007*cycle,
                "Tmean_C":24+.7*phase+.001*cycle,
                "Tmax_C":25+.8*phase+.001*cycle,
                "QDischarge_end_Ah":q,
                "SOC_nominal_3Ah":1.-q/3,
                "SOC_proxy":1.-q/2.8,
            })
    return pd.DataFrame(rows)


class MissionUtilityTests(unittest.TestCase):
    def test_only_relative_charge_used(self):
        z=np.arange(18.,dtype=float).reshape(3,6)
        a,b=mission_differences(z,np.array([.1,.3,.6]))
        c,d=mission_differences(z,np.array([2.1,2.3,2.6]))
        np.testing.assert_allclose(a,c)
        np.testing.assert_allclose(b,d)

    def test_delta_charge_flags_unphysical_drop(self):
        z=np.ones((3,9))
        a,b=mission_differences(z,np.array([.2,.7,.3]))
        self.assertEqual(len(b),0)

    def test_initialization_does_not_read_RPT_proxy(self):
        a=toy_data()
        b=a.copy()
        b["SOC_proxy"]=(b["SOC_proxy"]*0.7).clip(0,1)
        x=prepare(a)
        y=prepare(b)
        np.testing.assert_allclose(x.sourceZ,y.sourceZ)
        np.testing.assert_allclose(x.frozen,y.frozen)
        np.testing.assert_allclose(x.sourceY,y.sourceY)
        # Only offline score arrays should change.
        np.testing.assert_allclose(x.missions[0]["Z"],y.missions[0]["Z"])

    def test_no_rpt_proxy_can_change_online_update_events(self):
        base=prepare(toy_data())
        mutated=copy.deepcopy(base)
        for task in mutated.missions:
            task["score_after_replay"]=1.-task["score_after_replay"]
        spec=("value_gate_005","value",3.,.05)
        a,ea,pa=evaluate_one(base,spec,"TEST")
        b,eb,pb=evaluate_one(mutated,spec,"TEST")
        np.testing.assert_allclose(pa,pb)
        self.assertEqual(ea,eb)
        self.assertNotAlmostEqual(a["MAE_pp"],b["MAE_pp"])

    def test_no_future_mission_can_change_past_update(self):
        src=prepare(toy_data())
        other=copy.deepcopy(src)
        for task in other.missions[150:]:
            task["dq"]=np.array([.9,1.8])
            task["dZ"]=np.ones((2,8))
        a,ea,_=evaluate_one(src,("periodic_C3","periodic",3.,None),"TEST")
        b,eb,_=evaluate_one(other,("periodic_C3","periodic",3.,None),"TEST")
        initial_a=[x for x in ea if x["mission_ordinal"]<=150]
        initial_b=[x for x in eb if x["mission_ordinal"]<=150]
        self.assertEqual(initial_a,initial_b)

    def test_capacity_prior_changes_physics_target_only(self):
        dZ=np.array([[1.,0.],[0.,1.]])
        observation=[{"dZ":dZ,"dq":np.array([.1,.2])}]
        frozen=np.array([.1,0.0,0.0])
        self.assertGreater(physics_residual(frozen,observation,2.7),
                           physics_residual(frozen,observation,3.3))

    def test_budget_never_exceeded(self):
        inputs=prepare(toy_data())
        for a in ARMS:
            out,_,_=evaluate_one(inputs,a,"TEST")
            self.assertLessEqual(out["accepted_updates"],BUDGET)
            self.assertLessEqual(out["attempted_updates"],BUDGET)

    def test_partition_stable_and_has_8_challenge(self):
        ids=[f"C{i:02d}" for i in range(22)]
        p=selected_cell_protocol(ids)
        self.assertEqual(list(p.values()).count("retrospective_challenge"),8)
        self.assertEqual(selected_cell_protocol(ids[::-1]),p)


if __name__=="__main__":
    unittest.main()
