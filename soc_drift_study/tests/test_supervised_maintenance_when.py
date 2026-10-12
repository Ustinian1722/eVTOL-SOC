"""Causal paid-label release, deterministic budgets, exact hindsight oracle."""
from __future__ import annotations
import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from supervised_maintenance_when import (
    CellData, supervised_one_mission, scheduled_opportunities,
    event_models, predictor_for_events, oracle_DP, online_schedule,
    make_cell, run_one,
)


def toy_cell(n=130):
    t=np.arange(n)
    phase=np.array([.9,.65,.37])
    frozen=np.clip(phase[None,:] - .0001*t[:,None],0,1)
    truth=np.clip(frozen-.025*(t[:,None]>40),0,1)
    return CellData(
        cell_id="TOY",frozen=frozen,label_broker=truth,
        nominal=frozen+.002,voltage_residual=np.full((n,3),.005),
        current=np.ones((n,3))*-9,temp=np.ones((n,3))*25,
        training_disagreement=np.array([.001,.002,.0015]*7),
        training_vresidual=np.array([.004,.005,.006]*7),
        dates=np.arange(n)+100,phase_rows=3,
    )


class SupervisedMaintenanceTests(unittest.TestCase):
    def test_supervised_calibration_uses_only_unlocked_curve(self):
        pred=np.array([.9,.6,.3])
        labels=np.array([.87,.57,.27])
        a,b=supervised_one_mission(pred,labels)
        self.assertLess(b,0)
        self.assertLessEqual(abs(b),.12)
        self.assertLessEqual(abs(a-1),.25)

    def test_fixed_event_effect_starts_after_reference_mission(self):
        cell=toy_cell()
        opp=scheduled_opportunities(cell)
        bank=event_models(cell,opp)
        t=opp[0]
        out=predictor_for_events(cell,bank,[t])
        np.testing.assert_allclose(out[:t+1],cell.frozen[:t+1])
        self.assertEqual(len(out),len(cell.frozen))

    def test_online_threshold_cannot_read_hidden_soc_reference(self):
        x=toy_cell()
        y=toy_cell()
        y.label_broker=np.clip(1-x.label_broker,0,1)
        events=scheduled_opportunities(x)
        for kind in ["periodic_horizon","periodic_150","random",
                     "model_disagreement","voltage_residual",
                     "cusum_voltage","risk_combo"]:
            self.assertEqual(
                online_schedule(x,events,kind,4,seed=37),
                online_schedule(y,events,kind,4,seed=37),
            )

    def test_oracle_exact_vs_all_combinations(self):
        cell=toy_cell(n=140)
        events=[24,49,74,99]
        bank=event_models(cell,events)
        oracle=oracle_DP(cell,bank,events,2)
        truth=cell.label_broker
        def loss(indices):
            pred=predictor_for_events(cell,bank,indices)
            return float(np.mean(np.abs(pred-truth)))
        allsets=[()] + [(i,) for i in events] + list(itertools.combinations(events,2))
        true_best=min(loss(s) for s in allsets)
        self.assertAlmostEqual(loss(oracle),true_best,places=10)
        self.assertLessEqual(len(oracle),2)

    def test_cost_budget_never_overdrawn(self):
        cell=toy_cell()
        opp=scheduled_opportunities(cell)
        bank=event_models(cell,opp)
        for budget in [0,2,4]:
            for p in ["frozen","periodic_horizon","periodic_150","random",
                      "model_disagreement","voltage_residual","cusum_voltage",
                      "risk_combo","offline_oracle"]:
                metric,indices=run_one(cell,bank,opp,p,budget,11)
                self.assertLessEqual(len(indices),budget)
                self.assertEqual(metric["labeled_phase_end_points"],3*len(indices))

    def test_actual_source_training_never_uses_later_label(self):
        rows=[]
        for cycle in range(1,141):
            for ns in [4,5,6]:
                q=(ns-3)*.17
                rows.append({
                    "cell_id":"TOY","charge_cycle_index":cycle,"phase_Ns":ns,
                    "duration_s":70*ns,"I_mean_A":-3*ns,"I_peak_A":4*ns,
                    "Vmin_V":4.0-.13*q-.0005*cycle,
                    "Vmax_V":4.1-.12*q-.0004*cycle,
                    "Tmean_C":24+.3*ns,"Tmax_C":25+.3*ns,
                    "SOC_nominal_3Ah":1-q/3,
                    "SOC_proxy":1-q/2.8,
                })
        frame=pd.DataFrame(rows)
        modified=frame.copy()
        modified.loc[modified.charge_cycle_index>20,"SOC_proxy"]*=.85
        before=make_cell(frame)
        after=make_cell(modified)
        np.testing.assert_allclose(before.frozen,after.frozen)
        np.testing.assert_allclose(before.voltage_residual,after.voltage_residual)
        # The purchased-reference labels differ, as intended.
        self.assertGreater(float(np.abs(before.label_broker-after.label_broker).max()),.05)

if __name__=="__main__":
    unittest.main()
