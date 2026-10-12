"""Safety tests: only development OOF outcomes select each price/budget strategy."""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from test_stateful_marginal_roi_maintenance import toy_cell
from test_deferral_regret_when import ConstantPredictor
from scenario_safe_when_selector import (
    SCENARIOS,MENU,PERIODIC,ACTIVE,evaluate_one,choose_from_source,
    evaluate_challenge,
)


class SafeScenarioTests(unittest.TestCase):
    def test_menu_contains_no_update_and_strong_sparse_periodic(self):
        self.assertIn("frozen",PERIODIC)
        self.assertIn("periodic500",MENU)
        self.assertIn("delay75_12",ACTIVE)
        self.assertEqual(len(SCENARIOS),8)

    def test_policy_source_selection_has_only_supplied_cells(self):
        c1=toy_cell(cid="C1",n=210)
        c2=toy_cell(cid="C2",n=275)
        pair=(ConstantPredictor(.9),ConstantPredictor(.5))
        detail,_,selected=choose_from_source(
            {"C1":c1,"C2":c2},{"C1":pair,"C2":pair})
        self.assertEqual(set(detail.cell_id),{"C1","C2"})
        self.assertEqual(len(selected),len(SCENARIOS))
        self.assertTrue(set(selected.source_chosen).issubset(MENU))
        self.assertTrue(all(selected.eligible_candidates>0))

    def test_when_acts_only_after_completed_normal_mission(self):
        c=toy_cell(cid="C3",n=210)
        pair=(ConstantPredictor(.9),ConstantPredictor(.5))
        action=evaluate_one(c,pair,"delay75_12",2,.1)
        self.assertLessEqual(action["purchases"],2)
        if action["events"]:
            ix=[int(t) for t in action["events"].split(";")]
            self.assertTrue(set(ix).issubset(set(c.opportunities)))
            self.assertTrue(all(j-i>=75 for i,j in zip(ix,ix[1:])))

    def test_challenge_not_needed_for_source_policy_selection(self):
        source=toy_cell(cid="C1",n=275)
        model=(ConstantPredictor(.9),ConstantPredictor(.5))
        _,_,selected=choose_from_source({"C1":source},{"C1":model})
        target=toy_cell(cid="TOTALLY_NEW",n=210)
        detail,summary=evaluate_challenge({"TOTALLY_NEW":target},model,selected)
        self.assertEqual(len(summary),8)
        self.assertTrue(detail.groupby(["cell_id","budget","price"]).selected.sum().eq(1).all())
        self.assertEqual(set(detail.cell_id),{"TOTALLY_NEW"})


if __name__=="__main__":unittest.main()
