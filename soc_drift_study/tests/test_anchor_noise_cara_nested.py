"""Causal quality perturbation and cumulative staleness baseline tests."""
import copy
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from test_stateful_marginal_roi_maintenance import toy_cell
from anchor_noise_cara_nested import (
    NOISE_CASES,ARM_OPTIONS,noisy_curve,NoisyCell,cara_events,cara_score,
    source_cara_choice,
)

class AnchorCARAReview(unittest.TestCase):
    def test_unperturbed_reference_reproduces_existing_head(self):
        c=toy_cell("AUDIT",n=210)
        w=NoisyCell(c,"nominal")
        self.assertEqual(len(w.bank),len(c.opportunities))
        for t in c.opportunities:
            # Toy bank is not fit from toy labels, so check only that
            # purchasing the unperturbed ORIGINAL y reproduces raw y.
            np.testing.assert_allclose(noisy_curve(c.test[t],"nominal",c.cid,t),
                                       c.test[t].y)

    def test_capacity_perturbation_moves_bought_reference_only(self):
        c=toy_cell("AUDIT",n=210)
        m=c.test[24]
        plus=noisy_curve(m,"capacity_plus2pct",c.cid,24)
        minus=noisy_curve(m,"capacity_minus2pct",c.cid,24)
        self.assertEqual(len(plus),m.n)
        self.assertTrue(np.all(plus>=m.y-1e-12))
        self.assertTrue(np.all(minus<=m.y+1e-12))
        before=m.y.copy()
        c2=NoisyCell(c,"capacity_plus2pct")
        np.testing.assert_allclose(c.test[24].y,before)
        self.assertNotEqual(c2.current_reference_fit(24),
                            NoisyCell(c,"capacity_minus2pct").current_reference_fit(24))

    def test_online_cara_not_consume_hidden_target_soc_labels(self):
        c=toy_cell("AUDIT",n=210)
        hidden=copy.deepcopy(c)
        for m in hidden.test:m.y[:]=0
        for th,allow in ARM_OPTIONS:
            self.assertEqual(cara_events(c,3,th,allow),
                             cara_events(hidden,3,th,allow))

    def test_source_choice_always_legal_and_uses_inner_cells_only(self):
        cell=toy_cell("AUDIT",n=210)
        (th,allow),records,ranking=source_cara_choice({"AUDIT":cell},2,.1)
        self.assertIn((th,allow),ARM_OPTIONS)
        self.assertEqual(set(records.cell_id),{"AUDIT"})
        self.assertEqual(len(ranking),len(ARM_OPTIONS))

    def test_noisy_online_state_is_released_only_after_paid_event(self):
        c=toy_cell("AUDIT",n=210)
        w=NoisyCell(c,"capacity_plus5pct")
        before=c.feature(24,-1,0)
        after=w.feature(24,-1,0)
        self.assertEqual(before,after)
        paid=w.feature(49,24,1)
        self.assertEqual(paid["references_acquired"],1.)
        self.assertEqual(paid["last_reference_raw_points"],50.)
        self.assertTrue(np.isfinite(list(paid.values())).all())

if __name__=="__main__":unittest.main()
