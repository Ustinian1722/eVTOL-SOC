"""Nested source/outer splits and delayed purchase correctness."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from test_stateful_marginal_roi_maintenance import toy_cell
from nested_cell_when_5fold import outer_splits,latency_replay
from dense_highrate_reference_maintenance import replay
from deferral_regret_when import CELL_IDS

class NestedTests(unittest.TestCase):
    def test_every_physical_cell_is_outer_once(self):
        splits=outer_splits()
        self.assertEqual(len(splits),5)
        alltest=[]
        for train,held in splits:
            self.assertFalse(set(train)&set(held))
            self.assertEqual(set(train)|set(held),set(CELL_IDS))
            self.assertGreaterEqual(len(train),17)
            alltest.extend(held)
        self.assertEqual(len(alltest),22)
        self.assertEqual(len(set(alltest)),22)

    def test_latency_zero_matches_original_head_replay(self):
        c=toy_cell(n=210)
        events=[24,99,174]
        out=latency_replay(c,events,0)
        original=replay(c.frozen_loss,c.loss,events)
        np.testing.assert_allclose(out,original,rtol=0,atol=1e-12)

    def test_purchased_head_never_activates_before_release(self):
        c=toy_cell(n=210)
        for lag in (5,25):
            got=latency_replay(c,[24],lag)
            np.testing.assert_allclose(got[:25+lag],c.frozen_loss[:25+lag])
            np.testing.assert_allclose(got[25+lag:],c.loss[24][25+lag:])

    def test_lag_is_nonnegative(self):
        with self.assertRaises(ValueError):
            latency_replay(toy_cell(),[24],-1)

if __name__=="__main__":
    unittest.main()
