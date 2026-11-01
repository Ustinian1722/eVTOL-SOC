"""Synthetic safeguards for report42; no label-truth claims."""
import unittest
import pandas as pd
from audit_terminal_duration_consistency import analyze

def fixture():
    rows=[]
    for cell in ['A','B']:
        for i in range(10):
            rows.append(dict(cell_id=cell,charge_cycle_index=i+1,ordinal=i,
                phase_duration_min=16 if i<9 else (15.2 if cell=='A' else 16),
                end_V=2.5001 if i==9 else (2.53 if i==8 else 2.9),
                end_T=40,end_ok=True,prior_charge_ordered=True,
                charge_peak_V=4.2))
    return pd.DataFrame(rows)

class Tests(unittest.TestCase):
    def test_joint_not_threshold_only(self):
        d,s,c,t=analyze(fixture())
        v=dict(zip(s.group,s.tasks))
        self.assertEqual(v['strict'],4)
        self.assertEqual(v['joint_last_near25_short10'],1)
        self.assertFalse(d.independent_SOC_truth.any())
        self.assertFalse(d.verified_cutoff_reason.any())
    def test_row_order_invariance(self):
        a=analyze(fixture())
        b=analyze(fixture().sample(frac=1,random_state=31))
        for x,y in zip(a,b):
            pd.testing.assert_frame_equal(x,y)
    def test_duplicate_rejected(self):
        d=fixture()
        with self.assertRaises(ValueError):
            analyze(pd.concat([d,d.iloc[[0]]]))
    def test_bad_charge_rejected(self):
        d=fixture();d.loc[0,'prior_charge_ordered']=False
        with self.assertRaises(ValueError):analyze(d)
    def test_bad_endpoint_rejected(self):
        d=fixture();d.loc[0,'end_ok']=False
        with self.assertRaises(ValueError):analyze(d)

if __name__=='__main__':unittest.main()
