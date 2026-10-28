"""Synthetic regression tests: no actual SOC truth is invented."""
import unittest
import pandas as pd
from audit_mission_discharge_extent import inventory,stats

def fixture():
    lab=[];seg=[]
    for cell,cy,cap,qs in [("VAH01",1,3,[.5,1,1.4]),("VAH02",2,2.8,[.7,1.7,2.85])]:
        for ns,q in zip((4,5,6),qs):
            lab.append(dict(cell_id=cell,charge_cycle_index=cy,phase_Ns=ns,
                QDischarge_end_Ah=q,capacity_used_Ah=cap,
                capacity_source="latest_completed_candidate_RPT",
                SOC_proxy=max(0,1-q/cap),rows_in_phase=10,duration_s=100))
            seg.append(dict(cell_id=cell,charge_cycle_index=cy,Ns=ns,rows=10,duration_s=100))
    return pd.DataFrame(lab),pd.DataFrame(seg)
class TestMissionExtent(unittest.TestCase):
    def test_clipped_proxy_not_full_diagnostic(self):
        a,b=fixture();m,_=inventory(a,b)
        self.assertEqual(len(m),2)
        self.assertEqual(int(m.proxy_ge100.sum()),1)
        self.assertFalse(m.independent_full_highrate_diagnostic_verified.any())
        self.assertAlmostEqual(float(m.iloc[0].fraction_of_historical_anchor),1.4/3)
        self.assertEqual(int(stats(m).missions),2)
    def test_duplicate_rejected(self):
        a,b=fixture();a=pd.concat([a,a.iloc[[0]]],ignore_index=True)
        m,meta=inventory(a,b)
        self.assertEqual(len(m),1)
        self.assertEqual(meta["excluded_by_reason"]["duplicate_phase"],1)
    def test_row_mismatch_rejected(self):
        a,b=fixture();b.loc[(b.cell_id=="VAH01")&(b.Ns==4),"rows"]=5
        m,meta=inventory(a,b)
        self.assertEqual(len(m),1)
        self.assertEqual(meta["excluded_by_reason"]["row_mismatch"],1)
    def test_nonmonotonic_counter_rejected(self):
        a,b=fixture();a.loc[(a.cell_id=="VAH01")&(a.phase_Ns==5),"QDischarge_end_Ah"]=.1
        m,meta=inventory(a,b)
        self.assertEqual(len(m),1)
        self.assertEqual(meta["excluded_by_reason"]["counter_reversal"],1)
if __name__=="__main__":unittest.main()
