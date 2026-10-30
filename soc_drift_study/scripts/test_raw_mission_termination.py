"""Synthetic safeguards for original-sample terminal audit."""
import unittest
import pandas as pd
from audit_raw_mission_termination import match_chunk, combine, targets_from_segments

class TestRawTerminal(unittest.TestCase):
    def test_timestamp_and_ns_required(self):
        raw=pd.DataFrame({"time_s":[1,2,2,3],"Ns":[4,4,6,6],
            "Ecell_V":[4.2,4.1,3.1,2.6],"I_mA":[-2000]*4,
            "Temperature__C":[25]*4,"QDischarge_mA_h":[0,10,10,20]})
        t=pd.DataFrame({"target_t":[1,2,3,4],"Ns":[4,6,6,6]},index=[10,11,12,13])
        h=match_chunk(raw,t)
        self.assertEqual([x["target_id"] for x in h],[10,11,12])
        self.assertEqual([x["V"] for x in h],[4.2,3.1,2.6])
    def test_no_nearest_interpolation(self):
        raw=pd.DataFrame({"time_s":[1,3],"Ns":[4,6],
            "Ecell_V":[4,2.5],"I_mA":[-1000]*2,
            "Temperature__C":[20]*2,"QDischarge_mA_h":[0,1]})
        self.assertEqual(match_chunk(raw,pd.DataFrame({"target_t":[2.5],"Ns":[6]})),[])
    def test_ambiguous_match_rejected(self):
        m=pd.DataFrame({"cell_id":["X"],"charge_cycle_index":[1],"Q_end_Ah":[1.]})
        t=pd.DataFrame({"cell_id":["X","X"],"charge_cycle_index":[1,1],
            "kind":["start","end"],"Ns":[4,6],"target_t":[1,2]})
        h=pd.DataFrame({"target_id":[0,1],"match_count":[1,2],
            "V":[4.1,2.5],"T":[25,70],"I_A":[-3,-3],"Q_Ah":[0,1]})
        result=combine(m,t,h)
        self.assertFalse(bool(result.end_ok.iloc[0]))
        self.assertFalse(bool(result.independent_highrate_full_diagnostic_verified.iloc[0]))
    def test_target_boundary_selection(self):
        m=pd.DataFrame({"cell_id":["X"],"charge_cycle_index":[1]})
        s=pd.DataFrame({"cell_id":["X","X"],"charge_cycle_index":[1,1],
            "Ns":[4,6],"time_start_s":[2,6],"time_end_s":[3,7]})
        t=targets_from_segments(m,s)
        self.assertEqual(t.target_t.tolist(),[2,7])
if __name__=="__main__":unittest.main()
