"""Validate source GRU masks and ensure chunk boundaries do not cross cells."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from train_gru_proxy_crosscell import FEATURES, _make_chunks
from retrain_gru_boundary_safe import pack_cellwise


def toy_data():
    rows=[]
    for cell,n in [("X1",13),("X2",14),("X3",3)]:
        for k in range(1,n+1):
            rows.append({
                "cell_id":cell,"charge_cycle_index":k,"phase_Ns":4,
                "duration_s":40+k,"I_mean_A":-10.,
                "I_peak_A":11.,"Vmin_V":3.8-.01*k,
                "Vmax_V":4.1-.01*k,"Tmean_C":25.,"Tmax_C":27.,
                "SOC_proxy":.8-.01*k,
            })
    return pd.DataFrame(rows)


class ChunkSafetyTest(unittest.TestCase):
    def test_all_real_phase_rows_preserved(self):
        raw=toy_data()
        scaler=StandardScaler().fit(raw[FEATURES].to_numpy(dtype=np.float32))
        x,y,m,owners=pack_cellwise(raw,scaler)
        self.assertEqual(int(m.sum()),len(raw))
        self.assertEqual(x.shape[0],len(owners))
        self.assertEqual(len(set(owners)),3)

    def test_boundary_local_chunks_differ_from_old_flat_packing(self):
        raw=toy_data()
        scaler=StandardScaler().fit(raw[FEATURES])
        x,y,m,owners=pack_cellwise(raw,scaler)
        old_x,old_y,old_m=_make_chunks(raw,scaler)
        self.assertGreater(len(x),len(old_x))
        self.assertEqual(owners,["X1","X1","X2","X2","X3"])
        self.assertEqual([int(v) for v in m.sum(axis=1)],[12,1,12,2,3])
        # A flat chunk after index 12 would mix rows from X1 and X2.
        self.assertEqual([int(v) for v in old_m.sum(axis=1)],[12,12,6])

    def test_source_permutation_and_cell_order_not_affect_real_count(self):
        raw=toy_data()
        scaler=StandardScaler().fit(raw[FEATURES])
        x,y,m,owners=pack_cellwise(raw.sample(frac=1,random_state=123),scaler)
        self.assertEqual(int(m.sum()),len(raw))


if __name__=="__main__":
    unittest.main()
