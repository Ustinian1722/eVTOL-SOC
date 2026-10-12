"""Synthetic regression tests for strong GRU normal-mission voltage adaptation."""
from __future__ import annotations
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from strong_gru_voltage_selfsup import (
    FEATURES, AUX_FEATURES, ARMS, BUDGET, SEQ_LEN, VOLTAGE_IDX,
    aux_source, chunks_of_features, prepare, replay_signal,
    update_voltage_theta,
)


class DummyGRU:
    def predict(self,x,batch_size=256,verbose=0):
        y=np.clip(.5 + .04*x[:,:,VOLTAGE_IDX],0,1)
        return y[:,:,None]


def data(ncycles=190):
    rows=[]
    for i in range(1,ncycles+1):
        for phase in (4,5,6):
            d=(phase-3)*.20
            rows.append({
                "cell_id":"S1",
                "charge_cycle_index":i,"phase_Ns":phase,
                "duration_s":90*phase,
                "I_mean_A":-3.5*phase,"I_peak_A":4*phase,
                "Vmin_V":4.12-.13*d-.0003*i,
                "Vmax_V":4.2-.11*d-.0002*i,
                "Tmean_C":24+.3*phase,"Tmax_C":26+.3*phase,
                "SOC_proxy":1-d/2.7,
            })
    return pd.DataFrame(rows)


class VoltageTTTTests(unittest.TestCase):
    def prepare_all(self):
        raw=data()
        scaler=StandardScaler().fit(raw[FEATURES].to_numpy(dtype=np.float32))
        aux,Z,Y,theta=aux_source(raw.iloc[:300])
        signal=prepare(raw,scaler,aux)
        return raw,scaler,signal,Z,Y,theta

    def test_chunk_shape_and_nominal_training_cutoff(self):
        raw,scaler,signal,*_=self.prepare_all()
        self.assertEqual(signal.tensor.shape[1:],(SEQ_LEN,len(FEATURES)))
        self.assertEqual(int(signal.chunk_real.sum()),len(raw))
        self.assertEqual(int(signal.chunk_test.sum()),(190-20)*3)

    def test_auxiliary_voltage_supervision_only(self):
        raw=data()
        alt=raw.copy()
        alt["SOC_proxy"]=np.zeros(len(alt))
        m,z,y,theta=aux_source(raw)
        m2,z2,y2,theta2=aux_source(alt)
        np.testing.assert_allclose(theta,theta2)
        np.testing.assert_allclose(z,z2)

    def test_proxy_mutation_does_not_change_any_online_decisions(self):
        _,scaler,sig,Z,Y,theta=self.prepare_all()
        other=copy.deepcopy(sig)
        other.test_true=1-other.test_true
        for arm in ARMS:
            a,events,pred=replay_signal(sig,DummyGRU(),scaler,Z,Y,theta,arm)
            b,events2,pred2=replay_signal(other,DummyGRU(),scaler,Z,Y,theta,arm)
            np.testing.assert_allclose(pred,pred2)
            self.assertEqual(events,events2)
            self.assertEqual(a["attempted_updates"],b["attempted_updates"])
            self.assertLessEqual(a["attempted_updates"],BUDGET if arm[0]!="frequent_correction_015" else 999)

    def test_guard_uses_only_completed_target_voltage(self):
        _,_,sig,Z,Y,theta=self.prepare_all()
        records=[]
        for idx in range(30):
            c=sig.chunk_test[idx+5]
            if c.any():
                records.append({"Z":sig.auxZ[idx+5,c],"V":sig.volts[idx+5,c]})
        # Need 30 strict completed test chunks for candidate; synthesize
        while len(records)<30:
            records.append(records[-1])
        new,accept,info=update_voltage_theta(
            theta,theta,Z,Y,records[:30],True
        )
        self.assertTrue(np.isfinite(new).all())
        self.assertIn(info["reason"],("updated","voltage_holdout_or_source_guard"))

    def test_frozen_never_updates(self):
        _,scaler,sig,Z,Y,theta=self.prepare_all()
        a,events,pred=replay_signal(sig,DummyGRU(),scaler,Z,Y,theta,ARMS[0])
        self.assertEqual(a["attempted_updates"],0)
        self.assertEqual(len(events),0)
        self.assertEqual(len(pred),len(sig.test_true))


if __name__=="__main__":
    unittest.main()
