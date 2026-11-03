#!/usr/bin/env python3
"""Report 43: audit charge/discharge observability using Report 42 derived CSV."""
import argparse
from pathlib import Path
import pandas as pd

def main(source, output):
    df = pd.read_csv(source)
    assert len(df) == 20848 and df.cell_id.nunique() == 22
    assert not df.duplicated(['cell_id', 'charge_cycle_index']).any()
    c = df.loc[df.joint_last_near25_short10.astype(bool)].copy()
    assert len(c) == 15 and c.cell_id.nunique() == 15
    c['charge_to_discharge_Q_ratio'] = c.charge_Qmax_Ah / c.Q_end_Ah
    c['charge_minus_discharge_Ah'] = c.charge_Qmax_Ah - c.Q_end_Ah
    assert (c.charge_Qmax_Ah < c.Q_end_Ah).all()
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    c[['cell_id','charge_cycle_index','charge_peak_V','charge_Qmax_Ah','Q_end_Ah',
       'charge_to_discharge_Q_ratio','charge_minus_discharge_Ah','charge_to_mission_gap_min',
       'capacity_anchor_Ah','fraction_of_historical_anchor','duration_deficit_s']].to_csv(
        out / 'report43_candidates.csv', index=False, float_format='%.10f')
    summary = {'n':len(c),'cells':c.cell_id.nunique(),
       'charge_less_than_discharge':int((c.charge_Qmax_Ah<c.Q_end_Ah).sum()),
       'ratio_median':c.charge_to_discharge_Q_ratio.median(),
       'ratio_min':c.charge_to_discharge_Q_ratio.min(),
       'ratio_max':c.charge_to_discharge_Q_ratio.max(),
       'peak_ge_4p18':int((c.charge_peak_V>=4.18).sum())}
    pd.DataFrame([summary]).to_csv(out / 'report43_summary.csv',index=False)
    print(summary)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();main(a.input,a.output)
