"""Reprice archived causal-queue 22-cell results, without rerunning or changing HOW."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

GAMMAS = (0, .1, .5, 1, 2)
NOISES = ('nominal', 'capacity_plus2pct', 'capacity_plus5pct', 'capacity_iid2pct')
DELAYS = (0, 5, 25)

def paired_ci(delta):
    v=np.asarray(delta,dtype=float)
    rng=np.random.default_rng(20261021)
    means=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(axis=1)
    return np.quantile(means,[.025,.975])

def calculate(inputs):
    outer=pd.read_csv(inputs/'outer_22_cell_queue_gate.csv')
    source=pd.read_csv(inputs/'source_gate_grid.csv')
    choices=pd.read_csv(inputs/'source_gate_choices.csv')
    assert outer.cell_id.nunique()==22
    for df in (outer,source):
        assert np.max(np.abs(df.J-(df.MAE_pp+df.price*df.purchases)))<1e-8
    held=outer[(outer.budget==4)&(np.isclose(outer.price,.1))&(outer.variant=='ungated')].copy()
    assert not held.duplicated(['cell_id','delay','reference_noise','role']).any()
    assert held.groupby(['delay','reference_noise','role']).cell_id.nunique().eq(22).all()
    results=[]; selections=[]
    for delay in DELAYS:
        for gamma in GAMMAS:
            fold_choice={}
            for fold in range(5):
                chosen_arm=choices[(choices.outer_fold==fold)&(choices.budget==4)&
                    (np.isclose(choices.price,.1))&(choices.delay==delay)].source_arm.iloc[0]
                src=source[(source.outer_fold==fold)&(source.budget==4)&
                    (np.isclose(source.price,.1))&(source.delay==delay)&
                    (source.noise=='nominal')&(np.isinf(source.gate))&
                    (source.arm.isin([chosen_arm,'periodic200']))].copy()
                assert src.cell_id.nunique()>=17
                assert set(src.cell_id).isdisjoint(set(held[held.outer_fold==fold].cell_id))
                src['newJ']=src.MAE_pp+.1*src.purchases+gamma*src.label_points/1000
                means=src.groupby('arm').newJ.mean()
                choice=min(means.index,key=lambda arm:(means.loc[arm],arm!='periodic200'))
                fold_choice[fold]=choice
                selections.append(dict(delay=delay,gamma=gamma,fold=fold,choice=choice,
                    active_arm=chosen_arm,source_n_cells=src.cell_id.nunique()))
            for noise in NOISES:
                d=held[(held.delay==delay)&(held.reference_noise==noise)].copy()
                d['newJ']=d.MAE_pp+.1*d.purchases+gamma*d.label_points/1000
                wide=d.pivot(index='cell_id',columns='role',values='newJ')
                delta=wide.source_selected-wide.source_fixed
                lo,hi=paired_ci(delta)
                for role,g in d.groupby('role'):
                    results.append(dict(delay=delay,gamma=gamma,noise=noise,role=role,
                        meanJ=float(g.newJ.mean()),meanMAE=float(g.MAE_pp.mean()),
                        meanPurchases=float(g.purchases.mean()),meanPoints=float(g.label_points.mean()),
                        active_minus_fixed=float(delta.mean()),ci_low=float(lo),ci_high=float(hi)))
    return pd.DataFrame(results),pd.DataFrame(selections)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    r,s=calculate(a.input)
    a.out.mkdir(parents=True,exist_ok=True)
    r.to_csv(a.out/'workload_cost_summary.csv',index=False)
    s.to_csv(a.out/'source_only_choices.csv',index=False)
    key=r.query('delay==0 and gamma==0 and role=="source_selected"').set_index('noise')
    assert abs(key.loc['nominal','meanJ']-1.954443935)<.0001
    assert abs(key.loc['capacity_plus5pct','meanJ']-2.4848)<.0002
    (a.out/'metadata.json').write_text(json.dumps(dict(
        origin='GitHub Actions 39102686653 archived CSV, NOT new raw VAH replay',
        physical_cells=22,independent_test=False,original_HOW_unchanged=True,
        RPT_based_proxy=True,actual_agricultural_cost=False,
        source_selection_nominal_only=True),indent=2),encoding='utf8')
    print('PASS',len(r),'scenario rows',len(s),'source choices')
    print(key[['meanJ','meanPurchases','meanPoints','active_minus_fixed','ci_low','ci_high']])
if __name__=='__main__':main()
