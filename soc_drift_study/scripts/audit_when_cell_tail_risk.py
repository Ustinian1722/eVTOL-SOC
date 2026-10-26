#!/usr/bin/env python3
"""Physical-cell paired tail-risk audit of archived 22-cell causal WHEN replay.

Retrospective descriptive audit only. No new policy fitting, HOW changes, or raw CSV edits.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROLES = ('source_selected', 'source_fixed', 'periodic150', 'periodic200', 'frozen')
NOISES = ('nominal', 'capacity_plus2pct', 'capacity_plus5pct', 'capacity_iid2pct')
KEY = ['cell_id', 'outer_fold', 'budget', 'price', 'delay', 'reference_noise', 'role']


def validate_and_filter(raw, choices):
    x = raw[(raw.variant == 'ungated') & np.isinf(raw.gate)].copy()
    assert not x.duplicated(KEY).any(), 'duplicate physical-cell arm records'
    assert x.cell_id.nunique() == 22 and x.outer_fold.nunique() == 5
    assert set(x.reference_noise) == set(NOISES) and set(x.role) == set(ROLES)
    for (budget, price, delay, noise), g in x.groupby(['budget', 'price', 'delay', 'reference_noise']):
        assert len(g) == 22 * len(ROLES)
        assert g.cell_id.nunique() == 22
        assert g.groupby('cell_id').role.nunique().eq(len(ROLES)).all()
        assert g.groupby('cell_id').outer_fold.nunique().eq(1).all()
        assert np.allclose(g.J, g.MAE_pp + price * g.purchases, atol=1e-8)
    c = choices.set_index(['outer_fold', 'budget', 'price', 'delay'])
    for (fold, budget, price, delay), g in x.groupby(['outer_fold', 'budget', 'price', 'delay']):
        selected = c.loc[(fold, budget, price, delay)]
        assert set(g[g.role == 'source_selected'].arm) == {selected.source_arm}
        assert set(g[g.role == 'source_fixed'].arm) == {selected.source_fixed_arm}
    return x.sort_values(KEY).reset_index(drop=True)


def upper_tail(a, count=5):
    a = np.asarray(a, dtype=float)
    assert len(a) == 22
    return float(np.sort(a)[-count:].mean())


def analyze(x, nboot=10000, seed=20261025):
    rng = np.random.default_rng(seed)
    comparisons, cells, reversals = [], [], []
    group_keys = ['budget', 'price', 'delay', 'reference_noise']
    for (budget, price, delay, noise), g in x.groupby(group_keys, sort=True):
        p = g.pivot(index='cell_id', columns='role', values='J').sort_index()
        m = g.pivot(index='cell_id', columns='role', values='MAE_pp').sort_index()
        n = g.pivot(index='cell_id', columns='role', values='purchases').sort_index()
        assert p.shape == (22, 5) and not p.isna().any().any()
        for baseline in ROLES[1:]:
            d = (p.source_selected - p[baseline]).to_numpy()
            indices = rng.integers(0, 22, size=(nboot, 22))
            draws = d[indices]
            boot_mean = draws.mean(axis=1)
            boot_tail = np.sort(draws, axis=1)[:, -5:].mean(axis=1)
            comparisons.append(dict(budget=budget, price=price, delay=delay, reference_noise=noise,
                baseline=baseline, n_cells=22, mean_delta_J=float(d.mean()),
                ci_mean_low=float(np.quantile(boot_mean, .025)), ci_mean_high=float(np.quantile(boot_mean, .975)),
                upper_tail5_delta_J=upper_tail(d), ci_tail_low=float(np.quantile(boot_tail, .025)),
                ci_tail_high=float(np.quantile(boot_tail, .975)),
                worst_cell_delta_J=float(d.max()), worst_cell=str(p.index[np.argmax(d)]),
                active_better=int((d < -1e-12).sum()), active_worse=int((d > 1e-12).sum()),
                active_tied=int((np.abs(d) <= 1e-12).sum()),
                loo_mean_min=float((d.sum()-d).min()/21), loo_mean_max=float((d.sum()-d).max()/21),
                active_mean_J=float(p.source_selected.mean()), baseline_mean_J=float(p[baseline].mean()),
                active_mean_MAE=float(m.source_selected.mean()), baseline_mean_MAE=float(m[baseline].mean()),
                active_mean_purchases=float(n.source_selected.mean()), baseline_mean_purchases=float(n[baseline].mean())))
            for cell, delta in zip(p.index, d):
                cells.append(dict(cell_id=cell, budget=budget, price=price, delay=delay,
                    reference_noise=noise, baseline=baseline, delta_J=float(delta)))
    comp = pd.DataFrame(comparisons).sort_values(group_keys + ['baseline']).reset_index(drop=True)
    cell_df = pd.DataFrame(cells).sort_values(group_keys + ['baseline','cell_id']).reset_index(drop=True)
    for (budget, price, delay, baseline), g in cell_df.groupby(['budget','price','delay','baseline']):
        p = g.pivot(index='cell_id', columns='reference_noise', values='delta_J')
        d0, d5 = p.nominal, p.capacity_plus5pct
        reversals.append(dict(budget=budget, price=price, delay=delay, baseline=baseline,
            nominal_active_better=int((d0 < -1e-12).sum()), plus5_active_worse=int((d5 > 1e-12).sum()),
            benefit_to_harm_reversals=int(((d0 < -1e-12) & (d5 > 1e-12)).sum()),
            harm_to_benefit_reversals=int(((d0 > 1e-12) & (d5 < -1e-12)).sum()),
            reversed_cell_ids=';'.join(p.index[(d0 < -1e-12) & (d5 > 1e-12)])))
    rev = pd.DataFrame(reversals).sort_values(['budget','price','delay','baseline']).reset_index(drop=True)
    return comp, cell_df, rev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    folder, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(folder/'outer_22_cell_queue_gate.csv')
    choices = pd.read_csv(folder/'source_gate_choices.csv')
    x = validate_and_filter(raw, choices)
    comp, cells, rev = analyze(x)
    comp.to_csv(out/'paired_cell_tail_summary.csv', index=False, float_format='%.12g')
    cells.to_csv(out/'paired_cell_deltas.csv', index=False, float_format='%.12g')
    rev.to_csv(out/'noise_sign_reversals.csv', index=False)
    meta = {'source': 'archived successful raw-VAH-derived CI run 39102686653',
            'kind': 'retrospective physical-cell paired audit, NOT independent validation',
            'n_cells': 22, 'n_folds': 5, 'n_bootstrap': 10000,
            'bootstrap_unit': 'physical cell', 'seed': 20261025,
            'upper_tail_definition': 'mean of five largest per-cell paired J deltas among 22',
            'notes': ['historical SOC_proxy uses previous RPT capacity',
                      'lambda is hypothetical pp/purchase, not real operating cost',
                      'not a 100-to-0 high-rate diagnostic cycle or agricultural UAV field test']}
    (out/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('OK:', len(comp), 'comparisons,', len(cells), 'cell deltas,', len(rev), 'reversal summaries')


if __name__ == '__main__':
    main()
