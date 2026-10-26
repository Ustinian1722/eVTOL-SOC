#!/usr/bin/env python3
"""Regression and anti-duplication tests for descriptive tail audit."""
import argparse
from pathlib import Path
import pandas as pd
from audit_when_cell_tail_risk import validate_and_filter, analyze


def run(folder):
    raw = pd.read_csv(folder/'outer_22_cell_queue_gate.csv')
    choices = pd.read_csv(folder/'source_gate_choices.csv')
    x = validate_and_filter(raw, choices)
    assert len(x) == 2640
    comp, cells, rev = analyze(x, nboot=500)
    assert len(comp) == 96 and len(cells) == 2112 and len(rev) == 24
    q = comp.query("budget==4 and price==0.1 and delay==0 and baseline=='source_fixed'")
    nominal = q[q.reference_noise=='nominal'].iloc[0]
    plus5 = q[q.reference_noise=='capacity_plus5pct'].iloc[0]
    assert abs(nominal.mean_delta_J + .0931) < .001
    assert nominal.active_better == 21 and nominal.active_worse == 1
    assert abs(plus5.mean_delta_J - .0657) < .001
    assert plus5.active_worse == 20 and plus5.active_better == 2
    z = rev.query("budget==4 and price==0.1 and delay==0 and baseline=='source_fixed'").iloc[0]
    assert z.benefit_to_harm_reversals == 19
    shuffled = x.sample(frac=1, random_state=37)
    c2, d2, r2 = analyze(shuffled, nboot=500)
    pd.testing.assert_frame_equal(comp, c2)
    pd.testing.assert_frame_equal(cells, d2)
    pd.testing.assert_frame_equal(rev, r2)
    corrupted = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    try:
        validate_and_filter(corrupted, choices)
    except AssertionError:
        pass
    else:
        raise AssertionError('duplicate records were not rejected')
    altered = raw.copy()
    altered.loc[altered.variant=='source_gate', 'J'] += 999999
    x2 = validate_and_filter(altered, choices)
    pd.testing.assert_frame_equal(x, x2)
    print('PASS: 22 physical cells, paired tails, 19 reversals, order invariance, duplicates and variant isolation')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    run(Path(ap.parse_args().input))
