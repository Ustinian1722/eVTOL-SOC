#!/usr/bin/env python3
"""Source/heldout split, invariance, and negative-result checks."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from audit_robust_when_selection import source_decisions, heldout_evaluation, summarize


def run(input_dir):
    s = pd.read_csv(input_dir/"source_gate_grid.csv")
    c = pd.read_csv(input_dir/"source_gate_choices.csv")
    o = pd.read_csv(input_dir/"outer_22_cell_queue_gate.csv")
    decisions = source_decisions(s,c)
    assert len(decisions)==30 and decisions.source_n_cells.between(17,18).all()
    for r in decisions.itertuples(index=False):
        source_cells = set(s[(s.outer_fold==r.outer_fold)&(s.budget==r.budget)&
                             (s.price==r.price)&(s.delay==r.delay)].cell_id)
        target_cells = set(o[(o.outer_fold==r.outer_fold)&(o.budget==r.budget)&
                             (o.price==r.price)&(o.delay==r.delay)].cell_id)
        assert len(source_cells|target_cells)==22 and not (source_cells&target_cells)
    shuffled = source_decisions(s.sample(frac=1,random_state=19),
                                c.sample(frac=1,random_state=20))
    pd.testing.assert_frame_equal(decisions,shuffled,check_exact=True)
    h=heldout_evaluation(o,decisions)
    assert len(h)==3168 and h.cell_id.nunique()==22
    q=h[(h.budget==4)&(h.price==.1)&(h.delay==0)]
    def mean(policy,noise):
        return q[(q.policy==policy)&(q.reference_noise==noise)].J.mean()
    assert abs(mean("active","nominal")-1.954432)<1e-5
    assert abs(mean("active","capacity_plus5pct")-2.484840)<1e-5
    assert mean("minimax_regret","capacity_plus5pct")>mean("active","capacity_plus5pct")
    assert mean("minimax_cost","nominal")>mean("active","nominal")
    corrupt=o.copy()
    corrupt.loc[corrupt.role=="source_selected","J"]+=123456
    # The source-only decisions do not accept target data as an argument.
    pd.testing.assert_frame_equal(decisions,source_decisions(s,c),check_exact=True)
    assert heldout_evaluation(corrupt,decisions).query('policy=="active"').J.mean()>1e5
    summary=summarize(h,200)
    assert len(summary)==144 and summary.n_cells.eq(22).all()
    print("PASS: 22 physical cells, source-only selection, shuffled rows, heldout perturbation, NO-GO")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    run(p.parse_args().input)
