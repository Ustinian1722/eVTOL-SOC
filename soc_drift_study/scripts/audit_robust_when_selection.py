#!/usr/bin/env python3
"""Source-only minimax WHEN analysis of archived 22-cell queue experiment.
No retraining or original CSV mutation; no held-out data used for policy selection.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SELECT_NOISES = ("nominal", "capacity_plus5pct")
EVAL_NOISES = ("nominal", "capacity_plus2pct", "capacity_plus5pct", "capacity_iid2pct")
POLICIES = ("active", "periodic200", "source_fixed", "source_nominal", "minimax_cost", "minimax_regret")


def source_decisions(source, choices):
    rows = []
    src = source[np.isinf(source.gate)].copy()
    keys = ["budget", "price", "delay", "outer_fold"]
    for (budget, price, delay, fold), row in choices.set_index(keys).iterrows():
        active = row.source_arm
        q = src[(src.budget == budget) & (src.price == price) &
                (src.delay == delay) & (src.outer_fold == fold) &
                src.noise.isin(SELECT_NOISES) & src.arm.isin((active, "periodic200"))]
        p = q.pivot(index=["cell_id", "noise"], columns="arm", values="J")
        n = p.index.get_level_values("cell_id").nunique()
        assert n in (17, 18) and len(p) == 2 * n
        assert p.notna().all().all()
        assert set(p.index.get_level_values("noise")) == set(SELECT_NOISES)
        assert set(p.columns) == {active, "periodic200"} or active == "periodic200"
        mean = p.groupby(level="noise").mean()
        regret = p.subtract(p.min(axis=1), axis=0).groupby(level="noise").mean()
        options = [active] if active == "periodic200" else [active, "periodic200"]
        def choose(scores):
            return min(options, key=lambda a: (round(float(scores[a]), 12), a != "periodic200"))
        rows.append(dict(budget=budget, price=price, delay=delay, outer_fold=fold,
                         source_n_cells=n, source_arm=active, source_fixed_arm=row.source_fixed_arm,
                         source_nominal_arm=choose(mean.loc["nominal"]),
                         minimax_cost_arm=choose(mean.max(axis=0)),
                         minimax_regret_arm=choose(regret.max(axis=0)),
                         source_nominal_delta_active_minus_p200=float(mean.loc["nominal", active] - mean.loc["nominal", "periodic200"]),
                         source_plus5_delta_active_minus_p200=float(mean.loc["capacity_plus5pct", active] - mean.loc["capacity_plus5pct", "periodic200"]),
                         source_worst_regret_active=float(regret[active].max()),
                         source_worst_regret_p200=float(regret["periodic200"].max())))
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def heldout_evaluation(outer, decisions):
    x = outer[(outer.variant == "ungated") & outer.reference_noise.isin(EVAL_NOISES)].copy()
    keys = ["cell_id", "outer_fold", "budget", "price", "delay", "reference_noise", "role"]
    assert not x.duplicated(keys).any()
    out = []
    for dec in decisions.itertuples(index=False):
        z = x[(x.outer_fold == dec.outer_fold) & (x.budget == dec.budget) &
              (x.price == dec.price) & (x.delay == dec.delay)]
        assert z.cell_id.nunique() == 22 - dec.source_n_cells
        for (cell, noise), g in z.groupby(["cell_id", "reference_noise"]):
            vals = g.set_index("role")
            assert set(vals.index) == {"source_selected", "periodic200", "periodic150", "source_fixed", "frozen"}
            roles = {"active": "source_selected", "periodic200": "periodic200",
                     "source_fixed": "source_fixed",
                     "source_nominal": "periodic200" if dec.source_nominal_arm == "periodic200" else "source_selected",
                     "minimax_cost": "periodic200" if dec.minimax_cost_arm == "periodic200" else "source_selected",
                     "minimax_regret": "periodic200" if dec.minimax_regret_arm == "periodic200" else "source_selected"}
            oracle = min(float(vals.loc["source_selected", "J"]), float(vals.loc["periodic200", "J"]))
            for policy, role in roles.items():
                r = vals.loc[role]
                out.append(dict(cell_id=cell, outer_fold=dec.outer_fold, budget=dec.budget,
                                price=dec.price, delay=dec.delay, reference_noise=noise,
                                policy=policy, selected_role=role, MAE_pp=float(r.MAE_pp),
                                J=float(r.J), purchases=int(r.purchases),
                                label_points=int(r.label_points),
                                J_minus_two_arm_oracle=float(r.J)-oracle))
    h = pd.DataFrame(out)
    assert len(h) == 2*3*22*4*len(POLICIES)
    assert h.groupby(["budget","price","delay","reference_noise","policy"]).cell_id.nunique().eq(22).all()
    return h


def summarize(h, nboot=10000):
    rng = np.random.default_rng(20261023)
    out = []
    for (budget, price, delay, noise), group in h.groupby(["budget","price","delay","reference_noise"]):
        p = group.pivot(index="cell_id", columns="policy", values="J").sort_index()
        for policy in POLICIES:
            delta = (p[policy]-p["active"]).to_numpy()
            boot = delta[rng.integers(0, len(delta), size=(nboot,len(delta)))].mean(axis=1)
            q = group[group.policy == policy]
            out.append(dict(budget=budget,price=price,delay=delay,reference_noise=noise,
                            policy=policy,J_mean=float(q.J.mean()),MAE_mean=float(q.MAE_pp.mean()),
                            purchases_mean=float(q.purchases.mean()),
                            J_delta_vs_active=float(delta.mean()),
                            CI_low=float(np.quantile(boot,.025)),CI_high=float(np.quantile(boot,.975)),
                            cells_better_than_active=int((delta < -1e-12).sum()),
                            mean_J_minus_two_arm_oracle=float(q.J_minus_two_arm_oracle.mean()),n_cells=22))
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input/"source_gate_grid.csv")
    choices = pd.read_csv(args.input/"source_gate_choices.csv")
    outer = pd.read_csv(args.input/"outer_22_cell_queue_gate.csv")
    decisions = source_decisions(source, choices)
    for r in decisions.itertuples(index=False):
        a = set(source[(source.outer_fold==r.outer_fold)&(source.budget==r.budget)&
                       (source.price==r.price)&(source.delay==r.delay)].cell_id)
        b = set(outer[(outer.outer_fold==r.outer_fold)&(outer.budget==r.budget)&
                      (outer.price==r.price)&(outer.delay==r.delay)].cell_id)
        assert not a & b and len(a | b)==22
    heldout = heldout_evaluation(outer, decisions)
    summary = summarize(heldout,args.bootstrap)
    decisions.to_csv(args.out/"source_only_robust_choices.csv",index=False,float_format="%.12g")
    heldout.to_csv(args.out/"heldout_robust_policy_cells.csv",index=False,float_format="%.12g")
    summary.to_csv(args.out/"robust_policy_summary.csv",index=False,float_format="%.12g")
    meta = dict(source="CI 39102686653 causal-queue-22cell-quality-results",
                n_cells=22,source_decisions=len(decisions),heldout_rows=len(heldout),
                summary_rows=len(summary),bootstrap=args.bootstrap,seed=20261023,
                source_heldout_disjoint=True,target_used_for_selection=False,
                selection_noises=SELECT_NOISES,
                limitation="Retrospective internal five-fold analysis; not new raw-VAH retraining or agricultural field validation.")
    (args.out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(meta,ensure_ascii=False))
    print(summary[(summary.budget==4)&(summary.price==.1)&(summary.delay==0)&
                  summary.reference_noise.isin(SELECT_NOISES)][
                      ["reference_noise","policy","J_mean","J_delta_vs_active","CI_low","CI_high"]].to_string(index=False))


if __name__=="__main__":
    main()
