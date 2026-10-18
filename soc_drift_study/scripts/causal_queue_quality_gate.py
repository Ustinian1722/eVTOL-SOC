"""Nested 22-cell evaluation of strictly causal reference queue and quality gate."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from anchor_noise_cara_nested import NoisyCell
from deferral_regret_when import build_cell, source_examples, source_models
from nested_cell_when_5fold import outer_splits
from scenario_safe_when_selector import choose_from_source, evaluate_one
from causal_queue_core import queue_replay

SCENARIOS=((4,.1),(2,.3))
DELAYS=(0,5,25)
NOISE=("nominal","capacity_plus2pct","capacity_plus5pct","capacity_iid2pct")
GATES=(float("inf"),2.,4.,6.,10.)

def select_source_gate(dev,oof,selected,budget,price,delay):
    """One gate for active and periodic200; choose on source OOF only."""
    rows=[]
    for cid,original in sorted(dev.items()):
        for noise,cell in (("nominal",original),
                           ("capacity_plus5pct",NoisyCell(original,"capacity_plus5pct"))):
            for gate in GATES:
                for arm in dict.fromkeys((selected,"periodic200")):
                    z=queue_replay(cell,oof[cid],arm,budget,price,delay,gate)
                    rows.append(dict(cell_id=cid,noise=noise,arm=arm,gate=gate,**z))
    scores=pd.DataFrame(rows)
    means=scores.groupby(["gate","noise"],as_index=False).J.mean()
    worst=means.groupby("gate").J.max()
    gate=min(GATES,key=lambda g:(float(worst.loc[g]),0 if np.isinf(g) else 1,-g))
    return gate,scores,float(worst.loc[gate])

def run_fold(fold,out):
    source_ids,held_ids=outer_splits()[fold]
    dev,held,source={},{},[]
    for cid in sorted(set(source_ids)|set(held_ids)):
        cell=build_cell(cid)
        if cid in source_ids:
            dev[cid]=cell
            source.extend(source_examples(cell))
        else: held[cid]=cell
        print("PARSED",cid,"source" if cid in dev else "held",flush=True)
    assert set(dev).isdisjoint(held)
    _,cv,oof,final=source_models(dev,pd.DataFrame(source))
    _,inner,selected=choose_from_source(dev,oof)
    rows,grids,choices=[],[],[]
    for budget,price in SCENARIOS:
        chosen=str(selected[(selected.budget==budget)&
                            np.isclose(selected.price,price)].iloc[0].source_chosen)
        fixed=inner[(inner.budget==budget)&np.isclose(inner.price,price)&
                    inner.arm.str.startswith("periodic")].copy()
        fixed["score"]=.7*fixed.meanJ+.3*fixed.worstJ
        source_fixed=str(fixed.sort_values(["score","meanJ","arm"]).iloc[0].arm)
        for delay in DELAYS:
            gate,grid,src_worst=select_source_gate(dev,oof,chosen,budget,price,delay)
            grid["outer_fold"]=fold
            grid["budget"]=budget
            grid["price"]=price
            grid["delay"]=delay
            grids.append(grid)
            choices.append(dict(outer_fold=fold,budget=budget,price=price,
                                delay=delay,gate=gate,source_arm=chosen,
                                source_fixed_arm=source_fixed,
                                source_worst_noise_J=src_worst))
            for cid,original in sorted(held.items()):
                for noise in NOISE:
                    cell=original if noise=="nominal" else NoisyCell(original,noise)
                    arms=(("source_selected",chosen),("periodic150","periodic150"),
                          ("periodic200","periodic200"),
                          ("source_fixed",source_fixed),("frozen","frozen"))
                    for role,arm in arms:
                        for variant,threshold in (("ungated",float("inf")),
                                                  ("source_gate",gate)):
                            z=queue_replay(cell,final,arm,budget,price,delay,threshold)
                            if delay==0 and variant=="ungated":
                                legacy=evaluate_one(cell,final,arm,budget,price)
                                assert abs(z["MAE_pp"]-legacy["MAE_pp"])<1e-8,(
                                    cid,noise,arm,z,legacy)
                                assert z["events"]==legacy["events"],(
                                    cid,noise,arm,z,legacy)
                            rows.append(dict(cell_id=cid,outer_fold=fold,
                                budget=budget,price=price,delay=delay,
                                reference_noise=noise,role=role,arm=arm,
                                variant=variant,gate=threshold,**z))
                print("HELD",fold,cid,budget,price,delay,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"outer_queue_gate.csv",index=False)
    pd.concat(grids,ignore_index=True).to_csv(out/"source_gate_grid.csv",index=False)
    pd.DataFrame(choices).to_csv(out/"source_gate_choices.csv",index=False)
    cv.to_csv(out/"source_regressor_cv.csv",index=False)
    (out/"metadata.json").write_text(json.dumps(dict(
        fold=fold,source_cells=list(source_ids),held_cells=list(held_ids),
        all_22_cells_previously_explored=True,not_independent_test=True,
        historical_SOC_proxy_uses_previous_RPT_capacity=True,
        normal_mission_not_100_to_0=True,no_real_agriculture=True,
        gate_source_OOF_only=True,rejected_labels_still_cost=True,
        delayed_feedback_affects_WHEN=True,original_CSV_unchanged=True),
        indent=2,ensure_ascii=False),encoding="utf8")
    print("FOLD_DONE",fold,len(held),len(rows),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--outer-fold",type=int,choices=range(5),required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run_fold(a.outer_fold,a.out)
