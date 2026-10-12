"""Five outer physical-cell folds; inner cell-group OOF model/policy selection.

Original 22 cells have been repeatedly studied: internal retrospective
cross-fitting, NOT a pristine independent test. Historical SOC_proxy uses
previous RPT capacity; reference acquisition costs are hypothetical.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from deferral_regret_when import CELL_IDS,SEED,build_cell,source_examples,source_models
from scenario_safe_when_selector import SCENARIOS,ACTIVE,choose_from_source,evaluate_one
from stateful_marginal_roi_maintenance import run_cell

def outer_splits(ids=CELL_IDS):
    arr=np.array(sorted(ids))
    return [(tuple(arr[tr]),tuple(arr[te])) for tr,te in
            KFold(n_splits=5,shuffle=True,random_state=SEED).split(arr)]

def latency_replay(cell,events,delay):
    """New paid head can first serve at mission t+delay+1."""
    if delay<0:raise ValueError("delay must be nonnegative")
    losses=cell.frozen_loss.copy()
    activation=sorted((int(t)+delay+1,int(t)) for t in events)
    last=None;j=0
    for k in range(cell.n):
        while j<len(activation) and activation[j][0]<=k:
            last=activation[j][1];j+=1
        if last is not None:losses[k]=cell.loss[last][k]
    return losses

def run(fold,outdir):
    training_ids,test_ids=outer_splits()[fold]
    training={};held={};sources=[];manifest=[]
    for cid in CELL_IDS:
        cell=build_cell(cid)
        if cid in training_ids:
            training[cid]=cell;sources.extend(source_examples(cell))
        else:held[cid]=cell
        manifest.append({"cell_id":cid,"outer_fold":fold,
            "role":"training" if cid in training_ids else "outer_test",
            **cell.raw_stats})
        print("LOADED",cid,cell.n,flush=True)
    src=pd.DataFrame(sources)
    assert set(src.cell_id)==set(training_ids)
    assert set(training).isdisjoint(held)
    winners,cv,oof,model=source_models(training,src)
    _,inner_arms,selected=choose_from_source(training,oof)
    assert len(selected)==len(SCENARIOS)
    records=[];latency=[]
    for cid,cell in sorted(held.items()):
        for sc in selected.itertuples():
            arms={"frozen","periodic150","periodic200","periodic250",
                  "periodic300","periodic400","periodic500",
                  "offline_oracle",sc.source_chosen}
            for arm in sorted(arms):
                if arm=="offline_oracle":
                    raw,_=run_cell(cell,arm,sc.budget,sc.price)
                    z={"MAE_pp":raw["SOCproxy_MAE_pp"],"J":raw["J"],
                       "purchases":raw["num_reference_purchases"],
                       "label_points":raw["num_raw_labels_purchased"],
                       "events":raw["reference_idx"]}
                else:z=evaluate_one(cell,model,arm,sc.budget,sc.price)
                records.append({**z,"cell_id":cid,"outer_fold":fold,
                    "budget":sc.budget,"price":sc.price,
                    "arm":arm,"is_selected":int(arm==sc.source_chosen),
                    "is_active":int(arm in ACTIVE)})
                if arm in {sc.source_chosen,"periodic150","periodic200"}:
                    times=[int(x) for x in str(z["events"]).split(";") if x]
                    for delay in (0,5,25):
                        loss=latency_replay(cell,times,delay)
                        if delay==0:
                            assert abs(loss.mean()-z["MAE_pp"])<1e-8
                        latency.append({"cell_id":cid,"outer_fold":fold,
                            "budget":sc.budget,"price":sc.price,"arm":arm,
                            "is_selected":int(arm==sc.source_chosen),
                            "delay":delay,"MAE_pp":float(loss.mean()),
                            "J":float(loss.mean()+sc.price*len(times)),
                            "purchases":len(times)})
        print("OUTER EVALUATED",cid,flush=True)
    rows=pd.DataFrame(records);lag=pd.DataFrame(latency)
    expected={(c,b,p) for c in test_ids for b,p in SCENARIOS}
    chosen=rows[rows.is_selected.eq(1)]
    assert len(chosen)==len(expected)
    assert set(zip(chosen.cell_id,chosen.budget,chosen.price))==expected
    assert set(rows.cell_id)==set(test_ids)
    outdir.mkdir(parents=True,exist_ok=True)
    output={"inner_cv":cv,"inner_policy_arms":inner_arms,
            "inner_policy_selected":selected,
            "outer_policy":rows,"outer_latency":lag,
            "manifest":pd.DataFrame(manifest)}
    for name,df in output.items():
        df.to_csv(outdir/(name+".csv"),index=False)
    meta={"fold":fold,"outer_cells":test_ids,"inner_cells":training_ids,
          "winners":winners,"num_source_examples":len(src),
          "all_22_previously_explored":True,
          "reference":"normal-rate original samplewise SOC_proxy",
          "capacity_source":"preceding RPT anchor; not independent SOC",
          "label_cost":"hypothetical MAE-pp per purchase",
          "online_never_reads_unpurchased_SOC_or_future_lifetime":True}
    (outdir/"metadata.json").write_text(
        json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf8")
    (outdir/"report.md").write_text(
        "# Nested internal outer fold "+str(fold)+"\n\n"+
        "Source-only inner regressor selection and OOF policy comparison.\n"+
        "Previously explored cells: not a pristine external test.\n\n"+
        selected.to_string(index=False)+"\n",encoding="utf8")
    print("COMPLETE",fold,"source",len(training),"held",len(held),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--outer-fold",type=int,choices=range(5),required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.outer_fold,a.out)
