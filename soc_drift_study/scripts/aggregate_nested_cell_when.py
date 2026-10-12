"""Aggregate five disjoint physical-cell outer-fold experiment results.

The 22 cells were studied previously. Nested CV is internal retrospective
validation, never pristine external generalization or independently measured SOC.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

def aggregate(folders,out):
    data=[];lat=[];policy=[]
    for path in folders:
        a=pd.read_csv(path/"outer_policy.csv")
        data.append(a)
        lat.append(pd.read_csv(path/"outer_latency.csv"))
        policy.append(pd.read_csv(path/"inner_policy_selected.csv").assign(
            outer_fold=int(a.outer_fold.iloc[0])))
    rows=pd.concat(data,ignore_index=True)
    delays=pd.concat(lat,ignore_index=True)
    choices=pd.concat(policy,ignore_index=True)
    chosen=rows[rows.is_selected.eq(1)]
    assert chosen.cell_id.nunique()==22
    assert len(chosen)==22*8
    assert chosen.groupby(["cell_id","budget","price"]).size().eq(1).all()
    assert rows.groupby("cell_id").outer_fold.nunique().eq(1).all()
    paired=[];summaries=[]
    for (b,p),frame in rows.groupby(["budget","price"]):
        ref=frame[frame.arm.eq("periodic150")][["cell_id","J","MAE_pp","purchases"]]
        ref=ref.rename(columns={"J":"refJ","MAE_pp":"refMAE",
                                "purchases":"refBuys"})
        a=frame[frame.is_selected.eq(1)].merge(ref,on="cell_id",validate="one_to_one")
        a["deltaJ"]=a.J-a.refJ
        a["deltaMAE"]=a.MAE_pp-a.refMAE
        a["deltaBuys"]=a.purchases-a.refBuys
        fixed=frame[frame.arm.isin(["frozen","periodic150","periodic200",
             "periodic250","periodic300","periodic400","periodic500"])]
        hindsight=fixed.groupby("arm").J.mean().min()
        v=a.deltaJ.to_numpy(float)
        rng=np.random.default_rng(20261012+b*100+int(p*100))
        ix=rng.integers(0,len(v),(12000,len(v)))
        lo,hi=np.quantile(v[ix].mean(axis=1),[.025,.975])
        summaries.append({"budget":b,"price":p,"n_cells":len(a),
            "mean_J":a.J.mean(),"mean_MAE_pp":a.MAE_pp.mean(),
            "mean_purchases":a.purchases.mean(),
            "active_cells":int(a.is_active.sum()),
            "fixed_or_frozen_cells":int((a.is_active==0).sum()),
            "mean_deltaJ_vs150":v.mean(),"CI_low":lo,"CI_high":hi,
            "mean_deltaMAE_vs150":a.deltaMAE.mean(),
            "mean_deltaBuys_vs150":a.deltaBuys.mean(),
            "better_than150_cells":int((v<0).sum()),
            "gap_vs_hindsight_best_outer_periodic":a.J.mean()-hindsight})
        paired.append(a)
    agg=pd.DataFrame(summaries).sort_values(["budget","price"])
    selected_lag=delays[delays.is_selected.eq(1)]
    assert selected_lag.groupby(["budget","price","delay"]).cell_id.nunique().eq(22).all()
    lag=selected_lag.groupby(["budget","price","delay"],as_index=False).agg(
        mean_J=("J","mean"),mean_MAE_pp=("MAE_pp","mean"),
        mean_purchases=("purchases","mean"))
    out.mkdir(parents=True,exist_ok=True)
    for name,table in [
        ("outer_all_policies",rows),("outer_selected_paired",pd.concat(paired)),
        ("outer_22_cell_summary",agg),("outer_latency_summary",lag),
        ("inner_selected_by_fold",choices)]:
        table.to_csv(out/(name+".csv"),index=False)
    (out/"metadata.json").write_text(json.dumps({
        "n_outer_cells":22,"outer_folds":5,"each_cell_held_once":True,
        "all_cells_previously_explored":True,
        "SOC_proxy_uses_previous_RPT_capacity":True,
        "real_cost_or_independent_true_SOC":False,
        "bootstrap_not_corrected_for_previous_human_model_selection":True
    },indent=2),encoding="utf8")
    report=("# Nested 22-cell internal SOC WHEN evaluation\n\n"+
        "Previously explored cells, not independent external validation.\n"+
        "Each cell is scored in its outer fold only.\n"+
        "Original SOC_proxy depends on previous RPT capacity and hypothetical label price.\n\n"+
        "## Source-only selected active/fixed policies by price/budget\n\n"+
        agg.to_string(index=False)+"\n\n"+
        "## Delayed calibration activation 0 / 5 / 25 missions\n\n"+
        lag.to_string(index=False)+"\n")
    (out/"report.md").write_text(report,encoding="utf8")
    print(report,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fold-dirs",type=Path,nargs=5,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    aggregate(a.fold_dirs,a.out)
