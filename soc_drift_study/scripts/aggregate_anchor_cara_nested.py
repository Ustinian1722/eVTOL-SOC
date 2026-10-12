"""Aggregate five nested held-cell CARA-like and noisy purchased SOC reference tests."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

def aggregate(folders,out):
    allrows=[];policy=[]
    for path in folders:
        x=pd.read_csv(path/"outer_cara_capacity_latency.csv")
        allrows.append(x)
        policy.append(pd.read_csv(path/"inner_source_cara_choices.csv"))
    data=pd.concat(allrows,ignore_index=True)
    selected=pd.concat(policy,ignore_index=True)
    assert data.cell_id.nunique()==22
    assert data.groupby("cell_id").outer_fold.nunique().eq(1).all()
    assert selected.outer_fold.nunique()==5 if "outer_fold" in selected else selected.fold.nunique()==5
    # All physical cells have one nested evaluation for every scenario:
    assert data.groupby(["budget","price","reference_noise","arm","lag"]).cell_id.nunique().eq(22).all()
    out.mkdir(parents=True,exist_ok=True)
    data.to_csv(out/"all22_reference_noise_cara_latency.csv",index=False)
    selected.to_csv(out/"source_selected_cara_thresholds.csv",index=False)
    summaries=[]
    for (b,p,noise,lag),block in data.groupby(["budget","price","reference_noise","lag"]):
        fixed=block[block.arm=="periodic150"][["cell_id","J","MAE_pp","purchases"]].rename(
            columns={"J":"J150","MAE_pp":"MAE150","purchases":"buys150"})
        simple=block[block.arm=="periodic200"][["cell_id","J"]].rename(columns={"J":"J200"})
        for arm,g in block.groupby("arm"):
            g=g.merge(fixed,on="cell_id",validate="one_to_one").merge(simple,on="cell_id",validate="one_to_one")
            d=(g.J-g.J150).to_numpy(float)
            d200=(g.J-g.J200).to_numpy(float)
            rnd=np.random.default_rng(20261012+int(b*100)+int(p*1000)+lag)
            ix=rnd.integers(0,len(d),(8000,len(d)))
            lo,hi=np.quantile(d[ix].mean(axis=1),[.025,.975])
            summaries.append({
                "budget":b,"price":p,"reference_noise":noise,"lag":lag,
                "arm":arm,"cells":len(g),
                "mean_J":float(g.J.mean()),
                "mean_MAE_pp":float(g.MAE_pp.mean()),
                "mean_purchases":float(g.purchases.mean()),
                "delta_J_vs_fixed150":float(d.mean()),
                "CI_low":float(lo),"CI_high":float(hi),
                "delta_J_vs_fixed200":float(d200.mean()),
                "cells_better_than_fixed150":int((d<0).sum()),
            })
    summary=pd.DataFrame(summaries).sort_values(
        ["budget","price","reference_noise","lag","arm"])
    summary.to_csv(out/"nested_cell_paired_reference_quality_cost.csv",index=False)
    md=summary[(summary.budget==4)&(np.isclose(summary.price,.1))]
    focus=md[md.lag.eq(0)][[
        "reference_noise","arm","mean_J","mean_MAE_pp","mean_purchases",
        "delta_J_vs_fixed150","CI_low","CI_high","cells_better_than_fixed150"]]
    lag=md[md.reference_noise.eq("nominal")][[
        "lag","arm","mean_J","delta_J_vs_fixed150","CI_low","CI_high"]]
    # Regression against report26 EXACT main scenario, old nested case:
    check=summary[(summary.budget==4)&(np.isclose(summary.price,.1))&
                  summary.reference_noise.eq("nominal")&summary.lag.eq(0)&
                  summary.arm.eq("nested_selected")].iloc[0]
    assert abs(check.mean_J-1.954432)<.00015,check.mean_J
    assert abs(check.mean_purchases-2.772727)<.00015
    meta={
        "all22_outer_unique":True,"five_source_only_folds":True,
        "same_delay_applied_to_active_AND_fixed_baselines":True,
        "source_CARA_proxy_thresholds_not_target_tuned":True,
        "capacity_noise_at_purchased_reference_only":True,
        "benchmark_not_faithful_CARA_reimplementation":True,
        "not_independently_measured_SOC":True,
        "prior_repeated_researcher_exposure_to_same22":True,
    }
    (out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    report=("# Nested 22-cell capacity reference perturbation and CARA-inspired benchmark\n\n"+
       "22 physical cells unique outer fold each, source-only cost-staleness thresholds.\n"+
       "Previously explored cells: INTERNAL retrospective, NOT external test.\n"+
       "RPT capacity in historical SOC_proxy denominator, NOT measured SOC truth.\n"+
       "Capacity perturbation only on simulated purchased reference, benchmark scoring holds historical proxy fixed.\n"+
       "CARA-style accumulated nominal discrepancy is a heuristic, NOT exact CARA method reproduction.\n\n"+
       "## B4 lambda0.1 reference capacity perturbation (zero deployment latency)\n\n"+
       focus.to_string(index=False)+"\n\n"+
       "## Fair comparison with BOTH policies delayed under B4 lambda0.1\n\n"+
       lag.to_string(index=False)+"\n")
    (out/"report.md").write_text(report,encoding="utf8")
    print(report,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold-dirs",type=Path,nargs=5,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    aggregate(args.fold_dirs,args.out)
