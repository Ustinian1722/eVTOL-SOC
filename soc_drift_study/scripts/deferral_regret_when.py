"""DEV-only wait-vs-buy regret, deployed via causal stateful SOC maintenance.

Normal high-power complete flight SOC_proxy is historical Coulomb counting
anchored by a prior RPT capacity, not independent SOC truth. No RPT waveforms
train the high-power head. Test labels never enter decisions unless purchased.
All 22 cells were previously investigated: challenges are retrospective.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from stateful_marginal_roi_maintenance import (
    CELL_IDS,HORIZON,FIELDS,SEED,cell_split,build_cell,meta_examples,
    model_constructors,learned_events,run_cell,PRICE_GRID,BUDGET_GRID)
from dense_highrate_reference_maintenance import replay

HOLD=25
SRC_PRICE=.1
SRC_BUDGET=4
MODES=("delay","reserve","delay_only","old_roi")
GRID=tuple((mode,gap,thr) for mode in MODES for gap in (75,125,175,225)
           for thr in (-.1,0.,.04,.12) if mode!="old_roi")
GRID+=(("old_roi",175,0.),)


def counterfactual_delay(cell,t,last):
    """Source teacher: loss if wait next 25 missions then buy MINUS buy now."""
    n=t+HOLD
    if n not in cell.loss or n+1>=cell.n:return np.nan
    end=min(cell.n,t+1+HORIZON)
    old=cell.frozen_loss if last<0 else cell.loss[last]
    wait=np.r_[old[t+1:n+1],cell.loss[n][n+1:end]]
    now=cell.loss[t][t+1:end]
    assert len(now)==len(wait)
    return float(np.mean(wait-now))


def source_examples(cell):
    x=meta_examples(cell)
    result=[]
    for row in x:
        v=counterfactual_delay(cell,int(row["event_idx"]),
                               int(row["prev_purchased_idx"]))
        if np.isfinite(v):result.append({**row,"delay25_regret_pp":v})
    return result


def source_models(dev,src):
    """Physical-cell heldout models, then final on 14 source cells only."""
    names=list(FIELDS)
    unique=np.array(sorted(dev))
    cv=GroupKFold(n_splits=5)
    result=[]
    choices={}
    for label in ("incremental_future100_ROI_pp","delay25_regret_pp"):
        for name,ctor in model_constructors().items():
            err=[];pred=[];truth=[]
            for tr,vl in cv.split(unique,groups=unique):
                trset=set(unique[tr]);vlset=set(unique[vl])
                a=src[src.cell_id.isin(trset)]
                b=src[src.cell_id.isin(vlset)]
                model=ctor()
                model.fit(a[names],a[label])
                pp=model.predict(b[names])
                bb=b.assign(_pred=pp)
                for _,d in bb.groupby("cell_id"):
                    err.append(float(np.mean((d._pred-d[label])**2)))
                pred.extend(pp);truth.extend(b[label])
            rank=spearmanr(pred,truth).statistic if np.std(pred)>1e-10 else 0.
            result.append({"target":label,"regressor":name,
                           "cell_macro_MSE":float(np.mean(err)),
                           "OOF_spearman":float(rank)})
        d=pd.DataFrame([r for r in result if r["target"]==label])
        choices[label]=str(d.sort_values(["cell_macro_MSE","regressor"]).iloc[0].regressor)
    oof={}
    for tr,vl in cv.split(unique,groups=unique):
        a=src[src.cell_id.isin(set(unique[tr]))]
        models=[]
        for label in ("incremental_future100_ROI_pp","delay25_regret_pp"):
            m=model_constructors()[choices[label]]()
            m.fit(a[names],a[label])
            models.append(m)
        for cid in unique[vl]:oof[cid]=tuple(models)
    final=[]
    for label in ("incremental_future100_ROI_pp","delay25_regret_pp"):
        m=model_constructors()[choices[label]]()
        m.fit(src[names],src[label]);final.append(m)
    return choices,pd.DataFrame(result),oof,tuple(final)


def decide(cell,models,config,budget,price,trace=False):
    roi,delay=models
    mode,gap,threshold=config
    if mode=="old_roi":
        return learned_events(cell,roi,budget,price,gap,.15,trace=trace)
    last=-1;chosen=[];log=[]
    for t in cell.opportunities:
        if len(chosen)>=budget:break
        if last>=0 and t-last<gap:continue
        f=cell.feature(t,last,len(chosen))
        frame=pd.DataFrame([f])[list(FIELDS)]
        rr=float(roi.predict(frame)[0])
        dd=float(delay.predict(frame)[0])
        # Cost 5*lambda assumes a stated 500-task design horizon.
        # It NEVER reads the target's actual future dataset length.
        roi_threshold=.15+5.*price
        reserve=.1*len(chosen)/max(1,budget) if mode=="reserve" else 0.
        buy=(dd>threshold+reserve) and (mode=="delay_only" or rr>roi_threshold)
        if trace:
            log.append({"cell_id":cell.cid,"t":t,"last_purchased":last,
                "price":price,"budget":budget,"mode":mode,"pred_roi":rr,
                "pred_buy_now_benefit_over_wait25":dd,"reserve":reserve,
                "decision_buy":int(buy),**f})
        if buy:chosen.append(t);last=t
    return chosen,log


def replay_policy(cell,models,config,budget,price,trace=False):
    events,log=decide(cell,models,config,budget,price,trace)
    loss=replay(cell.frozen_loss,cell.loss,events)
    return {"cell_id":cell.cid,"mode":config[0],"gap":config[1],
            "threshold":config[2],"budget":budget,"price":price,
            "policy":"learning","MAE_pp":float(loss.mean()),
            "J":float(loss.mean()+price*len(events)),
            "purchases":len(events),
            "label_points":sum(cell.bank[e][1] for e in events),
            "events":";".join(map(str,events))},log


def tune_on_source(dev,oof):
    res=[]
    for cid,cell in sorted(dev.items()):
        for c in GRID:
            x,_=replay_policy(cell,oof[cid],c,SRC_BUDGET,SRC_PRICE)
            res.append(x)
        print("Source OOF policies evaluated",cid,flush=True)
    percell=pd.DataFrame(res)
    summary=percell.groupby(["mode","gap","threshold"],as_index=False).agg(
        mean_J=("J","mean"),mean_MAE=("MAE_pp","mean"),
        mean_purchases=("purchases","mean")).sort_values(
        ["mean_J","mode","gap","threshold"])
    best=summary.iloc[0]
    return (str(best["mode"]),int(best["gap"]),float(best["threshold"])),summary,percell


def challenge_replay(challenge,models,chosen):
    out=[];trace=[]
    for cid,cell in sorted(challenge.items()):
        for budget in BUDGET_GRID:
            for price in PRICE_GRID:
                for arm in ("source_chosen","previous_stateful","periodic150",
                            "periodic200","periodic250","periodic300",
                            "periodic400","periodic500","offline_oracle"):
                    if arm in ("source_chosen","previous_stateful"):
                        conf=chosen if arm=="source_chosen" else ("old_roi",175,0.)
                        r,d=replay_policy(cell,models,conf,budget,price,True)
                        r["policy"]=arm
                        trace.extend(d)
                    else:
                        p,_=run_cell(cell,arm,budget,price)
                        r={"cell_id":cid,"mode":"","gap":-1,
                           "threshold":float("nan"),"budget":budget,
                           "price":price,"policy":arm,
                           "MAE_pp":p["SOCproxy_MAE_pp"],"J":p["J"],
                           "purchases":p["num_reference_purchases"],
                           "label_points":p["num_raw_labels_purchased"],
                           "events":p["reference_idx"]}
                    out.append(r)
        print("Challenge replay",cid,flush=True)
    return pd.DataFrame(out),pd.DataFrame(trace)


def paired_scores(df):
    rows=[]
    for (b,p),g in df.groupby(["budget","price"]):
        base=g[g.policy=="periodic150"][["cell_id","J"]].rename(columns={"J":"fixedJ"})
        for policy,arm in g.groupby("policy"):
            m=arm.merge(base,on="cell_id",validate="one_to_one")
            delta=(m.J-m.fixedJ).to_numpy()
            rng=np.random.default_rng(SEED)
            idx=rng.integers(0,len(delta),(8000,len(delta)))
            lo,hi=np.quantile(delta[idx].mean(axis=1),[.025,.975])
            rows.append({"budget":b,"price":p,"policy":policy,
                "n_cells":len(m),"median_MAE_pp":m.MAE_pp.median(),
                "mean_J":m.J.mean(),"mean_purchases":m.purchases.mean(),
                "deltaJ_fixed150":delta.mean(),"CI_low":lo,"CI_high":hi,
                "n_better_than150":int(np.sum(delta<0))})
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    split=cell_split()
    dev={};challenge={};examples=[];manifest=[]
    for cid in CELL_IDS:
        cell=build_cell(cid)
        if split[cid]=="development":
            dev[cid]=cell
            examples.extend(source_examples(cell))
        else:challenge[cid]=cell
        manifest.append({"cell_id":cid,"partition":split[cid],**cell.raw_stats})
        print("Raw normal flights",cid,cell.n,split[cid],flush=True)
    assert len(dev)==14 and len(challenge)==8 and len(examples)>500
    source=pd.DataFrame(examples)
    winners,cv,oof,final=source_models(dev,source)
    chosen,grid,source_percell=tune_on_source(dev,oof)
    challenge_df,trace=challenge_replay(challenge,final,chosen)
    paired=paired_scores(challenge_df)
    files={"dev_teacher_counterfactual_NOT_ONLINE":source,
           "dev_groupCV":cv,"dev_OOF_policy_grid":grid,
           "dev_OOF_policy_percell":source_percell,
           "challenge_policies":challenge_df,
           "challenge_prepurchase_log":trace,
           "challenge_physicalcell_bootstrap":paired,
           "22cell_raw_manifest":pd.DataFrame(manifest)}
    for name,df in files.items():df.to_csv(args.out/(name+".csv"),index=False)
    meta={"source14":sorted(dev),"challenge8":sorted(challenge),
          "models_selected_on_source_only":winners,
          "policy_selected_on_source_OOF_only":chosen,
          "no_unpaid_SOC_target_or_future_horizon_in_ONLINE":True,
          "reference":"recorded native-sampled normal high-power mission",
          "SOC_proxy":"historical Coulomb count with prior RPT capacity denominator",
          "not_independent_SOC":True,
          "cost":"assumed MAE-pp equivalent per reference"}
    (args.out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    sub=paired[(paired.budget==4)&(np.isclose(paired.price,.1))].sort_values("mean_J")
    lines=["# Buy now versus wait 25 normal flights: source-only OOF policy",
      "",f"Source14 counterfactual rows: {len(source)}; retrospective challenge8.",
      "This study is not independent external battery validation.",
      "SOC proxy comes from Coulomb counting and previously measured RPT",
      "capacity anchor. No RPT low-rate curves enter the supervised Head.",
      f"Source-only regression winners: {winners}.",
      f"Source-only OOF selected decision: {chosen}.",
      "",
      "## Regression CV",cv.to_markdown(index=False),"",
      "## Source OOF top 10 policy configurations",
      grid.head(10).to_markdown(index=False),"",
      "## Challenge B4 cost lambda=.1",
      sub.to_markdown(index=False),"",
      "Do not claim active better unless paired cell bootstrap and both",
      "periodic150 and sparse periodic200 support it; eight challenge",
      "cells were explored in previous studies and cannot be called pristine.",
      "Online buyer may use ONLY past purchased references; the source-only",
      "counterfactual teacher uses future labels purely for offline training.",
      "Dataset end is unknown to online policies (no final-20 censoring)."]
    report="\n".join(lines)+"\n"
    (args.out/"report.md").write_text(report,encoding="utf8")
    print(report,flush=True)

if __name__=="__main__":main()
