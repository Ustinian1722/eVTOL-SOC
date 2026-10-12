"""Reference-capacity perturbation and CARA-inspired cumulative-staleness baseline.

Same pre-existing 22 physical cells, nested physical-cell evaluation,
NO pristine external cells. This is NOT faithful CARA paper reproduction:
the accessible cumulative prediction discrepancy is a *proxy* for loss.

Reference perturbation changes ONLY *purchased* normal high-power SOC
reference labels, not the historical evaluation SOC_proxy. Online strategy
is REPLAYED after perturbation with only previous PURCHASED noisy labels.
No prior RPT waveform trains the calibration head, but historical
SOC_proxy Q/C denominator still uses previous RPT capacity.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from deferral_regret_when import build_cell,source_examples,source_models
from nested_cell_when_5fold import outer_splits,latency_replay
from scenario_safe_when_selector import choose_from_source,evaluate_one
from dense_highrate_reference_maintenance import fit_calibration,replay
from stateful_marginal_roi_maintenance import Cell

SCENARIOS=((4,.1),(2,.3))
GATES=(25.,50.,100.,200.,400.)
ALLOWANCES=(.1,.3,.6)
ARM_OPTIONS=tuple((g,a) for g in GATES for a in ALLOWANCES)
NOISE_CASES=("nominal","capacity_plus2pct","capacity_minus2pct",
             "capacity_plus5pct","capacity_iid2pct")

def cara_events(cell,budget,threshold,allowance):
    """A causal accumulated proxy staleness signal using ordinary mission
    voltage/current/temp SOC estimates vs nominal3Ah Coulomb integral.

    The baseline nominal3Ah is a PROXY, not independent SOC accuracy.
    No target SOC_proxy, future labels, or observed test horizon accessed.
    """
    base=np.mean([100.*np.mean(np.abs(m.predicted-m.qnom))
                  for m in cell.train])
    gain=0.
    events=[];last=-1
    opportunity=set(cell.opportunities)
    for t,m in enumerate(cell.test):
        a,b=cell.deployed_params(last)
        signal=100.*np.mean(np.abs(np.clip(a*m.predicted+b,0,1)-m.qnom))
        gain=max(0.,gain+max(0.,signal-base)-allowance)
        if t in opportunity and gain>threshold and len(events)<budget:
            events.append(t);last=t;gain=0.
    return events

def cara_score(cell,budget,price,threshold,allowance):
    times=cara_events(cell,budget,threshold,allowance)
    loss=replay(cell.frozen_loss,cell.loss,times)
    return {
        "J":float(loss.mean()+price*len(times)),
        "MAE_pp":float(loss.mean()),
        "purchases":len(times),
        "events":";".join(map(str,times)),
    }

def source_cara_choice(cells,budget,price):
    table=[]
    for threshold,allowance in ARM_OPTIONS:
        for cid,cell in sorted(cells.items()):
            v=cara_score(cell,budget,price,threshold,allowance)
            table.append({"cell_id":cid,"threshold":threshold,
                          "allowance":allowance,**v})
    df=pd.DataFrame(table)
    macro=df.groupby(["threshold","allowance"],as_index=False).agg(
        mean_J=("J","mean"),worst_cell_J=("J","max"),
        mean_buy=("purchases","mean"))
    # Policy chosen ONLY from available source/inner cells; no held cells.
    macro["source_robust"]=.7*macro.mean_J+.3*macro.worst_cell_J
    top=macro.sort_values(["source_robust","mean_J","threshold","allowance"]).iloc[0]
    return (float(top.threshold),float(top.allowance)),df,macro

def error_for_calibrated_head(cell,params):
    a,b=params
    return np.array([
        100.*np.mean(np.abs(np.clip(a*m.predicted+b,0,1)-m.y))
        for m in cell.test
    ])

def noisy_curve(m,case,cid,t):
    if case=="nominal":return m.y
    if case=="capacity_plus2pct":delta=.02
    elif case=="capacity_minus2pct":delta=-.02
    elif case=="capacity_plus5pct":delta=.05
    elif case=="capacity_iid2pct":
        # Does not depend on target real SOC or future task count.
        key=hashlib.sha256((cid+":"+str(t)+":noise-seed").encode()).hexdigest()
        delta=.02 if int(key[:8],16)%2 else -.02
    else:raise ValueError("Unsupported perturbation case: "+str(case))
    # Reconstructed discharge Ah from the previously existing proxy curve;
    # perturb ONLY reference capacity denominator, not the evaluation truth.
    ah=(1.-m.y)*m.source_capacity
    return np.clip(1.-ah/(m.source_capacity*(1.+delta)),0,1)

class NoisyCell:
    """Minimum wrapper to keep the *paid feedback feature* label consistent
    with perturbed label; an online policy must not silently see true bought y.
    """
    def __init__(self,original,case):
        self.orig=original
        self.case=case
        self.bank={}
        self.loss={}
        self._paid_labels={}
        for t in original.opportunities:
            m=original.test[t]
            y=noisy_curve(m,case,original.cid,t)
            params=fit_calibration(m.predicted,y)
            self.bank[t]=(params,len(y))
            self.loss[t]=error_for_calibrated_head(original,params)
            self._paid_labels[t]=y

    def __getattr__(self,name):
        return getattr(self.orig,name)

    def deployed_params(self,last):
        return (1.,0.) if last<0 else self.bank[last][0]

    def current_reference_fit(self,last):
        if last<0:return (0.,0.,0.)
        a,b=self.bank[last][0]
        m=self.test[last]
        err=100.*(np.clip(a*m.predicted+b,0,1)-self._paid_labels[last])
        return float(np.mean(np.abs(err))),float(np.mean(err)),float(len(err))

    def feature(self,t,last,num_paid):
        # Original implementation calls self.deployed_params and
        # self.current_reference_fit, so both respect released noisy labels.
        return Cell.feature(self,t,last,num_paid)

def outer_run(fold,outdir):
    train_ids,test_ids=outer_splits()[fold]
    dev={};target={};sources=[]
    for cid in sorted(set(train_ids)|set(test_ids)):
        cell=build_cell(cid)
        if cid in train_ids:
            dev[cid]=cell;sources.extend(source_examples(cell))
        else:target[cid]=cell
        print("PARSED",cid,"train" if cid in dev else "outer",flush=True)
    source=pd.DataFrame(sources)
    assert set(source.cell_id)==set(train_ids)
    models,regcv,oof,final=source_models(dev,source)
    _,_,choices=choose_from_source(dev,oof)
    rows=[];source_cara=[]
    for budget,price in SCENARIOS:
        active_arm=choices[
            (choices.budget==budget)&(np.isclose(choices.price,price))
        ].iloc[0].source_chosen
        cara,details,mean=source_cara_choice(dev,budget,price)
        source_cara.append({
            "fold":fold,"budget":budget,"price":price,
            "source_active_arm":active_arm,
            "CARA_proxy_threshold":cara[0],"CARA_proxy_allowance":cara[1],
            "source_CARA_proxy_J":float(mean[(mean.threshold==cara[0])&
                                            (mean.allowance==cara[1])].mean_J.iloc[0]),
        })
        for cid,cell in sorted(target.items()):
            for case in NOISE_CASES:
                # Strictly reruns ACTIVE WHEN decisions with noisy feedback
                # already paid, preserving source-trained frozen predictor.
                effective=cell if case=="nominal" else NoisyCell(cell,case)
                for arm in ("nested_selected","CARA_proxy","periodic150",
                            "periodic200"):
                    if arm=="nested_selected":
                        value=evaluate_one(effective,final,active_arm,budget,price)
                        kind=("active" if active_arm.startswith(("delay","reserve","prior"))
                              else "periodic_or_none")
                    elif arm=="CARA_proxy":
                        value=cara_score(effective,budget,price,*cara)
                        kind="sensor_cumulative"
                    else:
                        value=evaluate_one(effective,final,arm,budget,price)
                        kind="fixed"
                    events=[int(x) for x in str(value["events"]).split(";") if x]
                    for lag in (0,5,25):
                        losses=latency_replay(effective,events,lag)
                        if lag==0:
                            assert abs(losses.mean()-value["MAE_pp"])<1e-8
                        rows.append({
                            "cell_id":cid,"outer_fold":fold,
                            "budget":budget,"price":price,
                            "reference_noise":case,
                            "arm":arm,"policy_type":kind,
                            "chosen_under_source":active_arm,
                            "lag":lag,
                            "MAE_pp":float(losses.mean()),
                            "J":float(losses.mean()+price*len(events)),
                            "purchases":len(events),
                            "event_times":";".join(map(str,events)),
                        })
            print("OUTER_COMPLETE",fold,cid,budget,price,flush=True)
    outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir/"outer_cara_capacity_latency.csv",index=False)
    pd.DataFrame(source_cara).to_csv(outdir/"inner_source_cara_choices.csv",index=False)
    regcv.to_csv(outdir/"inner_regressorCV.csv",index=False)
    (outdir/"metadata.json").write_text(json.dumps({
        "fold":fold,"outer_cells":test_ids,"source_cells":train_ids,
        "no_target_outcome_for_selection":True,
        "capacity_noise_applied_only_to_bought_highrate_SOC_proxy":True,
        "evaluation_reference_unperturbed_historical_SOC_proxy":True,
        "CARA_proxy_not_exact_CARA_reproduction":True,
        "labels_not_true_SOC":True,
        "data_repeatedly_explored_not_pristine":True,
    },indent=2),encoding="utf8")
    print("FOLD_DONE",fold,len(target),len(rows),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--outer-fold",type=int,choices=range(5),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    outer_run(args.outer_fold,args.out)
