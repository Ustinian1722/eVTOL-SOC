"""Do unlabeled *normal flight missions* improve an already strong source GRU?

The precomputed baseline is a 5-fold leave-cell-out, 3-seed SOURCE GRU.
GRU was PRETRAINED on other cells' historical SOC_proxy labels (including
offline prior-RPT-derived capacity). This experiment NEVER retrains that GRU
and NEVER reads RPT curves/capacities/SOC labels for online updates. A bounded
1-parameter SOC slope adapter is updated using within-mission delta Q only.

Important: baseline is not the RPT-free from-scratch nominal N20 Ridge.
The offline score is SOC_proxy, which is Coulomb-counted with historical RPT
capacity: it is NOT an external independent SOC measurement.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/"results"
OUTPUT_METHODS = [
    "frozen",
    "periodic_C3",
    "periodic_C27",
    "periodic_C33",
    "value_gate_C3",
    "stream_C3",
    "periodic_shuffled_q_C3",
]
N20=20
C_REF=3.0
SEED=20261012
BUDGET=4
BUFFER=35
HELD=8
COOLDOWN=35
PRIOR=.08
MIN_RELATIVE_GAIN=.05


def assemble() -> pd.DataFrame:
    pred=pd.read_csv(RESULTS/"gru_proxy_crosscell_predictions.csv")
    labels=pd.read_csv(RESULTS/"soc_proxy_phase_labels.csv",
                       usecols=["cell_id","charge_cycle_index","phase_Ns","QDischarge_end_Ah"])
    keys=["cell_id","charge_cycle_index","phase_Ns"]
    assert not labels.duplicated(keys).any(), "ambiguous mission phase labels"
    assert not pred.duplicated(keys+["seed"]).any(), "ambiguous seed prediction"
    df=pred.merge(labels,on=keys,how="left",validate="many_to_one",indicator=True)
    assert df["_merge"].eq("both").all(), "missing mission Q values"
    df=df.drop(columns=["_merge"])
    assert df["source_GRU_prediction"].between(0,1).all()
    assert df["QDischarge_end_Ah"].notna().all()
    assert df["cell_id"].nunique()==22 and df["seed"].nunique()==3
    return df


def missions_from_predictions(grp: pd.DataFrame):
    grp=grp.sort_values(["charge_cycle_index","phase_Ns"])
    records=[]
    for cycle,mission in grp.groupby("charge_cycle_index",sort=True):
        p=mission["source_GRU_prediction"].to_numpy(dtype=float)
        q=mission["QDischarge_end_Ah"].to_numpy(dtype=float)
        if len(p)<2 or np.any(np.diff(q)<-0.003):
            dP=np.array([]);dQ=np.array([])
        else:
            rawdQ=q[1:]-q[0]
            ok=np.isfinite(rawdQ)&(rawdQ>.005)&(rawdQ<2.8)
            dP=(p[1:]-p[0])[ok]
            dQ=rawdQ[ok]
        records.append({
            "cycle":int(cycle),
            "p":p,
            "dP":dP,
            "dQ":dQ,
            "score_only_soc_proxy":mission["SOC_proxy"].to_numpy(dtype=float),
        })
    return records


def candidate_slope(records:list[dict],C:float,seed:int,shuffle_q:bool=False)->float|None:
    dp=np.concatenate([x["dP"] for x in records if len(x["dP"])])
    dq=np.concatenate([x["dQ"] for x in records if len(x["dP"])])
    if len(dp)<12:
        return None
    if shuffle_q:
        rng=np.random.default_rng(seed)
        dq=rng.permutation(dq)
    y=-dq/C
    # Strong source-to-target trust regularization; only adjust relative slope.
    ratio=(PRIOR+float(dp@y))/(PRIOR+float(dp@dp))
    return float(np.clip(ratio,0.65,1.35))


def physics_loss(a:float,records:list[dict],C:float)->float:
    res=[]
    for x in records:
        if len(x["dP"]):
            res.extend(np.abs(a*x["dP"] + x["dQ"]/C).tolist())
    return float(np.mean(res)) if len(res) else float("nan")


def try_update(a:float,history:list[dict],C:float,policy:str,seed:int):
    if len(history)<20:
        return a,False,{"reason":"too_few_missions"}
    train=history[:-HELD]; hold=history[-HELD:]
    fitted=candidate_slope(train,C,seed,policy=="periodic_shuffled_q_C3")
    if fitted is None:
        return a,False,{"reason":"too_few_phase_pairs"}
    proposal=float(np.clip(0.5*a+0.5*fitted,0.65,1.35))
    old=physics_loss(a,hold,C)
    new=physics_loss(proposal,hold,C)
    utility=float((old-new)/old) if old>.004 and np.isfinite(new) else float("-inf")
    # Gate evaluated on completed missions only, BEFORE next mission output.
    approved=policy!="value_gate_C3" or utility>=MIN_RELATIVE_GAIN
    return (proposal if approved else a), bool(approved), {
        "fitted_slope":fitted,
        "candidate_slope":proposal,
        "physical_loss_before":old,
        "physical_loss_after":new,
        "relative_gain_on_completed_holdout":utility,
        "reason":"accepted" if approved else "utility_rejected",
    }


def replay(missions:list[dict],arm:str,cell:str,seed:int):
    c=2.7 if arm=="periodic_C27" else 3.3 if arm=="periodic_C33" else C_REF
    rng=np.random.default_rng(SEED+int(hashlib.sha256(f"{cell}/{seed}".encode()).hexdigest()[:8],16))
    a=1.
    history=deque(maxlen=BUFFER)
    last=0; attempts=accepted=0
    estimates=[];eventlog=[]
    for idx,m in enumerate(missions,start=1):
        # Score is deliberately NOT read until replay has completed.
        estimates.append(np.clip(1.0+a*(m["p"]-1.0),0,1))
        history.append({k:m[k] for k in ["cycle","dP","dQ"]})
        if arm=="frozen" or (arm!="stream_C3" and attempts>=BUDGET):
            continue
        if idx-last<COOLDOWN or len(history)<BUFFER:
            continue
        if arm=="stream_C3":
            fire=(idx%40==0)
        elif arm=="value_gate_C3":
            fire=(idx%20==0)
        else:
            fire=(idx%150==0)
        if not fire:
            continue
        attempts+=1
        new,ok,diag=try_update(a,list(history),c,arm,
                               int(rng.integers(1,2**31-1)))
        if ok:
            a=new;accepted+=1;last=idx
        eventlog.append({"cell_id":cell,"seed":seed,"arm":arm,
                         "event_index":idx,"event_cycle":m["cycle"],
                         "attempted":attempts,"accepted":int(ok),"updated_scale":a,
                         **diag})

    truth=np.concatenate([m["score_only_soc_proxy"] for m in missions])
    pred=np.concatenate(estimates)
    percycle=np.asarray([np.mean(np.abs(p-m["score_only_soc_proxy"]))*100
                         for p,m in zip(estimates,missions)])
    err=(pred-truth)*100
    return {
        "cell_id":cell,"seed":seed,"arm":arm,
        "mission_cycles":len(missions),"phase_rows":len(pred),
        "attempts":attempts,"accepted":accepted,
        "SOC_proxy_MAE_pp":float(np.mean(np.abs(err))),
        "SOC_proxy_RMSE_pp":float(np.sqrt(np.mean(err**2))),
        "late_cycle_MAE_pp":float(np.mean(percycle[int(.75*len(percycle)):])),
        "P95_cycle_MAE_pp":float(np.quantile(percycle,.95)),
        "worst_cycle_MAE_pp":float(np.max(percycle)),
    },eventlog,np.concatenate(estimates)


def report(cell:pd.DataFrame,by_seed:pd.DataFrame,macro:pd.DataFrame):
    out=[
        "# Normal-mission physics adapter on pre-trained cell-held-out GRU",
        "",
        "- Frozen backbone: existing leave-cell-out GRU, PRETRAINED on other-cell SOC_proxy history (including offline old RPT references).",
        "- Online: past mission ΔQ and previously committed GRU prediction only; ZERO RPT curve, RPT capacity, or current SOC label in update logic.",
        "- Online adapter is bounded single scale about SOC=1; no direct GRU weight updates, and no independent absolute SOC identification.",
        "- Offline evaluation is historical prior-RPT-anchored SOC_proxy, not independent BMS truth.",
        "- All scores are phase-end, 3 pre-existing seeds per physical cell, aggregation at physical-cell level.",
        "- Four attempted updates max (stream is intentionally a much higher-cost non-matched control).",
        "",
        "## Per-cell aggregated (3 seeds averaged before statistics)",
        "",
        "| Arm | Cells | Median MAE (pp) | Mean Δ vs frozen (pp) | 95% cell-bootstrap CI | Improved cells | Median attempts/accepts per seed |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for v in macro.sort_values("median_MAE_pp").itertuples():
        out.append(f"| {v.arm} | {v.ncells} | {v.median_MAE_pp:.3f} | "
                   f"{v.mean_delta_pp:+.3f} | [{v.ci_lo:+.3f},{v.ci_hi:+.3f}] | "
                   f"{v.improved_cells}/{v.ncells} | {v.median_attempts:.0f}/{v.median_accepted:.0f} |")
    out +=[
        "",
        "## Interpretation",
        "",
        "- Explicitly compare the strong GRU frozen baseline; improvements relative to a weak N20 Ridge are insufficient justification for deployment.",
        "- The SOC-proxy reference is coulomb-counted; updating a slope around nominal 3 Ah may be misspecified for aged cells.",
        "- C-prior sensitivity and shuffled ΔQ placebo test are critical for causal attribution.",
        "- This is a fixed source GRU + online scalar adapter, not neural backbone test-time training or SOC ground truth.",
        "- Selecting the best policy here is exploratory because this 22-cell cohort was previously examined in method development.",
        "- Active timing must beat budget-matched periodic updating; no assumption of active superiority.",
    ]
    return "\n".join(out)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    full=assemble()
    rows=[];eventrows=[]
    for (cell,seed),grp in full.groupby(["cell_id","seed"],sort=True):
        m=missions_from_predictions(grp)
        for arm in OUTPUT_METHODS:
            result,events,_=replay(m,arm,cell,int(seed))
            rows.append(result);eventrows.extend(events)
    by_seed=pd.DataFrame(rows)
    assert len(by_seed)==22*3*len(OUTPUT_METHODS)
    assert by_seed.query("arm!='stream_C3'").attempts.le(BUDGET).all()
    cell=by_seed.groupby(["cell_id","arm"],as_index=False).agg(
        MAE_pp=("SOC_proxy_MAE_pp","mean"),
        RMSE_pp=("SOC_proxy_RMSE_pp","mean"),
        late_MAE_pp=("late_cycle_MAE_pp","mean"),
        median_attempts=("attempts","median"),
        median_accepted=("accepted","median"),
    )
    frozen=cell[cell.arm=="frozen"].set_index("cell_id")
    cell=cell.join(frozen[["MAE_pp"]],on="cell_id",rsuffix="_frozen")
    cell["paired_delta_pp"]=cell["MAE_pp"]-cell["MAE_pp_frozen"]
    stats=[]
    for arm,g in cell.groupby("arm"):
        d=g.paired_delta_pp.to_numpy()
        rng=np.random.default_rng(SEED)
        boot=rng.choice(d,size=(8000,len(d)),replace=True).mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        stats.append({
            "arm":arm,"ncells":len(g),
            "median_MAE_pp":g.MAE_pp.median(),
            "mean_MAE_pp":g.MAE_pp.mean(),
            "median_late_MAE_pp":g.late_MAE_pp.median(),
            "mean_delta_pp":d.mean(),
            "ci_lo":lo,"ci_hi":hi,
            "improved_cells":int((d<0).sum()),
            "median_attempts":g.median_attempts.median(),
            "median_accepted":g.median_accepted.median(),
        })
    macro=pd.DataFrame(stats)
    by_seed.to_csv(out/"strong_gru_by_cell_seed.csv",index=False)
    cell.to_csv(out/"strong_gru_cell_macro.csv",index=False)
    macro.to_csv(out/"strong_gru_comparison.csv",index=False)
    pd.DataFrame(eventrows).to_csv(out/"strong_gru_events.csv",index=False)
    (out/"report.md").write_text(report(cell,by_seed,macro),encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "model":"fixed 5-fold three-seed leave-cell-out pretrained source GRU",
        "backbone_source_training_uses_historical_RPT_SOC_proxy":True,
        "online_update_never_uses_RPT":True,
        "online_update_never_uses_test_SOC_proxy":True,
        "source_predictions_available_after_target_N20":True,
        "adapter":"bounded scale about 1.0, fitted from completed mission relative dQ",
        "conditional_physical_capacity_assumptions":[2.7,3.0,3.3],
        "score":"SOC_proxy only",
        "source_model_semantics":"Other-cell full-life supervised offline, as in original report 02",
        "arm_names":OUTPUT_METHODS,
        "max_attempts_except_stream":BUDGET,
        "aggregation":"within-cell three seeds before 22-cell-bootstrap",
    },indent=2,ensure_ascii=False),encoding="utf-8")
    print((out/"report.md").read_text(encoding="utf-8"))


if __name__=="__main__":
    main()
