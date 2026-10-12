"""Cross-cell learned update-value trigger for normal eVTOL mission SOC adaptation.

WHY: voltage drift / physics mismatch is not equivalent to improvement from a
particular model update. Source-development cells with historical *offline*
SOC_proxy labels let us learn a mapping from observable pre-update signals
to the NEXT 50 mission counterfactual improvement of a physics adapter.
At deployment to 8 different cells the trigger and adapter observe only
normal V/I/T and within-mission delta-Q, no RPT, SOH or true SOC.

Scientific caution: source meta-supervision DOES rely on historical RPT-
capacity-normalized SOC_proxy for defining training ROI. This is allowed only
as OFFLINE source-cell training, NOT live RPT-based updating. All 22 cells
have been used in previous exploratory studies: the 8-cell cell-block test
is a RETROSPECTIVE challenge, not a never-inspected lockbox.

Fixed meta model and thresholds; this is a research pilot, not proven active
trigger superiority or full neural backprop test-time training.
"""
from __future__ import annotations
import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mission_update_utility import (
    DATA, SEED, N20, BUDGET, WINDOW, HOLDOUT as HELD, COOLDOWN,
    prepare, selected_cell_protocol, fit_and_gate,
)

ROOT=Path(__file__).resolve().parents[1]
FUTURE=50
TOLERANCE_PP=.05
C_AH=3.0
SOURCE_DEGRADATION_GUARD=.012
ARMS=[
    ("frozen",None),
    ("periodic_150",None),
    ("physics_gain_guard",.10),
    ("learned_roi_050",.50),
    ("learned_roi_065",.65),
    ("learned_roi_080",.80),
]
SIGNALS=[
    "holdout_rel_gain", "physical_old", "physical_new",
    "source_degradation", "theta_displacement",
    "candidate_effect_recent_pp",
    "voltage_recent_z", "voltage_recent_trend_z",
    "phase_Vmin_z", "phase_current_z", "phase_temperature_z",
    "deltaQ_mean_Ah", "deltaQ_std_Ah",
]


def proposal_features(inp,history:list[dict],theta:np.ndarray):
    """Only completed normal-mission signals; does not touch scoring target."""
    if len(history)<WINDOW:
        return None,None,None
    candidate,diag,status=fit_and_gate(theta,inp,history,C_AH,min_gain=-float("inf"))
    if candidate is None or diag is None:
        return None,None,None
    recent=history[-HELD:]
    past=history[:-(HELD)]
    def mean_safe(x): return float(np.mean(x)) if len(x) else 0.0
    vmean=mean_safe([r["voltage_abs_resid_mV"] for r in recent])
    old_v=mean_safe([r["voltage_abs_resid_mV"] for r in past[-HELD:]])
    recent_Z=np.concatenate([r["Z"] for r in recent],axis=0)
    dq=np.concatenate([r["dq"] for r in recent if len(r["dq"])]) if any(len(r["dq"]) for r in recent) else np.array([])
    magnitude=mean_safe([float(np.mean(np.abs(r["Z"]@(candidate-theta)))) for r in recent])*100
    def zmean(k):return float(np.mean(recent_Z[:,k]))
    features={
        "holdout_rel_gain":float(np.clip(diag["physics_rel_gain"],-5,5)),
        "physical_old":float(diag["physics_loss_before"]),
        "physical_new":float(diag["physics_loss_after"]),
        "source_degradation":float(diag["source_degradation"]),
        "theta_displacement":float(np.linalg.norm(candidate-inp.frozen)),
        "candidate_effect_recent_pp":magnitude,
        "voltage_recent_z":float(np.clip((vmean-inp.v_center)/inp.v_scale,-30,30)),
        "voltage_recent_trend_z":float(np.clip((vmean-old_v)/inp.v_scale,-30,30)),
        "phase_Vmin_z":zmean(5),
        "phase_current_z":zmean(3),
        "phase_temperature_z":zmean(7),
        "deltaQ_mean_Ah":mean_safe(dq),
        "deltaQ_std_Ah":float(np.std(dq)) if len(dq) else 0.,
    }
    assert all(np.isfinite(val) for val in features.values())
    return candidate,diag,features


def future_roi(inp,k:int,theta:np.ndarray,candidate:np.ndarray)->float:
    """OFFLINE ONLY: never called from deployment rollout. Positive means
    the candidate gives *lower* SOC_proxy MAE than frozen for next 50 tasks."""
    future=inp.missions[k:min(len(inp.missions),k+FUTURE)]
    if not future: return float("nan")
    old=np.concatenate([np.clip(r["Z"]@theta,0,1) for r in future])
    new=np.concatenate([np.clip(r["Z"]@candidate,0,1) for r in future])
    gold=np.concatenate([r["score_after_replay"] for r in future])
    return float(100*(np.mean(np.abs(old-gold))-np.mean(np.abs(new-gold))))


def build_meta_training(prepared:dict,dev:set[str]):
    examples=[]
    for cid in sorted(dev):
        inp=prepared[cid]
        for k in range(WINDOW,min(len(inp.missions)-FUTURE,2000),25):
            hist=[{key:m[key] for key in ["cycle","Z","dZ","dq","voltage_abs_resid_mV"]}
                  for m in inp.missions[k-WINDOW:k]]
            candidate,diag,features=proposal_features(inp,hist,inp.frozen)
            if candidate is None: continue
            improvement=future_roi(inp,k,inp.frozen,candidate)
            if not np.isfinite(improvement):continue
            examples.append({
                "cell_id":cid,"decision_after_mission":k,
                "future_50_SOC_proxy_gain_pp":improvement,
                "beneficial":int(improvement>TOLERANCE_PP),
                **features,
            })
    return pd.DataFrame(examples)


def make_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.4,max_iter=2000,class_weight="balanced",random_state=SEED)
    )


def honest_group_cv(frame:pd.DataFrame):
    x=frame[SIGNALS].to_numpy(dtype=float)
    y=frame["beneficial"].to_numpy(dtype=int)
    groups=frame["cell_id"].to_numpy()
    if len(np.unique(y))<2: return float("nan"),float("nan"),"single-class"
    cv=GroupKFold(n_splits=5)
    pred=np.zeros(len(frame))
    supported=np.zeros(len(frame),dtype=bool)
    for train,val in cv.split(x,y,groups):
        if len(np.unique(y[train]))<2:
            continue
        model=make_classifier()
        model.fit(x[train],y[train])
        pred[val]=model.predict_proba(x[val])[:,1]
        supported[val]=True
    if not supported.any() or len(np.unique(y[supported]))<2:
        return float("nan"),float("nan"),"insufficient-fold-classes"
    return (float(roc_auc_score(y[supported],pred[supported])),
            float(average_precision_score(y[supported],pred[supported])),
            f"valid_rows={supported.sum()}/{len(frame)}")


def rollout(inp,cell:str,arm:tuple,model):
    name,threshold=arm
    theta=inp.frozen.copy()
    history=deque(maxlen=WINDOW)
    attempted=accepted=0
    since_accept=0
    outputs=[];logs=[]
    for k,m in enumerate(inp.missions,start=1):
        # Evaluate current SOC before the mission can influence adaptation.
        outputs.append(np.clip(m["Z"]@theta,0,1))
        history.append({key:m[key] for key in ["cycle","Z","dZ","dq","voltage_abs_resid_mV"]})
        since_accept+=1
        if name=="frozen" or attempted>=BUDGET or len(history)<WINDOW or since_accept<COOLDOWN:
            continue
        if name=="periodic_150":fire=k%150==0
        else:fire=k%20==0
        if not fire:continue
        proposed,diag,features=proposal_features(inp,list(history),theta)
        if proposed is None:continue
        p=float(model.predict_proba(pd.DataFrame([features])[SIGNALS])[:,1][0])
        positive_physics=bool(features["holdout_rel_gain"]>=.10)
        eligible=features["source_degradation"]<=SOURCE_DEGRADATION_GUARD
        if name=="periodic_150":
            approved=True
        elif name=="physics_gain_guard":
            approved=positive_physics and eligible
        else:
            approved=p>=float(threshold) and eligible
        attempted+=1
        if approved:
            theta=proposed
            since_accept=0
            accepted+=1
        logs.append({
            "cell_id":cell,"arm":name,"decision_mission":k,
            "prob_roi":p,"approved":int(approved),
            "attempted_no":attempted,"accepted_total":accepted,
            **features,
        })
    scored=np.concatenate([m["score_after_replay"] for m in inp.missions])
    ypred=np.concatenate(outputs)
    percycle=np.array([100*np.mean(np.abs(p-m["score_after_replay"]))
                       for p,m in zip(outputs,inp.missions)])
    errors=(ypred-scored)*100
    return {
        "cell_id":cell,"arm":name,
        "MAE_pp":float(np.mean(np.abs(errors))),
        "RMSE_pp":float(np.sqrt(np.mean(errors**2))),
        "late_MAE_pp":float(percycle[int(.75*len(percycle)):].mean()),
        "P95_cycle_MAE_pp":float(np.quantile(percycle,.95)),
        "attempts":attempted,"accepted":accepted,
        "missions":len(inp.missions),
    },logs


def challenge_candidate_diagnostics(prepared:dict,challenge:set[str],model):
    # ROI future target is used ONLY for posthoc diagnostic after classifier
    # has been completely fitted on DEV; this is NOT an online update.
    frame=build_meta_training(prepared,challenge)
    if not len(frame):return frame,{"roc_auc":None,"n":0}
    frame["predicted_benefit_prob"]=model.predict_proba(frame[SIGNALS])[:,1]
    y=frame.beneficial.to_numpy()
    auc=float(roc_auc_score(y,frame.predicted_benefit_prob)) if len(np.unique(y))>1 else None
    return frame,{"roc_auc":auc,"candidate_events":len(frame),
                  "beneficial_fraction":float(np.mean(y))}


def summarize(metrics:pd.DataFrame):
    frozen=metrics[metrics.arm=="frozen"].set_index("cell_id")
    paired=metrics.join(frozen[["MAE_pp"]],on="cell_id",rsuffix="_frozen")
    paired["delta_pp"]=paired["MAE_pp"]-paired["MAE_pp_frozen"]
    grouped=[]
    for arm,g in paired.groupby("arm"):
        d=g.delta_pp.to_numpy(dtype=float)
        rng=np.random.default_rng(SEED)
        boot=rng.choice(d,size=(10000,len(d)),replace=True).mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        grouped.append({
            "arm":arm,"cells":len(g),
            "median_MAE_pp":float(g.MAE_pp.median()),
            "mean_delta_pp":float(d.mean()),
            "CI_low":float(lo),"CI_high":float(hi),
            "improved_cells":int((d<0).sum()),
            "median_attempts":float(g.attempts.median()),
            "median_accepted":float(g.accepted.median()),
        })
    return paired,pd.DataFrame(grouped)


def report(train:pd.DataFrame,cv:tuple,challenge_stats:dict,agg:pd.DataFrame,challenge_events:pd.DataFrame):
    npos=int(train.beneficial.sum())
    lines=[
        "# Cross-cell learned update-value pilot (NO target RPT online)",
        "",
        "- Target online update sees only mission V/I/T and within-mission ΔQ. All target SOC_proxy labels are scoring-only.",
        "- DEV offline meta-supervision uses next 50 mission SOC_proxy score to define which historically observed candidate would help. These dev labels rely on prior RPT capacity by their construction.",
        "- Fixed SHA256 ID 14 development / 8 RETROSPECTIVE challenge cells. NOT pristine, since the full cohort was already explored in previous work.",
        "- Per-target baseline is N20 nominal-3Ah Ridge, no RPT in SOC backbone initial training.",
        "- Utility classifier: fixed standardized logistic regression with 13 observable pre-update features, class balancing, group-cell CV.",
        "- Meta-ROI estimates are based on counterfactual ONE-UPDATE proposals from frozen dev models; rollout may differ after cumulative updates.",
        "- Budget B=4 ATTEMPTS, not guaranteed four accepted updates.",
        "",
        "## Development utility labels and group-cell CV",
        f"- Development candidate events: {len(train)}; positive (>0.05pp future gain): {npos}/{len(train)}.",
        f"- Group-cell CV ROC-AUC: {cv[0]:.3f}; Average Precision: {cv[1]:.3f} ({cv[2]}).",
        f"- Retrospective 8-cell challenge candidate event ROC-AUC (diagnostic only): {challenge_stats['roc_auc']}.",
        "",
        "## Deployment-style chronological replay on eight challenge cells",
        "| Trigger | Cell-median SOC-proxy MAE (pp) | Paired mean Δ to frozen (pp) [95% cell CI] | Improved | Median attempts/accepted |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in agg.sort_values("median_MAE_pp").itertuples():
        lines.append(f"| {x.arm} | {x.median_MAE_pp:.3f} | {x.mean_delta_pp:+.3f} [{x.CI_low:+.3f},{x.CI_high:+.3f}] | "
                     f"{x.improved_cells}/{x.cells} | {x.median_attempts:.0f}/{x.median_accepted:.0f} |")
    lines += [
        "",
        "## Observed transfer failure diagnostics (offline-only)",
        "",
        f"- DEV candidate positive fraction: {float(train.beneficial.mean()):.3f}; challenge candidate fraction: {challenge_stats.get('beneficial_fraction')}.",
        "- Challenge per-cell ROC-AUC requires BOTH positive and negative candidate events;",
        "  cells with one class only are marked NA and are not pooled as independently valid AUCs.",
        "",
        "| Challenge cell | Candidate windows | Positive fraction | ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for c,g in challenge_events.groupby("cell_id"):
        y=g.beneficial.to_numpy()
        auc=float(roc_auc_score(y,g.predicted_benefit_prob)) if len(np.unique(y))>1 else None
        lines.append(
            f"| {c} | {len(g)} | {np.mean(y):.3f} | "
            f"{f'{auc:.3f}' if auc is not None else 'NA'} |"
        )
    scales=train[SIGNALS].std().clip(lower=1e-6)
    shift=((challenge_events[SIGNALS].mean()-train[SIGNALS].mean())/scales).abs().sort_values(ascending=False)
    lines.append("")
    lines.append("- Largest DEV-to-challenge standardized feature mean shifts:")
    for fname,amount in shift.head(5).items():
        lines.append(f"  - {fname}: {amount:.2f} training SD")
    lines += [
        "",
        "## Interpretation",
        "- Explicitly compare against the 4-attempt periodic policy; if learned WHEN cannot outperform it, retain active-timing NO-GO.",
        "- AUC measures ranking of one-off historical counterfactual *proxy* gains, not the improvement of a full deployed active schedule.",
        "- Source labels for meta-model are historical SOC_proxy and therefore not a zero-RPT-anywhere system. Target live updates remain RPT-free.",
        "- Never treat this post-hoc eight-cell challenge as external independent proof; a truly new dataset/battery collection is needed for confirmation.",
        "- No neural backprop or real-time per-sample SOC estimation is claimed.",
    ]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(DATA)
    cohorts=selected_cell_protocol(data.cell_id.unique().tolist())
    assert len(cohorts)==22
    dev={c for c,t in cohorts.items() if t=="development"}
    challenge=set(cohorts)-dev
    prepared={c:prepare(g) for c,g in data.groupby("cell_id",sort=True)}
    source=build_meta_training(prepared,dev)
    if len(source)<50 or source.beneficial.nunique()<2:
        raise RuntimeError("Development has insufficient utility positives / negatives")
    cv=honest_group_cv(source)
    model=make_classifier()
    model.fit(source[SIGNALS],source.beneficial)
    challenge_events,challenge_auc=challenge_candidate_diagnostics(prepared,challenge,model)
    rows=[];events=[]
    for cid in sorted(challenge):
        for arm in ARMS:
            metrics,timeline=rollout(prepared[cid],cid,arm,model)
            rows.append(metrics);events+=timeline
    frame=pd.DataFrame(rows)
    paired,agg=summarize(frame)
    assert frame.cell_id.nunique()==8
    assert frame.groupby("cell_id").size().eq(len(ARMS)).all()
    assert frame.attempts.le(BUDGET).all()
    assert (frame.accepted<=frame.attempts).all()
    source.to_csv(out/"offline_source_utility_training.csv",index=False)
    challenge_events.to_csv(out/"retrospective_challenge_candidate_diagnostic.csv",index=False)
    paired.to_csv(out/"challenge_policy_by_cell.csv",index=False)
    agg.to_csv(out/"challenge_policy_summary.csv",index=False)
    pd.DataFrame(events).to_csv(out/"challenge_online_events.csv",index=False)
    (out/"report.md").write_text(report(source,cv,challenge_auc,agg,challenge_events),encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "split":cohorts,
        "only_source_dev_SOC_proxy_for_meta_supervised_training":True,
        "target_online_RPT":False,
        "target_online_SOC_proxy":False,
        "target_initial_soc_nominal3Ah_no_RPT":True,
        "retrospective_challenge_not_pristine":True,
        "source_training_target_label":"future_50_SOC_proxy_gain_pp",
        "observable_signal_columns":SIGNALS,
        "fixed_classifier":"standardized logistic class_weight balanced C=0.4",
        "fixed_deployment_methods":ARMS,
        "max_attempts":BUDGET,
        "test_cells":sorted(challenge),
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print((out/"report.md").read_text(encoding="utf-8"))


if __name__=="__main__":
    main()
