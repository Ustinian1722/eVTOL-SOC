"""Active *supervised* SOC-maintenance timing with retrospective label release.

This benchmark intentionally changes the central question from unsupervised
HOW to label-purchase WHEN. At a selected normal HIGH-RATE mission, an offline
label broker releases exactly ONE completed mission's SOC_proxy reference
(typically 3 phase-end points), enabling a FIXED supervised affine calibrator.
No low-rate RPT curve is used to calibrate the live model.

Facts/limits:
- This derived dataset has 3 PHASE-END reference values per mission, not a
  densely sampled, independently measured 'true SOC curve'.
- SOC_proxy labels are historical previous-RPT-capacity-normalized Coulomb
  integration; this fact applies to initial N20 training AND offline labels.
- This is a HYPOTHETICAL paid-label-acquisition experiment: an operational
  mechanism providing real high-rate labels at maintenance is NOT in data.
- Online detectors only get current completed mission V/I/T and normal
  discharge counters; SOC_proxy is kept by a separate broker and scorer.
- Offline ORACLE has deliberately clairvoyant access to future scoring labels
  and is an upper bound on timing choices, never an online algorithm.
- Horizon-even periodic policy knows full retrospective horizon, modeled as an
  operationally PLANNED maintenance schedule; fixed-150 periodic does not.
- The calibration form, source N20, budgets, candidate grid, thresholds and
  costs are PREDECLARED. This sweep is not held-cell external model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT/"results"/"soc_proxy_phase_labels.csv"
SOURCE_MISSIONS = 20
CANDIDATE_FIRST = 25
CANDIDATE_STRIDE = 25
MIN_AFTER_EVENT = 20
BUDGETS=(0,2,4)
SEEDS=(11,23,37)
COST_PER_EVENT_PP=(0.0,0.1,0.3,0.6,1.0,2.0)
FEATURES=[
    "phase_Ns","duration_s","I_mean_A","I_peak_A",
    "Vmin_V","Vmax_V","Tmean_C","Tmax_C",
]
VOLTAGE_FEATURES=[
    "phase_Ns","duration_s","I_mean_A","I_peak_A",
    "Vmax_V","Tmean_C","Tmax_C",
]
# Fixed WHAT: one high-rate 3-phase mission as bought supervision.
# One update ignores older bought labels, making all WHEN policies use the
# exact same update operator; this also permits exact dynamic programming.
SLOPE_PRIOR=3.0
BIAS_PRIOR=0.5
MAX_SLOPE_SHIFT=0.25
MAX_BIAS=0.12
SEED=20261012
POLICIES=("frozen","periodic_horizon","periodic_150","random",
          "model_disagreement","voltage_residual","cusum_voltage",
          "risk_combo","offline_oracle")


def robust(x):
    z=np.asarray(x,dtype=float)
    center=float(np.median(z))
    mad=float(np.median(np.abs(z-center)))
    return center,max(1.4826*mad,0.20*float(np.std(z)),1e-3)


def supervised_one_mission(base_predictions, newly_unlocked_soc):
    """Regularized 2-parameter head: a*p+b, with frozen identity prior.

    Called ONLY after a policy purchases that mission's 3 phase-end labels.
    This is a single fixed supervised CALIBRATION mechanism, not fine-tuning
    a GRU or dynamically changing per-policy training conditions.
    """
    p=np.asarray(base_predictions,dtype=float)
    y=np.asarray(newly_unlocked_soc,dtype=float)
    if len(p)!=len(y) or len(y)<2 or not np.isfinite(y).all():
        raise ValueError("At least two matched calibrated phase SOC labels required")
    X=np.column_stack((p,np.ones_like(p)))
    prior=np.diag([SLOPE_PRIOR,BIAS_PRIOR])
    rhs=X.T@y+prior@np.array([1.0,0.0])
    a,b=np.linalg.solve(X.T@X+prior,rhs)
    a=float(np.clip(a,1-MAX_SLOPE_SHIFT,1+MAX_SLOPE_SHIFT))
    b=float(np.clip(b,-MAX_BIAS,MAX_BIAS))
    return a,b


@dataclass
class CellData:
    cell_id:str
    frozen:np.ndarray
    label_broker:np.ndarray
    nominal:np.ndarray
    voltage_residual:np.ndarray
    current:np.ndarray
    temp:np.ndarray
    training_disagreement:np.ndarray
    training_vresidual:np.ndarray
    dates:np.ndarray
    phase_rows:int
    incomplete_heldout_missions:int=0

    @property
    def n(self):return len(self.frozen)


def make_cell(frame):
    frame=frame.sort_values(["charge_cycle_index","phase_Ns"])
    cid=str(frame.cell_id.iloc[0])
    ids=np.sort(frame.charge_cycle_index.unique())
    if len(ids)<=SOURCE_MISSIONS+MIN_AFTER_EVENT+1:
        raise ValueError(f"Not enough missions {cid}")
    first=frame[frame.charge_cycle_index.isin(ids[:SOURCE_MISSIONS])]
    held=frame[~frame.charge_cycle_index.isin(ids[:SOURCE_MISSIONS])]
    model=make_pipeline(StandardScaler(),Ridge(alpha=5.0))
    model.fit(first[FEATURES],first["SOC_proxy"])
    # Frozen at the target cell's N20. No future target SOC used to fit.
    volt=make_pipeline(StandardScaler(),Ridge(alpha=5.0))
    volt.fit(first[VOLTAGE_FEATURES],first["Vmin_V"])
    dtrain=first.copy()
    dtrain["frozen"]=np.clip(model.predict(first[FEATURES]),0,1)
    dtrain["disag"]=np.abs(dtrain["frozen"]-dtrain["SOC_nominal_3Ah"])
    dtrain["v_err"]=np.abs(volt.predict(first[VOLTAGE_FEATURES])-first["Vmin_V"])
    train_disag=dtrain.groupby("charge_cycle_index")["disag"].mean().to_numpy()
    train_ve=dtrain.groupby("charge_cycle_index")["v_err"].mean().to_numpy()

    test=held.copy()
    test["frozen"]=np.clip(model.predict(test[FEATURES]),0,1)
    test["v_err"]=np.abs(volt.predict(test[VOLTAGE_FEATURES])-test["Vmin_V"])
    test["nominal"]=test["SOC_nominal_3Ah"]
    # ONLY the label broker/scoring constructor obtains this target column;
    # detectors never receive CellData.label_broker.
    raw_groups=[g for _,g in test.groupby("charge_cycle_index",sort=True)]
    # Exactly three normal task phases give every hypothetical purchase the
    # SAME number of high-rate SOC labels. Source dataset has a handful of
    # incomplete missions; make their exclusion explicit in metadata/report.
    groups=[g for g in raw_groups if len(g)==3]
    removed=len(raw_groups)-len(groups)
    if removed:
        print(f"DATA_QUALITY: {cid} excludes {removed} incomplete heldout mission(s); "
              f"3 phases per calibration opportunity are mandatory",flush=True)
    if len(groups)<CANDIDATE_FIRST+MIN_AFTER_EVENT:
        raise AssertionError(f"Too few complete normal missions for {cid}")
    take=lambda key:np.vstack([g[key].to_numpy(dtype=float) for g in groups])
    return CellData(
        cell_id=cid, frozen=take("frozen"),
        label_broker=take("SOC_proxy"),
        nominal=take("nominal"),
        voltage_residual=take("v_err"),
        current=take("I_mean_A"),temp=take("Tmean_C"),
        training_disagreement=train_disag,training_vresidual=train_ve,
        dates=np.array([int(g.charge_cycle_index.iloc[0]) for g in groups]),
        phase_rows=3,incomplete_heldout_missions=removed
    )


def scheduled_opportunities(cell):
    n=cell.n
    # Mission t is observed/predicted BEFORE labels become available and the
    # new head affects only t+1 onward. Leave at least 20 future missions.
    return [t for t in range(CANDIDATE_FIRST-1,n-MIN_AFTER_EVENT,CANDIDATE_STRIDE)]


def event_models(cell,indices):
    """Hypothetical label escrow: precompute options offline only.

    Online policies must obtain a calibration from the escrow after selecting
    an event. Precomputing does not grant them any access to these labels.
    """
    bank={}
    for i in indices:
        a,b=supervised_one_mission(cell.frozen[i],cell.label_broker[i])
        bank[i]=(a,b)
    return bank


def window_trigger_stats(cell):
    # All thresholds from source N20; no test truth or RPT is inspected.
    d0,ds=robust(cell.training_disagreement)
    v0,vs=robust(cell.training_vresidual)
    return d0,ds,v0,vs


def public_risks(cell):
    d0,ds,v0,vs=window_trigger_stats(cell)
    disagreement=np.mean(np.abs(cell.frozen-cell.nominal),axis=1)
    vres=np.mean(cell.voltage_residual,axis=1)
    zd=np.clip((disagreement-d0)/ds,-10,30)
    zv=np.clip((vres-v0)/vs,-10,30)
    return zd,zv


def online_schedule(cell,opportunities,policy,budget,seed=11):
    """Policy observes completed sensors/predictions, NEVER SOC_proxy.

    At each available candidate mission t it decides to pay for t's labels.
    New calibrated head becomes active strictly on the following mission.
    """
    if policy=="frozen" or budget==0:return []
    cand=list(opportunities)
    if not cand:return []
    if policy=="periodic_horizon":
        # Uses a planned *full life horizon*, allowed as a predeclared
        # retrospective planned-maintenance baseline, not online detection.
        positions=np.rint(np.linspace(0,len(cand)-1,min(budget,len(cand))+2)[1:-1]).astype(int)
        return sorted(set(cand[int(p)] for p in positions))
    if policy=="random":
        rng=np.random.default_rng(seed+int(hashlib.sha256(cell.cell_id.encode()).hexdigest()[:8],16))
        selected=rng.choice(cand,size=min(budget,len(cand)),replace=False)
        return sorted(int(x) for x in selected)
    zd,zv=public_risks(cell)
    chosen=[]
    h=0.
    for t in range(cell.n):
        # Detector state can update on all completed normal missions; it has
        # no access to that mission's withheld SOC labels.
        h=max(0.,h+max(0.,zv[t])-1.0)
        if t not in cand or len(chosen)>=budget:continue
        if policy=="periodic_150":fire=((t+1)%150==0)
        elif policy=="model_disagreement":fire=zd[t]>3.0
        elif policy=="voltage_residual":fire=zv[t]>3.0
        elif policy=="cusum_voltage":fire=h>18.
        elif policy=="risk_combo":fire=(zd[t]>2.0 and zv[t]>2.0)
        else:raise ValueError(policy)
        if fire:
            chosen.append(t)
            h=0.
    return chosen


def predictor_for_events(cell,bank,event_indices):
    """Use event's supervised labels ONLY after that event's online purchase."""
    output=cell.frozen.copy()
    for j,t in enumerate(event_indices):
        next_t=event_indices[j+1] if j+1<len(event_indices) else cell.n
        a,b=bank[t]
        output[t+1:next_t+1]=np.clip(a*cell.frozen[t+1:next_t+1]+b,0,1)
    return output


def oracle_DP(cell,bank,opportunities,budget):
    """Exact hindsight MIN cumulative absolute error given a one-curve HOW.

    The oracle reads future held-out label scores and chooses at most budget
    events; no deployable policy or online detector has such information.
    """
    n=cell.n
    nodes=[None]+list(opportunities)
    starts=[0]+[t+1 for t in opportunities]
    err={}
    c=cell.label_broker
    for j,ev in enumerate(nodes):
        p=cell.frozen if ev is None else np.clip(bank[ev][0]*cell.frozen+bank[ev][1],0,1)
        err[j]=np.r_[0.,np.cumsum(np.mean(np.abs(p-c),axis=1)*100.)]
    def interval(j,end):
        return float(err[j][end]-err[j][starts[j]])
    # F[j][k]: minimum past cost with exactly k purchased calibrations,
    # the last anchor event at node j. Cost from this anchor to horizon
    # is not yet included.
    m=len(nodes)
    F=np.full((m,budget+1),np.inf)
    paths={}
    F[0,0]=0.; paths[(0,0)]=()
    for k in range(1,budget+1):
        for j in range(1,m):
            best=np.inf; winner=None
            for i in range(j):
                prev=F[i,k-1]
                if not np.isfinite(prev):continue
                candidate=prev+interval(i,starts[j])
                if candidate<best:
                    best=candidate;winner=paths[(i,k-1)]+(nodes[j],)
            if winner is not None:
                F[j,k]=best;paths[(j,k)]=winner
    scores=[(interval(0,n),())]
    for k in range(1,budget+1):
        for j in range(1,m):
            if np.isfinite(F[j,k]):
                scores.append((F[j,k]+interval(j,n),paths[(j,k)]))
    objective,chosen=min(scores,key=lambda x:x[0])
    check=np.mean(np.abs(predictor_for_events(cell,bank,list(chosen))-cell.label_broker))*100*cell.n
    assert abs(check-objective)<1e-5,(check,objective)
    return list(chosen)


def cycle_scores(cell,pred):
    e=100*np.abs(pred-cell.label_broker)
    return e.mean(axis=1),float(np.sqrt(np.mean((100*(pred-cell.label_broker))**2)))


def run_one(cell,bank,opportunities,policy,budget,seed):
    if policy=="offline_oracle":
        events=oracle_DP(cell,bank,opportunities,budget)
    else:
        events=online_schedule(cell,opportunities,policy,budget,seed)
    predictions=predictor_for_events(cell,bank,events)
    cyclemae,rmse=cycle_scores(cell,predictions)
    return {
        "cell_id":cell.cell_id,"policy":policy,"budget":budget,
        "seed":seed,"missions":cell.n,
        "MAE_pp":float(cyclemae.mean()),
        "RMSE_pp":rmse,
        "late_MAE_pp":float(cyclemae[int(.75*cell.n):].mean()),
        "P95_mission_MAE_pp":float(np.quantile(cyclemae,.95)),
        "high_error_3pp_rate":float(np.mean(cyclemae>3.)),
        "high_error_5pp_rate":float(np.mean(cyclemae>5.)),
        "maintenance_events":len(events),
        "labeled_phase_end_points":3*len(events),
        "event_positions":";".join(map(str,events)),
        "avg_labeled_missions_per_event":1,
    },events


def aggregate(df):
    # Three random seeds are paired within the PHYSICAL CELL before CI.
    cell=df.groupby(["cell_id","policy","budget"],as_index=False).agg(
        MAE_pp=("MAE_pp","mean"),
        RMSE_pp=("RMSE_pp","mean"),
        late_MAE_pp=("late_MAE_pp","mean"),
        high_error_3pp_rate=("high_error_3pp_rate","mean"),
        high_error_5pp_rate=("high_error_5pp_rate","mean"),
        events=("maintenance_events","mean"),
        qpoints=("labeled_phase_end_points","mean"),
    )
    base=cell[cell.policy.eq("frozen")][["cell_id","MAE_pp"]].rename(columns={"MAE_pp":"frozen_mae"})
    paired=cell.merge(base,on="cell_id",how="left",validate="many_to_one")
    paired["delta_frozen_pp"]=paired.MAE_pp-paired.frozen_mae
    periodic=cell[cell.policy.eq("periodic_horizon")][["cell_id","budget","MAE_pp"]].rename(columns={"MAE_pp":"periodic_mae"})
    paired=paired.merge(periodic,on=["cell_id","budget"],how="left",validate="many_to_one")
    paired["delta_periodic_pp"]=paired.MAE_pp-paired.periodic_mae
    summ=[]
    for (budget,policy),g in paired.groupby(["budget","policy"]):
        d=g.delta_frozen_pp.to_numpy()
        dper=g.delta_periodic_pp.to_numpy()
        rng=np.random.default_rng(SEED)
        rows=rng.integers(0,len(d),size=(6000,len(d)))
        lo,hi=np.quantile(d[rows].mean(axis=1),[.025,.975])
        plo,phi=np.quantile(dper[rows].mean(axis=1),[.025,.975])
        summ.append({
            "budget":int(budget),"policy":policy,"physical_cells":len(g),
            "median_MAE_pp":float(g.MAE_pp.median()),
            "mean_MAE_pp":float(g.MAE_pp.mean()),
            "mean_paired_delta_frozen_pp":float(d.mean()),
            "CI_frozen_low":float(lo),"CI_frozen_high":float(hi),
            "mean_paired_delta_periodic_pp":float(dper.mean()),
            "CI_periodic_low":float(plo),"CI_periodic_high":float(phi),
            "cells_better_frozen":int((d<-1e-9).sum()),
            "cells_better_periodic":int((dper<-1e-9).sum()),
            "median_maintenance_events":float(g.events.median()),
            "median_late_MAE_pp":float(g.late_MAE_pp.median()),
            "mean_error_over3_rate":float(g.high_error_3pp_rate.mean()),
        })
    return paired,pd.DataFrame(summ)


def cost_grid(cell_metrics):
    rows=[]
    for cost in COST_PER_EVENT_PP:
        df=cell_metrics.copy()
        df["J_pp"]=df.MAE_pp+cost*df.events
        g=df.groupby(["budget","policy"],as_index=False).agg(
            physical_cells=("cell_id","nunique"),
            mean_loss_pp=("MAE_pp","mean"),
            mean_events=("events","mean"),
            mean_total_cost_pp=("J_pp","mean"),
        )
        g["cost_per_calibration_pp_equivalent"]=cost
        rows.append(g)
    return pd.concat(rows,ignore_index=True)


def report(summary,grid,meta):
    lines=[
        "# Active SUPERVISED normal-mission SOC label maintenance pilot",
        "",
        "## Critical interpretation",
        "",
        "- This is a VTC6 laboratory eVTOL SINGLE-CELL and three-phase-END pilot, not agricultural drone field deployment.",
        "- Non-three-phase incomplete missions are excluded rather than silently padded; exact count appears in metadata.",
        "- The 'bought SOC curve' presently means one normal high-rate mission with THREE phase-end SOC_proxy points. Full dense SOC curves were NOT bought or replayed.",
        "- SOC_proxy is Coulomb-counted with the previous RPT candidate capacity as denominator, not independent true SOC; prior RPT curves are NEVER used for the high-rate maintenance calibration.",
        "- N20 source training uses historical SOC_proxy labels; all ONLINE triggers see only V/I/T and the normal QDischarge counter, no SOC_proxy until paying for a reference event.",
        "- After an event, only its 3 SOC_proxy labels are unlocked to the FIXED supervised affine head. One purchased mission = 3 newly exposed target labels.",
        "- The detector thresholds use early N20 training signal statistics, never future error.",
        "- The periodic_horizon baseline is a retrospective PLANNED schedule using known total test horizon, not a streaming drift detector.",
        "- The offline_oracle does *know all future heldout proxy labels* and supplies an UNDEPLOYABLE upper bound on timing with the same calibration action.",
        "- The cost parameter is a hypothetical MAE-pp EQUIVALENT price of one 3-point calibration; no empirical dollars, technician time, or real field labels were measured.",
        "",
        "## Matched budget = 4: policies (22 physical cells)",
        "",
        "| Policy | Median SOC_proxy MAE pp | Mean delta vs frozen (95% cell CI) | Mean delta vs planned periodic (95% cell CI) | Median used labels | Cells better periodic |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in summary[summary.budget.eq(4)].sort_values("median_MAE_pp").itertuples():
        lines.append(
            f"| {row.policy} | {row.median_MAE_pp:.3f} | "
            f"{row.mean_paired_delta_frozen_pp:+.3f} [{row.CI_frozen_low:+.3f},{row.CI_frozen_high:+.3f}] | "
            f"{row.mean_paired_delta_periodic_pp:+.3f} [{row.CI_periodic_low:+.3f},{row.CI_periodic_high:+.3f}] | "
            f"{row.median_maintenance_events*3:.0f} | {row.cells_better_periodic}/22 |"
        )
    lines+=["","## Cost vs error (hypothetical calibration price)","",
            "| Cost per calibration (MAE-pp-equivalent) | Budget ceiling | Lowest deployable-policy mean total J | Deployable policy | Offline oracle total J |",
            "|---:|---:|---:|---|---:|"]
    for (cost,budget),g in grid.groupby(["cost_per_calibration_pp_equivalent","budget"]):
        if budget==0:continue
        eligible=g[~g.policy.eq("offline_oracle")]
        best=eligible.sort_values("mean_total_cost_pp").iloc[0]
        oracle=g[g.policy.eq("offline_oracle")].iloc[0]
        lines.append(f"| {cost:.1f} | {budget} | {best.mean_total_cost_pp:.3f} | {best.policy} | {oracle.mean_total_cost_pp:.3f} |")
    lines +=[
        "",
        "## What this pilot can and cannot establish",
        "- It tests WHEN under a SINGLE predeclared supervised three-point update operation, not a full neural-network retraining or actual dense SOC reference acquisition.",
        "- We need to verify that this update operator improves Frozen on most cells before trying sophisticated timing policies.",
        "- Offline oracle can optionally refuse all maintenance. Online policies can use less than full budget; read actual counts before attributing efficiency.",
        "- The planned periodic's perfect knowledge of full-horizon length must not be mistaken for a causal online predictor; use periodic_150 as the strict no-future-horizon alternative.",
        "- This is an exploratory all-cell cohort previously used in other studies, not unseen held-cell final testing.",
        "- Future extensions: matching full dense normal high-rate SOC paths, varying labels per calibration (3/30/300), true field cost, maintenance latency, and corrected strong source GRU.",
    ]
    return "\n".join(lines)+"\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",required=True,type=Path)
    args=p.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(LABELS)
    needed=set(FEATURES+["cell_id","charge_cycle_index","SOC_proxy","SOC_nominal_3Ah"])
    if not needed.issubset(raw.columns):raise AssertionError(f"Missing columns {needed-set(raw.columns)}")
    all_results=[];timeline=[];per_calib=[]
    started=time.perf_counter()
    incomplete_missions=0
    for cid,data in raw.groupby("cell_id",sort=True):
        cell=make_cell(data)
        incomplete_missions+=cell.incomplete_heldout_missions
        events=scheduled_opportunities(cell)
        bank=event_models(cell,events)
        assert all(0<=k<cell.n-1 for k in events)
        for budget in BUDGETS:
            for policy in POLICIES:
                seeds=SEEDS if policy=="random" else (SEEDS[0],)
                for seed in seeds:
                    result,selected=run_one(cell,bank,events,policy,budget,seed)
                    all_results.append(result)
                    for t in selected:
                        timeline.append({
                            "cell_id":cid,"budget":budget,"policy":policy,"seed":seed,
                            "event_test_mission_index":t,"event_cycle_index":int(cell.dates[t]),
                            "visible_label_count":3,"first_affected_test_mission_index":t+1,
                            "calibration_a":bank[t][0],"calibration_b":bank[t][1],
                        })
        base=np.mean(np.abs(cell.frozen-cell.label_broker),axis=1)*100
        for e in events:
            p=np.clip(bank[e][0]*cell.frozen+bank[e][1],0,1)
            corrected=np.mean(np.abs(p-cell.label_broker),axis=1)*100
            hi=min(cell.n,e+1+50)
            per_calib.append({
                "cell_id":cid,"candidate_test_mission":e,
                "future_50_missions":hi-(e+1),
                "frozen_future_MAE_pp":float(base[e+1:hi].mean()),
                "calibrated_future_MAE_pp":float(corrected[e+1:hi].mean()),
                "counterfactual_future_gain_pp":float((base[e+1:hi]-corrected[e+1:hi]).mean()),
            })
        print(f"completed {cid}, {cell.n} test missions, {len(events)} normal-task reference opportunities",flush=True)
    results=pd.DataFrame(all_results)
    assert results.cell_id.nunique()==22
    assert results.maintenance_events.le(results.budget).all()
    paired,summary=aggregate(results)
    grid=cost_grid(paired)
    timeline=pd.DataFrame(timeline)
    candidate_table=pd.DataFrame(per_calib)
    results.to_csv(out/"all_cell_policy_seed.csv",index=False)
    paired.to_csv(out/"paired_cell_metrics.csv",index=False)
    summary.to_csv(out/"budget_policy_summary.csv",index=False)
    grid.to_csv(out/"hypothetical_cost_frontier.csv",index=False)
    timeline.to_csv(out/"calibration_release_timeline.csv",index=False)
    candidate_table.to_csv(out/"counterfactual_candidate_diagnostics_OFFLINE_ONLY.csv",index=False)
    meta={
        "goal":"timing and cost of supervised SOC calibration using historical high-rate mission labels",
        "dataset":"CMU eVTOL VTC6; three phase-end points per mission, not dense real curve",
        "reference_label":"SOC_proxy historical prior RPT capacity CC; no independent electrochemical ground truth",
        "online_update_uses_RPT_phases":False,
        "online_update_receives_labels_ONLY_at_event":True,
        "normal_mission_label_release_size":3,
        "fixed_maintenance_operator":"bounded regularized affine a*p+b fitted on one current high-rate mission's 3 points",
        "base_model":"same-cell first20 normal mission Ridge, historical prior-RPT anchored offline SOC proxy",
        "first_online_reference_eligible_after_test_missions":CANDIDATE_FIRST,
        "maintenance_grid_stride_missions":CANDIDATE_STRIDE,
        "first_affected_mission":"strictly next normal mission",
        "plannable_periodic_horizon_access":True,
        "offline_oracle_cheats_future_scores":True,
        "budgets":BUDGETS,"hypothetical_calibration_prices":COST_PER_EVENT_PP,
        "selection_does_not_use_test_SOC":True,
        "no_online_reference_SOC_without_purchase":True,
        "physical_cells":22,
        "incomplete_heldout_missions_excluded":incomplete_missions,
        "elapsed_s":time.perf_counter()-started,
        "research_scope":"pilot only; no actual calibrated full high-rate SOC curve or measured maintenance price",
    }
    (out/"metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    (out/"report.md").write_text(report(summary,grid,meta),encoding="utf-8")
    print((out/"report.md").read_text(encoding="utf-8"),flush=True)


if __name__=="__main__":
    main()
