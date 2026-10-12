"""Cell-block cross-validated ROI predictor for PURCHASED normal-flight SOC references.

Learning phase: on 14 DEV cells, hindsight SOC_proxy labels can quantify a
one-off future benefit of acquiring a high-rate reference mission. These
labels (which are originally Coulomb-integrated with prior RPT capacity) are
SOURCE-ONLY offline meta-supervision. On 8 retrospective challenge cells,
online policy sees V/I/T and frozen model signals before purchase, but no
target SOC_proxy or future outcomes. The broker releases a three-phase-end
SOC_proxy label after selecting a maintenance opportunity, and a FIXED
supervised affine calibration action is used on subsequent missions.

This is exploratory, 22 cells previously studied. A cell-hash split is
not a pristine never-seen test. No field SOC truth or monetary maintenance
cost claims. The 100-mission ROI is counterfactual against the *FROZEN*
baseline; after multiple calibrations actual policy benefit differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from supervised_maintenance_when import (
    LABELS, BUDGETS, SEED, FEATURES, COST_PER_EVENT_PP,
    make_cell, scheduled_opportunities, event_models, public_risks,
    predictor_for_events, oracle_DP, online_schedule, cycle_scores,
)
from supervised_maintenance_when import aggregate as previous_aggregate

FUTURE=100
LOOKBACK=30
SOURCE_COUNT=14
MIN_PAUSE=75
COSTS=(0.0,.10,.30)
DECISIONS=("frozen","periodic_horizon","periodic_150",
           "learned_positive","learned_conservative",
           "learned_75pct_OOF","learned_90pct_OOF",
           "learned_cost_value","offline_oracle")
PLANNING_HORIZON=500  # declared horizon ASSUMPTION, not target future knowledge
FEATURE_NAMES=[
    "mission_seen",
    "last_disagreement_z","disagreement_30_mean","disagreement_30_slope",
    "last_voltage_z","voltage_30_mean","voltage_30_slope",
    "voltage_30_std","current_30_mean","current_30_std",
    "temp_30_mean","temp_30_trend",
    "novelty_current","novelty_30_mean",
    "frozen_soc_30_mean","frozen_soc_30_range",
]


def frozen_partition(ids):
    # Preserve the same ID-only 14/8 partition as previous experiments,
    # never inspect challenge SOC_proxy to select the meta-model.
    order=sorted(ids,key=lambda c:hashlib.sha256(("split_"+c).encode()).hexdigest())
    return {c:("development" if i<SOURCE_COUNT else "retrospective_challenge")
            for i,c in enumerate(order)}


def visible_features(cell,t):
    """All features measurable at end of mission t before any paid label.
    Nothing can inspect cell.label_broker or a future time step.
    """
    lo=max(0,t+1-LOOKBACK)
    zd,zv=public_risks(cell)
    zd=zd[lo:t+1];zv=zv[lo:t+1]
    cur=cell.current[lo:t+1].mean(axis=1)
    temp=cell.temp[lo:t+1].mean(axis=1)
    fp=cell.frozen[lo:t+1]
    x=cell.encoded_test_features
    if x is not None and cell.labeled_source_features is not None:
        base=cell.labeled_source_features
        novelty=[]
        for mission in x[lo:t+1]:
            val=[]
            for phase in mission:
                compat=base[np.isclose(base[:,0],phase[0])]
                if not len(compat):compat=base
                val.append(float(np.sqrt(np.mean((compat-phase)**2,axis=1)).min()))
            novelty.append(np.mean(val))
        nov_now=float(novelty[-1])
        nov_avg=float(np.mean(novelty))
    else:
        nov_now=nov_avg=0.
    def slope(v):
        return float(np.mean(v[len(v)//2:])-np.mean(v[:len(v)//2])) if len(v)>3 else 0.
    obs={
        "mission_seen":float(t+1),
        "last_disagreement_z":float(zd[-1]),
        "disagreement_30_mean":float(np.mean(zd)),
        "disagreement_30_slope":slope(zd),
        "last_voltage_z":float(zv[-1]),
        "voltage_30_mean":float(np.mean(zv)),
        "voltage_30_slope":slope(zv),
        "voltage_30_std":float(np.std(zv)),
        "current_30_mean":float(np.mean(cur)),
        "current_30_std":float(np.std(cur)),
        "temp_30_mean":float(np.mean(temp)),
        "temp_30_trend":slope(temp),
        "novelty_current":nov_now,
        "novelty_30_mean":nov_avg,
        "frozen_soc_30_mean":float(fp.mean()),
        "frozen_soc_30_range":float(fp.max()-fp.min()),
    }
    assert all(np.isfinite(z) for z in obs.values())
    return obs


def counterfactual_roi(cell,bank,t):
    """This function is STRICTLY OFFLINE and must NEVER be invoked by a
    challenge deployment policy; 100 future mission labels enter supervision.
    """
    end=min(cell.n,t+1+FUTURE)
    f=cell.frozen[t+1:end]
    a,b=bank[t]
    c=np.clip(a*f+b,0,1)
    truth=cell.label_broker[t+1:end]
    gain=float((np.abs(f-truth)-np.abs(c-truth)).mean()*100.)
    return gain


def make_source_meta(cells,dev):
    rows=[]
    for cid in sorted(dev):
        cell=cells[cid]
        bank=event_models(cell,scheduled_opportunities(cell))
        for t in scheduled_opportunities(cell):
            rows.append({
                "cell_id":cid,"event_test_index":t,
                "future_100_SoCproxy_gain_pp":counterfactual_roi(cell,bank,t),
                **visible_features(cell,t),
            })
    return pd.DataFrame(rows)


def candidates():
    return {
        "ridge":lambda:make_pipeline(StandardScaler(),Ridge(alpha=30)),
        "huber":lambda:make_pipeline(StandardScaler(),HuberRegressor(alpha=10.,epsilon=1.75,max_iter=500)),
        "constant":lambda:DummyRegressor(strategy="mean"),
    }


def train_select_development(meta):
    X=meta[FEATURE_NAMES]
    y=meta.future_100_SoCproxy_gain_pp.to_numpy()
    ids=meta.cell_id.to_numpy()
    nfolds=min(5,len(np.unique(ids)))
    cv=GroupKFold(n_splits=nfolds)
    cvrows=[]
    oof={}
    for name,ctor in candidates().items():
        preds=np.full(len(meta),np.nan)
        for tr,vl in cv.split(X,y,ids):
            m=ctor()
            m.fit(X.iloc[tr],y[tr])
            preds[vl]=m.predict(X.iloc[vl])
        oof[name]=preds
        df=pd.DataFrame({"cell_id":ids,"true":y,"pred":preds})
        # Optimizing fold-held-cell error, never challenge labels.
        cell_mse=df.assign(mse=(df.true-df.pred)**2).groupby("cell_id").mse.mean()
        corr=float(spearmanr(y,preds).statistic) if np.std(preds)>1e-10 else 0.
        cvrows.append({"name":name,"CV_cell_macro_MSE":float(cell_mse.mean()),
                       "CV_spearman_all_events":corr,"dev_cells":len(cell_mse)})
    out=pd.DataFrame(cvrows)
    winner=out.sort_values(["CV_cell_macro_MSE","name"]).iloc[0]["name"]
    model=candidates()[winner]()
    model.fit(X,y)
    return model,out,str(winner),oof[str(winner)]


def learned_schedule(cell,opportunities,model,budget,threshold):
    if budget==0:return [],[]
    selected=[];preds=[];last=-MIN_PAUSE-1
    for t in opportunities:
        if len(selected)>=budget:break
        if t-last<MIN_PAUSE:continue
        f=visible_features(cell,t)
        prob=float(model.predict(pd.DataFrame([f])[FEATURE_NAMES])[0])
        # ROI is estimated future local improvement pp (not true lifetime J).
        preds.append({"event_test_index":t,"predicted_100mission_gain_pp":prob,
                      "purchase":int(prob>threshold),
                      **f})
        if prob>threshold:
            selected.append(t)
            last=t
    return selected,preds


def replay(cell,bank,opportunities,policy,budget,price,model,thresholds):
    if policy=="offline_oracle":
        events=oracle_DP(cell,bank,opportunities,budget,cost_per_event_pp=price)
        pred=[]
    elif policy=="learned_positive":
        events,pred=learned_schedule(cell,opportunities,model,budget,threshold=.0)
    elif policy=="learned_conservative":
        events,pred=learned_schedule(cell,opportunities,model,budget,threshold=.12)
    elif policy=="learned_75pct_OOF":
        events,pred=learned_schedule(cell,opportunities,model,budget,threshold=thresholds["q75"])
    elif policy=="learned_90pct_OOF":
        events,pred=learned_schedule(cell,opportunities,model,budget,threshold=thresholds["q90"])
    elif policy=="learned_cost_value":
        # The model forecasts average improvement over 100 missions. To
        # compare with total-lifetime MAE-pp-equivalent event price, assume
        # a fixed 500-mission planning denominator known at design time.
        # This heuristic must never read the challenge cell's true horizon.
        events,pred=learned_schedule(
            cell,opportunities,model,budget,threshold=price*PLANNING_HORIZON/FUTURE
        )
    else:
        events=online_schedule(cell,opportunities,policy,budget)
        pred=[]
    estimate=predictor_for_events(cell,bank,events)
    mae,rmse=cycle_scores(cell,estimate)
    return {
        "cell_id":cell.cell_id,"policy":policy,"price":price,"budget":budget,
        "MAE_pp":float(np.mean(mae)),"RMSE_pp":rmse,
        "late_MAE_pp":float(np.mean(mae[int(.75*len(mae)):])),
        "events":len(events),
        "J":float(np.mean(mae)+price*len(events)),
        "event_positions":";".join(str(e) for e in events),
    },pred


def pair_ci(frame,price=.1,budget=4):
    v=frame[(frame.price==price)&(frame.budget==budget)]
    base=v[v.policy=="periodic_150"][["cell_id","MAE_pp","J"]].rename(
        columns={"MAE_pp":"periodic_mae","J":"periodic_J"})
    paired=v.merge(base,on="cell_id",validate="many_to_one")
    paired["delta_MAE"]=paired.MAE_pp-paired.periodic_mae
    paired["delta_J"]=paired.J-paired.periodic_J
    macro=[]
    for name,g in paired.groupby("policy"):
        dd=g.delta_MAE.to_numpy()
        dc=g.delta_J.to_numpy()
        rng=np.random.default_rng(SEED)
        ix=rng.integers(0,len(dd),size=(7000,len(dd)))
        lo,hi=np.quantile(dd[ix].mean(axis=1),[.025,.975])
        clo,chi=np.quantile(dc[ix].mean(axis=1),[.025,.975])
        macro.append({
            "policy":name,"cells":len(g),
            "median_SOCproxy_MAE_pp":float(g.MAE_pp.median()),
            "mean_delta_vs_periodic_MAE_pp":float(dd.mean()),
            "CI_low":lo,"CI_high":hi,
            "mean_J":float(g.J.mean()),
            "mean_J_vs_periodic":float(dc.mean()),
            "J_CI_low":clo,"J_CI_high":chi,
            "median_references_bought":float(g.events.median()),
            "cells_better_periodic":int((dd<0).sum()),
        })
    return paired,pd.DataFrame(macro)


def report(meta,cv,chosen,challenge,macro,diagnostics,thresholds):
    content=[
        "# Learned WHEN from purchased high-rate SOC reference counterfactuals",
        "",
        "## Contract and limitations",
        "- Offline meta-training can use development-cell future **SOC_proxy** to train the ROI predictor.",
        "- Challenge ONLINE policy sees only previously completed normal mission sensor features, age counter and initial frozen SOC outputs until it buys a 3-phase-end label.",
        "- Challenge SOC_proxy is **only** used by purchased-reference broker and after-the-fact evaluator, never trigger or threshold selection.",
        "- Deterministic SHA256 split: 14 DEV / 8 RETROSPECTIVE CHALLENGE (already studied previously, NOT a pristine independent test).",
        "- Cost λ here is hypothetical MAE-pp-equivalent per 3-point calibration, not field dollars or labor.",
        "- Source/meta future ROI is 100-task single-event improvement against FROZEN predictor; full multi-event future gains differ.",
        "- Same source N20 Ridge and same purchased one-curve 3-point regularized head for all WHEN policies.",
        "- Periodic 150 never sees terminal horizon. Cost-aware hindsight Oracle deliberately does.",
        "",
        "## Model choice on source cells ONLY",
        f"- Development source candidates: {len(meta)} over {meta.cell_id.nunique()} cells.",
        f"- 5-fold group-CV winner: {chosen}.",
        f"- Source group-CV prediction thresholds: upper quartile={thresholds['q75']:.3f} pp, upper decile={thresholds['q90']:.3f} pp. Fixed BEFORE 8-cell challenge.",
        "- Price-gated variant requires predicted 100-task MAE benefit above λ×500/100, with design-time 500-task planning horizon (not known target lifespan).",
        "| Method | 5-fold group-cell macro MSE (pp²) | CV rank correlation |",
        "|---|---:|---:|",
    ]
    for e in cv.itertuples():
        content.append(f"| {e.name} | {e.CV_cell_macro_MSE:.3f} | {e.CV_spearman_all_events:.3f} |")
    content += [
        "",
        "## Eight challenge cells: λ=0.1, max B=4 purchases",
        "| Policy | Median MAE (pp) | Mean Δ MAE vs strict periodic150 [95% cell CI] | Mean total J | Mean Δ J [CI] | Median purchases |",
        "|---|---:|---|---:|---|---:|",
    ]
    for e in macro.sort_values("median_SOCproxy_MAE_pp").itertuples():
        content.append(
            f"| {e.policy} | {e.median_SOCproxy_MAE_pp:.3f} | "
            f"{e.mean_delta_vs_periodic_MAE_pp:+.3f} [{e.CI_low:+.3f},{e.CI_high:+.3f}] | "
            f"{e.mean_J:.3f} | {e.mean_J_vs_periodic:+.3f} [{e.J_CI_low:+.3f},{e.J_CI_high:+.3f}] | "
            f"{e.median_references_bought:.0f} |"
        )
    content += [
        "",
        "## Generalization diagnostic OFFLINE ONLY",
        f"- Challenge single-update ROI MSE (frozen baseline target): {diagnostics['mse']:.3f} pp².",
        f"- Challenge ROI Spearman (pooling events; descriptive only): {diagnostics['spearman']:.3f}.",
        f"- Dev positive ROI frequency: {diagnostics['dev_positive_rate']:.3f}; challenge positive: {diagnostics['challenge_positive_rate']:.3f}.",
        "",
        "## Research decision",
        "- Only positive *budget/cost-matched, physical-cell* improvements versus periodic justify claiming active maintenance.",
        "- Oracle is hindsight unattainable. If learned WHEN fails, report failure and prioritize richer reference labels / better transferable utility features.",
        "- No full SOC curve, actual agriculture battery field data, or financial maintenance costs yet.",
    ]
    return "\n".join(content)+"\n"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(LABELS)
    cells={c:make_cell(g) for c,g in df.groupby("cell_id",sort=True)}
    split=frozen_partition(list(cells))
    dev={c for c,v in split.items() if v=="development"}
    test={c for c in split if c not in dev}
    train=make_source_meta(cells,dev)
    if len(train)<150:raise AssertionError("Insufficient source ROI examples")
    model,cv,chosen,oof=train_select_development(train)
    thresholds={"q75":float(np.quantile(oof,.75)),"q90":float(np.quantile(oof,.90))}
    rows=[];times=[];diagnostics=[]
    for cid in sorted(test):
        cell=cells[cid]
        opportunities=scheduled_opportunities(cell)
        bank=event_models(cell,opportunities)
        # OFFLINE, never used for decision or hyperparameter selection:
        for t in opportunities:
            x=visible_features(cell,t)
            true_gain=counterfactual_roi(cell,bank,t)
            pred=float(model.predict(pd.DataFrame([x])[FEATURE_NAMES])[0])
            diagnostics.append({"cell_id":cid,"event":t,
                                "true_frozen_future_roi_pp":true_gain,"predicted_roi_pp":pred})
        for price in COSTS:
            for budget in (2,4):
                for arm in DECISIONS:
                    r,steps=replay(cell,bank,opportunities,arm,budget,price,model,thresholds)
                    rows.append(r)
                    for e in steps:
                        times.append({"cell_id":cid,"policy":arm,
                                      "price":price,"budget":budget,**e})
    results=pd.DataFrame(rows)
    assert len(test)==8 and results.cell_id.nunique()==8
    assert results.events.le(results.budget).all()
    paired,macro=pair_ci(results)
    d=pd.DataFrame(diagnostics)
    spearman=float(spearmanr(d.true_frozen_future_roi_pp,d.predicted_roi_pp).statistic)
    diag={
        "mse":float(np.mean((d.true_frozen_future_roi_pp-d.predicted_roi_pp)**2)),
        "spearman":spearman if np.isfinite(spearman) else float("nan"),
        "dev_positive_rate":float((train.future_100_SoCproxy_gain_pp>0).mean()),
        "challenge_positive_rate":float((d.true_frozen_future_roi_pp>0).mean()),
    }
    train.to_csv(out/"source_offline_future_ROI_META_LABELS.csv",index=False)
    cv.to_csv(out/"dev_cell_block_CV.csv",index=False)
    results.to_csv(out/"challenge_when_policy_price_budget.csv",index=False)
    paired.to_csv(out/"challenge_paired_lambda01_B4.csv",index=False)
    macro.to_csv(out/"challenge_macro_lambda01_B4.csv",index=False)
    d.to_csv(out/"challenge_offline_ROI_diagnostic_DO_NOT_USE_FOR_SELECTION.csv",index=False)
    pd.DataFrame(times).to_csv(out/"challenge_pre_purchase_predictions_and_actions.csv",index=False)
    (out/"report.md").write_text(report(train,cv,chosen,results,macro,diag,thresholds),encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "split":split,"chosen_on_dev_only":chosen,"source_cv":cv.to_dict(orient="records"),
        "source_only_quantile_gates":thresholds,"planning_horizon_hypothesis_missions":PLANNING_HORIZON,
        "normal_mission_label_broker":"3 phase-end SOC_proxy points per maintenance",
        "ONLINE_DOES_NOT_USE":["SOC_proxy hidden until purchase","future ROI","RPT curve","true SOH"],
        "original_label_definition":"Coulomb count normalized by preceding RPT capacity",
        "offline_meta_target":"next 100 normal missions future proxy MAE relative frozen after single calibration",
        "no_pristine_test_due_to_prior_exploration":True,
        "costs_assumed_MAE_pp_equivalent_per_purchase":COSTS,
        "policies":DECISIONS,
    },indent=2,ensure_ascii=False),encoding="utf-8")
    print((out/"report.md").read_text(encoding="utf-8"),flush=True)


if __name__=="__main__":
    main()
