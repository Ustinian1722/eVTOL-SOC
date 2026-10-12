"""Stateful marginal-reference ROI learning for cost-aware SOC-proxy maintenance.

References are complete *observed-sample* normal high-power missions,
NOT low-rate RPT curves. Historical SOC_proxy still uses earlier RPT
capacity as an OFFLINE denominator; no real-world independent SOC truth.

The central distinction vs earlier experiments: the supervised source
target is the conditional 100-future-mission gain of buying at t versus
KEEPING THE ACTUALLY DEPLOYED calibrated head from previous purchase e,
not versus the initial frozen head. For online challenge cells, the
model never reads hidden target labels, hidden future outcomes, or the
counterfactual benefit. Only previously PURCHASED labels are usable.

14 development cells -> 8 retrospectively held cells. Regression and
policy thresholds/spacing chosen on source-cell GroupKFold OOF replay.
No target challenge label drives hyperparameter/policy selection. All
previously explored cells, so this is a retrospective challenge, not
an unseen external test. Time horizon intentionally unknown online.
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
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dense_highrate_reference_maintenance import (
    ROOT,SEED,N20,GRID_START,GRID_STRIDE,select_windows,
    stream_raw_missions,make_missions,frozen_training,sensor_risks,
    build_broker,per_mission_errors,replay,optimal_hindsight,choose_when,
)

CELL_IDS=("VAH01","VAH02","VAH05","VAH06","VAH07","VAH09",
          "VAH10","VAH11","VAH12","VAH13","VAH15","VAH16",
          "VAH17","VAH20","VAH22","VAH23","VAH24","VAH25",
          "VAH26","VAH27","VAH28","VAH30")
DENSITY="native_all"
SOURCE_COUNT=14
HORIZON=100
PRICE_GRID=(0.,.1,.3)
BUDGET_GRID=(2,4)
TRAIN_OFFSETS=(2,4,8,12)
# A small predefined budget of policy families. Tune ONLY by source
# held-cell group-CV replay, never by the 8 challenge cell outcomes.
POLICY_GRID=tuple((gap,thr)
    for gap in (75,125,175,225)
    for thr in (-.10,0.,.15,.35,.65,1.00))
SOURCE_PRICE=.1
SOURCE_BUDGET=4
TEST_POLICIES=("frozen","periodic150","periodic200","periodic250",
               "periodic300","periodic400","periodic500",
               "model_disagreement","model_voltage","learned_stateful",
               "offline_oracle")
FIELDS=(
    "seen_missions","since_last_reference","references_acquired",
    "prior_head_a","prior_head_b","last_reference_fit_MAE_pp",
    "last_reference_signed_error_pp","last_reference_raw_points",
    "cur_abs_calibrated_vs_nominal_pp","last10_abs_calibrated_vs_nominal_pp",
    "last30_abs_calibrated_vs_nominal_pp","last30_trend_disagreement_pp",
    "cur_abs_frozen_vs_nominal_pp",
    "last30_frozen_vs_nominal_pp",
    "cur_voltage_residual","last30_voltage_residual",
    "last30_trend_voltage_residual",
    "last30_mean_voltage","last30_mean_abs_current",
    "last30_mean_temp","last30_max_temp",
    "last30_duration_s","last30_temp_trend",
    "current_phase_voltage_delta",
    "current_phase_temp_delta",
)


def cell_split(ids=CELL_IDS):
    order=sorted(ids,key=lambda cid:hashlib.sha256(
        ("split_"+cid).encode()).hexdigest())
    return {cid:("development" if i<SOURCE_COUNT else "retrospective_challenge")
            for i,cid in enumerate(order)}


def safe_mean(arr):
    return float(np.mean(arr)) if len(arr) else 0.


def trend(a):
    return float(np.mean(a[len(a)//2:])-np.mean(a[:len(a)//2])) if len(a)>3 else 0.


@dataclass
class Cell:
    cid:str
    train:list
    test:list
    bank:dict
    loss:dict
    frozen_loss:np.ndarray
    risk:dict
    base_disagreement:np.ndarray
    per_mission_voltage:np.ndarray
    per_mission_I:np.ndarray
    per_mission_T:np.ndarray
    per_mission_Tmax:np.ndarray
    per_mission_delta_V:np.ndarray
    per_mission_delta_T:np.ndarray
    opportunities:list
    raw_stats:dict

    @property
    def n(self):return len(self.test)

    def deployed_params(self,last):
        return (1.,0.) if last<0 else self.bank[last][0]

    def current_reference_fit(self,last):
        """Past purchased reference is available to online decision.
        Current/unpurchased SOC labels must not be referenced here.
        """
        if last<0:return (0.,0.,0.)
        p=self.test[last];a,b=self.bank[last][0]
        idx=np.arange(p.n)  # native_all reference already BOUGHT
        cal=np.clip(a*p.predicted[idx]+b,0,1)
        err=(cal-p.y[idx])*100
        return float(np.abs(err).mean()),float(err.mean()),float(p.n)

    def feature(self,t,last,num_paid):
        if t<=last:raise ValueError("Feature queried before last purchase")
        # Never look at future missions. Measurement arrays derive
        # exclusively from V/I/T, nominal integrals, frozen models.
        lo=max(0,t-29)
        recent=self.test[lo:t+1]
        a,b=self.deployed_params(last)
        delta=np.array([100*np.mean(np.abs(
            np.clip(a*m.predicted+b,0,1)-m.qnom)) for m in recent])
        frozen_delta=self.base_disagreement[lo:t+1]
        volt=self.per_mission_voltage[lo:t+1]
        fits,signed,nrows=self.current_reference_fit(last)
        r={
            "seen_missions":float(t+1),
            "since_last_reference":float(t-last) if last>=0 else float(t+1),
            "references_acquired":float(num_paid),
            "prior_head_a":float(a),
            "prior_head_b":float(b),
            "last_reference_fit_MAE_pp":fits,
            "last_reference_signed_error_pp":signed,
            "last_reference_raw_points":nrows,
            "cur_abs_calibrated_vs_nominal_pp":float(delta[-1]),
            "last10_abs_calibrated_vs_nominal_pp":safe_mean(delta[-10:]),
            "last30_abs_calibrated_vs_nominal_pp":safe_mean(delta),
            "last30_trend_disagreement_pp":trend(delta),
            "cur_abs_frozen_vs_nominal_pp":float(self.base_disagreement[t]),
            "last30_frozen_vs_nominal_pp":safe_mean(frozen_delta),
            "cur_voltage_residual":float(self.per_mission_voltage[t]),
            "last30_voltage_residual":safe_mean(volt),
            "last30_trend_voltage_residual":trend(volt),
            "last30_mean_voltage":safe_mean([
                float(np.mean(m.X[:,0])) for m in recent]),
            "last30_mean_abs_current":safe_mean(self.per_mission_I[lo:t+1]),
            "last30_mean_temp":safe_mean(self.per_mission_T[lo:t+1]),
            "last30_max_temp":float(np.max(self.per_mission_Tmax[lo:t+1])),
            "last30_duration_s":safe_mean([m.duration for m in recent]),
            "last30_temp_trend":trend(self.per_mission_T[lo:t+1]),
            "current_phase_voltage_delta":float(self.per_mission_delta_V[t]),
            "current_phase_temp_delta":float(self.per_mission_delta_T[t]),
        }
        assert tuple(r)==FIELDS,("Feature contract mismatch",r.keys())
        assert np.isfinite(list(r.values())).all()
        return r


def build_cell(cid):
    windows=select_windows(cid)
    allocated,raw_stats=stream_raw_missions(ROOT/f"{cid}.csv",windows)
    missions,bad=make_missions(windows,allocated)
    train,test=frozen_training(missions)
    n=len(test)
    # An online policy must not know whether it is within 20 missions of
    # the experimentally observed cell end-of-life. Thus do NOT reuse
    # opportunities_for(n) from old pilot, which censored its last 20.
    ops=list(range(GRID_START,n,GRID_STRIDE))
    bank,loss=build_broker(test,ops,DENSITY)
    risk=sensor_risks(train,test)
    out=Cell(
        cid,train,test,bank,loss,per_mission_errors(test,(1.,0.)),risk,
        np.array([100*np.abs(m.predicted-m.qnom).mean() for m in test]),
        np.array([np.abs(m.voltage_resid).mean() for m in test]),
        np.array([np.abs(m.X[:,1]).mean() for m in test]),
        np.array([m.X[:,2].mean() for m in test]),
        np.array([m.X[:,2].max() for m in test]),
        np.array([m.X[m.ns==6,0].mean()-m.X[m.ns==4,0].mean()
                  for m in test]),
        np.array([m.X[m.ns==6,2].mean()-m.X[m.ns==4,2].mean()
                  for m in test]),
        ops,{"raw_rows":raw_stats["read_rows"],"bad_missions":len(bad),
             "full_normal_missions":len(missions),"test_missions":n}
    )
    return out


def meta_target(cell,t,last):
    """SOURCE-ONLY OFFLINE supervised counterfactual.
    Advantage of a NEW purchased reference at t vs EXISTING head last.
    Never call in challenge online_policy().
    """
    future_end=min(cell.n,t+1+HORIZON)
    if future_end<=t+1:return np.nan
    old=cell.frozen_loss if last<0 else cell.loss[last]
    new=cell.loss[t]
    return float(np.mean(old[t+1:future_end]-new[t+1:future_end]))


def meta_examples(cell):
    rows=[]
    for j,t in enumerate(cell.opportunities):
        if cell.n-t-1<25:continue
        states=[-1]
        states.extend(cell.opportunities[j-k] for k in TRAIN_OFFSETS if j>=k)
        for last in states:
            rows.append({
                "cell_id":cell.cid,"event_idx":t,"prev_purchased_idx":last,
                "incremental_future100_ROI_pp":meta_target(cell,t,last),
                **cell.feature(t,last,1 if last>=0 else 0)
            })
    return rows


def model_constructors():
    return {
        "constant":lambda:DummyRegressor(strategy="mean"),
        "ridge":lambda:make_pipeline(StandardScaler(),Ridge(alpha=80.)),
        "histgb":lambda:HistGradientBoostingRegressor(
            max_iter=90,max_leaf_nodes=12,min_samples_leaf=35,
            l2_regularization=5.,learning_rate=.06,random_state=SEED),
    }


def dev_model_selection(meta):
    cv=GroupKFold(n_splits=5)
    X=meta[list(FIELDS)]
    y=meta.incremental_future100_ROI_pp.to_numpy()
    groups=meta.cell_id.to_numpy()
    result=[]
    for name,ctor in model_constructors().items():
        pred=np.full(len(meta),np.nan)
        for tr,vl in cv.split(X,y,groups):
            model=ctor()
            model.fit(X.iloc[tr],y[tr])
            pred[vl]=model.predict(X.iloc[vl])
        df=pd.DataFrame({"cell_id":groups,"diff2":(pred-y)**2})
        mse=float(df.groupby("cell_id").diff2.mean().mean())
        rho=spearmanr(pred,y).statistic if np.std(pred)>1e-12 else 0.
        result.append({"model":name,"CV_cell_macro_MSE_pp2":mse,
                       "CV_all_event_rank_corr":float(rho)})
    table=pd.DataFrame(result).sort_values(["CV_cell_macro_MSE_pp2","model"])
    winner=str(table.iloc[0]["model"])
    return winner,table


def learned_events(cell,model,budget,price,gap,threshold,trace=False):
    events=[];decisions=[];last=-1
    for t in cell.opportunities:
        if len(events)>=budget:break
        if last>=0 and t-last<gap:continue
        features=cell.feature(t,last,len(events))
        estimate=float(model.predict(pd.DataFrame([features])[list(FIELDS)])[0])
        # Source-only threshold is in future-100-mission mean MAE pp.
        # Cost-price equivalent threshold is added without reading
        # target total-horizon. 500 is a DECLARED design horizon, not
        # the actual observed cell end.
        tau=threshold+price*5.
        buy=estimate>tau
        if trace:
            decisions.append({"cell_id":cell.cid,"event_idx":t,
                              "prior_purchased_idx":last,"budget":budget,
                              "price":price,"predicted_incremental_ROI_pp":estimate,
                              "threshold_pp":tau,"buy":int(buy),
                              **features})
        if buy:
            events.append(t)
            last=t
    return events,decisions


def run_cell(cell,policy,budget,price,model=None,config=None):
    if policy=="learned_stateful":
        if model is None or config is None:raise ValueError("Missing trained policy")
        events,decisions=learned_events(
            cell,model,budget,price,config["gap"],config["threshold"],trace=True)
    elif policy=="offline_oracle":
        events=optimal_hindsight(cell.frozen_loss,cell.loss,
                                 cell.opportunities,budget,price)
        decisions=[]
    elif policy=="frozen":
        events=[];decisions=[]
    elif policy=="model_disagreement":
        events=choose_when(cell.risk,cell.opportunities,cell.n,
                           budget,"disagreement3",cell.cid)
        decisions=[]
    elif policy=="model_voltage":
        events=choose_when(cell.risk,cell.opportunities,cell.n,
                           budget,"voltage3",cell.cid)
        decisions=[]
    else:
        events=choose_when(cell.risk,cell.opportunities,cell.n,
                           budget,policy,cell.cid)
        decisions=[]
    loss=replay(cell.frozen_loss,cell.loss,events)
    return {
        "cell_id":cell.cid,"policy":policy,"budget":budget,"price":price,
        "SOCproxy_MAE_pp":float(loss.mean()),
        "late_SOCproxy_MAE_pp":float(loss[int(.75*len(loss)):].mean()),
        "task_over3pp_fraction":float((loss>3.).mean()),
        "num_reference_purchases":len(events),
        "num_raw_labels_purchased":int(sum(cell.bank[t][1] for t in events)),
        "J":float(loss.mean()+price*len(events)),
        "reference_idx":";".join(map(str,events)),
    },decisions


def train_development_oof_policy(cells,meta,winner):
    """Nested source-cell OOF replay optimizes policy config without
    any challenge SOC labels. Model family chosen source-only CV.
    """
    ids=sorted(cells)
    fold=GroupKFold(n_splits=5)
    idsarr=np.array(ids)
    models={}
    predictions=[]
    for tr,vl in fold.split(idsarr,groups=idsarr):
        train_ids=set(idsarr[tr])
        m=model_constructors()[winner]()
        d=meta[meta.cell_id.isin(train_ids)]
        m.fit(d[list(FIELDS)],d.incremental_future100_ROI_pp)
        for cid in idsarr[vl]:
            models[cid]=m
    trials=[]
    for gap,threshold in POLICY_GRID:
        for cid in ids:
            row,_=run_cell(cells[cid],"learned_stateful",
                           SOURCE_BUDGET,SOURCE_PRICE,models[cid],
                           {"gap":gap,"threshold":threshold})
            trials.append({"gap":gap,"threshold":threshold,**row})
    table=pd.DataFrame(trials)
    macro=table.groupby(["gap","threshold"],as_index=False).agg(
        mean_source_OOF_J=("J","mean"),
        mean_source_OOF_MAE_pp=("SOCproxy_MAE_pp","mean"),
        mean_source_OOF_purchases=("num_reference_purchases","mean"),
    ).sort_values(["mean_source_OOF_J","gap","threshold"])
    best=macro.iloc[0]
    config={"gap":int(best.gap),"threshold":float(best.threshold)}
    return config,macro,table


def paired_challenge(results):
    # Same physical challenge cells, match same cost and B.
    out=[]
    for (p,b),g in results.groupby(["price","budget"]):
        base=g[g.policy=="periodic150"][["cell_id","J","SOCproxy_MAE_pp"]].rename(
            columns={"J":"baseline_J","SOCproxy_MAE_pp":"baseline_MAE"})
        for arm,rows in g.groupby("policy"):
            same=rows.merge(base,on="cell_id",validate="one_to_one")
            dc=(same.J-same.baseline_J).to_numpy()
            de=(same.SOCproxy_MAE_pp-same.baseline_MAE).to_numpy()
            rnd=np.random.default_rng(SEED)
            bootstrap=rnd.integers(0,len(dc),size=(5000,len(dc)))
            ci=np.quantile(dc[bootstrap].mean(axis=1),[.025,.975])
            out.append({
                "price":p,"budget":b,"policy":arm,
                "held_cells":len(same),
                "median_MAE_pp":float(same.SOCproxy_MAE_pp.median()),
                "mean_J":float(same.J.mean()),
                "mean_delta_J_vs_periodic150":float(dc.mean()),
                "J_CI_low":float(ci[0]),"J_CI_high":float(ci[1]),
                "mean_delta_MAE_vs_periodic150_pp":float(de.mean()),
                "mean_label_purchases":float(same.num_reference_purchases.mean()),
                "cells_with_lower_J_than_periodic150":int((dc<0).sum()),
            })
    return pd.DataFrame(out)


def report(config,cv,meta,source_policy,challenge,comparisons,split,diagnostic):
    sub=comparisons[(np.isclose(comparisons.price,.1))&
                    (comparisons.budget==4)].sort_values("mean_J")
    lines=[
        "# Stateful marginal ROI supervised SOC_proxy WHEN: retrospective split",
        "",
        "## Strict setup",
        "- Develop 14 physical cells; retrospectively challenge 8 physical cells. Prior literature/experiments already examined these 22, so this is NOT a pristine test.",
        "- Fixed same-cell initial N20 supervised Ridge, normal high-rate mission's actual native-sampling SOC_proxy as ONE paid reference; fixed regularized two-parameter calibration Head; calibration affects NEXT mission only.",
        "- Development supervised target: subsequent 100 mission SOC_proxy gain NEW calibration minus CURRENT deployed head after a simulated earlier purchased mission. Offline source labels allowed, including future outcomes, to create this supervised target.",
        "- Challenge policy never consults current unbought SOC_proxy, target future labels, target true remaining lifetime or counterfactual. Past bought mission labels permitted for deployed head state and past paid fit.",
        "- Candidates every 25 normal tasks starting t=24; NO censoring based on remaining lifetime. A candidate near the end may waste a purchase.",
        "- Price λ is hypothetical MAE-pp-equivalent PER reference event, not currency or time. Historical preceding RPT capacity creates SOC_proxy targets, but RPT waveform is not a purchased high-rate calibration reference.",
        "- The model uses nominal 3Ah coulomb features and target uses past RPT capacity; this evaluates historical reference adaptation, not independently verified electrochemical SOC.",
        "",
        f"14 development cells, {len(meta):,} counterfactual state-conditioned examples.",
        f"Chosen regressor by 5-fold held-cell CV: {cv.iloc[0].model}.",
        f"Source OOF-tuned policy: gap={config['gap']} tasks and minimum predicted 100-task incremental ROI={config['threshold']:+.2f} pp + 5λ design cost surcharge.",
        "",
        "| Source model | Cell-macro held-cell MSE (pp²) | Source event rank rho |",
        "|---|---:|---:|",
    ]
    for z in cv.itertuples():
        lines.append(f"| {z.model} | {z.CV_cell_macro_MSE_pp2:.3f} | {z.CV_all_event_rank_corr:.3f} |")
    lines.extend([
        "",
        f"Dev incremental ROI positive frequency: {diagnostic['positive_source']:.3f}.",
        f"Dev incremental ROI median: {diagnostic['median_source']:+.3f} pp; 10–90%: [{diagnostic['p10_source']:+.3f},{diagnostic['p90_source']:+.3f}] pp.",
        "",
        "## Eight retrospective challenge cells: B=4, λ=.1",
        "| Policy | SOCproxy MAE median (pp) | Mean J | Mean ΔJ vs fixed150 [95% cell CI] | Mean purchases | Better / 8 cells |",
        "|---|---:|---:|---|---:|---:|",
    ])
    for z in sub.itertuples():
        lines.append(
            f"| {z.policy} | {z.median_MAE_pp:.3f} | {z.mean_J:.3f} | "
            f"{z.mean_delta_J_vs_periodic150:+.3f} [{z.J_CI_low:+.3f},{z.J_CI_high:+.3f}] | "
            f"{z.mean_label_purchases:.2f} | {z.cells_with_lower_J_than_periodic150}/{z.held_cells} |")
    learned=sub[sub.policy=="learned_stateful"].iloc[0]
    periodic=sub[sub.policy.str.startswith("periodic")]
    best_fixed=periodic.sort_values("mean_J").iloc[0]
    lines.extend([
        "",
        "## Decision rules",
        f"- Learned-minus-**best retrospective fixed periodic** mean J = {learned.mean_J-best_fixed.mean_J:+.3f} (best fixed was {best_fixed.policy}; NOT a preselected deployment model).",
        "- For any claim that learned strategy improves cost: require a positive held-cell paired result against source-selected fixed-period baseline as well as strict periodic150; do not optimize post hoc on challenge scores.",
        "- Offline Oracle is clairvoyant using target future proxy labels. It is a lower bound only.",
        "- If learned policy fails, report it as a negative result; improve reference-price modeling, state-aware regret forecasting and correct cost assumptions instead of claiming success.",
    ])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    split=cell_split()
    cells={}
    meta_rows=[];manifest=[]
    for cid in CELL_IDS:
        clock=time.monotonic()
        cell=build_cell(cid)
        cells[cid]=cell
        manifest.append({"cell_id":cid,"partition":split[cid],**cell.raw_stats})
        if split[cid]=="development":
            source=meta_examples(cell);meta_rows.extend(source)
            print(f"DEV {cid}: {cell.n} test missions; {len(source)} source meta examples; {time.monotonic()-clock:.1f}s",flush=True)
        else:
            print(f"CHALLENGE {cid}: {cell.n} missions, future reference labels ESCROWED; {time.monotonic()-clock:.1f}s",flush=True)
    meta=pd.DataFrame(meta_rows)
    assert meta.cell_id.nunique()==14 and len(meta)>500
    assert list(meta[list(FIELDS)].columns)==list(FIELDS)
    winner,cv=dev_model_selection(meta)
    development={cid:c for cid,c in cells.items() if split[cid]=="development"}
    challenge={cid:c for cid,c in cells.items() if split[cid]!="development"}
    config,policy_cv,policy_details=train_development_oof_policy(development,meta,winner)
    final=model_constructors()[winner]()
    final.fit(meta[list(FIELDS)],meta.incremental_future100_ROI_pp)
    # Source-only select strongest low-maintenance fixed periodic interval.
    baseline_rows=[]
    for cid,cell in development.items():
        for pol in ("periodic150","periodic200","periodic250","periodic300",
                    "periodic400","periodic500"):
            row,_=run_cell(cell,pol,SOURCE_BUDGET,SOURCE_PRICE)
            baseline_rows.append(row)
    dev_fixed=pd.DataFrame(baseline_rows).groupby("policy",as_index=False).J.mean()
    best_fixed=str(dev_fixed.sort_values(["J","policy"]).iloc[0].policy)
    policies=("frozen","periodic150","periodic200","periodic250","periodic300",
              "periodic400","periodic500","model_disagreement","model_voltage",
              "learned_stateful","source_selected_periodic","offline_oracle")
    rows=[];decisions=[]
    for cid,cell in sorted(challenge.items()):
        for p in PRICE_GRID:
            for b in BUDGET_GRID:
                for policy in policies:
                    actual=best_fixed if policy=="source_selected_periodic" else policy
                    result,trace=run_cell(cell,actual,b,p,final,config)
                    result["policy"]=policy
                    rows.append(result)
                    decisions.extend(trace)
        print(f"CHALLENGE EVALUATED {cid}",flush=True)
    outcome=pd.DataFrame(rows)
    paired=paired_challenge(outcome)
    diag={
        "positive_source":float((meta.incremental_future100_ROI_pp>0).mean()),
        "median_source":float(meta.incremental_future100_ROI_pp.median()),
        "p10_source":float(meta.incremental_future100_ROI_pp.quantile(.10)),
        "p90_source":float(meta.incremental_future100_ROI_pp.quantile(.90)),
    }
    meta.to_csv(out/"DEV_offline_marginal_ROI_meta_supervision_DO_NOT_DEPLOY.csv",index=False)
    cv.to_csv(out/"development_cell_block_CV_regressors.csv",index=False)
    policy_cv.to_csv(out/"development_heldcell_OOF_policy_selection.csv",index=False)
    policy_details.to_csv(out/"development_heldcell_OOF_policy_percell.csv",index=False)
    dev_fixed.to_csv(out/"development_fixed_periodic_interval_selection.csv",index=False)
    pd.DataFrame(manifest).to_csv(out/"all22_partition_and_raw_quality.csv",index=False)
    outcome.to_csv(out/"challenge_stateful_WHEN_budget_cost.csv",index=False)
    paired.to_csv(out/"challenge_cell_paired_CI.csv",index=False)
    pd.DataFrame(decisions).to_csv(out/"challenge_pre_purchase_policy_trace.csv",index=False)
    (out/"metadata.json").write_text(json.dumps({
        "cells":split,"regressor_chosen_using_source_only":winner,
        "policy_chosen_using_source_heldcell_OOF":config,
        "fixed_periodic_selected_using_source_only":best_fixed,
        "model_features":FIELDS,
        "online_no_hidden_SOC_labels_or_future_horizon":True,
        "per_purchase_reference":"one NORMAL raw high-rate full mission, native samplewise",
        "reference_label":"Coulomb counted SOC_proxy using historical previous RPT capacity, not independently measured SOC",
        "sweep_prices":PRICE_GRID,"budget_grid":BUDGET_GRID,
        "prior_exploration_means_not_pristine_holdout":True,
        "horizon_for_offline_ROI_label":HORIZON,
        "online_cost_scaling_design_horizon_assumption":500,
    },indent=2,ensure_ascii=False),encoding="utf-8")
    text=report(config,cv,meta,policy_cv,outcome,paired,split,diag)
    (out/"report.md").write_text(text,encoding="utf-8")
    print(text,flush=True)


if __name__=="__main__":
    main()
