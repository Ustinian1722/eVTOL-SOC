"""Mission-only adaptive voltage dynamics as SOC feature correction: an online study.

All voltage-model updates train on *measured high-rate mission* Vmin, after
the mission's predictions have been recorded; no RPT or SOC labels are used
in online adaptation. Voltage prediction accuracy is an independent sensor
measurement objective; SOC scores use separately-held scoring-only SOC_proxy.

This is a lightweight RLS/source-replay surrogate inspired by battery
incremental model identification and TTT auxiliary voltage reconstruction,
NOT a full ECM, Kalman filter, OS-ELM, CoTTA, Transformer, or publication replica.
"""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"results"/"soc_proxy_phase_labels.csv"
FEATURES=["phase_Ns","duration_s","I_mean_A","I_peak_A","Vmin_V","Vmax_V","Tmean_C","Tmax_C"]
VX=["phase_Ns","duration_s","I_mean_A","I_peak_A","Vmax_V","Tmean_C","Tmax_C"]
N20=20
SEED=20261012
B=4
ARM_SPECS=[
    ("frozen","none","none"),
    ("stream_RLS_0995","rls_0995","stream"),
    ("stream_RLS_098","rls_098","stream"),
    ("periodic_replay","replay","periodic"),
    ("ph_replay","replay","ph"),
    ("cusum_replay","replay","cusum"),
    ("threshold_replay","replay","threshold"),
    ("guarded_replay","guard","ph"),
]
# For every voltage-model policy compare no SOC correction with exact
# linear adjustment of the SOC voltage feature to its N20 voltage regime.
SOC_CORRECTION_STRENGTHS=[0.0,0.5]


def robust(v: np.ndarray)->tuple[float,float]:
    v=np.asarray(v,dtype=float)
    med=float(np.median(v))
    mad=float(np.median(np.abs(v-med)))
    return med,max(1.4826*mad,0.25*float(np.std(v)),1e-4)


def fit_design(frame:pd.DataFrame,columns:list[str],target:str):
    base=make_pipeline(StandardScaler(),Ridge(alpha=5))
    base.fit(frame[columns],frame[target])
    return base


def design(model, frame:pd.DataFrame,features:list[str])->np.ndarray:
    return np.c_[np.ones(len(frame)),model.named_steps["standardscaler"].transform(frame[features])]


def coef(model)->np.ndarray:
    m=model.named_steps["ridge"]
    return np.r_[m.intercept_,m.coef_]


def weighted_ridge(Z:np.ndarray,y:np.ndarray,w:np.ndarray,prior:np.ndarray,lam:float)->np.ndarray:
    r=np.sqrt(w)
    X=Z*r[:,None]
    v=y*r
    eye=np.eye(Z.shape[1])
    return np.linalg.solve(X.T@X+lam*eye+1e-10*np.eye(Z.shape[1]),X.T@v+lam*prior)


def replay_update(theta:np.ndarray,frozen:np.ndarray,Z_source:np.ndarray,v_source:np.ndarray,
                  memory:list[tuple[np.ndarray,np.ndarray]], guarded:bool)->tuple[np.ndarray,bool]:
    if len(memory)<12:
        return theta.copy(),False
    fit=memory[:-5] if guarded else memory
    recentZ=np.concatenate([x for x,_ in fit],axis=0)
    recentV=np.concatenate([y for _,y in fit])
    Z=np.r_[Z_source,recentZ]
    target=np.r_[v_source,recentV]
    weights=np.r_[np.full(len(v_source),2.0),np.full(len(recentV),3.0)]
    candidate=weighted_ridge(Z,target,weights,frozen,lam=12)
    shift=candidate-frozen
    norm=float(np.linalg.norm(shift))
    if norm>1.5:
        candidate=frozen+shift/norm*1.5
    if not np.isfinite(candidate).all():
        return theta.copy(),False
    if guarded:
        holdZ=np.concatenate([x for x,_ in memory[-5:]],axis=0)
        holdY=np.concatenate([y for _,y in memory[-5:]])
        old=float(np.mean(np.abs(holdZ@theta-holdY)))
        new=float(np.mean(np.abs(holdZ@candidate-holdY)))
        orig=float(np.mean(np.abs(Z_source@theta-v_source)))
        src=float(np.mean(np.abs(Z_source@candidate-v_source)))
        if new>=0.98*old or src-orig>0.020:
            return theta.copy(),False
    return candidate,True


def rls_step(theta:np.ndarray,P:np.ndarray,z:np.ndarray,target:float,lam:float):
    k=P@z/(lam+z@P@z)
    prediction=float(z@theta)
    residual=float(np.clip(target-prediction,-0.25,0.25))
    updated=theta+k*residual
    newP=(P-np.outer(k,z)@P)/lam
    newP=0.5*(newP+newP.T)
    return updated,newP


def _prepare(cell:pd.DataFrame,track:str):
    cell=cell.sort_values(["charge_cycle_index","phase_Ns"]).copy()
    ids=sorted(cell["charge_cycle_index"].astype(int).unique())
    train=cell[cell.charge_cycle_index.isin(ids[:N20])]
    test=cell[~cell.charge_cycle_index.isin(ids[:N20])]
    voltage=fit_design(train,VX,"Vmin_V")
    soc=fit_design(train,FEATURES,track)
    Vsrc=design(voltage,train,VX)
    Ssrc=design(soc,train,FEATURES)
    Vtest=design(voltage,test,VX)
    Stest=design(soc,test,FEATURES)
    theta_v=coef(voltage)
    theta_soc=coef(soc)
    vs=float(soc.named_steps["standardscaler"].scale_[FEATURES.index("Vmin_V")])
    v_truth=train.Vmin_V.to_numpy(dtype=float)
    baseline_v_bycycle=pd.DataFrame({
        "cycle":train.charge_cycle_index.to_numpy(),
        "resid_mV":np.abs(Vsrc@theta_v-v_truth)*1000,
    }).groupby("cycle").resid_mV.mean().to_numpy()
    center,scale=robust(baseline_v_bycycle)
    # Full target preparation is preprocessing only. Each mission's features
    # are consumed for prediction/updated only at its chronological position.
    groups=[]
    cycle_id=test.charge_cycle_index.to_numpy()
    for cycle in np.unique(cycle_id):
        mask=cycle_id==cycle
        groups.append((
            int(cycle),Vtest[mask],Stest[mask],
            test.Vmin_V.to_numpy(dtype=float)[mask],
            test.SOC_proxy.to_numpy(dtype=float)[mask],
        ))
    return Vsrc,v_truth,theta_v,theta_soc,vs,center,scale,groups


def one_run(info,track:str,policy:tuple[str,str,str],corr:float,cell_id:str)->dict:
    Zsource,v_source,vfrozen,wsoc,vs,center,scale,groups=info
    name,alg,trigger=policy
    theta=vfrozen.copy()
    P=np.linalg.inv(Zsource.T@Zsource+4*np.eye(Zsource.shape[1]))
    history=deque(maxlen=35)
    last=0
    attempts=accepted=0
    ph=cusum=0.0
    elevated=0
    pred_volt=[];pred_soc=[];per_cycle_soc=[]
    for t,(cycle,VZ,SZ,Vobs,SOCtarget) in enumerate(groups,1):
        # Pre-update prediction of *entire* mission, in arrival order.
        curV=VZ@theta
        initialV=VZ@vfrozen
        pred_volt.extend((curV-Vobs)*1000)
        # EXACT linear correction to standardized Vmin input, using learned
        # change in high-rate voltage dynamics, NOT new SOC labels.
        initialSOC=SZ@wsoc
        correction=(curV-initialV)/vs*wsoc[1+FEATURES.index("Vmin_V")]
        p=np.clip(initialSOC-corr*np.clip(correction,-0.2,0.2),0,1)
        errors=(p-SOCtarget)*100
        pred_soc.extend(errors)
        per_cycle_soc.append(float(np.mean(np.abs(errors))))

        # Newly observed mission voltage becomes available *after* prediction.
        history.append((VZ,Vobs))
        score=float(np.mean(np.abs(curV-Vobs))*1000)
        z=(score-center)/scale
        elevated=elevated+1 if z>3 else 0
        ph=max(0.0,ph+z-0.5)
        cusum=max(0.0,cusum+z-1.0)
        eligible=t-last>=35 and len(history)>=12
        if alg=="none":
            update=False
        elif trigger=="stream":
            update=True
        elif trigger=="periodic":
            update=eligible and (t%150==0)
        elif trigger=="ph":
            update=eligible and (ph>=12)
        elif trigger=="cusum":
            update=eligible and (cusum>=20)
        elif trigger=="threshold":
            update=eligible and (elevated>=3)
        else:
            raise ValueError(trigger)
        if trigger!="stream" and attempts>=B: update=False
        if not update: continue
        attempts+=1
        if alg.startswith("rls"):
            lam=.995 if alg=="rls_0995" else .98
            for i in range(len(Vobs)):
                theta,P=rls_step(theta,P,VZ[i],float(Vobs[i]),lam)
            sh=theta-vfrozen
            n=float(np.linalg.norm(sh))
            if n>1.5: theta=vfrozen+sh/n*1.5
            accepted+=1
        else:
            next_theta,ok=replay_update(
                theta,vfrozen,Zsource,v_source,list(history),alg=="guard"
            )
            if ok:
                theta=next_theta
                accepted+=1
        last=t
        ph=cusum=0.0
        elevated=0

    soc=np.asarray(pred_soc)
    volt=np.asarray(pred_volt)
    late=np.asarray(per_cycle_soc)[int(.75*len(per_cycle_soc)):]
    return {
        "cell_id":cell_id,"track":track,
        "arm":name,"voltage_update":alg,"trigger":trigger,"soc_voltage_normalization_strength":corr,
        "attempts":attempts,"accepted":accepted,"missions":len(groups),
        "measured_Vmin_MAE_mV":float(np.mean(np.abs(volt))),
        "measured_Vmin_RMSE_mV":float(np.sqrt(np.mean(volt**2))),
        "SOC_proxy_MAE_pp":float(np.mean(np.abs(soc))),
        "SOC_proxy_RMSE_pp":float(np.sqrt(np.mean(soc**2))),
        "late_SOC_proxy_MAE_pp":float(np.mean(late)),
        "P95_cycle_SOC_proxy_MAE_pp":float(np.quantile(per_cycle_soc,.95)),
    }


def summarize(rows:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    frozen=rows[(rows.arm=="frozen") & (rows.soc_voltage_normalization_strength==0)].set_index(["cell_id","track"])
    joined=rows.join(frozen[["measured_Vmin_MAE_mV","SOC_proxy_MAE_pp"]],
                     on=["cell_id","track"],rsuffix="_frozen")
    joined["delta_voltage_mV"]=joined.measured_Vmin_MAE_mV-joined.measured_Vmin_MAE_mV_frozen
    joined["delta_soc_pp"]=joined.SOC_proxy_MAE_pp-joined.SOC_proxy_MAE_pp_frozen
    agg=joined.groupby(["track","arm","soc_voltage_normalization_strength"],as_index=False).agg(
        cells=("cell_id","nunique"),
        median_voltage_mV=("measured_Vmin_MAE_mV","median"),
        median_soc_proxy_pp=("SOC_proxy_MAE_pp","median"),
        late_soc_proxy_pp=("late_SOC_proxy_MAE_pp","median"),
        delta_voltage_mV=("delta_voltage_mV","mean"),
        delta_soc_pp=("delta_soc_pp","mean"),
        improved_voltage_cells=("delta_voltage_mV",lambda x:int((x<0).sum())),
        improved_soc_cells=("delta_soc_pp",lambda x:int((x<0).sum())),
        median_attempts=("attempts","median"),
        median_accepted=("accepted","median"),
    )
    ci_rows=[]
    for (track,arm,corr), subset in joined.groupby(["track","arm","soc_voltage_normalization_strength"]):
        delta_soc=subset["delta_soc_pp"].to_numpy()
        delta_v=subset["delta_voltage_mV"].to_numpy()
        rng=np.random.default_rng(SEED)
        draws=rng.integers(0,len(subset),size=(5000,len(subset)))
        b_soc=delta_soc[draws].mean(axis=1)
        b_v=delta_v[draws].mean(axis=1)
        scilo,scihi=np.quantile(b_soc,[.025,.975])
        vlo,vhi=np.quantile(b_v,[.025,.975])
        ci_rows.append({
            "track":track,"arm":arm,"soc_voltage_normalization_strength":corr,
            "soc_ci_lo":scilo,"soc_ci_hi":scihi,
            "voltage_ci_lo":vlo,"voltage_ci_hi":vhi,
        })
    agg=agg.merge(pd.DataFrame(ci_rows),on=["track","arm","soc_voltage_normalization_strength"])
    return joined,agg


def report(agg:pd.DataFrame,by_cell:pd.DataFrame)->str:
    lines=[
        "# Mission-only online voltage-model identification and SOC normalization",
        "",
        "High-rate *mission-observed* Vmin is the voltage target; no RPT measurement or test SOC label enters updates.",
        "SOC_proxy is scoring-only and RPT-anchored. Nominal SOC track has no RPT even in its initial training.",
        "Both voltage and SOC predictions precede this mission's own updates.",
        "RLS and regularized source replay are lightweight analogues, not the exact OS-ELM / ECM-UKF / transformer TTT architectures.",
        "All coefficients and thresholds predeclared; max 4 triggers (except stream RLS).",
        "",
    ]
    for track,g in agg.groupby("track"):
        lines.extend([
            f"## N20 SOC initialization = {track}",
            "",
            "| Voltage adaptation | SOC correction | Voltage MAE (mV) | Voltage Δ vs Frozen (mV) | SOC-proxy MAE (pp) | SOC Δ vs Frozen (pp) | Improved voltage / SOC cells | Attempts / accepted |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for x in g.sort_values(["soc_voltage_normalization_strength","median_soc_proxy_pp"]).itertuples():
            lines.append(
                f"| {x.arm} | {x.soc_voltage_normalization_strength:.1f} | "
                f"{x.median_voltage_mV:.2f} | {x.delta_voltage_mV:+.2f} | "
                f"{x.median_soc_proxy_pp:.3f} | {x.delta_soc_pp:+.3f} | "
                f"{x.improved_voltage_cells}/{x.cells} / {x.improved_soc_cells}/{x.cells} | "
                f"{x.median_attempts:.0f}/{x.median_accepted:.0f} |"
            )
        lines.append("")
    lines += [
        "## Paired cell-bootstrap uncertainty (mean change relative to frozen)",
        "",
        "| N20 labels | Voltage model | SOC correction | SOC mean delta & 95% CI (pp) | Voltage mean delta & 95% CI (mV) |",
        "|---|---|---:|---|---|",
    ]
    for x in agg[agg.soc_voltage_normalization_strength.eq(.5)].sort_values(
        ["track","delta_soc_pp"]
    ).itertuples():
        lines.append(
            f"| {x.track} | {x.arm} | 0.5 | "
            f"{x.delta_soc_pp:+.3f} [{x.soc_ci_lo:+.3f},{x.soc_ci_hi:+.3f}] | "
            f"{x.delta_voltage_mV:+.2f} [{x.voltage_ci_lo:+.2f},{x.voltage_ci_hi:+.2f}] |"
        )
    lines += [
        "## Online trigger versus fixed periodic (same How and nominal B=4)",
        "",
        "These comparisons condition on *exactly matched four accepted updates* per physical cell;",
        "fewer valid cells are evaluated if either policy does not complete four.",
        "Per-cell bootstrap intervals compare the *triggering policy minus periodic*.",
        "",
        "| N20 SOC labels | Trigger | Matched cells | Mean delta to periodic (pp) | 95% paired CI (pp) | Better cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for track,g in by_cell[
        by_cell.soc_voltage_normalization_strength.eq(.5)
    ].groupby("track"):
        base=g[(g.arm=="periodic_replay")&(g.accepted.eq(B))].set_index("cell_id")
        for alt in ["threshold_replay","ph_replay","cusum_replay","guarded_replay"]:
            sub=g[(g.arm==alt)&(g.accepted.eq(B))].set_index("cell_id")
            common=base.index.intersection(sub.index)
            if len(common)<5: continue
            diff=(
                sub.loc[common,"SOC_proxy_MAE_pp"]-base.loc[common,"SOC_proxy_MAE_pp"]
            ).to_numpy()
            rng=np.random.default_rng(SEED)
            draws=rng.choice(diff,size=(5000,len(diff)),replace=True).mean(axis=1)
            lo,hi=np.quantile(draws,[.025,.975])
            lines.append(
                f"| {track} | {alt} | {len(diff)} | {diff.mean():+.3f} | "
                f"[{lo:+.3f},{hi:+.3f}] | {(diff<0).sum()}/{len(diff)} |"
            )
    lines+= [
        "## Interpretation",
        "- An improved voltage model need not improve SOC: correcting a Vmin input to its early-regime reference may delete useful aging information.",
        "- Voltage is directly measured at the same phase end and Vmax from that phase is a feature; this is observable auxiliary Vmin RECONSTRUCTION, not forecasting a future unobserved voltage.",
        "- Voltage observation is non-RPT and available in operation; SOC reference remains charge counting, with past RPT used only in offline scoring.",
        "- Stream RLS has substantially more updates and must NOT be compared as budget-equivalent to four-shot methods.",
        "- PH/CUSUM, voltage threshold and periodic arms may use different update counts; compare both accuracy and count.",
        "- If voltage identification improves but SOC does not, learn a jointly trained voltage reconstruction + SOC representation rather than post-hoc feature subtraction.",
        "- These are phase-end experiments, not continuously updated ECM states or a replicated adaptive Kalman filter.",
    ]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(DATA)
    rows=[]
    for cell_id,cell in data.groupby("cell_id",sort=True):
        for track in ["SOC_proxy","SOC_nominal_3Ah"]:
            info=_prepare(cell,track)
            for arm in ARM_SPECS:
                for correction in SOC_CORRECTION_STRENGTHS:
                    rows.append(one_run(info,track,arm,correction,cell_id))
    frame=pd.DataFrame(rows)
    assert frame.cell_id.nunique()==22
    assert frame.groupby(["cell_id","track"]).size().eq(len(ARM_SPECS)*2).all()
    percell,summary=summarize(frame)
    percell.to_csv(out/"online_voltage_by_cell.csv",index=False)
    summary.to_csv(out/"online_voltage_summary.csv",index=False)
    summary_text=report(summary,percell)
    (out/"report.md").write_text(summary_text,encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "no_RPT_online":True,
        "available_online":"completed mission V/I/T and measured Vmin",
        "SOC_proxy_scoring_only":True,
        "initial_SOC_tracks":["SOC_proxy","SOC_nominal_3Ah"],
        "predict_then_update":True,
        "voltage_measurement_is_independent_from_SOC_target":True,
        "predeclared_policies":ARM_SPECS,
        "source_N20":N20,
        "fixed_budget":B,
        "physical_cells":int(frame.cell_id.nunique())
    },indent=2,ensure_ascii=False),encoding="utf-8")
    print(summary_text)


if __name__=="__main__":
    main()
