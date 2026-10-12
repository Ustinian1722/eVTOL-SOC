"""Dense NORMAL high-rate flight SOC-proxy *purchased calibration* WHEN benchmark.

- Uses actual raw time-series at its NATIVE sampling rate (often 1s early,
  ~10s later), never invents missing 1Hz measurements.
- Frozen same-cell Ridge trained on first N=20 complete normal missions;
  causal measured V/I/T, nominal-3Ah within-flight Q, elapsed time, Ns.
- Reference-broker purchase: one whole completed normal Ns4/5/6 mission's
  Coulomb-counted SOC_proxy, using a historical PREVIOUS RPT capacity anchor.
  This is an OFFLINE RETROSPECTIVE REFERENCE, not real independently measured
  SOC or a field SOC acquisition mechanism. No RPT waveform used to calibrate.
- One fixed supervised regularized affine head (bounded a,b) fitted to EITHER
  3 phase-end points, 16 uniform native samples or *all* recorded samples,
  then held fixed until next paid maintenance. NO target SOC leakage into WHEN.
- Time order is prequential: predict mission t before buying t's labels;
  updated head starts at t+1. Offline clairvoyant oracle is a bound only.
- This first pilot evaluates physical cells via explicit --cells selection.
  Never claim results generalize to all 22 based on a small selected pilot.
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

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent
RESULTS=STUDY/"results"
N20=20
MIN_FUTURE=20
GRID_START=24
GRID_STRIDE=25
BUDGETS=(0,2,4)
DENSITIES=("endpoints3","uniform16","native_all")
POLICIES=("frozen","periodic_planned","periodic150","random",
          "disagreement3","voltage3","joint3","offline_oracle")
COSTS=(0.0,0.1,0.3)
FIELDS=["time_s","Ecell_V","I_mA","Temperature__C",
        "QDischarge_mA_h","Ns"]
REQUIRED_LABELS={"cell_id","charge_cycle_index","phase_Ns",
                 "capacity_used_Ah","SOC_proxy","QDischarge_end_Ah"}
SEED=20261012


def robust(x):
    z=np.asarray(x,float)
    med=float(np.median(z))
    return med,max(float(np.median(np.abs(z-med)))*1.4826,
                   float(np.std(z))*.20,1e-4)


def select_windows(cellid):
    lab=pd.read_csv(RESULTS/"soc_proxy_phase_labels.csv")
    seg=pd.read_csv(RESULTS/"segment_summary.csv")
    if not REQUIRED_LABELS.issubset(lab.columns):
        raise AssertionError("Missing pre-existing SOC reference columns")
    lab=lab[lab.cell_id.eq(cellid)].copy()
    seg=seg[seg.cell_id.eq(cellid)&seg.Ns.isin([4,5,6])
            &seg.I_signed_mean_mA.le(-2000)&seg.duration_s.ge(10)].copy()
    keys=["charge_cycle_index","phase_Ns"]
    seg=seg.rename(columns={"Ns":"phase_Ns"})
    nseg=seg.groupby(keys).size()
    valid_key=set(nseg[nseg.eq(1)].index.tolist())
    rows=[]
    for cycle,g in lab.groupby("charge_cycle_index",sort=True):
        if set(g.phase_Ns)!={4,5,6} or len(g)!=3:continue
        if any((cycle,int(ns)) not in valid_key for ns in [4,5,6]):continue
        parts=[]
        for r in g.itertuples():
            s=seg[(seg.charge_cycle_index.eq(cycle))&
                  (seg.phase_Ns.eq(r.phase_Ns))].iloc[0]
            parts.append({
                "cell_id":cellid,"cycle_id":int(cycle),"ns":int(r.phase_Ns),
                "begin":float(s.time_start_s),"end":float(s.time_end_s),
                "expected_rows":int(s.rows),
                "expected_end_q":float(r.QDischarge_end_Ah),
                "capacity":float(r.capacity_used_Ah),
                "expected_end_soc":float(r.SOC_proxy),
                "capacity_source":str(r.capacity_source),
            })
        if all(np.isfinite(p["capacity"]) and p["capacity"]>0 for p in parts):
            rows.extend(parts)
    windows=pd.DataFrame(rows)
    if windows.empty:raise ValueError(f"no normal high-rate windows for {cellid}")
    windows=windows.sort_values(["begin","end"]).reset_index(drop=True)
    if not np.all(np.diff(windows.begin.to_numpy())>=0):
        raise AssertionError("segment starts not sorted")
    return windows


def stream_raw_missions(path,windows):
    """O(raw_rows*log(windows)) matching; no O(raw_rows*windows) scan."""
    header=path.open("rb").read(130)
    if b"git-lfs.github.com/spec/v1" in header:
        raise RuntimeError(f"{path} is unresolved LFS pointer")
    starts=windows.begin.to_numpy(float)
    ends=windows.end.to_numpy(float)
    nslist=windows.ns.to_numpy(int)
    allocated={i:[] for i in range(len(windows))}
    stats={"read_rows":0,"matched_rows":0,"bad_time_deltas":0}
    last=None
    for chunk in pd.read_csv(path,usecols=FIELDS,chunksize=200_000):
        t=chunk.time_s.to_numpy(float)
        ns=chunk.Ns.to_numpy(float)
        if len(t):
            stats["bad_time_deltas"]+=int(np.sum(np.diff(t)<=0))
            if last is not None:stats["bad_time_deltas"]+=int(t[0]<=last)
            last=t[-1]
        stats["read_rows"]+=len(chunk)
        candidates=np.searchsorted(starts,t,side="right")-1
        valid=candidates>=0
        idx=candidates.clip(0,len(starts)-1)
        valid &= t<=ends[idx]+1e-6
        valid &= ns==nslist[idx]
        if not np.any(valid):continue
        cand=candidates[valid]
        vals=chunk.loc[valid,FIELDS].to_numpy(dtype=np.float64)
        stats["matched_rows"]+=len(vals)
        # Stable grouping within chunk, preserving raw time ordering.
        order=np.argsort(cand,kind="stable")
        sorted_ids=cand[order]
        begin=np.r_[0,np.flatnonzero(np.diff(sorted_ids))+1]
        finish=np.r_[begin[1:],len(order)]
        for b,e in zip(begin,finish):
            k=int(sorted_ids[b])
            allocated[k].append(vals[order[b:e]])
    return allocated,stats


@dataclass
class Mission:
    cycle:int
    X:np.ndarray
    y:np.ndarray
    qnom:np.ndarray
    ns:np.ndarray
    duration:float
    source_capacity:float
    raw_rows:int
    predicted:np.ndarray|None=None
    voltage_resid:np.ndarray|None=None

    @property
    def n(self):return len(self.y)


def make_missions(windows,allocated):
    assembled=[]
    bad=[]
    for cid,g in windows.groupby("cycle_id",sort=True):
        items=[]
        for i,w in g.iterrows():
            pieces=allocated[i]
            if not pieces:
                bad.append([cid,w.ns,"no_raw_match"]);break
            rows=np.concatenate(pieces,axis=0)
            rows=rows[np.argsort(rows[:,0],kind="stable")]
            q=rows[:,FIELDS.index("QDischarge_mA_h")]/1000
            endpoint=1-np.max(q)/w.capacity
            n=len(rows)
            # Native logger cadence can be ~10s and takeoff stages <10 rows;
            # no arbitrary minimum of 20 samples per phase.
            ok=(n>=5 and (np.diff(rows[:,0])>0).mean()>.98
                and abs(np.max(q)-w.expected_end_q)<.02
                and abs(np.clip(endpoint,0,1)-w.expected_end_soc)<.0055
                and np.isfinite(rows).all())
            if not ok:
                bad.append([cid,w.ns,f"quality,raw={n},expected={w.expected_rows}"])
                break
            items.append((int(w.ns),rows,float(w.capacity)))
        if len(items)!=3:continue
        items=sorted(items,key=lambda x:x[0])
        # Every calibrated curve is the entire actually sampled
        # high-power mission (3 phases), not manufactured 1Hz labels.
        allrows=np.concatenate([t[1] for t in items])
        Q=allrows[:,FIELDS.index("QDischarge_mA_h")]/1000
        capacity=np.array([t[2] for t in items])
        if np.ptp(capacity)>1e-6:continue
        V=allrows[:,FIELDS.index("Ecell_V")]
        I=allrows[:,FIELDS.index("I_mA")]/1000
        T=allrows[:,FIELDS.index("Temperature__C")]
        ns=allrows[:,FIELDS.index("Ns")].astype(int)
        sec=allrows[:,FIELDS.index("time_s")]
        sec0=sec[0]
        qnom=np.clip(1-Q/3.,0,1)
        # Sensor predictors are available causally within phase. The
        # nominal 3Ah integral is intentionally explicit to permit an
        # age/capacity-drift baseline; report this label dependence.
        X=np.column_stack([
            V,I,T,qnom,np.clip((sec-sec0)/60,0,200),
            (ns==4).astype(float),(ns==5).astype(float),(ns==6).astype(float),
        ])
        y=np.clip(1-Q/capacity[0],0,1)
        assembled.append(Mission(int(cid),X,y,qnom,ns,
                          float(sec[-1]-sec0),capacity[0],len(y)))
    return sorted(assembled,key=lambda x:x.cycle),bad


def select_reference_rows(mission,density):
    if density=="native_all":return np.arange(mission.n)
    if density=="uniform16":return np.unique(np.linspace(0,mission.n-1,
                                            min(16,mission.n),dtype=int))
    if density=="endpoints3":
        return np.array([np.flatnonzero(mission.ns==ph)[-1] for ph in (4,5,6)])
    raise ValueError(density)


def fit_calibration(base,label):
    """Same bounded/regularized HOW for all WHEN arms and all densities.
    Normalize per-curve objective so native 1Hz vs 10s cadence does not
    multiply effective supervision merely because the logger ran faster.
    """
    x=np.asarray(base,float);y=np.asarray(label,float)
    if len(x)<3:raise ValueError("Not enough normal-task reference samples")
    A=np.c_[x,np.ones(len(x))]
    prior=np.diag([.10,.02])
    theta=np.linalg.solve(A.T@A/len(x)+prior,
                  A.T@y/len(x)+prior@np.array([1.,0.]))
    return (float(np.clip(theta[0],.70,1.30)),
            float(np.clip(theta[1],-.15,.15)))


def frozen_training(missions):
    if len(missions)<=N20+MIN_FUTURE+GRID_START:
        raise ValueError("Cell lacks sufficient complete high-rate missions")
    source=missions[:N20];test=missions[N20:]
    # Every initial mission receives equal contribution in the source
    # supervised training regardless of original logger's sampling cadence.
    train_x=[];train_y=[]
    for item in source:
        ix=np.unique(np.linspace(0,item.n-1,min(64,item.n),dtype=int))
        train_x.append(item.X[ix]);train_y.append(item.y[ix])
    base=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    base.fit(np.vstack(train_x),np.concatenate(train_y))
    # Measured voltage auxiliary residual, trained without future labels.
    vm=make_pipeline(StandardScaler(),Ridge(alpha=10.))
    vix=[0,1,2,3,4,5,6,7]
    # Predict V from I,T,nominal Q,elapsed,Ns; exclude measured V itself.
    vf=[1,2,3,4,5,6,7]
    vm.fit(np.vstack([x[:,vf] for x in train_x]),
           np.concatenate([x[:,0] for x in train_x]))
    for m in missions:
        m.predicted=np.clip(base.predict(m.X),0,1)
        m.voltage_resid=np.abs(vm.predict(m.X[:,vf])-m.X[:,0])
    return source,test


def sensor_risks(source,test):
    """NO SOC_proxy input. Sensor array is the only input to WHEN."""
    d_source=[float(np.mean(np.abs(m.predicted-m.qnom))) for m in source]
    v_source=[float(np.mean(m.voltage_resid)) for m in source]
    d0,ds=robust(d_source);v0,vs=robust(v_source)
    return {
        "disagreement_z":np.array([(np.mean(np.abs(m.predicted-m.qnom))-d0)/ds for m in test]),
        "voltage_z":np.array([(np.mean(m.voltage_resid)-v0)/vs for m in test]),
        "mission_duration":np.array([m.duration for m in test]),
    }


def opportunities_for(n):
    return list(range(GRID_START,n-MIN_FUTURE,GRID_STRIDE))


def choose_when(sensors,opportunities,n,budget,policy,cid):
    if not budget or policy=="frozen":return []
    cand=list(opportunities)
    if not cand:return []
    if policy=="periodic_planned":
        xx=np.rint(np.linspace(0,len(cand)-1,min(budget,len(cand))+2)[1:-1]).astype(int)
        return sorted(set(cand[k] for k in xx))
    if policy=="random":
        salt=int(hashlib.sha256(cid.encode()).hexdigest()[:8],16)
        return sorted(np.random.default_rng(SEED+salt).choice(
            cand,size=min(budget,len(cand)),replace=False).tolist())
    dz=sensors["disagreement_z"]
    vz=sensors["voltage_z"]
    selected=[];dcount=vcount=0
    last=-9999
    for i in range(n):
        dcount=dcount+1 if dz[i]>2.5 else 0
        vcount=vcount+1 if vz[i]>2.5 else 0
        if i not in cand or len(selected)>=budget or i-last<50:continue
        if policy=="periodic150":fire=(i+1)%150==0
        elif policy=="disagreement3":fire=dcount>=3
        elif policy=="voltage3":fire=vcount>=3
        elif policy=="joint3":fire=(dcount>=3 and vcount>=3)
        else:raise ValueError(policy)
        if fire:
            selected.append(i);last=i;dcount=vcount=0
    return selected


def per_mission_errors(test,params):
    a,b=params
    return np.array([np.mean(np.abs(
          np.clip(a*m.predicted+b,0,1)-m.y))*100 for m in test])


def build_broker(test,options,density):
    # OFFLINE vault: scores all counterfactual actions to compute true
    # hindsight Oracle. Online detection never sees bank or test.y.
    bank={};losses={}
    for e in options:
        m=test[e]
        ix=select_reference_rows(m,density)
        pars=fit_calibration(m.predicted[ix],m.y[ix])
        bank[e]=(pars,int(len(ix)))
        losses[e]=per_mission_errors(test,pars)
    return bank,losses


def replay(base_loss,event_losses,chosen):
    curve=base_loss.copy()
    for j,e in enumerate(chosen):
        end=chosen[j+1] if j+1<len(chosen) else len(curve)
        curve[e+1:end+1]=event_losses[e][e+1:end+1]
    return curve


def optimal_hindsight(base_loss,event_losses,opportunities,budget,price=0.):
    """Exact minimum lifecycle mean MAE + price*(number of buys).
    An oracle uses FUTURE scoring labels; NEVER a deployable trigger.
    """
    n=len(base_loss)
    starts=[0]+[e+1 for e in opportunities]
    ids=[None]+list(opportunities)
    nnode=len(ids)
    pref=[np.r_[0,np.cumsum(base_loss)]]
    pref.extend([np.r_[0,np.cumsum(event_losses[e])] for e in opportunities])
    dp=np.full((budget+1,nnode),np.inf)
    paths={(0,0):()}
    dp[0,0]=0.
    for k in range(1,budget+1):
        for j in range(1,nnode):
            for i in range(j):
                if not np.isfinite(dp[k-1,i]):continue
                # Existing model i handles mission up to the newly bought
                # event j, inclusive; new model starts next mission.
                carry=pref[i][starts[j]]-pref[i][starts[i]]
                cand=dp[k-1,i]+carry+price*n
                if cand<dp[k,j]:
                    dp[k,j]=cand;paths[(k,j)]=paths[(k-1,i)]+(ids[j],)
    options=[(float(pref[0][-1]),())]
    for k in range(1,budget+1):
        for j in range(1,nnode):
            if np.isfinite(dp[k,j]):
                obj=dp[k,j]+pref[j][-1]-pref[j][starts[j]]
                options.append((float(obj),paths[(k,j)]))
    objective,choice=min(options,key=lambda z:z[0])
    check=float(replay(base_loss,event_losses,choice).mean()+price*len(choice))
    if abs(check-objective/n)>1e-7:
        raise AssertionError(f"DP mismatch {check} vs {objective/n}")
    return list(choice)


def compare_cell(cid,missions,bad,density):
    source,test=frozen_training(missions)
    n=len(test);ops=opportunities_for(n)
    sensors=sensor_risks(source,test)
    baseline=per_mission_errors(test,(1.,0.))
    bank,losses=build_broker(test,ops,density)
    rows=[];events=[]
    for budget in BUDGETS:
        for price in COSTS:
            for policy in POLICIES:
                if policy=="offline_oracle":
                    chosen=optimal_hindsight(baseline,losses,ops,budget,price)
                else:
                    chosen=choose_when(sensors,ops,n,budget,policy,cid)
                err=replay(baseline,losses,chosen)
                bought=sum(bank[e][1] for e in chosen)
                result={
                    "cell_id":cid,"density":density,"budget":budget,"price":price,
                    "policy":policy,"missions":n,"MAE_pp":float(err.mean()),
                    "late_MAE_pp":float(err[int(.75*n):].mean()),
                    "over3pp_rate":float((err>3).mean()),
                    "maintenance_events":len(chosen),
                    "labels_bought":bought,
                    "total_J_eventcost":float(err.mean()+price*len(chosen)),
                    "event_test_indices":";".join(map(str,chosen)),
                }
                rows.append(result)
                for e in chosen:
                    events.append({
                        "cell_id":cid,"density":density,"budget":budget,"price":price,
                        "policy":policy,"event_test_mission_idx":e,
                        "event_original_cycle":test[e].cycle,
                        "first_corrected_idx":e+1,
                        "raw_reference_points":test[e].n,
                        "labels_purchased":bank[e][1],
                        "calibration_a":bank[e][0][0],
                        "calibration_b":bank[e][0][1],
                    })
    return rows,events,{
        "cell_id":cid,"n_accepted_normal_missions":len(missions),
        "n_raw_quality_rejections":len(bad),
        "n_test_missions":n,"median_raw_points":float(np.median([m.n for m in test])),
        "N20_training_missions":len(source),
        "available_reference_grid":len(ops),
        "frozen_dense_MAE_pp":float(baseline.mean()),
    }


def all_data(cells,out,skip_missing=False):
    stage=[]
    for cid in cells:
        raw=ROOT/f"{cid}.csv"
        if not raw.exists():
            if skip_missing:continue
            raise FileNotFoundError(raw)
        start=time.perf_counter()
        windows=select_windows(cid)
        samples,stream=stream_raw_missions(raw,windows)
        missions,bad=make_missions(windows,samples)
        if len(missions)<N20+GRID_START+MIN_FUTURE:
            if skip_missing:continue
            raise AssertionError(f"{cid} not enough valid flights: {len(missions)}")
        outrows,outevents,meta=[],[],None
        for density in DENSITIES:
            x,e,metadata=compare_cell(cid,missions,bad,density)
            outrows.extend(x);outevents.extend(e);meta=metadata
        meta.update(stream);meta["wall_s"]=time.perf_counter()-start
        stage.append((outrows,outevents,meta))
        print(f"OK {cid}: {len(missions)} normal missions; {meta['frozen_dense_MAE_pp']:.3f}pp frozen, "
              f"{meta['median_raw_points']:.0f} raw points median, elapsed={meta['wall_s']:.1f}s",flush=True)
        del samples,missions
    if not stage:raise AssertionError("No valid raw-cell studies completed")
    results=pd.DataFrame([r for rows,_,_ in stage for r in rows])
    events=pd.DataFrame([r for _,rows,_ in stage for r in rows])
    manifest=pd.DataFrame([m for _,_,m in stage])
    return results,events,manifest


def report(results,manifest):
    lines=[
        "# Native full-flight high-rate SOC_proxy purchased calibration: WHOLE-CURVE WHEN pilot",
        "",
        "- PHYSICAL CELLS: "+", ".join(manifest.cell_id)+".",
        "- Each purchased reference is one complete recorded normal high-rate Ns4/5/6 mission. This is a HYPOTHETICAL historically escrowed Coulomb-count SOC_proxy curve, not independent true SOC/field labeling.",
        "- All methods have the same fixed initial same-cell N20 Ridge trained from normal mission V/I/T and NOMINAL 3Ah coulomb count, not reference test waveforms.",
        "- All WHEN detectors see only normal measured signals/frozen predictions; the bought labels enter the fixed regularized two-parameter affine calibration ONLY after event prediction.",
        "- Three point, sixteen native sample, and all native sample maintenance use the SAME curve, with varying label subsample number. Native sample count varies over life due to 1->10s downsampling.",
        "- Cost λ per EVENT is hypothetical MAE-pp-equivalent, not actual field cost. Labels bought is also reported. The Oracle uses future hidden labels deliberately and is not deployable.",
        "- All cells selected for pilot were previously explored and are NOT new external test cells.",
        "",
        "## Data and model quality",
        "",
        "| Cell | Usable complete missions | Online held missions | Median raw samples/mission | Frozen dense SOC_proxy MAE (pp) | Raw phase rejects |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for x in manifest.itertuples():
        lines.append(f"| {x.cell_id} | {x.n_accepted_normal_missions} | {x.n_test_missions} | {x.median_raw_points:.0f} | {x.frozen_dense_MAE_pp:.3f} | {x.n_raw_quality_rejections} |")
    lines+=["","## MAX 4 purchases, λ=.1 cost per task (per selected cell, then median)",
    "","| Reference labels | WHEN policy | Median SOC proxy MAE (pp) | Mean cost objective J | Median events | Median labels bought |",
    "|---|---|---:|---:|---:|---:|"]
    for density in DENSITIES:
        v=results[(results.density==density)&(results.budget==4)&(np.isclose(results.price,.1))]
        for policy,g in v.groupby("policy"):
            lines.append(f"| {density} | {policy} | {g.MAE_pp.median():.3f} | {g.total_J_eventcost.mean():.3f} | {g.maintenance_events.median():.0f} | {g.labels_bought.median():.0f} |")
    lines+=[
        "",
        "## Decision",
        "- First test whether fixed supervised WHOLE-CURVE calibration improves Frozen for the majority of cells. If not, no credible WHEN paper result follows.",
        "- Even with positive WHEN result, compare strict fixed-horizon-blind periodic150 and explicitly planned periodic; the latter knows retrospective lifespan.",
        "- An Oracle advantage demonstrates theoretical headroom only; a positive implementable policy requires cell-block heldout comparison with uncertainty.",
        "- The nominal capacity integral is an explicit feature AND historic target proxy uses prior-RPT capacity; model may predominantly identify changing effective capacity rather than independently recover true SOC. Interpret claims accordingly.",
    ]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cells",nargs="+",default=["VAH01","VAH02","VAH17","VAH27"])
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    results,events,manifest=all_data(args.cells,out)
    results.to_csv(out/"dense_curve_supervised_WHEN_cell_policy.csv",index=False)
    events.to_csv(out/"reference_release_times_and_sample_cost.csv",index=False)
    manifest.to_csv(out/"dense_curve_raw_quality_manifest.csv",index=False)
    metadata={
        "purpose":"complete native sampled normal high-power flight reference supervised calibration WHEN",
        "cells":args.cells,
        "source_raw_not_modified":True,
        "online_labels_only_if_reference_purchased":True,
        "base_source_training":"same physical cell first20 complete normal missions",
        "baseline_features":"observed voltage/current/temperature/nominal Coulomb-integral/time/Ns",
        "labels":"historic prior-RPT-capacity Coulomb count SOC_proxy; NOT independent true SOC",
        "calibration_reference":"one normal high-rate full flight actual logged samplewise task",
        "calibration_densities":DENSITIES,
        "costs_hypothetical":COSTS,
        "reference_event_cost_not_normalized_to_label_count":True,
        "oracle_clairvoyant_only":True,
        "reference_broker_precomputed_labels_offline":"only scorer and fixed post-purchase calibrator receive the label bank",
    }
    (out/"metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    text=report(results,manifest)
    (out/"report.md").write_text(text,encoding="utf-8")
    print(text,flush=True)


if __name__=="__main__":
    main()
