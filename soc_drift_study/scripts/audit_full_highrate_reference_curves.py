"""Raw eVTOL high-rate SOC-proxy *curve* feasibility audit for one LFS cell.

Read-only audit for VAH01 raw Arbin-style logs. Match three complete normal
mission cycles' Ns=4/5/6 high-rate windows at early/middle/late lifetime to
existing derived phase-end labels. No new labels are created or published as
independent SOC ground truth.

The raw file contains V, I, T, time, QDischarge_mA_h. Thus full samplewise
Coulomb-count *proxies* can be reconstructed if time and counter are valid.
Aging-capacity anchor still originates from preceding RPT candidate; no
online calibration mechanism supplies such capacity during live operations.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent
RAW_FIELDS=["time_s","Ecell_V","I_mA","QDischarge_mA_h","Temperature__C","Ns"]


def select_windows(cell_id):
    seg=pd.read_csv(STUDY/"results"/"segment_summary.csv")
    lab=pd.read_csv(STUDY/"results"/"soc_proxy_phase_labels.csv")
    seg=seg[seg.cell_id.eq(cell_id)&seg.Ns.isin([4,5,6])].copy()
    lab=lab[lab.cell_id.eq(cell_id)].copy()
    complete=[]
    for cy,g in lab.groupby("charge_cycle_index",sort=True):
        if len(g)!=3 or set(g.phase_Ns)!={4,5,6}:continue
        if not np.isfinite(g.capacity_used_Ah).all():continue
        matched=seg[seg.charge_cycle_index.eq(cy)]
        if matched.Ns.nunique()!=3:continue
        periods=[]
        for _,r in g.iterrows():
            ms=matched[matched.Ns.eq(r.phase_Ns)]
            if len(ms)!=1:break
            s=ms.iloc[0]
            periods.append({
                "cell_id":cell_id,"cycle":int(cy),"phase":int(r.phase_Ns),
                "begin":float(s.time_start_s),"end":float(s.time_end_s),
                "expected_rows":int(s.rows),
                "expected_q_end_Ah":float(r.QDischarge_end_Ah),
                "capacity_ah":float(r.capacity_used_Ah),
                "expected_end_soc_proxy":float(r.SOC_proxy),
                "capacity_source":r.capacity_source,
            })
        if len(periods)==3:complete.append(periods)
    if not complete:raise ValueError("No complete labeled normal missions")
    pick=sorted(set([0,len(complete)//2,len(complete)-1]))
    return [win for ix in pick for win in complete[ix]],len(complete)


def stream_extract(raw_file,windows):
    # Explicit time interval, Ns and cycle-window matching; this audit
    # depends on non-overlapping/monotonic cell time_s and validates that.
    tables={i:[] for i in range(len(windows))}
    nrows=0;last_time=None;bad_dt=0;time_reversals=0
    for chunk in pd.read_csv(raw_file,usecols=RAW_FIELDS,chunksize=100_000):
        nrows+=len(chunk)
        t=chunk.time_s.to_numpy(dtype=float)
        if last_time is not None and len(t) and t[0]<=last_time:
            time_reversals+=int(t[0]<last_time)
            bad_dt+=int(t[0]<=last_time)
        bad_dt+=int((np.diff(t)<=0).sum())
        if len(t):last_time=float(t[-1])
        for i,w in enumerate(windows):
            sel=chunk.loc[
                chunk.time_s.between(w["begin"],w["end"],inclusive="both")
                &chunk.Ns.eq(w["phase"]),
                RAW_FIELDS
            ]
            if len(sel):tables[i].append(sel.copy())
    return {
        i:pd.concat(tables[i],ignore_index=True) if tables[i] else pd.DataFrame(columns=RAW_FIELDS)
        for i in tables
    },{"raw_total_rows":nrows,"nonpositive_dt_count":bad_dt,
        "interchunk_backward_time_count":time_reversals}


def inspect(curves,windows,out):
    summary=[];preview=[]
    for i,w in enumerate(windows):
        frame=curves[i].sort_values("time_s")
        n=len(frame)
        cap=w["capacity_ah"]
        if not n:
            summary.append({**w,"raw_rows":0,"phase_valid":False,"reason":"empty_match"})
            continue
        v=frame.Ecell_V.to_numpy(dtype=float)
        I=frame.I_mA.to_numpy(dtype=float)
        q=frame.QDischarge_mA_h.to_numpy(dtype=float)/1000
        temp=frame.Temperature__C.to_numpy(dtype=float)
        t=frame.time_s.to_numpy(dtype=float)
        soc=np.clip(1-q/cap,0,1)
        dtime=np.diff(t); dq=np.diff(q)
        endpoint=float(soc[np.argmax(q)])
        mismatch=100*abs(endpoint-w["expected_end_soc_proxy"])
        # The dataset logger varies sampling cadence with lifecycle:
        # early phases ~1 sec, later phases ~10 sec. A minimum of 20 rows
        # would falsely reject legitimate 7–12-sample takeoff/landing phases.
        median_dt=float(np.median(dtime)) if len(dtime) else float("nan")
        run_valid=bool(
            n>=5 and np.isfinite(soc).all()
            and np.isfinite(v).all() and np.isfinite(I).all()
            and len(dtime)>0 and (dtime>0).mean()>.98
            and median_dt<=30.
            and mismatch<0.55
            and abs(float(np.nanmax(q))-w["expected_q_end_Ah"])<.02
        )
        summary.append({
            **w,"raw_rows":n,
            "observed_duration_s":float(t[-1]-t[0]),
            "phase_valid":run_valid,
            "SOC_start_proxy":float(soc[0]),
            "SOC_end_proxy":endpoint,
            "end_mismatch_pp":mismatch,
            "max_counter_difference_Ah":float(np.nanmax(q)-w["expected_q_end_Ah"]),
            "negative_Q_step_count":int((dq<-.005).sum()),
            "nonpositive_time_steps":int((dtime<=0).sum()),
            "gaps_over_10s":int((dtime>10).sum()),
            "gaps_over_3x_cadence":int((dtime>3*median_dt).sum()) if np.isfinite(median_dt) else 0,
            "median_sampling_s":median_dt,
            "mean_abs_current_A":float(np.nanmean(np.abs(I))/1000),
            "Vmin_V":float(np.nanmin(v)),"Vmax_V":float(np.nanmax(v)),
            "Tmin_C":float(np.nanmin(temp)),"Tmax_C":float(np.nanmax(temp)),
            "Q_end_Ah":float(np.nanmax(q)),
            "reason":"pass" if run_valid else "incomplete_or_mismatch",
        })
        for j in np.linspace(0,n-1,min(80,n),dtype=int):
            preview.append({
                "cycle":w["cycle"],"phase":w["phase"],"time_s":float(t[j]),
                "V":v[j],"I_A":I[j]/1000,"T_C":temp[j],
                "SOC_proxy_raw":float(soc[j]),"QDischarge_Ah":q[j]
            })
    rows=pd.DataFrame(summary)
    rows.to_csv(out/"raw_mission_phase_curve_checks.csv",index=False)
    pd.DataFrame(preview).to_csv(out/"raw_curve_thinned_PREVIEW.csv",index=False)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw-file",type=Path,default=ROOT/"VAH01.csv")
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    path=args.raw_file.resolve()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    header=path.open("rb").read(120)
    if b"git-lfs.github.com/spec/v1" in header:
        raise RuntimeError("Raw VAH01.csv is an unmaterialized LFS POINTER, not test data")
    cell_id=path.stem
    win,complete=select_windows(cell_id)
    curves,meta=stream_extract(path,win)
    summary=inspect(curves,win,out)
    meta.update({
        "cell_id":cell_id,"complete_missions_derived":complete,
        "selected_lifetime_missions":3,"selected_high_rate_phases":len(win),
        "passing_phase_windows":int(summary.phase_valid.sum()),
        "reference_kind":"historic RPT-capacity-anchored CC SOC_proxy; NOT independent true SOC",
        "measured_field_cost":False,
    })
    (out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=[
        "# Raw samplewise high-rate flight SOC-proxy curve feasibility audit",
        "",
        f"- Physical cell: {cell_id}; streaming rows read: {meta['raw_total_rows']:,}; {complete} derived complete 3-phase missions.",
        f"- Selected 3 early/middle/late missions ({len(win)} high-rate phase windows); {summary.phase_valid.sum()}/{len(win)} windows match phase-end Coulomb SOC proxy.",
        f"- Raw time nonpositive steps: {meta['nonpositive_dt_count']}; backward chunk times: {meta['interchunk_backward_time_count']}.",
        "- Raw logger sampling cadence can change from ~1s early to ~10s in later life; quality criteria are resolution-aware. This is a *samplewise* complete-available-measurement curve, NOT uniformly 1 Hz across the full life.",
        "- Full measured V/I/T and cumulative discharged ampere-hours exist at each raw sample. If window joins and sampling continuity pass, a samplewise **SOC_proxy curve** reconstruction is feasible.",
        "- Such a curve is STILL NOT an independently measured true-SOC label: it needs a historical prior-RPT capacity anchor and a valid initial SOC/full-charge condition.",
        "- A field drone maintaining its BMS must obtain independent/credible SOC reference with a cost that is not present in this dataset. Do not silently call prior RPT denominator free.",
        "",
        "| Cycle | Phase | Raw samples | dt median (s) | End SOC delta vs derived (pp) | Time gaps >10 s | Quality |",
        "|---:|---:|---:|---:|---:|---:|---|"
    ]
    for v in summary.itertuples():
        lines.append(f"| {v.cycle} | {v.phase} | {v.raw_rows} | "
                     f"{v.median_sampling_s if hasattr(v,'median_sampling_s') else np.nan:.3f} | "
                     f"{v.end_mismatch_pp if hasattr(v,'end_mismatch_pp') else np.nan:.4f} | "
                     f"{v.gaps_over_10s if hasattr(v,'gaps_over_10s') else 0} | "
                     f"{v.reason} |")
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print((out/"report.md").read_text(encoding="utf-8"),flush=True)
    if summary.phase_valid.sum()<6:
        raise AssertionError("Fewer than 6/9 mission phases pass raw-vs-derived audit")


if __name__=="__main__":
    main()
