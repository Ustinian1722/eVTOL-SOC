"""All-22-cell full normal-flight V/I/T/SOC_proxy curve *availability* inventory.

Uses DERIVED, independently audited contiguous segment counts and phase SOC
endpoints; it is not the same as loading every original raw CSV and verifying
sample-by-sample joins. VAH01 direct raw LFS audit is a complementary check.
An individual proxy curve is reconstructible from raw QDischarge samples
only under a past capacity anchor and valid initial SOC condition.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def build():
    seg=pd.read_csv(ROOT/"results"/"segment_summary.csv")
    lab=pd.read_csv(ROOT/"results"/"soc_proxy_phase_labels.csv")
    keys=["cell_id","charge_cycle_index","Ns"]
    filtered=seg[seg.Ns.isin([4,5,6]) & seg.charge_cycle_index.notna()].copy()
    segs=filtered.groupby(keys,as_index=False).agg(
        segment_rows=("rows","sum"),num_contiguous_segments=("segment_run_id","nunique"),
        phase_duration_s=("duration_s","sum"),mean_Vmin=("V_min","min"),
    )
    phase=lab.rename(columns={"phase_Ns":"Ns","rows_in_phase":"label_rows"})
    merge=phase.merge(segs,on=keys,how="left",validate="one_to_one",indicator=True)
    merge["endpoints_matched"]=merge["_merge"].eq("both") & merge["segment_rows"].eq(merge["label_rows"])
    missions=[]
    for (cid,cycle),g in merge.groupby(["cell_id","charge_cycle_index"],sort=True):
        if len(g)!=3 or set(g.Ns)!={4,5,6}:continue
        if not g.endpoints_matched.all():continue
        mission_rows=int(g.segment_rows.sum())
        span=float(g.phase_duration_s.sum())
        stages=sorted(g.Ns)
        dts=[float(r.phase_duration_s/max(r.segment_rows-1,1)) for r in g.itertuples()]
        missions.append({
            "cell_id":cid,"charge_cycle_index":int(cycle),
            "raw_samples_expected":mission_rows,
            "takeoff_samples":int(g[g.Ns.eq(4)].segment_rows.iloc[0]),
            "cruise_samples":int(g[g.Ns.eq(5)].segment_rows.iloc[0]),
            "landing_samples":int(g[g.Ns.eq(6)].segment_rows.iloc[0]),
            "max_phase_implied_dt_s":float(max(dts)),
            "mean_phase_implied_dt_s":float(np.mean(dts)),
            "flight_duration_s":span,
            "historical_RPT_based_reference":int(g.capacity_source.eq("latest_completed_candidate_RPT").any()),
        })
    m=pd.DataFrame(missions)
    if m.cell_id.nunique()!=22:raise AssertionError("Missing physical cells")
    m["life_ordinal"]=m.groupby("cell_id").cumcount()
    m["is_after_N20"]=m.life_ordinal>=20
    m["life_fraction"]=m.groupby("cell_id").life_ordinal.transform(
        lambda v:v/max(len(v)-1,1)
    )
    m["life_stage"]=pd.cut(m.life_fraction,[-.001,.25,.75,1.001],
                            labels=["early","middle","late"])
    return m,merge


def analyze(m,phase):
    later=m[m.is_after_N20]
    counts=later.groupby("cell_id").agg(
        eligible_test_missions=("charge_cycle_index","size"),
        median_raw_samples=("raw_samples_expected","median"),
        P10_raw_samples=("raw_samples_expected",lambda x:float(x.quantile(.1))),
        median_flight_s=("flight_duration_s","median"),
        median_max_phase_dt_s=("max_phase_implied_dt_s","median"),
        candidate_ge_30_samples=("raw_samples_expected",lambda x:int((x>=30).sum())),
        candidate_ge_100_samples=("raw_samples_expected",lambda x:int((x>=100).sum())),
        candidate_ge_300_samples=("raw_samples_expected",lambda x:int((x>=300).sum())),
        candidate_ge_600_samples=("raw_samples_expected",lambda x:int((x>=600).sum())),
        fraction_RPT_anchor=("historical_RPT_based_reference","mean"),
    ).reset_index()
    life=later.groupby("life_stage",observed=True).agg(
        physical_cells=("cell_id","nunique"),
        missions=("charge_cycle_index","size"),
        median_expected_raw_samples=("raw_samples_expected","median"),
        median_phase_dt=("max_phase_implied_dt_s","median"),
        proportion_ge_100=("raw_samples_expected",lambda x:float((x>=100).mean())),
        proportion_ge_300=("raw_samples_expected",lambda x:float((x>=300).mean())),
    ).reset_index()
    return counts,life


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    m,phase=build()
    counts,life=analyze(m,phase)
    m.to_csv(out/"all_cells_normal_mission_raw_sample_manifest.csv",index=False)
    counts.to_csv(out/"all_cells_reference_density_by_cell.csv",index=False)
    life.to_csv(out/"all_cells_reference_density_by_lifestage.csv",index=False)
    report=[
        "# All 22 cells: normal-operation samplewise reference availability",
        "",
        f"- Complete 3-phase normal missions (all life): {len(m):,}; 22 physical cells.",
        f"- Test missions after N20: {int(m.is_after_N20.sum()):,}.",
        "- Expected raw points come from existing original-data segment counts, not a fresh per-cell raw file reparse.",
        "- A valid samplewise SOC_proxy needs QDischarge_mA_h and an effective-capacity anchor; source RPT capacity is used in the historical proxy and is NOT a freely acquired reference in a real BMS.",
        "- VAH01 LFS raw early/middle/late samplewise direct match is audited in an independent workflow job.",
        "",
        "## Reference data density across lifetime",
        "",
        "| Stage | Cells | Missions | Median expected raw points per mission | Max-phase implied dt median (s) | Fraction with ≥100 points | Fraction with ≥300 points |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for x in life.itertuples():
        report.append(f"| {x.life_stage} | {x.physical_cells} | {x.missions} | "
                      f"{x.median_expected_raw_samples:.0f} | {x.median_phase_dt:.1f} | "
                      f"{x.proportion_ge_100:.3f} | {x.proportion_ge_300:.3f} |")
    report += [
        "",
        "## Per-cell reference availability",
        "",
        "| Cell | Test missions | Median points per mission | ≥100 pt missions | ≥300 pt missions | RPT-anchored fraction |",
        "|---|---:|---:|---:|---:|---:|"
    ]
    for x in counts.itertuples():
        report.append(f"| {x.cell_id} | {x.eligible_test_missions} | {x.median_raw_samples:.0f} | "
                      f"{x.candidate_ge_100_samples} | {x.candidate_ge_300_samples} | {x.fraction_RPT_anchor:.3f} |")
    report += [
        "",
        "## What the numbers support",
        "- **YES:** much richer normal high-rate V/I/T/Q series than three phase endpoints exist in the original experiment, but per-stage logging cadence must be retained.",
        "- **NO:** this inventory is not verified real measured SOC, full actual field maintenance costs, or assurance that every high-rate mission has uniform 1Hz sampling.",
        "- The number and location of ≥300-point reference missions MUST be audited before promising equivalent full-curve calibration purchases across all battery ages.",
    ]
    (out/"report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "raw_files_reparsed_all_22":False,
        "source":"segment_summary.csv audited contiguous runs + soc_proxy_phase_labels.csv",
        "cells":22,
        "mission_count":len(m),
        "source_SOC_proxy_uses_prev_RPT_capacity":True,
        "single_cell_LFS_direct_audit":"VAH01, separate job",
        "independent_true_SOC":False,
    },indent=2,ensure_ascii=False),encoding="utf-8")
    print((out/"report.md").read_text(encoding="utf-8"),flush=True)


if __name__=="__main__":
    main()
