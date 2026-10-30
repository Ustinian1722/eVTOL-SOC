"""Read-only cross-check: prior Ns0 charging block before raw Ns4/Ns6 mission.
Peak charge voltage is evidence of a charge stage, not proof of SOC=100%.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
KEY=["cell_id","charge_cycle_index"]

def attach_prior_charge(m,seg):
    charge=seg[seg.Ns.eq(0)&seg.duration_s.gt(100)].groupby(KEY,as_index=False).agg(
        charge_runs=("Ns","size"),charge_end_s=("time_end_s","max"),
        charge_peak_V=("V_max","max"),charge_Qmax_Ah=("QCharge_max_Ah","max"),
        charge_duration_s=("duration_s","sum"))
    start=seg[seg.Ns.eq(4)].groupby(KEY,as_index=False).agg(
        mission_start_s=("time_start_s","min"))
    a=m.merge(start,on=KEY,how="left",validate="one_to_one").merge(
        charge,on=KEY,how="left",validate="one_to_one")
    a["prior_charge_ordered"]=a.charge_end_s.notna()&a.mission_start_s.notna()&a.charge_end_s.le(a.mission_start_s)
    a["charge_to_mission_gap_min"]=np.where(a.prior_charge_ordered,
        (a.mission_start_s-a.charge_end_s)/60,np.nan)
    a["prior_charge_peak_ge_4p0"]=a.prior_charge_ordered&a.charge_peak_V.ge(4.0)
    a["prior_charge_peak_ge_4p1"]=a.prior_charge_ordered&a.charge_peak_V.ge(4.1)
    a["prior_charge_peak_ge_4p18"]=a.prior_charge_ordered&a.charge_peak_V.ge(4.18)
    a["independent_full_charge_SOC_verified"]=False
    return a

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--study",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    m=pd.read_csv(a.out/"raw_terminal_missions.csv")
    seg=pd.read_csv(a.study/"results/segment_summary.csv")
    d=attach_prior_charge(m,seg)
    d.to_csv(a.out/"mission_prior_charge_evidence.csv",index=False)
    d["strict_cutoff_like"]=(d.end_ok&d.end_V.le(2.55))|(d.end_ok&d.end_T.ge(69))
    d["near_2p5"]=d.end_ok&d.end_V.sub(2.5).abs().le(.01)
    groups={"all":d[d.end_ok],"strict_cutoff_like":d[d.strict_cutoff_like],
            "near_2p5":d[d.near_2p5],"other":d[d.end_ok&~d.strict_cutoff_like]}
    rows=[]
    for name,g in groups.items():
        v=g[g.prior_charge_ordered]
        rows.append(dict(group=name,n=len(g),cells=g.cell_id.nunique(),
            prior_charge_ordered=int(g.prior_charge_ordered.sum()),
            peak_ge_4p0=int(g.prior_charge_peak_ge_4p0.sum()),
            peak_ge_4p1=int(g.prior_charge_peak_ge_4p1.sum()),
            peak_ge_4p18=int(g.prior_charge_peak_ge_4p18.sum()),
            median_peak_charge_V=float(v.charge_peak_V.median()) if len(v) else np.nan,
            median_charge_gap_min=float(v.charge_to_mission_gap_min.median()) if len(v) else np.nan,
            median_charge_Qmax_Ah=float(v.charge_Qmax_Ah.median()) if len(v) else np.nan))
    t=pd.DataFrame(rows)
    t.to_csv(a.out/"prior_charge_group_summary.csv",index=False)
    lines=["# 报告41补充：正常任务之前的Ns0充电块与满充可辨识性","",
        "根据原始分段审计表匹配每次Ns4/5/6正常任务之前的Ns0充电块，要求充电结束时间早于任务开始。此处Vmax是充电段峰值，不是已核实的静置OCV或精确初始SOC。","",
        "| 组别 | 任务 | 有前序充电时序 | 充电峰值≥4.0V | ≥4.1V | ≥4.18V | 峰值中位V | 充电后至任务间隔中位min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples(index=False):
        lines.append(f"| {r.group} | {r.n} | {r.prior_charge_ordered} | {r.peak_ge_4p0} | {r.peak_ge_4p1} | {r.peak_ge_4p18} | {r.median_peak_charge_V:.3f} | {r.median_charge_gap_min:.2f} |")
    lines+=["","## 解释边界",
        "高倍率Ns4开始时端电压明显低于充电峰值，是负载极化的正常可能现象；不能用Ns4起点是否≥4.1V判定此前是否满充。",
        "前序Ns0充电块与峰值电压只能支持充电协议的间接证据，不能独立证明每次达到统一100% SOC、没有中间异常、或任务确实按高倍率容量诊断截止。",
        "真正的100%→截止操作性SOC参考还需要独立核实充电终止、截止原因、完整电量积分和同倍率可用容量；本数据仍无独立SOC真值。",""]
    report="\n".join(lines)
    (a.out/"report41_prior_charge.md").write_text(report,encoding="utf-8")
    print(report)
if __name__=="__main__":main()
