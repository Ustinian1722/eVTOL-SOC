"""Exploratory follow-up: contextualize original raw cutoff-like mission endpoints.
Read-only descriptive audit. Not a model, policy, or diagnostic ground truth.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def summarize(m):
    m=m.copy()
    m["last_ordinal"]=m.groupby("cell_id").ordinal.transform("max")
    m["life_fraction"]=m.ordinal/m.last_ordinal.clip(lower=1)
    m["near_2p5"]=m.end_ok & m.end_V.sub(2.5).abs().le(.01)
    m["candidate_v"]=m.end_ok & m.end_V.le(2.55)
    m["candidate_t"]=m.end_ok & m.end_T.ge(69)
    m["candidate_any"]=m.candidate_v|m.candidate_t
    m["charged_like_start"]=m.start_ok & m.start_V.ge(4.10) & m.start_Q.abs().le(.05)
    m["last_mission"]=m.ordinal.eq(m.last_ordinal)
    m["last_10pct"]=m.life_fraction.ge(.90)
    m["possible_full_highrate_diagnostic_verified"]=False
    def row(g):
        return dict(n=int(len(g)),cells=int(g.cell_id.nunique()),
            charged_like=int(g.charged_like_start.sum()),
            last_mission=int(g.last_mission.sum()),
            last_10pct=int(g.last_10pct.sum()),
            median_start_v=float(g.start_V.median()) if len(g) else np.nan,
            median_start_q=float(g.start_Q.median()) if len(g) else np.nan,
            median_end_v=float(g.end_V.median()) if len(g) else np.nan,
            median_end_q=float(g.end_Q.median()) if len(g) else np.nan,
            median_fraction_anchor=float(g.fraction_of_historical_anchor.median()) if len(g) else np.nan,
            median_mission_minutes=float(g.phase_duration_min.median()) if len(g) else np.nan,
            median_end_current_A=float(g.end_I.median()) if len(g) else np.nan)
    groups={"all":m[m.end_ok],"v_le_2p55":m[m.candidate_v],
            "v_near_2p50":m[m.near_2p5],"temp_ge_69":m[m.candidate_t],
            "any_cutoff_like":m[m.candidate_any],
            "no_cutoff_like":m[m.end_ok&~m.candidate_any]}
    table=pd.DataFrame([{"group":name,**row(g)} for name,g in groups.items()])
    per_cell=m.groupby("cell_id").agg(missions=("end_ok","size"),
        voltage_le_2p55=("candidate_v","sum"),near_2p50=("near_2p5","sum"),
        temp_ge_69=("candidate_t","sum"),cutoff_like=("candidate_any","sum"),
        cutoff_like_last10pct=("last_10pct",lambda x:0)).reset_index()
    # Compute only among flagged rows (no accidental aggregation over all missions).
    late=m[m.candidate_any].groupby("cell_id").last_10pct.sum()
    per_cell["cutoff_like_last10pct"]=per_cell.cell_id.map(late).fillna(0).astype(int)
    return m,table,per_cell

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    m=pd.read_csv(a.out/"raw_terminal_missions.csv")
    m,t,c=summarize(m)
    t.to_csv(a.out/"terminal_pattern_groups.csv",index=False)
    c.to_csv(a.out/"terminal_pattern_by_cell.csv",index=False)
    flagged=m[m.candidate_any].copy()
    flagged.to_csv(a.out/"cutoff_like_events_exploratory.csv",index=False)
    lines=["# 报告41补充：原始末端截止样信号的生命周期与起点核验","",
        "这是一项探索性事后分层，不是监督参考质量标签，不参与WHEN策略选择或参数调整。","",
        "| 事件组 | 次数 | 电芯数 | 起点电压≥4.10V且Q起点≤0.05Ah | 电芯最后一次任务 | 后10%寿命任务 | 历史容量比例中位 | 任务时长中位min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t.itertuples(index=False):
        lines.append(f"| {r.group} | {r.n} | {r.cells} | {r.charged_like} | {r.last_mission} | {r.last_10pct} | {r.median_fraction_anchor:.3f} | {r.median_mission_minutes:.2f} |")
    lines+=["","## 关键解释",
        "电压接近2.5V的真实Ns6末端样本是可疑的高倍率截止候选，不能一概说原始正常任务没有任何截止事件。",
        "但是，单凭末端2.5V和初始4.1V不能证明该任务按照统一协议完成了满充→规定高倍率截止的完整诊断；需核对充电结束、终止原因码、动态负载和容量测量定义。",
        "正常任务SOC_proxy的分母仍来自历史低倍率RPT，不能把该比例当作独立同倍率容量利用率。",
        "任何候选事件都不构成已付费高倍率诊断的真实记录；没有农业无人机停机成本。",""]
    report="\n".join(lines)
    (a.out/"report41_cutoff_pattern.md").write_text(report,encoding="utf-8")
    print(report)
if __name__=="__main__":main()
