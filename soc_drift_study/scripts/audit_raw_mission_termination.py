"""Read-only original VAH sample-level audit of normal mission terminal conditions.
No new SOC truth, HOW change, or future-label access. Requires all 22 raw CSVs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from audit_mission_discharge_extent import inventory, KEY

COLS=["time_s","Ns","Ecell_V","I_mA","Temperature__C","QDischarge_mA_h"]
EPS=0.001  # seconds; reject nearest-neighbour interpolation

def targets_from_segments(m, seg):
    g=seg[seg.Ns.isin([4,6])].groupby(KEY+["Ns"],as_index=False).agg(
        t_start=("time_start_s","min"),t_end=("time_end_s","max"),
        runs=("Ns","size"))
    g=m[KEY].merge(g,on=KEY,validate="one_to_many")
    g["kind"]=np.where(g.Ns.eq(4),"start","end")
    g["target_t"]=np.where(g.Ns.eq(4),g.t_start,g.t_end)
    if g.duplicated(KEY+["kind"]).any():raise ValueError("duplicate targets")
    return g[KEY+["kind","Ns","target_t","runs"]].reset_index(drop=True)

def match_chunk(chunk,targets):
    """Return exact time+phase matches, including duplicates for later rejection."""
    if chunk.empty or targets.empty:return []
    t=chunk.time_s.to_numpy(dtype=float)
    order=np.argsort(t,kind="stable"); ts=t[order]
    tt=targets.target_t.to_numpy(dtype=float)
    ns=chunk.Ns.to_numpy()
    out=[]
    for j in np.flatnonzero(np.isfinite(tt)&(tt>=ts[0]-EPS)&(tt<=ts[-1]+EPS)):
        lo=np.searchsorted(ts,tt[j]-EPS,"left")
        hi=np.searchsorted(ts,tt[j]+EPS,"right")
        ids=order[lo:hi]
        ids=ids[ns[ids]==targets.Ns.iloc[j]]
        for ix in ids:
            r=chunk.iloc[int(ix)]
            out.append(dict(target_id=int(targets.index[j]),time=float(r.time_s),
                V=float(r.Ecell_V),T=float(r.Temperature__C),
                I_A=float(r.I_mA)/1000,Q_Ah=float(r.QDischarge_mA_h)/1000))
    return out

def scan_raw(path,targets):
    if path.stat().st_size<1000 or b"git-lfs.github.com/spec/v1" in path.open("rb").read(160):
        raise RuntimeError("unmaterialized LFS pointer: "+str(path))
    hits=[];total=0
    for chunk in pd.read_csv(path,usecols=COLS,chunksize=250000):
        total+=len(chunk);hits.extend(match_chunk(chunk,targets))
    if not hits:return pd.DataFrame(columns=["target_id","match_count"]),total
    h=pd.DataFrame(hits)
    counts=h.groupby("target_id").size().rename("match_count")
    h=h.drop_duplicates("target_id").merge(counts,left_on="target_id",right_index=True)
    return h,total

def combine(m,targets,hits):
    d=targets.merge(hits,left_index=True,right_on="target_id",how="left",validate="one_to_one")
    d["matched"]=d.match_count.eq(1)&d.V.notna()
    a=d[d.kind.eq("start")][KEY+["matched","V","Q_Ah"]].rename(
        columns={"matched":"start_ok","V":"start_V","Q_Ah":"start_Q"})
    b=d[d.kind.eq("end")][KEY+["matched","V","T","I_A","Q_Ah","match_count"]].rename(
        columns={"matched":"end_ok","V":"end_V","T":"end_T","I_A":"end_I","Q_Ah":"end_Q"})
    out=m.merge(a,on=KEY,how="left",validate="one_to_one").merge(
        b,on=KEY,how="left",validate="one_to_one")
    for k in ["start_ok","end_ok"]:out[k]=out[k].fillna(False).astype(bool)
    out["both_ok"]=out.start_ok&out.end_ok
    out["q_difference_Ah"]=out.end_Q-out.Q_end_Ah
    out["voltage_le_2p60"]=out.end_ok&out.end_V.le(2.60)
    out["voltage_le_2p55"]=out.end_ok&out.end_V.le(2.55)
    out["temperature_ge_69"]=out.end_ok&out.end_T.ge(69)
    out["possible_cutoff_signature"]=out.voltage_le_2p60|out.temperature_ge_69
    out["independent_highrate_full_diagnostic_verified"]=False
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--require-22",action="store_true")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    s=a.root/"soc_drift_study"
    m,meta=inventory(pd.read_csv(s/"results/soc_proxy_phase_labels.csv"),
        pd.read_csv(s/"results/segment_summary.csv"))
    seg=pd.read_csv(s/"results/segment_summary.csv")
    targets=targets_from_segments(m,seg)
    matches=[];raw_counts={}
    for cell in sorted(m.cell_id.unique()):
        h,n=scan_raw(a.root/f"{cell}.csv",targets[targets.cell_id.eq(cell)])
        matches.append(h);raw_counts[cell]=n
        print(f"{cell}: raw rows {n:,}, matched targets {len(h):,}",flush=True)
    out=combine(m,targets,pd.concat(matches,ignore_index=True))
    if a.require_22 and (out.cell_id.nunique()!=22 or
        out[out.end_ok].cell_id.nunique()!=22):
        raise AssertionError("raw terminal samples must cover all 22 cells")
    end=out[out.end_ok]
    counts=dict(cells=int(out.cell_id.nunique()),missions=len(out),
        end_matched=len(end),both_matched=int(out.both_ok.sum()),
        unmatched_or_ambiguous=int((~out.end_ok).sum()),
        end_voltage_le_2p60=int(end.voltage_le_2p60.sum()),
        end_voltage_le_2p55=int(end.voltage_le_2p55.sum()),
        end_temperature_ge_69=int(end.temperature_ge_69.sum()),
        possible_cutoff_signature=int(end.possible_cutoff_signature.sum()),
        median_terminal_V=float(end.end_V.median()) if len(end) else None,
        min_terminal_V=float(end.end_V.min()) if len(end) else None,
        max_terminal_T=float(end.end_T.max()) if len(end) else None,
        raw_rows=sum(raw_counts.values()),
        candidate_missions=meta["candidate_mission_keys"],
        excluded_derived=meta["excluded_count"])
    out.to_csv(a.out/"raw_terminal_missions.csv",index=False)
    out.groupby("cell_id").agg(missions=("end_ok","size"),
        end_matched=("end_ok","sum"),both_matched=("both_ok","sum"),
        voltage_le_2p60=("voltage_le_2p60","sum"),
        voltage_le_2p55=("voltage_le_2p55","sum"),
        temperature_ge_69=("temperature_ge_69","sum"),
        possible_cutoff_signature=("possible_cutoff_signature","sum"),
        median_terminal_V=("end_V","median"),
        min_terminal_V=("end_V","min"),
        max_terminal_T=("end_T","max")).reset_index().to_csv(
            a.out/"per_cell_terminal_summary.csv",index=False)
    (a.out/"metadata.json").write_text(json.dumps(
        {**counts,"raw_rows_per_cell":raw_counts,"match_tolerance_s":EPS,
        "thresholds_are_exploratory":True,"independent_SOC_truth":False},
        ensure_ascii=False,indent=2),encoding="utf-8")
    report=["# 报告41：22电芯原始正常任务末端电压与温度审计","",
        "仅只读原始VAH采样点；匹配Ns4起点和Ns6终点的时间戳与阶段。拒绝模糊匹配，重复命中判为歧义。HOW、WHEN、原始CSV均未改变。","",
        "| 指标 | 结果 |","|---|---:|",
        f"| 电芯 | {counts['cells']} |",
        f"| 派生表有效正常任务 | {counts['missions']:,} |",
        f"| 原始Ns6终点精确匹配 | {counts['end_matched']:,} |",
        f"| 起点终点均匹配 | {counts['both_matched']:,} |",
        f"| 末端未匹配或歧义 | {counts['unmatched_or_ambiguous']:,} |",
        f"| 末端电压≤2.60V | {counts['end_voltage_le_2p60']:,} |",
        f"| 末端电压≤2.55V | {counts['end_voltage_le_2p55']:,} |",
        f"| 末端温度≥69°C | {counts['end_temperature_ge_69']:,} |",
        f"| 候选截止信号合计（并集） | {counts['possible_cutoff_signature']:,} |",
        f"| 末端电压中位数 | {counts['median_terminal_V']:.4f}V |",
        "",
        "## 物理解释限制",
        "2.60/2.55V及69°C仅为排查阈值，不能代替每个工况的正式截止协议；触及阈值也不证明满充起点、独立高倍率可用容量或真实SOC。",
        "未触及阈值不排除其他保护动作；原始日志没有独立的终止原因码。Ns6结束可能是固定时长任务结束、保护截止或异常停止。",
        "SOC_proxy分母仍为先前低倍率RPT容量；这批正常任务不是已经验证的100%→截止高倍率维护诊断。农业无人机真实停机成本未测量。",
        "该22芯为反复使用的回顾性样本，不是全新独立外部测试。",
        "",
        "## 研究决策",
        "GO：原始采样点末端核验和异常保留。NO-GO：把阈值触发当成完整高倍率维护容量真值，或据此声称RPT-free、现场成本收益。",""]
    text="\n".join(report)
    (a.out/"report41.md").write_text(text,encoding="utf-8")
    print(text)
if __name__=="__main__":main()
