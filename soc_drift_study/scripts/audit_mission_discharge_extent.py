"""Read-only audit of normal Ns4/5/6 mission discharge extent, NOT full-rate diagnostic proof."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

KEY=["cell_id","charge_cycle_index"]
def inventory(labels,segments):
    req=set(KEY+["phase_Ns","QDischarge_end_Ah","capacity_used_Ah","capacity_source","SOC_proxy","rows_in_phase","duration_s"])
    if req-set(labels):raise ValueError("missing labels "+str(req-set(labels)))
    segreq=set(KEY+["Ns","rows","duration_s"])
    if segreq-set(segments):raise ValueError("missing segments "+str(segreq-set(segments)))
    a=labels[labels.phase_Ns.isin([4,5,6])].copy()
    b=segments[segments.Ns.isin([4,5,6])].groupby(KEY+["Ns"],as_index=False).agg(
        segment_rows=("rows","sum"))
    dup=set(map(tuple,a.loc[a.duplicated(KEY+["phase_Ns"],keep=False),KEY].itertuples(index=False,name=None)))
    a=a.merge(b,left_on=KEY+["phase_Ns"],right_on=KEY+["Ns"],how="left",validate="many_to_one")
    rows=[];reasons={}
    for key,g in a.groupby(KEY,sort=True):
        reason=None
        if tuple(key) in dup:reason="duplicate_phase"
        elif len(g)!=3 or set(g.phase_Ns)!={4,5,6}:reason="missing_phase"
        elif g.segment_rows.isna().any() or not np.allclose(g.rows_in_phase,g.segment_rows):reason="row_mismatch"
        elif not np.isfinite(g[["QDischarge_end_Ah","capacity_used_Ah","SOC_proxy","duration_s"]].to_numpy(float)).all():reason="nonfinite"
        elif (g.capacity_used_Ah<=0).any() or np.ptp(g.capacity_used_Ah.to_numpy(float))>1e-6:reason="capacity_inconsistent"
        else:
            g=g.sort_values("phase_Ns")
            q=g.QDischarge_end_Ah.to_numpy(float);cap=float(g.capacity_used_Ah.iloc[0])
            if np.any(np.diff(q)<-1e-5):reason="counter_reversal"
            elif np.max(np.abs(np.clip(1-q/cap,0,1)-g.SOC_proxy.to_numpy(float)))>.006:reason="proxy_mismatch"
        if reason:
            reasons[reason]=reasons.get(reason,0)+1
            continue
        f=float(q[-1]/cap)
        rows.append(dict(cell_id=key[0],charge_cycle_index=int(key[1]),
            capacity_anchor_Ah=cap,capacity_source=str(g.capacity_source.iloc[0]),
            Q_end_Ah=float(q[-1]),fraction_of_historical_anchor=f,
            landing_SOC_proxy=float(g.SOC_proxy.iloc[-1]),
            phase_duration_min=float(g.duration_s.sum())/60,
            raw_points_expected=int(g.rows_in_phase.sum()),
            proxy_ge95=bool(f>=.95),proxy_ge100=bool(f>=1),
            independent_full_highrate_diagnostic_verified=False))
    if not rows:raise ValueError("no valid missions")
    m=pd.DataFrame(rows).sort_values(KEY).reset_index(drop=True)
    m["ordinal"]=m.groupby("cell_id").cumcount()
    m["after_N20"]=m.ordinal.ge(20)
    return m,{"excluded_by_reason":reasons,"excluded_count":sum(reasons.values()),
              "candidate_mission_keys":int(a[KEY].drop_duplicates().shape[0])}

def stats(g):
    f=g.fraction_of_historical_anchor
    return pd.Series(dict(missions=len(g),median_fraction=f.median(),p05_fraction=f.quantile(.05),
        p95_fraction=f.quantile(.95),max_fraction=f.max(),
        proxy_ge95=int(g.proxy_ge95.sum()),proxy_ge100=int(g.proxy_ge100.sum()),
        median_landing_proxy=g.landing_SOC_proxy.median(),
        median_duration_min=g.phase_duration_min.median(),
        median_points=g.raw_points_expected.median(),
        fraction_prior_RPT=g.capacity_source.eq("latest_completed_candidate_RPT").mean()))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--labels",type=Path,required=True)
    p.add_argument("--segments",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--require-22",action="store_true")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    m,meta=inventory(pd.read_csv(a.labels),pd.read_csv(a.segments))
    if a.require_22 and m.cell_id.nunique()!=22:raise AssertionError("not 22 cells")
    cell=m.groupby("cell_id").apply(stats,include_groups=False).reset_index()
    summ=[]
    for name,g in [("all",m),("after_N20",m[m.after_N20])]:
        row=stats(g).to_dict();row["subset"]=name;row["cells"]=g.cell_id.nunique();summ.append(row)
    summary=pd.DataFrame(summ)
    m.to_csv(a.out/"mission_discharge_extent.csv",index=False)
    cell.to_csv(a.out/"per_cell_discharge_extent.csv",index=False)
    summary.to_csv(a.out/"summary_discharge_extent.csv",index=False)
    meta.update(physical_cells=int(m.cell_id.nunique()),valid_missions=len(m),
        capacity_denominator="prior candidate low-rate RPT or nominal 3Ah",
        verified_independent_highrate_diagnostics=0,
        evidence_limit="not verified from these derived tables; zero is NOT proof none exist in raw data")
    (a.out/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    s=summary.set_index("subset").loc["after_N20"];t=summary.set_index("subset").loc["all"]
    lines=["# 报告40：22电芯正常高倍率任务放电范围与完整维护诊断可辨识性审计","",
      "只读历史派生表复算；不是新电芯、独立真实SOC、或新高倍率完整诊断实验。HOW、WHEN及原始CSV未变。","",
      "| 指标 | 全寿命有效任务 | N20之后 |","|---|---:|---:|",
      f"| 物理电芯 | {int(t.cells)} | {int(s.cells)} |",
      f"| 三阶段任务 | {int(t.missions)} | {int(s.missions)} |",
      f"| Q_end/历史容量锚点中位数 | {t.median_fraction:.3f} | {s.median_fraction:.3f} |",
      f"| Q_end/历史容量锚点P05–P95 | {t.p05_fraction:.3f}–{t.p95_fraction:.3f} | {s.p05_fraction:.3f}–{s.p95_fraction:.3f} |",
      f"| 代理耗尽≥95%任务数 | {int(t.proxy_ge95)} | {int(s.proxy_ge95)} |",
      f"| 代理耗尽≥100%任务数 | {int(t.proxy_ge100)} | {int(s.proxy_ge100)} |",
      f"| 降落SOC_proxy中位数 | {t.median_landing_proxy:.3f} | {s.median_landing_proxy:.3f} |",
      f"| 三阶段时长中位数min | {t.median_duration_min:.2f} | {s.median_duration_min:.2f} |",
      f"| 原始点数中位数 | {t.median_points:.0f} | {s.median_points:.0f} |",
      f"| 历史RPT容量锚点比例 | {t.fraction_prior_RPT:.3f} | {s.fraction_prior_RPT:.3f} |",
      "",f"候选键数 {meta['candidate_mission_keys']}；排除 {meta['excluded_count']}；原因 {meta['excluded_by_reason']}。","",
      "## 关键限制",
      "Q_end/历史RPT容量是代理比例，不是独立同倍率可用容量或真实DOD。即使代理比例达到100%，也不证明执行过满电至安全截止的完整高倍率诊断。",
      "当前派生表无法独立证明满电初始SOC、持续到截止、同倍率容量分母、标签到账时间；独立核验的完整诊断数为0，意为未验证而非证明原始数据不存在。",
      "现有WHEN实验仅能支持付费解锁正常任务的部分SOC区间全采样代理曲线。其分母依赖先前RPT候选，不可称RPT-free真实SOC。",
      "## 维护协议",
      "真实目标方案需独立确认满电起点，按规定高倍率负载放至安全截止，完成后积分获得本次操作性容量，质控后才释放整段SOC参考；实际停机时间及代价不能从采样点数换算。",
      "## 决策",
      "GO：保留历史代理标签采购回放并明确标签可得性假设。NO-GO：不把它等同于100%至安全截止的高倍率维护检测，不虚构农业无人机现场验证。"]
    report="\n".join(lines)+"\n"
    (a.out/"report40.md").write_text(report,encoding="utf-8")
    print(report)
if __name__=="__main__":main()
