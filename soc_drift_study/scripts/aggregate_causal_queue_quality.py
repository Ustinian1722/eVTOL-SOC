"""Aggregate strictly causal 22-cell outer results; paired physical-cell bootstrap."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

BOOT=10000
SEED=20261017

def aggregate(dirs,out):
    raw=pd.concat([pd.read_csv(p/"outer_queue_gate.csv") for p in dirs],ignore_index=True)
    choices=pd.concat([pd.read_csv(p/"source_gate_choices.csv") for p in dirs],ignore_index=True)
    src=pd.concat([pd.read_csv(p/"source_gate_grid.csv") for p in dirs],ignore_index=True)
    assert raw.cell_id.nunique()==22
    assert raw.groupby("cell_id").outer_fold.nunique().eq(1).all()
    key=["cell_id","budget","price","delay","reference_noise","role","variant"]
    assert raw.groupby(key).size().eq(1).all()
    assert len(raw)==22*2*3*4*5*2,(len(raw),22*2*3*4*5*2)
    assert len(choices)==5*2*3
    assert raw.purchases.ge(raw.accepted+raw.rejected+raw.pending_end).all()
    assert raw.purchases.le(raw.budget).all()
    assert np.allclose(raw.J,raw.MAE_pp+raw.price*raw.purchases)
    check=raw[(raw.budget==4)&np.isclose(raw.price,.1)&(raw.delay==0)&
              (raw.reference_noise=="nominal")&
              (raw.role=="source_selected")&(raw.variant=="ungated")]
    assert abs(check.MAE_pp.mean()-1.677)<.004,check.MAE_pp.mean()
    assert abs(check.J.mean()-1.954)<.004,check.J.mean()
    rng=np.random.default_rng(SEED)
    comparisons=[]
    for (budget,price,delay,noise,variant),d in raw.groupby(
            ["budget","price","delay","reference_noise","variant"]):
        pivot=d.pivot(index="cell_id",columns="role",values="J").sort_index()
        for comparator in ("periodic150","periodic200","source_fixed","frozen"):
            diff=(pivot.source_selected-pivot[comparator]).to_numpy()
            idx=rng.integers(0,22,(BOOT,22))
            lo,hi=np.quantile(diff[idx].mean(axis=1),[.025,.975])
            active=d[d.role=="source_selected"]
            control=d[d.role==comparator]
            comparisons.append(dict(budget=budget,price=price,delay=delay,
                reference_noise=noise,variant=variant,comparator=comparator,
                mean_active_MAE_pp=float(active.MAE_pp.mean()),
                mean_active_J=float(active.J.mean()),
                mean_active_purchases=float(active.purchases.mean()),
                mean_active_rejected=float(active.rejected.mean()),
                mean_comparator_J=float(control.J.mean()),
                mean_comparator_purchases=float(control.purchases.mean()),
                paired_delta_J=float(diff.mean()),CI_low=float(lo),CI_high=float(hi),
                cells_better=int((diff<0).sum()),n_cells=22))
    comparisons=pd.DataFrame(comparisons)
    gate_effect=[]
    for (budget,price,delay,noise,role),d in raw.groupby(
            ["budget","price","delay","reference_noise","role"]):
        wide=d.pivot(index="cell_id",columns="variant",values="J").sort_index()
        diff=(wide.source_gate-wide.ungated).to_numpy()
        idx=rng.integers(0,22,(BOOT,22))
        lo,hi=np.quantile(diff[idx].mean(axis=1),[.025,.975])
        gate_effect.append(dict(budget=budget,price=price,delay=delay,
            reference_noise=noise,role=role,
            delta_J_gate_minus_ungated=float(diff.mean()),
            CI_low=float(lo),CI_high=float(hi),
            better_cells=int((diff<0).sum())))
    gate_effect=pd.DataFrame(gate_effect)
    out.mkdir(parents=True,exist_ok=True)
    for name,df in (("outer_22_cell_queue_gate",raw),
                    ("source_gate_choices",choices),
                    ("source_gate_grid",src),
                    ("paired_policy_22cell",comparisons),
                    ("gate_effect_22cell",gate_effect)):
        df.to_csv(out/(name+".csv"),index=False)
    select=choices[(choices.budget==4)&np.isclose(choices.price,.1)]
    focus=comparisons[(comparisons.budget==4)&np.isclose(comparisons.price,.1)&
                      comparisons.delay.isin([0,25])&
                      comparisons.reference_noise.isin(["nominal","capacity_plus5pct"])&
                      comparisons.comparator.isin(["periodic150","periodic200","source_fixed"])]
    effects=gate_effect[(gate_effect.budget==4)&np.isclose(gate_effect.price,.1)&
                        gate_effect.delay.isin([0,25])&
                        gate_effect.reference_noise.isin(["nominal","capacity_plus5pct"])&
                        (gate_effect.role=="source_selected")]
    report="\n".join([
        "# 30. 严格到账队列与源电芯参考质量门控（22电芯五折）",
        "",
        "证据级别：22颗电芯已被反复探索，属于回顾性内部交叉验证，不是全新独立测试。",
        "标签：正常高倍率任务的部分SOC范围SOC_proxy，分母仍来自先前低倍率RPT容量。",
        "没有100%→截止高倍率完整维护实测，没有农业无人机现场数据，λ仅为假设成本。",
        "同一到账队列：任务t先处理已到标签，预测后决定采购，最早t+delay+1生效。",
        "拒收标签仍付费且消耗预算；质量门控只看已购Head跃迁，不看隐藏目标SOC误差。",
        "",
        "## 源电芯门控选择 B4 λ0.1",
        select.to_string(index=False),
        "",
        "## 22电芯配对综合成本（负值=主动更低）",
        focus[["delay","reference_noise","variant","comparator",
               "mean_active_MAE_pp","mean_active_J","mean_active_purchases",
               "mean_active_rejected","paired_delta_J","CI_low","CI_high",
               "cells_better"]].to_string(index=False),
        "",
        "## 质量门控消融（负值=门控改善）",
        effects.to_string(index=False),
        "",
        "门控若选inf，代表源证据不支持拒收。区间跨零/成本增加须报告NO-GO。",
        "原始CSV只读；所有数值为历史SOC_proxy及假设价格，不代表真实SOC或人民币。"
    ])+"\n"
    (out/"report.md").write_text(report,encoding="utf8")
    (out/"metadata.json").write_text(json.dumps(dict(
        n_cells=22,bootstrap_unit="physical_cell",bootstrap_samples=BOOT,
        repeated_22_cells_not_external=True,strict_causal_arrival=True,
        quality_gate_source_only=True,rejected_reference_paid=True,
        historical_RPT_capacity_in_SOC_proxy=True,
        no_100_to_0_highrate_diagnostic=True,no_agriculture_field_data=True,
        original_CSV_untouched=True),indent=2),encoding="utf8")
    print(report,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fold-dirs",nargs=5,type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    aggregate(a.fold_dirs,a.out)
