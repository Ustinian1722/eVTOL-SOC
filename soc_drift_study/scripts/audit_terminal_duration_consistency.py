"""Report42: read-only terminal-duration cross-check on report41 raw-VAH audit.
Exploratory evidence only; NOT verified cutoff reason, SOC truth or maintenance.
"""
from pathlib import Path
import argparse
import json
import pandas as pd

KEY = ['cell_id', 'charge_cycle_index']
REQUIRED = KEY + ['ordinal','phase_duration_min','end_V','end_T',
                  'end_ok','prior_charge_ordered','charge_peak_V']

def analyze(src):
    if any(k not in src.columns for k in REQUIRED):
        raise ValueError('missing required report41 fields')
    if src.duplicated(KEY).any() or src.duplicated(['cell_id','ordinal']).any():
        raise ValueError('duplicate cell/task')
    if not src.end_ok.fillna(False).all() or not src.prior_charge_ordered.fillna(False).all():
        raise ValueError('unmatched endpoint or unordered charge')
    if src.phase_duration_min.isna().any() or src.phase_duration_min.le(0).any():
        raise ValueError('invalid duration')
    d = src.sort_values(['cell_id','ordinal'],kind='stable').reset_index(drop=True).copy()
    d['n_cell'] = d.groupby('cell_id').ordinal.transform('size')
    d['life_fraction'] = (d.ordinal+1)/d.n_cell
    d['last_recorded'] = d.ordinal.eq(d.groupby('cell_id').ordinal.transform('max'))
    # Protocol-specific descriptive baseline; no target SOC labels.
    early = d.loc[d.life_fraction.le(.8)].groupby('cell_id').phase_duration_min.median()
    if len(early) != d.cell_id.nunique():
        raise ValueError('missing early baseline')
    d['baseline_min'] = d.cell_id.map(early)
    d['duration_deficit_s'] = (d.baseline_min-d.phase_duration_min)*60
    d['near_2p50'] = d.end_V.sub(2.5).abs().le(.01)
    d['strict_candidate'] = d.end_V.le(2.55)|d.end_T.ge(69)
    d['short10'] = d.duration_deficit_s.gt(10)
    d['joint_candidate'] = d.last_recorded & d.near_2p50 & d.short10
    d['verified_cutoff_reason'] = False
    d['independent_SOC_truth'] = False
    masks = {
        'all': d.index==d.index,
        'strict': d.strict_candidate,
        'strict_nonlast': d.strict_candidate & ~d.last_recorded,
        'near25': d.near_2p50,
        'near25_nonlast': d.near_2p50 & ~d.last_recorded,
        'near25_last': d.near_2p50 & d.last_recorded,
        'joint_last_near25_short10': d.joint_candidate,
        'short10_other': d.short10 & ~d.joint_candidate,
    }
    summary = pd.DataFrame([{
        'group':k, 'tasks':int(v.sum()), 'cells':int(d.loc[v,'cell_id'].nunique()),
        'short10':int((v & d.short10).sum()),
        'low_charge_peak_lt4p18':int((v & d.charge_peak_V.lt(4.18)).sum()),
    } for k,v in masks.items()])
    cell = d.groupby('cell_id',sort=True).agg(
        missions=('ordinal','size'),baseline_min=('baseline_min','first'),
        strict=('strict_candidate','sum'),near25=('near_2p50','sum'),
        joint=('joint_candidate','sum'),short10=('short10','sum'),
        charge_peak_median=('charge_peak_V','median')).reset_index()
    sensitivity = pd.DataFrame([{
        'threshold_s':t, 'all_short':int(d.duration_deficit_s.gt(t).sum()),
        'joint_last_near25':int((d.duration_deficit_s.gt(t)&d.last_recorded&d.near_2p50).sum()),
        'nonlast_near25':int((d.duration_deficit_s.gt(t)&~d.last_recorded&d.near_2p50).sum())
    } for t in [5,10,20,30,45,60]])
    return d,summary,cell,sensitivity

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-22',action='store_true')
    a=p.parse_args()
    d,s,c,t=analyze(pd.read_csv(a.out/'mission_prior_charge_evidence.csv'))
    v=dict(zip(s.group,s.tasks))
    if a.require_22:
        assert len(d)==20848 and d.cell_id.nunique()==22
        for k,n in [('strict',94),('strict_nonlast',77),('near25',21),
                    ('near25_last',16),('joint_last_near25_short10',15)]:
            assert v[k]==n,(k,v[k],n)
    d.to_csv(a.out/'report42_duration_classification.csv',index=False)
    s.to_csv(a.out/'report42_duration_summary.csv',index=False)
    c.to_csv(a.out/'report42_duration_per_cell.csv',index=False)
    t.to_csv(a.out/'report42_threshold_sensitivity.csv',index=False)
    nonlast=d[d.strict_candidate&~d.last_recorded]
    joint=d[d.joint_candidate]
    lines=[
        '# 报告42：高倍率任务截止样终点与时长一致性',
        '',
        '报告41的原始22芯任务末端/充电审计结果再分析；每芯以前80%任务时长中位数为基线。只读，非独立外测。',
        '',
        '| 指标 | 结果 |','|---|---:|',
        f'| 电芯/任务 | {d.cell_id.nunique()}/{len(d):,} |',
        f'| 电压≤2.55V或温度≥69°C | {v["strict"]} |',
        f'| 上述候选中的非最后任务 | {v["strict_nonlast"]} |',
        f'| 末端2.50±0.01V | {v["near25"]} |',
        f'| 最后记录任务末端2.50±0.01V | {v["near25_last"]} |',
        f'| 最后任务、近2.5V且缩短>10秒 | {v["joint_last_near25_short10"]} |',
        f'| 其余缩短>10秒的任务 | {v["short10_other"]} |',
        '',
        '77/94个阈值候选不是最后记录任务，全部来自VAH07、VAH10；相对同芯早期时长基线均未缩短超过10秒。',
        f'15个联合候选的时长缩短范围{joint.duration_deficit_s.min():.1f}–{joint.duration_deficit_s.max():.1f}秒。',
        'VAH07最后任务接近2.5V但并未提前结束；另有一个非最后任务缩短约25秒但没有低压/高温末端信号。',
        '',
        '| 缩短阈值 | 全部缩短 | 最后近2.5V | 非最后近2.5V |',
        '|---:|---:|---:|---:|',
    ]
    for r in t.itertuples(index=False):
        lines.append(f'| >{r.threshold_s}s | {r.all_short} | {r.joint_last_near25} | {r.nonlast_near25} |')
    lines += [
        '',
        '## 可辨识性与NO-GO',
        '时长与末端电压的联合证据与提前截止一致，但不是独立保护终止原因码。',
        '先前充电峰值不等于已测SOC=100%；本批数据不能验证按需诊断的高倍率完整容量、可购买参考质量或真实停机时间。',
        'SOC_proxy分母仍为历史RPT容量；不称RPT-free、不虚构农业现场实测或人民币成本。',
        '不改变HOW、WHEN、原始CSV或策略超参数。保留阈值筛查的误判风险，不能把15个候选当成已购买的完整SOC参考。',
        '本分析是反复使用的22芯内部回顾性审计，不是独立测试。',
    ]
    report='\n'.join(lines)+'\n'
    (a.out/'report42.md').write_text(report,encoding='utf-8')
    (a.out/'report42_metadata.json').write_text(json.dumps({
        'cells':int(d.cell_id.nunique()),'missions':int(len(d)),
        'early_baseline_fraction':0.8,'descriptive_threshold_only':True,
        'independent_SOC_truth':False,'new_cells':False,
        'HOW_or_WHEN_changed':False},indent=2),encoding='utf-8')
    print(report)

if __name__=='__main__':
    main()
