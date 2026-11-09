from pathlib import Path
import pandas as pd
import numpy as np
p=Path(__file__).parent
d=pd.read_csv(p/'input.csv')
assert len(d)==20848 and d.cell_id.nunique()==22
assert not d.duplicated(['cell_id','charge_cycle_index']).any()
d['ratio']=d.charge_Qmax_Ah/d.Q_end_Ah
d['gap_calc_min']=(d.mission_start_s-d.charge_end_s)/60
d['gap_error_min']=d.gap_calc_min-d.charge_to_mission_gap_min
assert np.isfinite(d[['ratio','gap_calc_min','gap_error_min']].to_numpy()).all()
assert (d.charge_duration_s>0).all() and (d.Q_end_Ah>0).all()
summary=dict(n_tasks=len(d),n_cells=d.cell_id.nunique(),negative_gap=int((d.gap_calc_min<0).sum()),gap_discrepancy_gt_1s=int((d.gap_error_min.abs()>1/60).sum()),ratio_lt_one=int((d.ratio<1).sum()),ratio_median=d.ratio.median(),gap_median_min=d.gap_calc_min.median(),gap_p05_min=d.gap_calc_min.quantile(.05),gap_p95_min=d.gap_calc_min.quantile(.95),verified_initial_soc=int(d.independent_full_charge_SOC_verified.sum()),verified_highrate_reference=int(d.verified_full_highrate_SOC_reference.sum()))
pd.DataFrame([summary]).to_csv(p/'report46_summary.csv',index=False)
print(summary)
