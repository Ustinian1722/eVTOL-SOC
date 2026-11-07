import pandas as pd
from pathlib import Path
p=Path(__file__).parent
d=pd.read_csv(p/'input.csv')
assert len(d)==20848 and d.cell_id.nunique()==22
assert not d.duplicated(['cell_id','charge_cycle_index']).any()
assert (d.charge_runs==1).all()
ratio=d.charge_Qmax_Ah/d.Q_end_Ah
out=dict(n_tasks=len(d),n_cells=d.cell_id.nunique(),charge_runs_all_one=True,ratio_median=float(ratio.median()),ratio_below_one=int((ratio<1).sum()),duration_median_min=float(d.charge_duration_s.median()/60),duration_p95_min=float(d.charge_duration_s.quantile(.95)/60),peak_ge418=int((d.charge_peak_V>=4.18).sum()),verified_full_charge_soc=int(d.independent_full_charge_SOC_verified.sum()),verified_full_highrate_soc=int(d.verified_full_highrate_SOC_reference.sum()))
pd.DataFrame([out]).to_csv(p/'report45_summary.csv',index=False)
print(out)
