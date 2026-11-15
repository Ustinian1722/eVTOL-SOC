import pandas as pd
from pathlib import Path
p=Path(__file__).parent
d=pd.read_csv(p/'input.csv')
assert len(d)==20848 and d.cell_id.nunique()==22
assert not d.duplicated(['cell_id','ordinal']).any()
d=d[d.ordinal>=20].copy()
assert len(d)==20408
assert (d.raw_points_expected>0).all() and (d.phase_duration_min>0).all()
d['cost']=.1*(.5+.25*d.raw_points_expected/100+.25*d.phase_duration_min/16)
rows=[]
for period in [100,150,200,300]:
 for cell,g in d.groupby('cell_id'):
  s=g.sort_values('ordinal')
  s=s[(s.ordinal-19)%period==0].head(4)
  rows.append(dict(cell_id=cell,period=period,n=len(s),cost=s.cost.sum(),flat_cost=.1*len(s),points=s.raw_points_expected.sum(),minutes=s.phase_duration_min.sum()))
r=pd.DataFrame(rows)
r.to_csv(p/'fixed_cost.csv',index=False)
print(d[['raw_points_expected','phase_duration_min','cost']].quantile([0,.05,.5,.95,1]))
print(r.groupby('period')[['n','cost','flat_cost','points','minutes']].mean())
