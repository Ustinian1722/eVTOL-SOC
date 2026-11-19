"""Internal 22-cell replay; hypothetical costs, historical SOC_proxy only."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from deferral_regret_when import build_cell, source_examples, source_models
from nested_cell_when_5fold import outer_splits
from scenario_safe_when_selector import ACTIVE, PERIODIC
from stateful_marginal_roi_maintenance import FIELDS

def quote(m):
    # Ordinary completed mission metadata, not paid SOC labels.
    return .1*(.5+.25*m.X.shape[0]/100+.25*m.duration/960)

def simulate(cell, models, arm, cap, lag, variable=True):
    events=[]; pending=[]; last=-1; spent=0.; losses=[]; trace=[]
    ops=set(cell.opportunities)
    for t,m in enumerate(cell.test):
        while pending and pending[0][0]<=t:
            _,last=pending.pop(0)
        losses.append(cell.frozen_loss[t] if last<0 else cell.loss[last][t])
        if t not in ops or len(events)>=4 or pending: continue
        c=quote(m)
        if spent+c>cap+1e-12: continue
        if arm=="frozen": continue
        if arm.startswith("periodic"):
            period=int(arm[8:])
            buy=(t+1)%period==0
        else:
            mode,gap,threshold=ACTIVE[arm]
            if events and t-events[-1]<gap: continue
            f=pd.DataFrame([cell.feature(t,last,len(events))])[list(FIELDS)]
            rr=float(models[0].predict(f)[0])
            dd=float(models[1].predict(f)[0])
            price=c if variable else .1
            reserve=.1*len(events)/4 if mode=="reserve" else 0.
            buy=rr>.15+5*price if mode=="old_roi" else (dd>threshold+reserve and (mode=="delay_only" or rr>.15+5*price))
        if buy:
            events.append(t); spent+=c; pending.append((t+lag+1,t))
            trace.append((t,c,t+lag+1))
    assert spent<=cap+1e-12 and len(events)<=4
    mae=float(np.mean(losses))
    return dict(MAE_pp=mae,J=mae+spent,cost=spent,purchases=len(events),events=";".join(map(str,events)),quotes=";".join(str(x[1]) for x in trace))

def run(fold,out):
    tr,te=outer_splits()[fold]
    dev={c:build_cell(c) for c in tr}
    src=pd.DataFrame([r for c in dev.values() for r in source_examples(c)])
    assert set(src.cell_id)==set(tr)
    _,cv,oof,final=source_models(dev,src)
    held={c:build_cell(c) for c in te}
    rows=[]; selection=[]
    for cap in (.2,.4,.6):
        for lag in (0,5,25):
            scores=[]
            menu=list(PERIODIC)+list(ACTIVE)
            for cid,c in dev.items():
                for arm in menu:
                    scores.append(dict(cell_id=cid,arm=arm,**simulate(c,oof[cid],arm,cap,lag)))
            d=pd.DataFrame(scores).groupby("arm").J.agg(["mean","max"])
            d["score"]=.7*d["mean"]+.3*d["max"]
            fixed=d.loc[list(PERIODIC)].sort_values(["score","mean"]).index[0]
            active=d.loc[list(ACTIVE)].sort_values(["score","mean"]).index[0]
            selection.append(dict(fold=fold,cap=cap,lag=lag,fixed=fixed,active=active))
            for cid,c in held.items():
                for role,arm,var in (("source_fixed",fixed,True),("source_active",active,True),("active_flat_threshold",active,False),("fixed150","periodic150",True),("frozen","frozen",True)):
                    rows.append(dict(cell_id=cid,fold=fold,cap=cap,lag=lag,role=role,arm=arm,**simulate(c,final,arm,cap,lag,var)))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"outer.csv",index=False)
    pd.DataFrame(selection).to_csv(out/"selection.csv",index=False)
    cv.to_csv(out/"inner_cv.csv",index=False)

def aggregate(root,out):
    d=pd.concat([pd.read_csv(p) for p in sorted(root.glob("*/outer.csv"))])
    assert d.cell_id.nunique()==22 and len(d)==22*3*3*5
    assert not d.duplicated(["cell_id","cap","lag","role"]).any()
    assert (d.cost<=d.cap+1e-12).all()
    rows=[]
    for (cap,lag),g in d.groupby(["cap","lag"]):
        p=g.pivot(index="cell_id",columns="role",values="J")
        delta=(p.source_active-p.source_fixed).to_numpy()
        rng=np.random.default_rng(2026)
        lo,hi=np.quantile(delta[rng.integers(0,22,(8000,22))].mean(axis=1),[.025,.975])
        rows.append(dict(cap=cap,lag=lag,deltaJ=delta.mean(),lo=lo,hi=hi,better=int((delta<0).sum())))
    out.mkdir(parents=True,exist_ok=True)
    d.to_csv(out/"all.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"paired.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fold",type=int,choices=range(5))
    p.add_argument("--aggregate",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    if a.aggregate: aggregate(a.aggregate,a.out)
    else: run(a.fold,a.out)
