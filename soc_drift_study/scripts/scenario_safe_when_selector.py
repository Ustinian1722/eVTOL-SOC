"""Budget/price conditional WHEN: source-OOF model selection with safe periodic fallback.

No external cells. Eight challenge cells are previously examined, RETROSPECTIVE.
Normal high-rate purchased SOC_proxy is historical Coulomb count with prior
RPT capacity anchor, not independently measured SOC. λ is hypothetical.
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from deferral_regret_when import (
    CELL_IDS, SEED, cell_split, build_cell, source_examples, source_models,
    replay_policy)
from stateful_marginal_roi_maintenance import run_cell

SCENARIOS=tuple((b,c) for b in (2,4) for c in (0.,.1,.3,.6))
PERIODIC=("frozen","periodic150","periodic200","periodic250",
          "periodic300","periodic400","periodic500")
ACTIVE={
    "delay75_12":("delay",75,.12),
    "delay175_12":("delay",175,.12),
    "delay175_04":("delay",175,.04),
    "reserve175_12":("reserve",175,.12),
    "reserve175_04":("reserve",175,.04),
    "delay_only175_12":("delay_only",175,.12),
    "prior_ROI175":("old_roi",175,0.),
}
MENU=PERIODIC+tuple(ACTIVE)


def evaluate_one(cell,models,arm,budget,price):
    if arm in ACTIVE:
        z,_=replay_policy(cell,models,ACTIVE[arm],budget,price)
        return z
    v,_=run_cell(cell,arm,budget,price)
    return {"cell_id":cell.cid,"MAE_pp":v["SOCproxy_MAE_pp"],
            "J":v["J"],"purchases":v["num_reference_purchases"],
            "label_points":v["num_raw_labels_purchased"],
            "events":v["reference_idx"]}


def choose_from_source(dev,models):
    rows=[]
    for cid,cell in sorted(dev.items()):
        for b,c in SCENARIOS:
            for arm in MENU:
                z=evaluate_one(cell,models[cid],arm,b,c)
                rows.append({"cell_id":cid,"budget":b,"price":c,"arm":arm,**z})
        print("DEV evaluated",cid,flush=True)
    detail=pd.DataFrame(rows)
    avg=detail.groupby(["budget","price","arm"],as_index=False).agg(
        meanJ=("J","mean"),worstJ=("J","max"),
        meanMAE=("MAE_pp","mean"),meanPurchases=("purchases","mean"))
    selections=[]
    for (b,c),g in avg.groupby(["budget","price"]):
        g=g.copy()
        g["robustScore"]=.7*g.meanJ+.3*g.worstJ
        best_simple=float(g[g.arm.isin(PERIODIC)].meanJ.min())
        eligible=g[g.meanJ<=best_simple+.02]
        if eligible.empty:eligible=g
        best=eligible.sort_values(["robustScore","meanJ","arm"]).iloc[0]
        selections.append({"budget":b,"price":c,"source_chosen":str(best.arm),
            "strategy_class":"active" if str(best.arm) in ACTIVE else "periodic_or_none",
            "sourceOOF_meanJ":float(best.meanJ),
            "sourceOOF_worstJ":float(best.worstJ),
            "source_best_simple_meanJ":best_simple,
            "eligible_candidates":len(eligible)})
    return detail,avg,pd.DataFrame(selections)


def evaluate_challenge(test,model,selected):
    rows=[]
    for cid,cell in sorted(test.items()):
        for z in selected.itertuples():
            for arm in sorted(set(["periodic150","periodic200",
                                   "periodic250","periodic300",
                                   "offline_oracle",z.source_chosen])):
                r=evaluate_one(cell,model,arm,z.budget,z.price)
                rows.append({"cell_id":cid,"budget":z.budget,"price":z.price,
                             "policy":arm,"selected":int(arm==z.source_chosen),
                             **r})
        print("RETROSPECTIVE challenge",cid,flush=True)
    res=pd.DataFrame(rows)
    assert res.groupby(["cell_id","budget","price"]).selected.sum().eq(1).all()
    summary=[]
    for (b,c),g in res.groupby(["budget","price"]):
        base=g[g.policy=="periodic150"][["cell_id","J"]].rename(columns={"J":"referenceJ"})
        chosen=g[g.selected.eq(1)].merge(base,on="cell_id",validate="one_to_one")
        d=(chosen.J-chosen.referenceJ).to_numpy()
        rng=np.random.default_rng(SEED)
        boot=rng.integers(0,len(d),size=(8000,len(d)))
        lo,hi=np.quantile(d[boot].mean(axis=1),[.025,.975])
        best_diagnostic=float(g[g.policy.isin(["periodic150","periodic200",
                             "periodic250","periodic300"])].groupby("policy").J.mean().min())
        summary.append({"budget":b,"price":c,"chosen":chosen.policy.iloc[0],
            "chosen_type":"active" if chosen.policy.iloc[0] in ACTIVE else "periodic_or_none",
            "meanJ":float(chosen.J.mean()),
            "medianMAE":float(chosen.MAE_pp.median()),
            "meanPurchases":float(chosen.purchases.mean()),
            "deltaJ_vs_fixed150":float(d.mean()),
            "CI_low":float(lo),"CI_high":float(hi),
            "better_than_fixed150":int((d<0).sum()),
            "diagnostic_deltaJ_vs_best_retrospective_fixed":
                float(chosen.J.mean()-best_diagnostic)})
    return res,pd.DataFrame(summary)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    split=cell_split()
    dev={};test={};rows=[]
    for cid in CELL_IDS:
        cell=build_cell(cid)
        if split[cid]=="development":
            dev[cid]=cell
            rows.extend(source_examples(cell))
        else:test[cid]=cell
        print("DATA",cid,cell.n,split[cid],flush=True)
    data=pd.DataFrame(rows)
    winners,cv,oof,final=source_models(dev,data)
    srcdetail,srcsummary,selected=choose_from_source(dev,oof)
    testdetail,testsummary=evaluate_challenge(test,final,selected)
    artifacts={"source_OOF_each_cell":srcdetail,
               "source_OOF_scenario_policy_summary":srcsummary,
               "source_only_frozen_scenario_selections":selected,
               "retrospective_challenge_policy_costs":testdetail,
               "retrospective_challenge_cell_bootstrap":testsummary,
               "source_groupCV_models":cv}
    for name,table in artifacts.items():
        table.to_csv(args.out/(name+".csv"),index=False)
    metadata={
      "source_cell_count":len(dev),"retrospective_challenge_count":len(test),
      "source_model_winners":winners,
      "source_selected_arms":selected.to_dict(orient="records"),
      "menu_frozen_before_challenge":list(MENU),
      "active_fallback_to_fixed_or_frozen_possible":True,
      "future_target_SOC_not_used_in_action":True,
      "C_prev_RPT_needed_for_historical_label":True,
      "no_independent_SOC_reference_or_real_maintenance_price":True,
      "all_cells_previous_exploration_not_pristine":True
    }
    (args.out/"metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf8")
    report="\n".join([
      "# Cost/budget adaptive policy or periodic fallback, source-only selection",
      "",
      "These 8 challenge cells were previously inspected. Numbers are exploratory.",
      "Historical Coulomb SOC_proxy uses previous RPT capacity denominator.",
      "Hypothetical cost in MAE-pp units, not measured field expense.",
      "",
      "## Source-only menu decisions",
      selected.to_string(index=False),
      "",
      "## Previously examined challenge comparisons",
      testsummary.to_string(index=False),
      "",
      "When selection chooses periodic/frozen, this is NOT an active WHEN win.",
      "A bootstrap interval cannot undo prior repeated testing on the 8 cells.",
    ])+"\n"
    (args.out/"report.md").write_text(report,encoding="utf8")
    print(report,flush=True)


if __name__=="__main__":
    main()
