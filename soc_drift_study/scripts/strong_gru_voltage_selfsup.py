"""RPT-free online voltage reconstruction -> frozen strong-GRU feature correction.

Scientific design:
* 5 fold / 3 seed cell-boundary-corrected GRU is frozen and trained on SOURCE
  cells' historical (past-RPT-capacity-anchored) SOC_proxy. Source offline
  capacity labels are NOT equivalent to RPT-free *pretraining*.
* On the 22 target cells, the online voltage auxiliary Ridge sees only
  COMPLETED normal high-power mission-phase I/V/T signals. Its Vmin target
  is actually measured in the same phase; this is **voltage reconstruction**,
  not future voltage forecasting.
* SOC prediction uses current observed Vmin, while the voltage-derived input
  offset is learned strictly from earlier completed phases. No current/future
  target SOC label, RPT curve/capacity, SOH, life-stage enters the updating.
* A small bounded Vmin correction is an adaptation *hypothesis*, not an
  electrochemical identity. Compare no correction, direct, validation-guarded,
  reverse-sign placebo, and aggressive stream updating.
* SOC_proxy is used ONLY for scoring AFTER sequential replay.

The GRU predicts blocks of exactly 12 phase summaries, resetting hidden state
each block as in the original protocol. All predictions for a block are
COMMITTED before the corresponding block can enter the online update memory.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_gru_proxy_crosscell import (
    RESULTS, FEATURES, SEQ_LEN, FOLDS, SEEDS, _build_model,
)

STUDY = Path(__file__).resolve().parents[1]
AUX_FEATURES = [
    "phase_Ns", "duration_s", "I_mean_A", "I_peak_A",
    "Vmax_V", "Tmean_C", "Tmax_C",
]
VOLTAGE_IDX = FEATURES.index("Vmin_V")
SOURCE_EFFECTIVE_ROWS = 80.0
SOURCE_LOSS_GUARD_V = .015
TRUST_RADIUS_V = .30
MAX_OFFSET_V = .10
TRAIN_CHUNKS = 24
GUARD_CHUNKS = 6
WINDOW_CHUNKS = TRAIN_CHUNKS + GUARD_CHUNKS
PERIOD_CHUNKS = 40
BUDGET = 4
SEED = 20261012

# Predeclared comparisons. No test-SOC based parameter selection.
ARMS = [
    ("frozen", 0.0, False, False, 0),
    ("periodic_correction_015", .15, False, False, 40),
    ("periodic_correction_050", .50, False, False, 40),
    ("guarded_correction_015", .15, True, False, 40),
    ("guarded_correction_050", .50, True, False, 40),
    ("reverse_placebo_015", -.15, False, False, 40),
    ("frequent_correction_015", .15, True, True, 10),
]


def chunks_of_features(frame: pd.DataFrame, scale: StandardScaler):
    """Create the exact same chunk boundaries as the source-GRU estimator,
    without asking for ANY SOC target during inference."""
    z=scale.transform(frame[FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)
    n=len(z)
    count=int(np.ceil(n/SEQ_LEN))
    chunks=np.zeros((count,SEQ_LEN,len(FEATURES)),dtype=np.float32)
    is_real=np.zeros((count,SEQ_LEN),dtype=bool)
    for j,first in enumerate(range(0,n,SEQ_LEN)):
        k=min(SEQ_LEN,n-first)
        chunks[j,:k]=z[first:first+k]
        is_real[j,:k]=True
    return chunks,is_real


def aux_source(source: pd.DataFrame):
    """Source voltage regression, NO SOC labels (even offline)."""
    m=make_pipeline(StandardScaler(),Ridge(alpha=5.))
    m.fit(source[AUX_FEATURES],source["Vmin_V"])
    z=np.c_[np.ones(len(source)),m.named_steps["standardscaler"].transform(source[AUX_FEATURES])]
    y=source["Vmin_V"].to_numpy(dtype=float)
    f=m.named_steps["ridge"]
    theta=np.r_[f.intercept_,f.coef_]
    return m,z,y,theta


def build_aux_tensor(frame:pd.DataFrame,m):
    n=len(frame)
    z=np.c_[np.ones(n),m.named_steps["standardscaler"].transform(frame[AUX_FEATURES])]
    chunks=np.zeros((int(np.ceil(n/SEQ_LEN)),SEQ_LEN,z.shape[1]),dtype=float)
    y=np.zeros((len(chunks),SEQ_LEN),dtype=float)
    obs=frame["Vmin_V"].to_numpy(dtype=float)
    for j,first in enumerate(range(0,n,SEQ_LEN)):
        k=min(SEQ_LEN,n-first)
        chunks[j,:k]=z[first:first+k]
        y[j,:k]=obs[first:first+k]
    return chunks,y


def objective_aux(theta:np.ndarray,chunklist:list[dict])->float:
    residual=np.concatenate([r["Z"]@theta-r["V"] for r in chunklist])
    return float(np.mean(np.abs(residual))) if residual.size else float("nan")


def update_voltage_theta(
    old:np.ndarray, frozen:np.ndarray,
    sourceZ:np.ndarray,sourceY:np.ndarray,
    memory:list[dict],guard:bool,
)->tuple[np.ndarray,bool,dict]:
    """Fit on 24 past chunks and evaluate 6 later BUT ALREADY COMPLETED chunks.

    A weighted prior centered on the original auxiliary head constrains the
    estimator to stay within the source voltage domain.
    """
    if len(memory)<WINDOW_CHUNKS:
        return old.copy(),False,{"reason":"insufficient_prior_missions"}
    fit=memory[:-GUARD_CHUNKS]
    held=memory[-GUARD_CHUNKS:]
    X=np.concatenate([r["Z"] for r in fit],axis=0)
    y=np.concatenate([r["V"] for r in fit])
    src_factor=SOURCE_EFFECTIVE_ROWS/max(len(sourceY),1)
    anchor=3.0
    G=src_factor*(sourceZ.T@sourceZ)+4.0*(X.T@X)+anchor*np.eye(len(old))
    v=src_factor*(sourceZ.T@sourceY)+4.0*(X.T@y)+anchor*frozen
    proposal=np.linalg.solve(G+1e-9*np.eye(len(old)),v)
    delta=proposal-frozen
    norm=float(np.linalg.norm(delta))
    if norm>TRUST_RADIUS_V:
        proposal=frozen+delta/norm*TRUST_RADIUS_V
    if not np.isfinite(proposal).all():
        return old.copy(),False,{"reason":"nonfinite"}
    old_v=objective_aux(old,held)
    new_v=objective_aux(proposal,held)
    before_source=float(np.mean(np.abs(sourceZ@old-sourceY)))
    after_source=float(np.mean(np.abs(sourceZ@proposal-sourceY)))
    quality=float((old_v-new_v)/max(old_v,1e-6))
    diagnostic={
        "holdout_Vmin_MAE_before_mV":1000*old_v,
        "holdout_Vmin_MAE_candidate_mV":1000*new_v,
        "holdout_relative_improvement":quality,
        "source_Vmin_MAE_change_mV":1000*(after_source-before_source),
        "proposal_shift_norm":float(np.linalg.norm(proposal-frozen)),
        "reason":"updated",
    }
    if guard and (quality<.05 or after_source-before_source>SOURCE_LOSS_GUARD_V):
        diagnostic["reason"]="voltage_holdout_or_source_guard"
        return old.copy(),False,diagnostic
    return proposal,True,diagnostic


@dataclass
class CellSignal:
    id:str
    tensor:np.ndarray
    auxZ:np.ndarray
    volts:np.ndarray
    chunk_real:np.ndarray
    chunk_test:np.ndarray
    phase_cycles:np.ndarray
    test_true:np.ndarray
    test_keys:pd.DataFrame
    capacity_source:str = "not_read_for_online_update"


def prepare(cell:pd.DataFrame,gru_scaler:StandardScaler,aux_model):
    cell=cell.sort_values(["charge_cycle_index","phase_Ns"]).reset_index(drop=True)
    cycles=np.sort(cell.charge_cycle_index.unique())
    cutoff=cycles[19]
    mask=cell.charge_cycle_index.gt(cutoff).to_numpy()
    chunks,real=chunks_of_features(cell,gru_scaler)
    z,volts=build_aux_tensor(cell,aux_model)
    padded_test=np.zeros(real.shape,dtype=bool)
    padded_test.reshape(-1)[:len(cell)]=mask
    assert int(padded_test.sum())==int(mask.sum())
    return CellSignal(
        id=str(cell.cell_id.iloc[0]),
        tensor=chunks,auxZ=z,volts=volts,
        chunk_real=real,chunk_test=padded_test,
        phase_cycles=cell.charge_cycle_index.to_numpy(),
        test_true=cell.loc[mask,"SOC_proxy"].to_numpy(dtype=float), # scoring only
        test_keys=cell.loc[mask,["cell_id","charge_cycle_index","phase_Ns"]].copy(),
    )


def step_predict(model,inputs:np.ndarray)->np.ndarray:
    return model.predict(inputs,batch_size=256,verbose=0)[:,:,0]


def replay_signal(signal:CellSignal,model,gru_scaler,sourceZ,sourceY,source_theta,arm):
    name,strength,guard,frequent,step=arm
    current=source_theta.copy()
    memory=deque(maxlen=WINDOW_CHUNKS)
    nphase=int(signal.chunk_real.sum())
    pred=np.empty(nphase,dtype=np.float32)
    voltage_err=[]
    events=[]
    attempted=accepted=0
    testchunks=0
    # Phase-chunk blocks, model weights frozen. Changes only to input Vmin
    # are computed from PRIOR completed missions' voltage identification.
    cursor=0
    while cursor<len(signal.tensor):
        remaining=int(step) if step else len(signal.tensor)
        # Each method's update schedule is counted from first TEST chunk, not
        # source/N20 chunks. Process one chunk before the first target chunk.
        if not signal.chunk_test[cursor].any():
            limit=cursor+1
        else:
            upcoming = max(1,remaining-(testchunks%remaining)) if step else remaining
            # Never pass a non-test chunk inside the same adaptation block.
            limit=min(len(signal.tensor),cursor+upcoming)
        currentX=signal.tensor[cursor:limit].copy()
        currentAux=signal.auxZ[cursor:limit]
        valid=signal.chunk_real[cursor:limit]
        current_voltage=signal.volts[cursor:limit]
        online_diff=np.clip(currentAux@(current-source_theta),-MAX_OFFSET_V,MAX_OFFSET_V)
        if strength:
            currentX[:,:,VOLTAGE_IDX] -= np.asarray(
                strength*online_diff/max(float(gru_scaler.scale_[VOLTAGE_IDX]),1e-8),
                dtype=np.float32
            )*valid
        p=step_predict(model,currentX)
        for j in range(limit-cursor):
            idx=cursor+j
            row_ids=np.flatnonzero(signal.chunk_real[idx])
            start=idx*SEQ_LEN
            pred[start:start+len(row_ids)]=p[j,row_ids]
            if not signal.chunk_test[idx].any():
                continue
            # Predictions COMMITTED; now this whole measured phase chunk can
            # enter the self-supervised voltage-update memory.
            test_mask=signal.chunk_test[idx]
            Z=signal.auxZ[idx,test_mask].copy()
            V=signal.volts[idx,test_mask].copy()
            voltage_err.extend(np.abs(Z@current-V).tolist())
            memory.append({"Z":Z,"V":V})
            testchunks+=1
            if name=="frozen" or step==0:
                continue
            if not frequent and attempted>=BUDGET:
                continue
            if len(memory)<WINDOW_CHUNKS or testchunks%step != 0:
                continue
            attempted+=1
            candidate,good,diag=update_voltage_theta(
                current,source_theta,sourceZ,sourceY,list(memory),guard
            )
            if good:
                current=candidate
                accepted+=1
            events.append({
                "cell_id":signal.id,"arm":name,"event_test_chunk":testchunks,
                "attempt_no":attempted,"accepted":int(good),**diag,
            })
        cursor=limit
    # Score ONLY after replay is finished: input SOC_proxy cannot influence
    # any candidate voltage fit or active update decision.
    all_test_pred=pred[signal.chunk_test.reshape(-1)[:nphase]]
    errors=100*(all_test_pred-signal.test_true)
    percycle=pd.DataFrame({
        "cycle":signal.test_keys.charge_cycle_index.to_numpy(),
        "abs_error":np.abs(errors),
    }).groupby("cycle")["abs_error"].mean().to_numpy()
    return {
        "cell_id":signal.id,"arm":name,
        "SOC_proxy_MAE_pp":float(np.mean(np.abs(errors))),
        "SOC_proxy_RMSE_pp":float(np.sqrt(np.mean(errors*errors))),
        "late_cycle_MAE_pp":float(np.mean(percycle[int(.75*len(percycle)):])),
        "P95_cycle_MAE_pp":float(np.quantile(percycle,.95)),
        "Vmin_reconstruction_MAE_mV":float(np.mean(voltage_err)*1000),
        "attempted_updates":attempted,"accepted_updates":accepted,
        "test_chucks":testchunks,
    },events,all_test_pred


def aggregate(df:pd.DataFrame):
    percell=df.groupby(["cell_id","arm"],as_index=False).agg(
        MAE=("SOC_proxy_MAE_pp","mean"),
        voltage_MAE=("Vmin_reconstruction_MAE_mV","mean"),
        RMSE=("SOC_proxy_RMSE_pp","mean"),
        late=("late_cycle_MAE_pp","mean"),
        attempts=("attempted_updates","median"),
        accepted=("accepted_updates","median"),
    )
    frozen=percell[percell.arm=="frozen"].set_index("cell_id")
    percell=percell.join(frozen[["MAE","voltage_MAE"]],on="cell_id",rsuffix="_frozen")
    percell["delta_soc_pp"]=percell.MAE-percell.MAE_frozen
    percell["delta_voltage_mV"]=percell.voltage_MAE-percell.voltage_MAE_frozen
    rows=[]
    for arm,g in percell.groupby("arm"):
        diff=g.delta_soc_pp.to_numpy()
        vdiff=g.delta_voltage_mV.to_numpy()
        rnd=np.random.default_rng(SEED)
        ids=rnd.integers(0,len(diff),size=(6000,len(diff)))
        ci=np.quantile(diff[ids].mean(axis=1),[.025,.975])
        civ=np.quantile(vdiff[ids].mean(axis=1),[.025,.975])
        rows.append({
            "arm":arm,"physical_cells":len(g),
            "median_MAE_pp":float(g.MAE.median()),
            "mean_paired_SOC_delta_pp":float(diff.mean()),
            "SOC_CI_low":float(ci[0]),"SOC_CI_high":float(ci[1]),
            "SOC_improved_cells":int((diff<0).sum()),
            "median_voltage_MAE_mV":float(g.voltage_MAE.median()),
            "mean_paired_voltage_delta_mV":float(vdiff.mean()),
            "voltage_CI_low":float(civ[0]),"voltage_CI_high":float(civ[1]),
            "voltage_improved_cells":int((vdiff<0).sum()),
            "median_attempts":float(g.attempts.median()),
            "median_accepted":float(g.accepted.median()),
        })
    return percell,pd.DataFrame(rows)


def report(summary:pd.DataFrame,byseed:pd.DataFrame,events:pd.DataFrame)->str:
    lines=[
        "# Strong cell-boundary-safe GRU + self-supervised voltage online adapter",
        "",
        "- Frozen backbone is the exact same boundary-corrected source GRU as the published experimental reference, 5 folds × 3 seeds.",
        "- SOURCE offline GRU training was historical SOC_proxy (prior-RPT capacity anchored); this is NOT an RPT-free pretraining model.",
        "- TARGET online voltage reconstruction uses ONLY completed normal high-power mission V/I/T phase summaries; no RPT or online SOC targets.",
        "- Vmin auxiliary task reconstructs measured same-phase voltage given current Vmax/current/temp, not future voltage prediction.",
        "- At each chunk, all 12 phase SOC predictions are recorded before any voltage data from that chunk becomes available for UPDATE.",
        "- Adapter modifies the standardized Vmin input through a bounded previously identified voltage-response difference.",
        "- The voltage correction mapping to SOC is a falsifiable engineering hypothesis, NOT a proven physical equivalence.",
        "- SOC_proxy read after replay for scoring only. No target true-SOC labels exist.",
        "- Periodic arms max four attempts; frequent 10-chunk update is intentionally NOT budget matched.",
        "",
        "| Policy | SOC proxy median MAE (pp) | Paired SOC delta (pp) & 95% cell CI | Better SOC cells | Voltage MAE (mV) | Paired voltage delta (mV) | Median updates tried/accepted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary.sort_values("median_MAE_pp").itertuples():
        lines.append(
            f"| {r.arm} | {r.median_MAE_pp:.3f} | {r.mean_paired_SOC_delta_pp:+.3f} "
            f"[{r.SOC_CI_low:+.3f},{r.SOC_CI_high:+.3f}] | "
            f"{r.SOC_improved_cells}/{r.physical_cells} | {r.median_voltage_MAE_mV:.2f} | "
            f"{r.mean_paired_voltage_delta_mV:+.2f} | "
            f"{r.median_attempts:.0f}/{r.median_accepted:.0f} |"
        )
    lines += [
        "",
        "## Decision rules",
        "- Frozen versus adapted SOC MUST be compared on exact same trained GRU and same phase test entries; script asserts original predictions match.",
        "- Positive self-supervised voltage reconstruction gains do not guarantee a SOC improvement.",
        "- Reverse placebo may expose whether a systematic voltage correction sign is informative.",
        "- If an adapter harms frozen GRU, mark NO-GO. Never 'pick a winner' via retrospectively tuning held-cell SOC_proxy.",
        "- For future deployment, genuinely independent SOC reference or credible controlled capacity protocol remains important.",
    ]
    return "\n".join(lines)+"\n"


def main():
    import tensorflow as tf
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir",type=Path,required=True)
    parser.add_argument("--corrected-predictions",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    t0=time.perf_counter()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(RESULTS/"soc_proxy_phase_labels.csv")
    df=df.sort_values(["cell_id","charge_cycle_index","phase_Ns"]).reset_index(drop=True)
    ref=pd.read_csv(args.corrected_predictions)
    cells=sorted(df.cell_id.unique())
    folds=[list(x) for x in np.array_split(np.random.default_rng(SEED).permutation(cells),FOLDS)]
    rows=[];events=[]
    for fold_i,held in enumerate(folds,start=1):
        source=df[~df.cell_id.isin(held)]
        auxm,auxSrcZ,auxSrcY,auxFrozen=aux_source(source)
        scaler=StandardScaler().fit(source[FEATURES].to_numpy(dtype=np.float32))
        for seed in SEEDS:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed+(fold_i-1)*1000)
            model=_build_model()
            model.load_weights(args.checkpoint_dir/f"safe_fold{fold_i}_seed{seed}.weights.h5")
            for cell,raw in df[df.cell_id.isin(held)].groupby("cell_id",sort=True):
                sig=prepare(raw,scaler,auxm)
                fresh_pred=None
                for arm in ARMS:
                    metric,history,preds=replay_signal(
                        sig,model,scaler,auxSrcZ,auxSrcY,auxFrozen,arm
                    )
                    metric.update({"seed":seed,"fold":fold_i})
                    rows.append(metric);events+=[
                        {"seed":seed,"fold":fold_i,**e} for e in history
                    ]
                    if arm[0]=="frozen":
                        fresh_pred=preds
                target_ref=ref[(ref.cell_id==cell)&(ref.seed==seed)].sort_values(
                    ["charge_cycle_index","phase_Ns"]
                )
                assert len(target_ref)==len(fresh_pred), (cell,seed,len(target_ref),len(fresh_pred))
                deviation=float(np.max(np.abs(fresh_pred-target_ref.source_GRU_prediction.to_numpy())))
                if deviation>1e-5:
                    raise AssertionError(f"Frozen reloaded model diverged from original corrected prediction: {cell},seed={seed}: {deviation}")
            print(f"Fold {fold_i} / {FOLDS}, seed {seed}, wall_s={time.perf_counter()-t0:.0f}",flush=True)
    results=pd.DataFrame(rows)
    if results.groupby(["cell_id","seed"]).size().nunique()!=1:
        raise AssertionError("Unequal arms/cells")
    assert len(results)==22*len(SEEDS)*len(ARMS)
    assert results[~results.arm.eq("frequent_correction_015")].attempted_updates.le(BUDGET).all()
    percell,summary=aggregate(results)
    results.to_csv(out/"strong_gru_online_voltage_cell_seed.csv",index=False)
    percell.to_csv(out/"strong_gru_online_voltage_cell_macro.csv",index=False)
    summary.to_csv(out/"strong_gru_online_voltage_summary.csv",index=False)
    pd.DataFrame(events).to_csv(out/"strong_gru_online_voltage_events.csv",index=False)
    text=report(summary,results,pd.DataFrame(events))
    (out/"report.md").write_text(text,encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "pretraining_RPT_proxy":"source SOC_proxy prior RPT anchored",
        "online_update_RPT_access":False,
        "online_update_SOC_label_access":False,
        "source_aux_voltage_training":"source physical cells full lifecycle V/I/T only",
        "online_update_target":"Vmin observed in completed normal mission phase",
        "prequential_chunk_size":SEQ_LEN,"frozen_predictions_matched_exact":True,
        "max_budget_B4_except_frequent":BUDGET,"source_cells_excluded_heldout":True,
        "n_physical_cells":22,"seeds":SEEDS,
        "control_arms":ARMS,
        "total_wall_sec":time.perf_counter()-t0,
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print(text,flush=True)


if __name__=="__main__":
    main()
