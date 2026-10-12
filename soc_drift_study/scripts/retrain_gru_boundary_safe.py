"""Reproduce the original cell-held-out source GRU with *cell-local* chunks.

Original train_gru_proxy_crosscell.py packs 12-step sequence chunks AFTER
concatenating all source cells, so some chunks can span cell boundaries.
This audit changes ONLY the source packing step. Target inference already
runs once per held-out cell and remains unchanged.

Source training uses original historical SOC_proxy (past RPT capacity
normalized CC). The GRU is a frozen offline model; target online updating is
a separate, RPT-free experiment (see strong_gru_mission_adapter.py).
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ.setdefault("TF_DETERMINISTIC_OPS","1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from train_gru_proxy_crosscell import (
    STUDY, RESULTS, FEATURES, SEQ_LEN, HIDDEN_SIZE, BATCH_SIZE,
    EPOCHS, SEEDS, FOLDS, _make_chunks, _build_model, _predict, _metric
)


def pack_cellwise(frame:pd.DataFrame,scaler:StandardScaler):
    """Never pack phase summaries from two different source cells in one sequence."""
    arr=[];targets=[];masks=[];ownership=[]
    sorted_cellwise=frame.sort_values(["cell_id","charge_cycle_index","phase_Ns"])
    for cid,g in sorted_cellwise.groupby("cell_id",sort=True):
        x,y,mask=_make_chunks(g,scaler)
        arr.append(x); targets.append(y); masks.append(mask)
        ownership.extend([cid]*len(x))
        assert int(mask.sum())==len(g)
    if not arr:
        raise ValueError("No labeled source data")
    x=np.concatenate(arr,axis=0)
    y=np.concatenate(targets,axis=0)
    m=np.concatenate(masks,axis=0)
    assert int(m.sum())==len(frame)
    assert len(ownership)==len(x)
    return x,y,m,ownership


def run(out:Path):
    t0=time.perf_counter()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    (out/"models").mkdir(exist_ok=True)
    df=pd.read_csv(RESULTS/"soc_proxy_phase_labels.csv").sort_values(
        ["cell_id","charge_cycle_index","phase_Ns"]
    ).reset_index(drop=True)
    cells=sorted(df.cell_id.unique())
    assert len(cells)==22
    folds=[list(x) for x in np.array_split(np.random.default_rng(20261012).permutation(cells),FOLDS)]
    stats=[];pred=[];packing=[]
    for seed in SEEDS:
        for idx,held in enumerate(folds):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed+idx*1000)
            train=df[~df.cell_id.isin(held)]
            target=df[df.cell_id.isin(held)]
            scaler=StandardScaler().fit(train[FEATURES].to_numpy(dtype=np.float32))
            x,y,m,owners=pack_cellwise(train,scaler)
            model=_build_model()
            model.fit(x,y[:,:,None],sample_weight=m,
                      batch_size=BATCH_SIZE,epochs=EPOCHS,shuffle=True,verbose=0)
            model.save_weights(out/"models"/f"safe_fold{idx+1}_seed{seed}.weights.h5")
            (out/"models"/f"safe_fold{idx+1}_seed{seed}_scaler.json").write_text(
                json.dumps({
                    "features":FEATURES,"mean":scaler.mean_.tolist(),
                    "scale":scaler.scale_.tolist(),
                    "source_cells":sorted(train.cell_id.unique()),
                    "test_cells":held,"sequence_len":SEQ_LEN,
                    "normalization_only_source":True,
                    "cell_local_source_chunks":True,
                },indent=2),encoding="utf-8")
            for cell,g in target.groupby("cell_id",sort=True):
                g=g.sort_values(["charge_cycle_index","phase_Ns"])
                n20=set(sorted(g.charge_cycle_index.unique())[:20])
                mask=~g.charge_cycle_index.isin(n20)
                test=g[mask]
                all_y=_predict(model,g,scaler)
                future=all_y[mask.to_numpy()]
                stats.append(_metric(test,future,cell,int(seed),idx+1,"source_GRU_cell_boundary_safe"))
                f=test[["cell_id","charge_cycle_index","phase_Ns","SOC_proxy"]].copy()
                f["seed"]=seed
                f["fold"]=idx+1
                f["source_GRU_prediction"]=future
                pred.append(f)
            packing.append({
                "fold":idx+1,"seed":seed,
                "cells_source":int(train.cell_id.nunique()),
                "cells_heldout":int(target.cell_id.nunique()),
                "source_valid_phase_rows":int(m.sum()),
                "source_chunks":len(owners),
                "cell_local_groups":len(set(owners)),
            })
            print(f"boundary-safe fold={idx+1}/{FOLDS} seed={seed} elapsed={time.perf_counter()-t0:.1f}s",flush=True)

    metrics=pd.DataFrame(stats)
    preds=pd.concat(pred,ignore_index=True)
    assert len(metrics)==22*len(SEEDS)
    assert preds.groupby(["cell_id","seed"]).size().shape[0]==22*len(SEEDS)
    assert not preds.duplicated(["cell_id","charge_cycle_index","phase_Ns","seed"]).any()
    metrics.to_csv(out/"boundary_safe_gru_metrics.csv",index=False)
    preds.to_csv(out/"boundary_safe_gru_predictions.csv",index=False)
    pd.DataFrame(packing).to_csv(out/"source_pack_manifest.csv",index=False)
    old=pd.read_csv(RESULTS/"gru_proxy_crosscell_metrics.csv")
    old=old[old.method=="source_GRU_leave_cells_out"][["cell_id","seed","proxy_MAE_pp"]]
    new=metrics[["cell_id","seed","proxy_MAE_pp"]].rename(columns={"proxy_MAE_pp":"safe"})
    matched=new.merge(old,on=["cell_id","seed"],validate="one_to_one")
    matched["delta_safe_minus_original_pp"]=matched.safe-matched.proxy_MAE_pp
    matched.to_csv(out/"boundary_effect_cell_seed.csv",index=False)
    cell=matched.groupby("cell_id").agg(
        safe_MAE=("safe","mean"),old_MAE=("proxy_MAE_pp","mean"),
        delta=("delta_safe_minus_original_pp","mean")
    )
    rng=np.random.default_rng(20261012)
    d=cell.delta.to_numpy()
    bs=rng.choice(d,size=(10000,len(d)),replace=True).mean(axis=1)
    ci=np.quantile(bs,[.025,.975])
    duration=time.perf_counter()-t0
    report_text="\n".join([
        "# Corrected source-cell boundaries for 5-fold × 3-seed GRU",
        "",
        "- Only training source sequence packing changed. All source labels remain RPT-anchored SOC_proxy.",
        "- Held-out target inference, folds, random seeds, hidden size, epochs and other hyperparameters unchanged.",
        "- Each 12-step source chunk belongs to EXACTLY one physical cell. Zero target data enters scaler/model fitting.",
        "- All metrics are SOC_proxy pp (not independent measured SOC).",
        "",
        "| Model | Physical cells | Median three-seed cell-MAE (pp) | Mean cell-MAE (pp) |",
        "|---|---:|---:|---:|",
        f"| Original concatenated-source GRU | 22 | {cell.old_MAE.median():.3f} | {cell.old_MAE.mean():.3f} |",
        f"| Source cell-local chunk GRU | 22 | {cell.safe_MAE.median():.3f} | {cell.safe_MAE.mean():.3f} |",
        "",
        f"- Paired cell-mean change (safe minus original): {cell.delta.mean():+.3f} pp, 95% cell Bootstrap [{ci[0]:+.3f},{ci[1]:+.3f}] pp.",
        f"- Source-cell-local model improves {int((cell.delta<0).sum())}/22 physical cells.",
        f"- Total train/infer wall time: {duration:.1f} s.",
        "- The old strong-source-GRU adapter comparison must be repeated with the **boundary-safe** predictions before any final statement.",
        "",
        "Artifacts: per-seed test predictions (for online RPT-free adapter replay), source packing manifest, saved weights/scalers, per-cell comparisons.",
    ])+"\n"
    (out/"report.md").write_text(report_text,encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "change":"source training cell boundaries only",
        "source_SOC_training":"historical completed-prior-RPT normalized SOC_proxy",
        "test_SOC_labels_only_scoring":True,
        "heldout_physical_cells":22,
        "folds":FOLDS,"seeds":SEEDS,"sequence_len":SEQ_LEN,
        "valid_source_mask_checked_each_fold":True,
        "per_cell_chunks":True,"duration_s":duration,
    },indent=2),encoding="utf-8")
    print(report_text,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,required=True)
    args=p.parse_args()
    run(args.output_dir.resolve())
