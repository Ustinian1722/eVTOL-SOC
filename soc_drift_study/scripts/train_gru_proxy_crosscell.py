"""Train a small causal GRU over mission-phase summaries with cell-held-out folds.

Targets are SOC_proxy labels. This is a proxy-label sequence baseline, not a
ground-truth SOC model or a replication of the FiberResearch GRU/SRA pipeline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results"
MODELS = STUDY / "models"
LOGS = STUDY / "logs"
FIGURES = STUDY / "figures"
FEATURES = [
    "phase_Ns", "duration_s", "I_mean_A", "I_peak_A",
    "Vmin_V", "Vmax_V", "Tmean_C", "Tmax_C",
]
SEQ_LEN = 12  # four mission cycles × three phase summaries
HIDDEN_SIZE = 16
EPOCHS = 20
BATCH_SIZE = 128
SEEDS = [17, 31, 43]
FOLDS = 5
HEAD_PRIOR_STRENGTH = 3.0


def _make_chunks(frame: pd.DataFrame, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = scaler.transform(frame[FEATURES].to_numpy(dtype=np.float32)).astype(np.float32)
    y = frame["SOC_proxy"].to_numpy(dtype=np.float32)
    count = int(np.ceil(len(frame) / SEQ_LEN))
    x_chunks = np.zeros((count, SEQ_LEN, len(FEATURES)), dtype=np.float32)
    y_chunks = np.zeros((count, SEQ_LEN), dtype=np.float32)
    mask = np.zeros((count, SEQ_LEN), dtype=np.float32)
    for chunk_i, start in enumerate(range(0, len(frame), SEQ_LEN)):
        stop = min(start + SEQ_LEN, len(frame))
        n = stop - start
        x_chunks[chunk_i, :n] = x[start:stop]
        y_chunks[chunk_i, :n] = y[start:stop]
        mask[chunk_i, :n] = 1.0
    return x_chunks, y_chunks, mask


def _build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(SEQ_LEN, len(FEATURES)), name="phase_sequence")
    masked = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    sequence = tf.keras.layers.GRU(HIDDEN_SIZE, return_sequences=True, name="gru")(masked)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="soc_proxy_head")(sequence)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
                  loss=tf.keras.losses.Huber())
    return model


def _predict(model: tf.keras.Model, frame: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    x, _, mask = _make_chunks(frame, scaler)
    prediction = model.predict(x, batch_size=512, verbose=0)[:, :, 0]
    return prediction[mask.astype(bool)]


def _fit_affine(prior_prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([prior_prediction, np.ones_like(prior_prediction)])
    prior = np.array([1.0, 0.0])
    params = np.linalg.solve(
        design.T @ design + HEAD_PRIOR_STRENGTH * np.eye(2),
        design.T @ target + HEAD_PRIOR_STRENGTH * prior,
    )
    return float(np.clip(params[0], 0.5, 1.5)), float(np.clip(params[1], -0.15, 0.15))


def _metric(frame: pd.DataFrame, prediction: np.ndarray, cell_id: str, seed: int, fold: int, method: str) -> dict:
    scored = frame[["charge_cycle_index", "SOC_proxy"]].copy()
    scored["prediction"] = prediction
    scored["abs_error_pp"] = (scored["prediction"] - scored["SOC_proxy"]).abs() * 100
    scored["sq_error_pp"] = np.square((scored["prediction"] - scored["SOC_proxy"]) * 100)
    cycles = scored.groupby("charge_cycle_index", as_index=False).agg(
        cycle_mae_pp=("abs_error_pp", "mean"),
        cycle_rmse_sq=("sq_error_pp", "mean"),
    )
    late = cycles.loc[cycles["charge_cycle_index"].ge(cycles["charge_cycle_index"].quantile(0.75))]
    return {
        "cell_id": cell_id,
        "seed": seed,
        "fold": fold,
        "method": method,
        "test_phase_rows": int(len(scored)),
        "test_cycles": int(len(cycles)),
        "proxy_MAE_pp": float(scored["abs_error_pp"].mean()),
        "proxy_RMSE_pp": float(np.sqrt(scored["sq_error_pp"].mean())),
        "late_quartile_proxy_MAE_pp": float(late["cycle_mae_pp"].mean()),
        "P95_cycle_MAE_pp": float(cycles["cycle_mae_pp"].quantile(0.95)),
        "worst_cycle_MAE_pp": float(cycles["cycle_mae_pp"].max()),
    }


def main() -> None:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    labels = pd.read_csv(RESULTS / "soc_proxy_phase_labels.csv")
    labels = labels.sort_values(["cell_id", "charge_cycle_index", "phase_Ns"]).reset_index(drop=True)
    cells = sorted(labels["cell_id"].unique().tolist())
    folds = [list(x) for x in np.array_split(np.random.default_rng(20261012).permutation(cells), FOLDS)]
    metrics = []
    predictions = []
    epoch_logs = []
    start_all = time.perf_counter()

    for seed in SEEDS:
        for fold_i, held_cells in enumerate(folds):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed + fold_i * 1000)
            train = labels.loc[~labels["cell_id"].isin(held_cells)].copy()
            test = labels.loc[labels["cell_id"].isin(held_cells)].copy()
            scaler = StandardScaler().fit(train[FEATURES].to_numpy(dtype=np.float32))
            x, y, mask = _make_chunks(train, scaler)
            model = _build_model()
            history = model.fit(
                x,
                y[:, :, None],
                sample_weight=mask,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                shuffle=True,
                verbose=0,
            )
            for epoch, loss in enumerate(history.history["loss"], start=1):
                epoch_logs.append({
                    "seed": seed,
                    "fold": fold_i,
                    "epoch": epoch,
                    "train_huber_loss": float(loss),
                    "source_cells": len(train.cell_id.unique()),
                    "heldout_cells": ";".join(held_cells),
                })

            model_path = MODELS / f"phase_gru_fold{fold_i + 1}_seed{seed}.weights.h5"
            model.save_weights(model_path)
            scaler_path = MODELS / f"phase_gru_fold{fold_i + 1}_seed{seed}_scaler.json"
            scaler_path.write_text(json.dumps({
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "features": FEATURES,
                "source_cells": sorted(train.cell_id.unique().tolist()),
                "heldout_cells": held_cells,
                "sequence_length": SEQ_LEN,
                "hidden_size": HIDDEN_SIZE,
                "seed": seed,
            }, indent=2), encoding="utf-8")

            for cell_id, cell in test.groupby("cell_id", sort=True):
                cell = cell.sort_values(["charge_cycle_index", "phase_Ns"]).reset_index(drop=True)
                unique_cycles = sorted(cell["charge_cycle_index"].astype(int).unique().tolist())
                train_cycles = set(unique_cycles[:20])
                target_test = cell.loc[~cell["charge_cycle_index"].isin(train_cycles)].copy()
                source_prediction = _predict(model, cell, scaler)
                n20_mask = cell["charge_cycle_index"].isin(train_cycles).to_numpy()
                future_mask = ~n20_mask
                scale, bias = _fit_affine(
                    source_prediction[n20_mask],
                    cell.loc[n20_mask, "SOC_proxy"].to_numpy(dtype=float),
                )
                adapted = np.clip(scale * source_prediction + bias, 0, 1)
                for method, values in [
                    ("source_GRU_leave_cells_out", source_prediction[future_mask]),
                    ("source_GRU_plus_target_N20_affine", adapted[future_mask]),
                ]:
                    metrics.append(_metric(target_test, values, cell_id, seed, fold_i + 1, method))
                out = target_test[["cell_id", "charge_cycle_index", "phase_Ns", "SOC_proxy"]].copy()
                out["seed"] = seed
                out["fold"] = fold_i + 1
                out["source_GRU_prediction"] = source_prediction[future_mask]
                out["source_GRU_plus_N20_prediction"] = adapted[future_mask]
                predictions.append(out)
            print(f"Finished GRU seed={seed}, fold={fold_i + 1}; cells={len(held_cells)}", flush=True)

    metrics_frame = pd.DataFrame(metrics)
    predictions_frame = pd.concat(predictions, ignore_index=True)
    metrics_frame.to_csv(RESULTS / "gru_proxy_crosscell_metrics.csv", index=False)
    predictions_frame.to_csv(RESULTS / "gru_proxy_crosscell_predictions.csv", index=False)
    pd.DataFrame(epoch_logs).to_csv(LOGS / "gru_proxy_training_epochs.csv", index=False)
    summary = (
        metrics_frame.groupby("method", as_index=False)
        .agg(
            cell_seed_evaluations=("cell_id", "size"),
            cells=("cell_id", "nunique"),
            median_proxy_MAE_pp=("proxy_MAE_pp", "median"),
            mean_proxy_MAE_pp=("proxy_MAE_pp", "mean"),
            median_proxy_RMSE_pp=("proxy_RMSE_pp", "median"),
            median_late_quartile_proxy_MAE_pp=("late_quartile_proxy_MAE_pp", "median"),
            median_P95_cycle_MAE_pp=("P95_cycle_MAE_pp", "median"),
            median_worst_cycle_MAE_pp=("worst_cycle_MAE_pp", "median"),
        )
    )
    summary.to_csv(RESULTS / "gru_proxy_crosscell_summary.csv", index=False)
    (RESULTS / "gru_proxy_crosscell_metadata.json").write_text(
        json.dumps({
            "model": "one-layer causal GRU over phase-summary observations",
            "sequence_length_phase_summaries": SEQ_LEN,
            "context_approximation": "current and up to 11 prior phase summaries within each 12-step chunk; hidden state resets at chunk boundaries",
            "hidden_size": HIDDEN_SIZE,
            "epochs": EPOCHS,
            "seeds": SEEDS,
            "cell_group_folds": len(folds),
            "target": "SOC_proxy, not independent true SOC",
            "test_protocol": "leave held-out physical cells out of scaler/model training; evaluate later target missions; affine N20 head uses only target first 20 missions",
            "training_rows_scope": "all labeled source-cell phase summaries; held-out target cells never enter source training",
            "limitation": "phase-summary GRU, not raw high-rate V/I/T sequence and not FiberResearch GRU/SRA replication",
            "wall_time_seconds": time.perf_counter() - start_all,
            "device": "cpu",
        }, indent=2), encoding="utf-8")

    per_cell = metrics_frame.groupby(["method", "cell_id"], as_index=False)["proxy_MAE_pp"].median()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = ["source_GRU_leave_cells_out", "source_GRU_plus_target_N20_affine"]
    data = [per_cell.loc[per_cell.method.eq(name), "proxy_MAE_pp"].to_numpy() for name in names]
    ax.boxplot(data, tick_labels=["Source GRU", "Source GRU + target N20 affine"], showfliers=False)
    ax.set_ylabel("Cell median post-N20 MAE vs SOC proxy (pp)")
    ax.set_title("Cell-held-out phase-summary GRU: proxy-label results")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "gru_proxy_crosscell.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {len(metrics_frame)} GRU cell-seed evaluations; elapsed {time.perf_counter() - start_all:.1f}s")


if __name__ == "__main__":
    main()
