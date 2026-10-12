"""Match the GRU cell-held-out folds with a full-source-lifecycle Ridge control."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results"
MODELS = STUDY / "models"
FEATURES = [
    "phase_Ns", "duration_s", "I_mean_A", "I_peak_A",
    "Vmin_V", "Vmax_V", "Tmean_C", "Tmax_C",
]


def _metrics(frame: pd.DataFrame, pred: np.ndarray, method: str, fold: int) -> dict:
    scored = frame[["cell_id", "charge_cycle_index", "SOC_proxy"]].copy()
    scored["pred"] = pred
    error = (scored["pred"] - scored["SOC_proxy"]) * 100
    scored["abs_error_pp"] = np.abs(error)
    scored["sq_error_pp"] = np.square(error)
    cycles = scored.groupby("charge_cycle_index", as_index=False).agg(cycle_mae_pp=("abs_error_pp", "mean"))
    late = cycles.loc[cycles.charge_cycle_index.ge(cycles.charge_cycle_index.quantile(0.75))]
    return {
        "cell_id": str(scored.cell_id.iloc[0]),
        "fold": fold,
        "method": method,
        "test_cycles": int(len(cycles)),
        "proxy_MAE_pp": float(scored.abs_error_pp.mean()),
        "proxy_RMSE_pp": float(np.sqrt(scored.sq_error_pp.mean())),
        "late_quartile_proxy_MAE_pp": float(late.cycle_mae_pp.mean()),
        "P95_cycle_MAE_pp": float(cycles.cycle_mae_pp.quantile(0.95)),
        "worst_cycle_MAE_pp": float(cycles.cycle_mae_pp.max()),
    }


def _fit_head(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([pred, np.ones_like(pred)])
    prior = np.array([1.0, 0.0])
    coeff = np.linalg.solve(design.T @ design + 3.0 * np.eye(2), design.T @ target + 3.0 * prior)
    return float(np.clip(coeff[0], 0.5, 1.5)), float(np.clip(coeff[1], -0.15, 0.15))


def main() -> None:
    labels = pd.read_csv(RESULTS / "soc_proxy_phase_labels.csv")
    labels = labels.sort_values(["cell_id", "charge_cycle_index", "phase_Ns"]).reset_index(drop=True)
    metric_rows = []
    prediction_rows = []
    for fold in range(1, 6):
        fold_config = json.loads((MODELS / f"phase_gru_fold{fold}_seed17_scaler.json").read_text(encoding="utf-8"))
        held_cells = fold_config["heldout_cells"]
        source = labels.loc[~labels.cell_id.isin(held_cells)].copy()
        model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        model.fit(source[FEATURES], source["SOC_proxy"])
        for cell_id, cell in labels.loc[labels.cell_id.isin(held_cells)].groupby("cell_id", sort=True):
            cell = cell.sort_values(["charge_cycle_index", "phase_Ns"]).reset_index(drop=True)
            cycles = sorted(cell.charge_cycle_index.astype(int).unique().tolist())
            n20 = set(cycles[:20])
            pred = np.clip(model.predict(cell[FEATURES]), 0, 1)
            n20_mask = cell.charge_cycle_index.isin(n20).to_numpy()
            test_mask = ~n20_mask
            test = cell.loc[test_mask].copy()
            metric_rows.append(_metrics(test, pred[test_mask], "source_full_lifecycle_Ridge", fold))
            scale, bias = _fit_head(pred[n20_mask], cell.loc[n20_mask, "SOC_proxy"].to_numpy(dtype=float))
            adapted = np.clip(scale * pred + bias, 0, 1)
            metric_rows.append(_metrics(test, adapted[test_mask], "source_full_lifecycle_Ridge_plus_target_N20_affine", fold))
            out = test[["cell_id", "charge_cycle_index", "phase_Ns", "SOC_proxy"]].copy()
            out["fold"] = fold
            out["source_full_lifecycle_Ridge_prediction"] = pred[test_mask]
            out["source_full_lifecycle_Ridge_plus_N20_prediction"] = adapted[test_mask]
            prediction_rows.append(out)
        print(f"Completed full-source Ridge control for fold {fold}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS / "source_full_ridge_crosscell_metrics.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(RESULTS / "source_full_ridge_crosscell_predictions.csv", index=False)
    summary = (
        metrics.groupby("method", as_index=False)
        .agg(
            cells=("cell_id", "nunique"),
            median_proxy_MAE_pp=("proxy_MAE_pp", "median"),
            mean_proxy_MAE_pp=("proxy_MAE_pp", "mean"),
            median_proxy_RMSE_pp=("proxy_RMSE_pp", "median"),
            median_late_quartile_proxy_MAE_pp=("late_quartile_proxy_MAE_pp", "median"),
            median_P95_cycle_MAE_pp=("P95_cycle_MAE_pp", "median"),
        )
    )
    summary.to_csv(RESULTS / "source_full_ridge_crosscell_summary.csv", index=False)
    print(f"Wrote {len(metrics)} source Ridge evaluations across {metrics.cell_id.nunique()} cells")


if __name__ == "__main__":
    main()
