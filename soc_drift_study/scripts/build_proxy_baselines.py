"""Build a strictly past-capacity SOC proxy and chronological phase baselines.

The labels are coulomb-counted proxies, not independent SOC ground truth. This
script keeps that distinction explicit in every output and uses no future RPT
capacity for a mission cycle.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results"
MODELS = STUDY / "models"
FIGURES = STUDY / "figures"
FEATURES = [
    "phase_Ns",
    "duration_s",
    "I_mean_A",
    "I_peak_A",
    "Vmin_V",
    "Vmax_V",
    "Tmean_C",
    "Tmax_C",
]
N20 = 20
NOMINAL_CAPACITY_AH = 3.0


def build_phase_labels() -> pd.DataFrame:
    segments = pd.read_csv(RESULTS / "segment_summary.csv")
    rpt = pd.read_csv(RESULTS / "capacity_test_candidates.csv")
    active = segments.loc[
        segments["Ns"].isin([4, 5, 6])
        & segments["I_signed_mean_mA"].le(-2000)
        & segments["duration_s"].ge(10)
    ].copy()
    active["I_mean_A"] = active["I_signed_mean_mA"] / 1000
    active["I_peak_A"] = active["I_abs_peak_mA"] / 1000
    active["Vmin_V"] = active["V_min"]
    active["Vmax_V"] = active["V_max"]
    active["Tmax_C"] = active["T_max"]

    label_rows: list[dict[str, object]] = []
    for cell_id, cell_active in active.groupby("cell_id", sort=True):
        cell_rpt = rpt.loc[rpt["cell_id"].eq(cell_id)].sort_values("charge_cycle_index")
        rpt_by_index = (
            cell_rpt.groupby("charge_cycle_index")["QDischarge_max_Ah"].max().sort_index()
        )
        rpt_indices = rpt_by_index.index.to_numpy(dtype=float)
        rpt_capacity = rpt_by_index.to_numpy(dtype=float)

        # One row per phase of each mission. RPT cycles themselves are not
        # treated as missions and never train or test the flight baseline.
        for cycle_index, cycle in cell_active.groupby("charge_cycle_index", sort=True):
            if pd.isna(cycle_index) or float(cycle_index) in set(rpt_indices.tolist()):
                continue
            cycle_index = int(cycle_index)
            previous_rpt_pos = int(np.searchsorted(rpt_indices, cycle_index, side="left") - 1)
            if previous_rpt_pos >= 0:
                capacity_ah = float(rpt_capacity[previous_rpt_pos])
                anchor_index = int(rpt_indices[previous_rpt_pos])
                capacity_source = "latest_completed_candidate_RPT"
            else:
                capacity_ah = NOMINAL_CAPACITY_AH
                anchor_index = 0
                capacity_source = "nominal_3Ah_before_first_RPT"
            if not np.isfinite(capacity_ah) or capacity_ah <= 0:
                continue

            for ns, phase in cycle.groupby("Ns", sort=True):
                weights = phase["rows"].astype(float)
                weight_sum = float(weights.sum())
                qd_ah = float(phase["QDischarge_max_Ah"].max())
                q_previous = qd_ah / capacity_ah
                q_nominal = qd_ah / NOMINAL_CAPACITY_AH
                label_rows.append(
                    {
                        "cell_id": cell_id,
                        "charge_cycle_index": cycle_index,
                        "phase_Ns": int(ns),
                        "rows_in_phase": int(phase["rows"].sum()),
                        "duration_s": float(phase["duration_s"].sum()),
                        "I_mean_A": float(np.average(phase["I_mean_A"], weights=weights)),
                        "I_peak_A": float(phase["I_peak_A"].max()),
                        "Vmin_V": float(phase["Vmin_V"].min()),
                        "Vmax_V": float(phase["Vmax_V"].max()),
                        "Tmean_C": float(np.average(phase["T_mean_C"], weights=weights)),
                        "Tmax_C": float(phase["Tmax_C"].max()),
                        "QDischarge_end_Ah": qd_ah,
                        "capacity_used_Ah": capacity_ah,
                        "capacity_anchor_index": anchor_index,
                        "capacity_source": capacity_source,
                        "SOC_proxy": float(np.clip(1 - q_previous, 0, 1)),
                        "SOC_nominal_3Ah": float(np.clip(1 - q_nominal, 0, 1)),
                        "nominal_capacity_abs_error_pp": float(
                            abs(np.clip(1 - q_nominal, 0, 1) - np.clip(1 - q_previous, 0, 1)) * 100
                        ),
                    }
                )

    labels = pd.DataFrame(label_rows).sort_values(["cell_id", "charge_cycle_index", "phase_Ns"])
    labels.to_csv(RESULTS / "soc_proxy_phase_labels.csv", index=False)
    return labels


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, object]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    model.fit(train[FEATURES], train["SOC_proxy"])
    return np.clip(model.predict(test[FEATURES]), 0, 1), model


def _metrics(test: pd.DataFrame, pred: np.ndarray, model_name: str, n_train: int) -> dict[str, object]:
    result = test[["cell_id", "charge_cycle_index", "phase_Ns", "SOC_proxy", "SOC_nominal_3Ah", "nominal_capacity_abs_error_pp"]].copy()
    result["prediction"] = pred
    result["abs_error_pp"] = (result["prediction"] - result["SOC_proxy"]).abs() * 100
    by_cycle = result.groupby(["cell_id", "charge_cycle_index"], as_index=False).agg(
        cycle_mae_pp=("abs_error_pp", "mean"),
        nominal_capacity_mae_pp=("nominal_capacity_abs_error_pp", "mean"),
    )
    ordered = by_cycle.sort_values("charge_cycle_index")
    late_start = ordered["charge_cycle_index"].quantile(0.75)
    late = ordered.loc[ordered["charge_cycle_index"].ge(late_start)]
    return {
        "cell_id": str(test["cell_id"].iloc[0]),
        "model": model_name,
        "train_phase_rows": int(n_train),
        "test_phase_rows": int(len(test)),
        "test_cycles": int(test["charge_cycle_index"].nunique()),
        "proxy_MAE_pp": float(result["abs_error_pp"].mean()),
        "proxy_RMSE_pp": float(np.sqrt(np.mean(np.square((result["prediction"] - result["SOC_proxy"]) * 100)))),
        "late_quartile_proxy_MAE_pp": float(late["cycle_mae_pp"].mean()) if len(late) else np.nan,
        "P95_cycle_MAE_pp": float(by_cycle["cycle_mae_pp"].quantile(0.95)),
        "cycles_over_2pp_pct": float((by_cycle["cycle_mae_pp"] > 2).mean() * 100),
        "cycles_over_3pp_pct": float((by_cycle["cycle_mae_pp"] > 3).mean() * 100),
        "nominal_3Ah_vs_proxy_MAE_pp": float(by_cycle["nominal_capacity_mae_pp"].mean()),
        "late_nominal_3Ah_vs_proxy_MAE_pp": float(late["nominal_capacity_mae_pp"].mean()) if len(late) else np.nan,
    }, result


def train_baselines(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    MODELS.mkdir(parents=True, exist_ok=True)
    per_cell_metrics: list[dict[str, object]] = []
    row_predictions: list[pd.DataFrame] = []
    models: dict[str, dict[str, object]] = {}

    cycle_first = (
        labels[["cell_id", "charge_cycle_index"]]
        .drop_duplicates()
        .sort_values(["cell_id", "charge_cycle_index"])
    )
    first20_by_cell = {
        cell_id: set(group.head(N20)["charge_cycle_index"].astype(int).tolist())
        for cell_id, group in cycle_first.groupby("cell_id", sort=True)
    }
    labels = labels.copy()
    labels["_in_first20"] = [
        int(cycle) in first20_by_cell.get(cell_id, set())
        for cell_id, cycle in zip(labels["cell_id"], labels["charge_cycle_index"])
    ]

    for cell_id, cell in labels.groupby("cell_id", sort=True):
        train_cycles = first20_by_cell[cell_id]
        train = cell.loc[cell["charge_cycle_index"].isin(train_cycles)]
        test = cell.loc[~cell["charge_cycle_index"].isin(train_cycles)]
        if len(train) < 10 or test.empty:
            continue
        pred, model = _fit_predict(train, test)
        metrics, result = _metrics(test, pred, "target_cell_N20_ridge", len(train))
        per_cell_metrics.append(metrics)
        result["model"] = "target_cell_N20_ridge"
        row_predictions.append(result)
        models.setdefault(cell_id, {})["target_cell_N20_ridge"] = model

        # Leave-one-cell-out source model, trained only on each source cell's
        # first 20 missions. Then add target N20 labels as a separate causal
        # initialization comparison.
        source_rows = labels.loc[labels["_in_first20"] & labels["cell_id"].ne(cell_id)].drop(columns="_in_first20")
        if not source_rows.empty:
            pred_src, src_model = _fit_predict(source_rows, test)
            metrics_src, result_src = _metrics(test, pred_src, "source_cells_first20", len(source_rows))
            per_cell_metrics.append(metrics_src)
            result_src["model"] = "source_cells_first20"
            row_predictions.append(result_src)
            models[cell_id]["source_cells_first20"] = src_model

            source_plus_target = pd.concat([source_rows, train], ignore_index=True)
            pred_n20, n20_model = _fit_predict(source_plus_target, test)
            metrics_n20, result_n20 = _metrics(test, pred_n20, "source_plus_target_N20", len(source_plus_target))
            per_cell_metrics.append(metrics_n20)
            result_n20["model"] = "source_plus_target_N20"
            row_predictions.append(result_n20)
            models[cell_id]["source_plus_target_N20"] = n20_model

    metric_frame = pd.DataFrame(per_cell_metrics)
    prediction_frame = pd.concat(row_predictions, ignore_index=True)
    metric_frame.to_csv(RESULTS / "soc_proxy_baseline_metrics.csv", index=False)
    prediction_frame.to_csv(RESULTS / "soc_proxy_baseline_predictions.csv", index=False)
    joblib.dump(models, MODELS / "phase_ridge_models.joblib", compress=3)
    (MODELS / "phase_ridge_metadata.json").write_text(
        json.dumps(
            {
                "model": "StandardScaler + Ridge(alpha=5.0)",
                "input_features": FEATURES,
                "target": "SOC_proxy",
                "target_definition": "1 - QDischarge / latest strictly earlier candidate RPT capacity",
                "training_protocol": "chronological per-cell first 20 missions; leave-one-cell-out source comparison",
                "evaluation": "later missions only; metrics are against a coulomb-counted proxy, not ground truth",
                "fixed_seed_needed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metric_frame, prediction_frame


def plot_errors(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    # Per-cell median metrics preserve the physical cell as the unit of analysis.
    summary = metrics.groupby("model")["proxy_MAE_pp"].agg(["median", "mean", "count"]).reset_index()
    summary.to_csv(RESULTS / "soc_proxy_baseline_group_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    order = ["source_cells_first20", "source_plus_target_N20", "target_cell_N20_ridge"]
    data = [metrics.loc[metrics["model"].eq(name), "proxy_MAE_pp"].dropna().to_numpy() for name in order]
    ax.boxplot(data, tick_labels=["Source only", "Source + target N20", "Target N20 only"], showmeans=True)
    ax.set_ylabel("Later-life phase MAE vs SOC proxy (percentage points)")
    ax.set_title("Chronological proxy baseline; one value per physical cell and model")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "soc_proxy_baseline_by_model.png", dpi=180)
    plt.close(fig)

    chosen = ["VAH01", "VAH07", "VAH12", "VAH27"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, cell_id in zip(axes.flat, chosen):
        part = predictions.loc[
            predictions["cell_id"].eq(cell_id)
            & predictions["model"].eq("target_cell_N20_ridge")
        ]
        cycle = part.groupby("charge_cycle_index")["abs_error_pp"].mean().sort_index()
        ax.plot(cycle.index, cycle.values, linewidth=0.9, alpha=0.65, label="cycle error")
        ax.plot(cycle.index, cycle.rolling(51, min_periods=10).median(), linewidth=2, label="51-cycle median")
        ax.set_title(cell_id)
        ax.set_xlabel("Sustained charge-block index (proxy)")
        ax.set_ylabel("Proxy MAE (pp)")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Target-cell N20 phase Ridge: later-life error against coulomb-counted proxy")
    fig.tight_layout()
    fig.savefig(FIGURES / "soc_proxy_error_lifecycle.png", dpi=180)
    plt.close(fig)


def main() -> None:
    labels = build_phase_labels()
    metrics, predictions = train_baselines(labels)
    plot_errors(metrics, predictions)
    print(f"Wrote {len(labels)} phase labels and {len(metrics)} cell-model evaluations")


if __name__ == "__main__":
    main()
