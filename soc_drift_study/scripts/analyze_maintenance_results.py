"""Summarize cell-paired maintenance effects and detection diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"
BOOTSTRAP_SEED = 20261012
BOOTSTRAP_DRAWS = 10000


def paired_bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _with_random_mean(metrics: pd.DataFrame) -> pd.DataFrame:
    random = metrics.loc[metrics["method"].str.startswith("Random_seed")]
    random_mean = (
        random.groupby(["cell_id", "budget"], as_index=False)
        .agg(**{col: (col, "mean") for col in [
            "proxy_MAE_pp", "late_quartile_proxy_MAE_pp", "P95_cycle_MAE_pp",
            "updates_used", "reference_test_hours_total", "cycles_over_2pp_pct",
            "cycles_over_3pp_pct",
        ]})
    )
    random_mean["method"] = "Random_mean5"
    random_mean["seed"] = -1
    frozen = metrics.loc[metrics["method"].eq("Frozen")].copy()
    frozen_rows = []
    for budget in [2, 4, 6]:
        duplicate = frozen.copy()
        duplicate["budget"] = budget
        frozen_rows.append(duplicate)
    return pd.concat([metrics, random_mean, *frozen_rows], ignore_index=True)


def summarize_paired_effects(metrics: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records = []
    comparison = _with_random_mean(metrics)
    for budget in [2, 4, 6]:
        part = comparison.loc[comparison["budget"].eq(budget)]
        frozen = part.loc[part["method"].eq("Frozen")].set_index("cell_id")
        periodic = part.loc[part["method"].eq("Periodic_K")].set_index("cell_id")
        for method, group in part.groupby("method", sort=True):
            if method == "Frozen":
                continue
            current = group.set_index("cell_id")
            cells = sorted(set(frozen.index) & set(periodic.index) & set(current.index))
            if len(cells) < 3:
                continue
            d_frozen = current.loc[cells, "proxy_MAE_pp"].to_numpy() - frozen.loc[cells, "proxy_MAE_pp"].to_numpy()
            d_periodic = current.loc[cells, "proxy_MAE_pp"].to_numpy() - periodic.loc[cells, "proxy_MAE_pp"].to_numpy()
            mean_frozen, low_frozen, high_frozen = paired_bootstrap(d_frozen, rng)
            mean_periodic, low_periodic, high_periodic = paired_bootstrap(d_periodic, rng)
            records.append({
                "method": method,
                "budget": budget,
                "cells": len(cells),
                "median_proxy_MAE_pp": float(current.loc[cells, "proxy_MAE_pp"].median()),
                "mean_delta_vs_frozen_pp": mean_frozen,
                "frozen_bootstrap_95_low_pp": low_frozen,
                "frozen_bootstrap_95_high_pp": high_frozen,
                "cells_better_than_frozen": int((d_frozen < 0).sum()),
                "mean_delta_vs_periodic_pp": mean_periodic,
                "periodic_bootstrap_95_low_pp": low_periodic,
                "periodic_bootstrap_95_high_pp": high_periodic,
                "cells_better_than_periodic": int((d_periodic < 0).sum()),
                "median_reference_hours": float(current.loc[cells, "reference_test_hours_total"].median()),
                "median_updates": float(current.loc[cells, "updates_used"].median()),
            })
    return pd.DataFrame(records)


def summarize_indicators() -> pd.DataFrame:
    corr = pd.read_csv(RESULTS / "drift_indicator_correlations.csv")
    return (
        corr.groupby("indicator", as_index=False)
        .agg(
            cells=("cell_id", "nunique"),
            median_current_spearman=("spearman_current_proxy_error", "median"),
            mean_current_spearman=("spearman_current_proxy_error", "mean"),
            positive_current_cells=("spearman_current_proxy_error", lambda x: int((x > 0).sum())),
            median_future50_spearman=("spearman_future_50_cycle_error", "median"),
            mean_future50_spearman=("spearman_future_50_cycle_error", "mean"),
            positive_future50_cells=("spearman_future_50_cycle_error", lambda x: int((x > 0).sum())),
        )
    )


def summarize_onset() -> pd.DataFrame:
    onset = pd.read_csv(RESULTS / "proxy_detection_onset.csv")
    return (
        onset.groupby(["method", "budget"], as_index=False)
        .agg(
            cells_with_first_reference=("cell_id", "nunique"),
            median_reference_minus_proxy_onset=("reference_minus_proxy_onset_cycles", "median"),
            median_first_reference=("first_reference_cycle_index", "median"),
            fraction_reference_before_proxy_onset=("pre_onset_reference", "mean"),
        )
    )


def plot_strategy_errors(metrics: pd.DataFrame) -> None:
    methods = ["Frozen", "Periodic_K", "Random_mean5", "ModelDisagreement", "VoltageResidual", "ADWINlike_Voltage", "Periodic_K_plus_Risk", "OfflineOracle_proxy_hindsight"]
    short_names = ["Frozen", "Periodic K", "Random mean", "Model disagreement", "Voltage residual", "ADWIN-like", "Periodic + risk", "Oracle (hindsight)"]
    plot_data = _with_random_mean(metrics)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {2: "#0072B2", 4: "#D55E00", 6: "#009E73"}
    x_offsets = {2: -0.18, 4: 0.0, 6: 0.18}
    for budget in [2, 4, 6]:
        subset = plot_data.loc[plot_data["budget"].eq(budget) & plot_data["method"].isin(methods)]
        values = subset.groupby("method")["proxy_MAE_pp"].median()
        for method, value in values.items():
            xpos = methods.index(method) + x_offsets[budget]
            ax.scatter(xpos, value, color=colors[budget], s=34, label=f"B={budget}" if method == methods[0] else None)
    ax.axhline(2.5225, color="#555555", linestyle="--", linewidth=0.8, label="Frozen B=0 reference")
    ax.set_xticks(range(len(methods)), short_names, rotation=24, ha="right")
    ax.set_ylabel("Median post-N20 test MAE vs SOC proxy (pp)")
    ax.set_title("Maintenance updates did not improve the frozen proxy baseline")
    ax.set_ylim(0, 10)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "maintenance_strategy_errors.png", dpi=180)
    plt.close(fig)


def plot_timeline() -> None:
    timelines = pd.read_csv(RESULTS / "proxy_maintenance_timeline.csv")
    cell = timelines.loc[timelines["cell_id"].eq("VAH01") & timelines["budget"].eq(4)].copy()
    methods = ["Periodic_K", "Random_seed11", "ModelDisagreement", "VoltageResidual", "PageHinkley_Voltage", "ADWINlike_Voltage", "Periodic_K_plus_Risk"]
    cell = cell.loc[cell["method"].isin(methods)]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    markers = [
        ("request_cycle_index", "^", "Request", "#D55E00"),
        ("reference_cycle_index", "s", "RPT complete", "#222222"),
        ("first_effective_cycle_index", "o", "Next mission applies", "#0072B2"),
    ]
    for y, method in enumerate(methods):
        rows = cell.loc[cell["method"].eq(method)]
        for _, row in rows.iterrows():
            if pd.notna(row["first_effective_cycle_index"]):
                ax.plot([row["request_cycle_index"], row["reference_cycle_index"], row["first_effective_cycle_index"]], [y, y, y], color="#BBBBBB", linewidth=0.7, zorder=1)
        for column, marker, label, color in markers:
            if method == methods[0] and y == 0:
                ax.scatter([], [], marker=marker, color=color, label=label)
            vals = rows[column].dropna()
            ax.scatter(vals, np.full(len(vals), y), marker=marker, s=30, color=color, zorder=2)
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xlabel("Reconstructed charge-cycle index (ordering proxy)")
    ax.set_title("VAH01, B=4: request → completed RPT → next mission update")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    fig.savefig(FIGURES / "maintenance_timeline_VAH01_B4.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(RESULTS / "proxy_maintenance_metrics.csv")
    oracle_path = RESULTS / "offline_oracle_metrics.csv"
    if oracle_path.exists():
        metrics = pd.concat([metrics, pd.read_csv(oracle_path)], ignore_index=True)
    summarize_paired_effects(metrics).to_csv(RESULTS / "proxy_maintenance_decision_summary.csv", index=False)
    summarize_indicators().to_csv(RESULTS / "drift_indicator_cell_summary.csv", index=False)
    summarize_onset().to_csv(RESULTS / "proxy_detection_group_summary.csv", index=False)
    plot_strategy_errors(metrics)
    plot_timeline()
    print("Wrote paired effect, indicator, onset summaries and maintenance figures")


if __name__ == "__main__":
    main()
