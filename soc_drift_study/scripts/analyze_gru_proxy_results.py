"""Cell-paired summary of leave-cell-out GRU proxy-label experiments."""

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
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261012


def main() -> None:
    gru = pd.read_csv(RESULTS / "gru_proxy_crosscell_metrics.csv")
    baseline = pd.read_csv(RESULTS / "soc_proxy_baseline_metrics.csv")
    per_cell_gru = (
        gru.groupby(["cell_id", "method"], as_index=False)
        .agg(proxy_MAE_pp=("proxy_MAE_pp", "median"),
             proxy_RMSE_pp=("proxy_RMSE_pp", "median"),
             late_quartile_proxy_MAE_pp=("late_quartile_proxy_MAE_pp", "median"),
             P95_cycle_MAE_pp=("P95_cycle_MAE_pp", "median"))
    )
    per_cell_gru.to_csv(RESULTS / "gru_proxy_cell_summary.csv", index=False)
    baseline = baseline.loc[baseline["model"].isin([
        "target_cell_N20_ridge", "source_cells_first20", "source_plus_target_N20"
    ])]
    baseline_pivot = baseline.pivot(index="cell_id", columns="model", values="proxy_MAE_pp")
    gru_pivot = per_cell_gru.pivot(index="cell_id", columns="method", values="proxy_MAE_pp")
    full_ridge = pd.read_csv(RESULTS / "source_full_ridge_crosscell_metrics.csv")
    full_ridge_pivot = full_ridge.pivot(index="cell_id", columns="method", values="proxy_MAE_pp")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    base_methods = [
        "target_cell_N20_ridge", "source_cells_first20", "source_plus_target_N20",
        "source_full_lifecycle_Ridge", "source_full_lifecycle_Ridge_plus_target_N20_affine",
    ]
    gru_methods = ["source_GRU_leave_cells_out", "source_GRU_plus_target_N20_affine"]
    for gru_method in gru_methods:
        for baseline_method in base_methods:
            cells = sorted(set(gru_pivot.index) & set(baseline_pivot.index))
            reference = baseline_pivot[baseline_method] if baseline_method in baseline_pivot else full_ridge_pivot[baseline_method]
            delta = gru_pivot.loc[cells, gru_method].to_numpy() - reference.loc[cells].to_numpy()
            draws = rng.choice(delta, size=(BOOTSTRAP_DRAWS, len(delta)), replace=True).mean(axis=1)
            rows.append({
                "gru_method": gru_method,
                "baseline_method": baseline_method,
                "cells": len(cells),
                "gru_median_proxy_MAE_pp": float(gru_pivot.loc[cells, gru_method].median()),
                "baseline_median_proxy_MAE_pp": float(reference.loc[cells].median()),
                "mean_paired_delta_gru_minus_baseline_pp": float(delta.mean()),
                "median_paired_delta_pp": float(np.median(delta)),
                "bootstrap_95_low_pp": float(np.quantile(draws, 0.025)),
                "bootstrap_95_high_pp": float(np.quantile(draws, 0.975)),
                "cells_gru_better": int((delta < 0).sum()),
            })
    comparison = pd.DataFrame(rows)
    gwide = per_cell_gru.pivot(index="cell_id", columns="method", values="proxy_MAE_pp")
    cell_delta = (
        gwide["source_GRU_plus_target_N20_affine"] - gwide["source_GRU_leave_cells_out"]
    ).to_numpy(dtype=float)
    draws = rng.choice(cell_delta, size=(BOOTSTRAP_DRAWS, len(cell_delta)), replace=True).mean(axis=1)
    comparison = pd.concat([comparison, pd.DataFrame([{
        "gru_method": "source_GRU_plus_target_N20_affine",
        "baseline_method": "source_GRU_leave_cells_out",
        "cells": len(cell_delta),
        "gru_median_proxy_MAE_pp": float(gwide["source_GRU_plus_target_N20_affine"].median()),
        "baseline_median_proxy_MAE_pp": float(gwide["source_GRU_leave_cells_out"].median()),
        "mean_paired_delta_gru_minus_baseline_pp": float(cell_delta.mean()),
        "median_paired_delta_pp": float(np.median(cell_delta)),
        "bootstrap_95_low_pp": float(np.quantile(draws, 0.025)),
        "bootstrap_95_high_pp": float(np.quantile(draws, 0.975)),
        "cells_gru_better": int((cell_delta < 0).sum()),
    }])], ignore_index=True)
    comparison.to_csv(RESULTS / "gru_proxy_paired_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    methods = [*base_methods, *gru_methods]
    data = [
        baseline_pivot[method].to_numpy() if method in baseline_pivot else
        full_ridge_pivot[method].to_numpy() if method in full_ridge_pivot else gru_pivot[method].to_numpy()
        for method in methods
    ]
    labels = [
        "Target N20 Ridge", "Source N20 Ridge", "Source + target N20 Ridge",
        "Source full-life Ridge", "Source full-life Ridge + N20",
        "Source GRU", "Source GRU + N20 affine",
    ]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Cell MAE vs SOC proxy (pp)")
    ax.set_title("Leave-cell-out GRU and Ridge baselines (proxy labels)")
    ax.tick_params(axis="x", labelrotation=23)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "gru_ridge_proxy_comparison.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {len(comparison)} paired GRU comparisons across {baseline_pivot.shape[0]} cells")


if __name__ == "__main__":
    main()
