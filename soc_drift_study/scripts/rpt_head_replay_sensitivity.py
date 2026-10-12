"""Compare latest completed RPT head calibration with bounded last-3 replay."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "scripts"))
import proxy_drift_maintenance as md  # noqa: E402


def main() -> None:
    results = STUDY / "results"
    figures = STUDY / "figures"
    segment_fields = [
        "rows", "time_start_s", "time_end_s", "I_abs_peak_mA", "Ns", "cell_id",
        "duration_s", "I_signed_mean_mA", "V_min", "V_max", "T_mean_C", "T_max",
        "QDischarge_max_Ah", "charge_cycle_index",
    ]
    segments = pd.read_csv(results / "segment_summary.csv", usecols=segment_fields)
    labels = pd.read_csv(results / "soc_proxy_phase_labels.csv")
    candidates = pd.read_csv(results / "capacity_test_candidates.csv")
    references_by_cell = md._make_rpt_reference_table(segments, candidates)
    metric_rows = []
    timeline_rows = []

    for cell_id, cell in labels.groupby("cell_id", sort=True):
        cell = cell.sort_values(["charge_cycle_index", "phase_Ns"]).copy()
        cycle_indices = sorted(cell["charge_cycle_index"].astype(int).unique().tolist())
        train_cycles = set(cycle_indices[:md.N20])
        train = cell.loc[cell["charge_cycle_index"].isin(train_cycles)].copy()
        test = cell.loc[~cell["charge_cycle_index"].isin(train_cycles)].copy()
        if test.empty:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        model.fit(train[md.FEATURES], train["SOC_proxy"])
        test["base_prediction"] = np.clip(model.predict(test[md.FEATURES]), 0, 1)
        refs = references_by_cell.get(cell_id, {})
        after_n20 = sorted(index for index in refs if index > max(train_cycles))
        test_indices = sorted(test["charge_cycle_index"].astype(int).unique().tolist())
        for budget in [2, 4, 6]:
            selected = md._timeline_selection(test_indices, after_n20, {}, np.array([]), budget, "periodic", 0)
            for window, name in [(1, "Periodic_K_LatestRPT"), (3, "Periodic_K_ReplayLast3")]:
                metric, updates, _ = md._simulate(cell_id, test, selected, refs, model, replay_window=window)
                if not metric:
                    continue
                metric.update(method=name, budget=budget, replay_window=window)
                metric_rows.append(metric)
                for update in updates:
                    update.update(method=name, budget=budget, replay_window=window)
                    timeline_rows.append(update)
        print(f"Completed head replay sensitivity for {cell_id}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    timelines = pd.DataFrame(timeline_rows)
    metrics.to_csv(results / "rpt_head_replay_metrics.csv", index=False)
    timelines.to_csv(results / "rpt_head_replay_timeline.csv", index=False)
    paired = metrics.pivot(index=["cell_id", "budget"], columns="method", values="proxy_MAE_pp").reset_index()
    paired["replay_last3_minus_latest_pp"] = paired["Periodic_K_ReplayLast3"] - paired["Periodic_K_LatestRPT"]
    paired.to_csv(results / "rpt_head_replay_paired.csv", index=False)
    rng = np.random.default_rng(20261012)
    paired_summary = []
    for budget, group in paired.groupby("budget", sort=True):
        delta = group["replay_last3_minus_latest_pp"].to_numpy(dtype=float)
        draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        paired_summary.append({
            "budget": int(budget),
            "cells": int(len(delta)),
            "mean_replay_minus_latest_pp": float(delta.mean()),
            "median_replay_minus_latest_pp": float(np.median(delta)),
            "bootstrap_95_low_pp": float(np.quantile(draws, 0.025)),
            "bootstrap_95_high_pp": float(np.quantile(draws, 0.975)),
            "cells_replay_better": int((delta < 0).sum()),
        })
    pd.DataFrame(paired_summary).to_csv(results / "rpt_head_replay_paired_summary.csv", index=False)

    summary = (
        metrics.groupby(["method", "budget"], as_index=False)
        .agg(
            cells=("cell_id", "nunique"),
            median_proxy_MAE_pp=("proxy_MAE_pp", "median"),
            mean_proxy_MAE_pp=("proxy_MAE_pp", "mean"),
            median_late_proxy_MAE_pp=("late_quartile_proxy_MAE_pp", "median"),
            median_reference_hours=("reference_test_hours_total", "median"),
        )
    )
    summary.to_csv(results / "rpt_head_replay_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    order = ["Periodic_K_LatestRPT", "Periodic_K_ReplayLast3"]
    colors = ["#0072B2", "#D55E00"]
    for i, budget in enumerate([2, 4, 6]):
        part = metrics.loc[metrics["budget"].eq(budget)]
        vals = [part.loc[part["method"].eq(method), "proxy_MAE_pp"] for method in order]
        boxes = ax.boxplot(vals, positions=[i * 3, i * 3 + 1], widths=0.65, showfliers=False, patch_artist=True,
                           boxprops={"facecolor": colors[0], "alpha": 0.25},
                           medianprops={"color": "black"})
        boxes["boxes"][1].set_facecolor(colors[1])
        # Overlay replay medians with distinct colors so the paired distributions are easy to compare.
        ax.scatter([i * 3 + 0] * len(vals[0]), vals[0], s=7, color=colors[0], alpha=0.45)
        ax.scatter([i * 3 + 1] * len(vals[1]), vals[1], s=7, color=colors[1], alpha=0.45)
        ax.text(i * 3 + 0.5, max(metrics["proxy_MAE_pp"]) + 0.2, f"B={budget}", ha="center", fontsize=9)
    ax.set_xticks([0, 1, 3, 4, 6, 7], ["Latest", "Replay 3", "Latest", "Replay 3", "Latest", "Replay 3"])
    ax.set_ylabel("Post-N20 MAE vs SOC proxy (pp)")
    ax.set_title("Last-3 RPT replay does not guarantee safer calibration")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "rpt_head_replay_sensitivity.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {len(metrics)} head-policy evaluations across {metrics.cell_id.nunique()} cells")


if __name__ == "__main__":
    main()
