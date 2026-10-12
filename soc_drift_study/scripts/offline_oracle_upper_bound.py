"""Post-hoc proxy-label oracle upper bound for selecting RPT update timing.

The oracle uses held-out SOC_proxy errors to choose reference dates. It is an
unavailable hindsight bound, never an online method or deployment result.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "scripts"))
import proxy_drift_maintenance as md  # noqa: E402


def _oracle_schedule(
    cycles: np.ndarray,
    frozen_sum: np.ndarray,
    event_indices: list[int],
    head_sums: dict[int, np.ndarray],
    max_budget: int,
) -> dict[int, list[int]]:
    """Exact DP for at most B replacement updates over ordered RPT opportunities."""
    event_count = len(event_indices)
    frozen_prefix = np.r_[0.0, np.cumsum(frozen_sum)]
    after = [int(np.searchsorted(cycles, event, side="right")) for event in event_indices]
    head_prefix = {event: np.r_[0.0, np.cumsum(head_sums[event])] for event in event_indices}
    dp: list[np.ndarray] = []
    back: list[list[int | None]] = []

    for n_updates in range(1, max_budget + 1):
        costs = np.full(event_count, np.inf)
        predecessors: list[int | None] = [None] * event_count
        for j, event in enumerate(event_indices):
            right = after[j]
            if n_updates == 1:
                costs[j] = frozen_prefix[right]
                continue
            for i in range(j):
                prior = dp[n_updates - 2][i]
                if not np.isfinite(prior):
                    continue
                previous_event = event_indices[i]
                prefix = head_prefix[previous_event]
                segment = prefix[right] - prefix[after[i]]
                candidate = prior + segment
                if candidate < costs[j]:
                    costs[j] = candidate
                    predecessors[j] = i
        dp.append(costs)
        back.append(predecessors)

    output: dict[int, list[int]] = {}
    frozen_total = float(frozen_prefix[-1])
    for budget in range(1, max_budget + 1):
        best_cost = frozen_total
        best_end: int | None = None
        best_updates = 0
        for n_updates in range(1, budget + 1):
            for j, event in enumerate(event_indices):
                prefix = head_prefix[event]
                total = dp[n_updates - 1][j] + prefix[-1] - prefix[after[j]]
                if total < best_cost:
                    best_cost = float(total)
                    best_end = j
                    best_updates = n_updates
        chosen: list[int] = []
        if best_end is not None:
            pos: int | None = best_end
            for level in range(best_updates - 1, -1, -1):
                if pos is None:
                    break
                chosen.append(event_indices[pos])
                pos = back[level][pos]
            chosen.reverse()
        output[budget] = chosen
    return output


def main() -> None:
    results = STUDY / "results"
    segment_fields = [
        "rows", "time_start_s", "time_end_s", "I_abs_peak_mA", "Ns", "cell_id",
        "duration_s", "I_signed_mean_mA", "V_min", "V_max", "T_mean_C", "T_max",
        "QDischarge_max_Ah", "charge_cycle_index",
    ]
    segments = pd.read_csv(results / "segment_summary.csv", usecols=segment_fields)
    candidates = pd.read_csv(results / "capacity_test_candidates.csv")
    labels = pd.read_csv(results / "soc_proxy_phase_labels.csv")
    references_by_cell = md._make_rpt_reference_table(segments, candidates)
    metric_rows: list[dict] = []
    timeline_rows: list[dict] = []

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
        after_n20_refs = sorted(index for index in refs if index > max(train_cycles))
        test_cycles = np.asarray(sorted(test["charge_cycle_index"].astype(int).unique()), dtype=int)
        frozen_error = (test["base_prediction"] - test["SOC_proxy"]).abs() * 100
        frozen_by_cycle = frozen_error.groupby(test["charge_cycle_index"].astype(int)).sum().reindex(test_cycles, fill_value=0).to_numpy()
        head_errors: dict[int, np.ndarray] = {}

        for event in after_n20_refs:
            reference = refs[event]["phases"]
            reference_prediction = np.clip(model.predict(reference[md.FEATURES]), 0, 1)
            scale, bias, _ = md._calibrated_head(
                reference_prediction,
                reference["SOC_reference_proxy"].to_numpy(dtype=float),
            )
            error = (
                np.clip(scale * test["base_prediction"].to_numpy(dtype=float) + bias, 0, 1)
                - test["SOC_proxy"].to_numpy(dtype=float)
            )
            error = np.abs(error) * 100
            head_errors[event] = (
                pd.Series(error, index=test.index)
                .groupby(test["charge_cycle_index"].astype(int))
                .sum()
                .reindex(test_cycles, fill_value=0)
                .to_numpy()
            )

        schedules = _oracle_schedule(
            test_cycles, frozen_by_cycle, after_n20_refs, head_errors, max_budget=6
        )
        for budget in [0, 2, 4, 6]:
            schedule = [] if budget == 0 else schedules[budget]
            selected = {
                event: max((cycle for cycle in test_cycles if cycle < event), default=int(event) - 1)
                for event in schedule
            }
            metric, updates, _ = md._simulate(cell_id, test, selected, refs, model)
            if not metric:
                continue
            metric.update(
                method="OfflineOracle_proxy_hindsight",
                budget=budget,
                seed=0,
                reference_opportunities=len(after_n20_refs),
                selection_signal="held-out SOC_proxy hindsight; unavailable online",
            )
            metric_rows.append(metric)
            for update in updates:
                update.update(method="OfflineOracle_proxy_hindsight", budget=budget, seed=0,
                              selection_signal="held-out SOC_proxy hindsight; unavailable online")
                timeline_rows.append(update)
        print(f"Completed hindsight upper bound for {cell_id}", flush=True)

    pd.DataFrame(metric_rows).to_csv(results / "offline_oracle_metrics.csv", index=False)
    pd.DataFrame(timeline_rows).to_csv(results / "offline_oracle_timeline.csv", index=False)
    summary = (
        pd.DataFrame(metric_rows)
        .groupby(["method", "budget"], as_index=False)
        .agg(
            cells=("cell_id", "nunique"),
            median_proxy_MAE_pp=("proxy_MAE_pp", "median"),
            mean_proxy_MAE_pp=("proxy_MAE_pp", "mean"),
            median_late_proxy_MAE_pp=("late_quartile_proxy_MAE_pp", "median"),
            median_updates=("updates_used", "median"),
            median_reference_hours=("reference_test_hours_total", "median"),
        )
    )
    summary.to_csv(results / "offline_oracle_summary.csv", index=False)
    (results / "offline_oracle_metadata.json").write_text(
        '{\n  "meaning": "post-hoc hindsight upper bound; not causal or deployable",\n'
        '  "selection_labels": "held-out SOC_proxy errors across the test lifetime",\n'
        '  "update_rule": "same bounded latest-RPT affine head as online strategies",\n'
        '  "selection": "dynamic program minimizes total post-N20 proxy absolute error using up to B completed reference opportunities",\n'
        '  "budgets": [0, 2, 4, 6]\n}\n',
        encoding="utf-8",
    )
    print(f"Wrote {len(metric_rows)} offline oracle evaluations across {len(labels.cell_id.unique())} cells")


if __name__ == "__main__":
    main()
