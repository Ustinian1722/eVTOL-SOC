"""Proxy-label drift indicators and causal, budget-matched maintenance replay.

All SOC errors in this script are errors against SOC_proxy, not independent
electrochemical ground truth. Online triggers use V/I/T-derived features or an
explicit disagreement with a nominal-capacity counter; proxy labels are used
only for offline scoring and after a completed reference RPT is released.
"""

from __future__ import annotations

import json
import time
from bisect import bisect_right
from collections import deque
from pathlib import Path

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
CONFIG = json.loads((STUDY / "configs" / "proxy_maintenance.json").read_text(encoding="utf-8"))
N20 = int(CONFIG["training_mission_cycles_per_cell"])
FEATURES = [
    "phase_Ns", "duration_s", "I_mean_A", "I_peak_A",
    "Vmin_V", "Vmax_V", "Tmean_C", "Tmax_C",
]
VOLTAGE_FEATURES = list(CONFIG["voltage_residual_features"])
RPT_PHASES = [int(value) for value in CONFIG["rpt_phases"]]
RPT_CURRENT_RANGE_MA = tuple(int(value) for value in CONFIG["rpt_current_range_mA"])
MAX_BUDGETS = [int(value) for value in CONFIG["max_reference_budgets"]]
RANDOM_SEEDS = [int(value) for value in CONFIG["random_seeds"]]
HEAD_PRIOR_STRENGTH = float(CONFIG["head_update"]["prior_strength"])
HEAD_SCALE_BOUNDS = tuple(float(value) for value in CONFIG["head_update"]["scale_bounds"])
HEAD_BIAS_BOUNDS = tuple(float(value) for value in CONFIG["head_update"]["bias_bounds"])
ROBUST_THRESHOLD_FACTOR = float(CONFIG["threshold_factor_robust_scale"])
PH_DELTA = float(CONFIG["page_hinkley_style"]["delta_standardized"])
PH_THRESHOLD = float(CONFIG["page_hinkley_style"]["threshold"])
CUSUM_ALLOWANCE = float(CONFIG["cusum"]["allowance_standardized"])
CUSUM_THRESHOLD = float(CONFIG["cusum"]["threshold"])
ADWIN_WINDOW = int(CONFIG["adwin_like"]["window_observations"])
ADWIN_HALF_WINDOW = int(CONFIG["adwin_like"]["half_window"])
ADWIN_BOUND_MULTIPLIER = float(CONFIG["adwin_like"]["bound_multiplier"])


def _make_rpt_reference_table(segments: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, dict[int, dict]]:
    references: dict[str, dict[int, dict]] = {}
    segment_groups = segments.groupby(["cell_id", "charge_cycle_index"], sort=False)
    for cell_id, cell_candidates in candidates.groupby("cell_id", sort=True):
        references[cell_id] = {}
        for cycle_index, candidate in cell_candidates.groupby("charge_cycle_index", sort=True):
            index = int(cycle_index)
            capacity_ah = float(candidate["QDischarge_max_Ah"].max())
            try:
                cycle_segments = segment_groups.get_group((cell_id, cycle_index))
            except KeyError:
                continue
            phases = cycle_segments.loc[
                cycle_segments["Ns"].isin(RPT_PHASES)
                & cycle_segments["I_signed_mean_mA"].between(*RPT_CURRENT_RANGE_MA)
                & cycle_segments["duration_s"].ge(100)
            ]
            rows = []
            for ns, group in phases.groupby("Ns", sort=True):
                weights = group["rows"].astype(float)
                qd = float(group["QDischarge_max_Ah"].max())
                rows.append(
                    {
                        "cell_id": cell_id,
                        "charge_cycle_index": index,
                        "phase_Ns": int(ns),
                        "duration_s": float(group["duration_s"].sum()),
                        "I_mean_A": float(np.average(group["I_signed_mean_mA"] / 1000, weights=weights)),
                        "I_peak_A": float(group["I_abs_peak_mA"].max() / 1000),
                        "Vmin_V": float(group["V_min"].min()),
                        "Vmax_V": float(group["V_max"].max()),
                        "Tmean_C": float(np.average(group["T_mean_C"], weights=weights)),
                        "Tmax_C": float(group["T_max"].max()),
                        "QDischarge_end_Ah": qd,
                        "capacity_reference_Ah": capacity_ah,
                        "SOC_reference_proxy": float(np.clip(1 - qd / capacity_ah, 0, 1)),
                    }
                )
            ref_frame = pd.DataFrame(rows)
            if len(ref_frame) >= 2:
                event_duration_h = (
                    float(cycle_segments["time_end_s"].max() - cycle_segments["time_start_s"].min()) / 3600
                    if len(cycle_segments) else np.nan
                )
                references[cell_id][index] = {
                    "phases": ref_frame.sort_values("phase_Ns").reset_index(drop=True),
                    "capacity_ah": capacity_ah,
                    "event_duration_h": event_duration_h,
                }
    return references


def _calibrated_head(base_prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    start = time.perf_counter()
    design = np.column_stack([base_prediction, np.ones_like(base_prediction)])
    prior = np.array([1.0, 0.0])
    lhs = design.T @ design + HEAD_PRIOR_STRENGTH * np.eye(2)
    rhs = design.T @ reference + HEAD_PRIOR_STRENGTH * prior
    scale, bias = np.linalg.solve(lhs, rhs)
    scale = float(np.clip(scale, *HEAD_SCALE_BOUNDS))
    bias = float(np.clip(bias, *HEAD_BIAS_BOUNDS))
    elapsed_ms = (time.perf_counter() - start) * 1000
    return scale, bias, elapsed_ms


def _robust_center_scale(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.nanmedian(values)) if len(values) else 0.0
    mad = float(np.nanmedian(np.abs(values - center))) if len(values) else 0.0
    scale = max(1.4826 * mad, float(np.nanstd(values)) * 0.25 if len(values) else 0.0, 1e-6)
    return center, scale, center + ROBUST_THRESHOLD_FACTOR * scale


def _evenly_spaced_positions(n: int, budget: int) -> list[int]:
    if n <= 0 or budget <= 0:
        return []
    count = min(n, budget)
    return sorted(set(np.rint(np.linspace(0, n - 1, count)).astype(int).tolist()))


def _timeline_selection(
    mission_cycles: list[int],
    reference_cycles: list[int],
    cycle_scores: dict[int, float],
    training_scores: np.ndarray,
    budget: int,
    detector: str,
    seed: int = 0,
) -> dict[int, int]:
    """Return selected RPT index -> request mission index, using only past scores."""
    events = sorted(reference_cycles)
    if not events or budget <= 0:
        return {}
    mission_cycles = sorted(mission_cycles)
    if detector == "periodic":
        positions = _evenly_spaced_positions(len(events), budget)
        return {
            events[pos]: max((m for m in mission_cycles if m < events[pos]), default=events[pos] - 1)
            for pos in positions
        }
    if detector == "random":
        rng = np.random.default_rng(seed)
        positions = sorted(rng.choice(len(events), size=min(budget, len(events)), replace=False).tolist())
        return {
            events[pos]: max((m for m in mission_cycles if m < events[pos]), default=events[pos] - 1)
            for pos in positions
        }

    train_center, train_scale, simple_threshold = _robust_center_scale(training_scores)
    pending_since: int | None = None
    selected: dict[int, int] = {}
    score_state = 0.0
    ph_cumulative = 0.0
    ph_minimum = 0.0
    window: deque[float] = deque(maxlen=ADWIN_WINDOW)
    reference_pos = 0
    hybrid_next_slot = _evenly_spaced_positions(len(events), budget)
    hybrid_slot_cursor = 0

    timeline = sorted(
        [(cycle, "mission") for cycle in mission_cycles]
        + [(cycle, "reference") for cycle in events],
        key=lambda item: (item[0], 0 if item[1] == "mission" else 1),
    )
    for cycle, event_type in timeline:
        if event_type == "mission":
            value = float(cycle_scores.get(cycle, train_center))
            if detector in {"threshold", "hybrid"}:
                triggered = value > simple_threshold
            elif detector == "cusum":
                score_state = max(0.0, score_state + (value - train_center) / train_scale - CUSUM_ALLOWANCE)
                triggered = score_state > CUSUM_THRESHOLD
            elif detector == "page_hinkley":
                ph_cumulative += (value - train_center) / train_scale - PH_DELTA
                ph_minimum = min(ph_minimum, ph_cumulative)
                triggered = (ph_cumulative - ph_minimum) > PH_THRESHOLD
            elif detector == "adwin_like":
                window.append(value)
                triggered = False
                if len(window) == ADWIN_WINDOW:
                    arr = np.asarray(window)
                    cut = ADWIN_HALF_WINDOW
                    bound = ADWIN_BOUND_MULTIPLIER * train_scale * np.sqrt(1 / cut + 1 / cut)
                    triggered = float(arr[cut:].mean() - arr[:cut].mean()) > bound
            else:
                raise ValueError(f"Unknown detector: {detector}")
            if triggered and pending_since is None:
                pending_since = cycle
            continue

        if detector == "hybrid":
            periodic_due = (
                hybrid_slot_cursor < len(hybrid_next_slot)
                and reference_pos >= hybrid_next_slot[hybrid_slot_cursor]
            )
            if pending_since is not None or periodic_due:
                request_at = pending_since
                if request_at is None:
                    request_at = max((m for m in mission_cycles if m < cycle), default=cycle - 1)
                selected[cycle] = int(request_at)
                pending_since = None
                score_state = 0.0
                ph_cumulative = 0.0
                ph_minimum = 0.0
                window.clear()
                hybrid_slot_cursor += 1
                if len(selected) >= budget:
                    break
                # Spread remaining periodic fallback slots over remaining RPT
                # opportunities after this update; the current decision is causal.
                remaining_budget = budget - len(selected)
                remaining_events = len(events) - reference_pos - 1
                if remaining_budget > 0 and remaining_events > 0:
                    step = max(1, int(np.ceil(remaining_events / (remaining_budget + 1))))
                    hybrid_next_slot = hybrid_next_slot[:hybrid_slot_cursor] + [reference_pos + step]
                    for j in range(1, remaining_budget):
                        nxt = hybrid_next_slot[-1] + max(1, int(np.ceil((len(events) - hybrid_next_slot[-1] - 1) / (remaining_budget - j + 1))))
                        hybrid_next_slot.append(min(nxt, len(events) - 1))
        elif pending_since is not None and len(selected) < budget:
            selected[cycle] = int(pending_since)
            pending_since = None
            score_state = 0.0
            ph_cumulative = 0.0
            ph_minimum = 0.0
            window.clear()
            if len(selected) >= budget:
                break
        reference_pos += 1
    return selected


def _simulate(
    cell_id: str,
    test: pd.DataFrame,
    selected: dict[int, int],
    references: dict[int, dict],
    model,
    replay_window: int = 1,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    event_indices = sorted(selected)
    mission_indices = sorted(test["charge_cycle_index"].astype(int).unique().tolist())
    updates = []
    update_ms = []
    ref_hours = []
    valid_indices = []
    scales = []
    biases = []
    seen_references: list[int] = []
    for ref_index in event_indices:
        if ref_index not in references:
            continue
        seen_references.append(ref_index)
        calibration_indices = seen_references[-max(1, int(replay_window)):]
        reference = pd.concat([references[index]["phases"] for index in calibration_indices], ignore_index=True)
        base_ref_prediction = np.clip(model.predict(reference[FEATURES]), 0, 1)
        scale, bias, elapsed_ms = _calibrated_head(
            base_ref_prediction,
            reference["SOC_reference_proxy"].to_numpy(dtype=float),
        )
        update_ms.append(elapsed_ms)
        event_hours = float(references[ref_index]["event_duration_h"])
        ref_hours.append(event_hours)
        next_position = bisect_right(mission_indices, ref_index)
        next_mission = mission_indices[next_position] if next_position < len(mission_indices) else np.nan
        updates.append(
            {
                "cell_id": cell_id,
                "request_cycle_index": int(selected[ref_index]),
                "reference_cycle_index": int(ref_index),
                "first_effective_cycle_index": int(next_mission) if np.isfinite(next_mission) else np.nan,
                "head_scale": scale,
                "head_bias": bias,
                "head_update_ms": elapsed_ms,
                "reference_test_hours": event_hours,
                "calibration_references_used": len(calibration_indices),
            }
        )
        valid_indices.append(ref_index)
        scales.append(scale)
        biases.append(bias)

    result = test.copy()
    if result.empty:
        return {}, updates, result
    cycle_index = result["charge_cycle_index"].to_numpy(dtype=int)
    row_scale = np.ones(len(result), dtype=float)
    row_bias = np.zeros(len(result), dtype=float)
    if valid_indices:
        update_position = np.searchsorted(np.asarray(valid_indices), cycle_index, side="left") - 1
        has_update = update_position >= 0
        scale_array = np.asarray(scales, dtype=float)
        bias_array = np.asarray(biases, dtype=float)
        row_scale[has_update] = scale_array[update_position[has_update]]
        row_bias[has_update] = bias_array[update_position[has_update]]
    result["adapted_prediction"] = np.clip(
        row_scale * result["base_prediction"].to_numpy(dtype=float) + row_bias, 0, 1
    )
    result["abs_error_pp"] = (result["adapted_prediction"] - result["SOC_proxy"]).abs() * 100
    result["sq_error_pp"] = np.square((result["adapted_prediction"] - result["SOC_proxy"]) * 100)
    cycles = result.groupby("charge_cycle_index", as_index=False).agg(
        cycle_mae_pp=("abs_error_pp", "mean"),
        cycle_rmse_pp=("sq_error_pp", lambda x: float(np.sqrt(x.mean()))),
    )
    late_threshold = float(cycles["charge_cycle_index"].quantile(0.75))
    late = cycles.loc[cycles["charge_cycle_index"].ge(late_threshold)]
    metric = {
        "cell_id": cell_id,
        "mission_test_cycles": int(len(cycles)),
        "updates_used": int(len(updates)),
        "reference_test_hours_total": float(np.nansum(ref_hours)),
        "reference_test_hours_mean": float(np.nanmean(ref_hours)) if ref_hours else 0.0,
        "head_update_ms_mean": float(np.mean(update_ms)) if update_ms else 0.0,
        "proxy_MAE_pp": float(result["abs_error_pp"].mean()),
        "proxy_RMSE_pp": float(np.sqrt(result["sq_error_pp"].mean())),
        "late_quartile_proxy_MAE_pp": float(late["cycle_mae_pp"].mean()) if len(late) else np.nan,
        "P95_cycle_MAE_pp": float(cycles["cycle_mae_pp"].quantile(0.95)),
        "worst_cycle_MAE_pp": float(cycles["cycle_mae_pp"].max()),
        "cycles_over_2pp_pct": float((cycles["cycle_mae_pp"] > 2).mean() * 100),
        "cycles_over_3pp_pct": float((cycles["cycle_mae_pp"] > 3).mean() * 100),
    }
    return metric, updates, result


def _corr_metrics(cycles: pd.DataFrame) -> list[dict[str, object]]:
    output = []
    for cell_id, group in cycles.groupby("cell_id", sort=True):
        group = group.sort_values("charge_cycle_index").reset_index(drop=True)
        error = group["proxy_cycle_mae_pp"].to_numpy(dtype=float)
        future = np.full(len(group), np.nan)
        for i in range(len(group)):
            future_window = error[i + 1 : min(len(group), i + 51)]
            if len(future_window) >= 20:
                future[i] = float(np.mean(future_window))
        group["future_50_cycle_proxy_MAE_pp"] = future
        for indicator in ("model_disagreement_pp", "voltage_residual_mV"):
            current_corr = group[indicator].corr(group["proxy_cycle_mae_pp"], method="spearman")
            future_corr = group[indicator].corr(group["future_50_cycle_proxy_MAE_pp"], method="spearman")
            output.append(
                {
                    "cell_id": cell_id,
                    "indicator": indicator,
                    "cycles": int(len(group)),
                    "spearman_current_proxy_error": float(current_corr) if pd.notna(current_corr) else np.nan,
                    "spearman_future_50_cycle_error": float(future_corr) if pd.notna(future_corr) else np.nan,
                }
            )
    return output


def _error_onset(cycles: pd.DataFrame) -> dict[str, int | None]:
    output: dict[str, int | None] = {}
    for cell_id, group in cycles.groupby("cell_id", sort=True):
        group = group.sort_values("charge_cycle_index")
        rolling = group["proxy_cycle_mae_pp"].rolling(window=51, min_periods=20).median()
        over = rolling.gt(2.0)
        # Require ten consecutive window endpoints over 2 pp to avoid a single
        # bad cycle defining lifecycle onset.
        run = over.astype(int).rolling(window=10, min_periods=10).sum().eq(10)
        idx = group.loc[run, "charge_cycle_index"]
        output[cell_id] = int(idx.iloc[0]) if len(idx) else None
    return output


def _aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["method", "budget"], as_index=False)
        .agg(
            cells=("cell_id", "nunique"),
            median_proxy_MAE_pp=("proxy_MAE_pp", "median"),
            mean_proxy_MAE_pp=("proxy_MAE_pp", "mean"),
            median_proxy_RMSE_pp=("proxy_RMSE_pp", "median"),
            median_late_proxy_MAE_pp=("late_quartile_proxy_MAE_pp", "median"),
            median_P95_cycle_MAE_pp=("P95_cycle_MAE_pp", "median"),
            median_worst_cycle_MAE_pp=("worst_cycle_MAE_pp", "median"),
            median_cycles_over_2pp_pct=("cycles_over_2pp_pct", "median"),
            median_cycles_over_3pp_pct=("cycles_over_3pp_pct", "median"),
            median_updates=("updates_used", "median"),
            median_reference_hours=("reference_test_hours_total", "median"),
        )
    )


def _bootstrap_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20261012)
    records = []
    for budget in MAX_BUDGETS:
        subset = metrics.loc[metrics["budget"].eq(budget)]
        periodic = subset.loc[subset["method"].eq("Periodic_K")].groupby("cell_id")["proxy_MAE_pp"].mean()
        for method in sorted(subset["method"].unique()):
            if method in {"Periodic_K", "Frozen"}:
                continue
            values = subset.loc[subset["method"].eq(method)].groupby("cell_id")["proxy_MAE_pp"].mean()
            cells = sorted(set(periodic.index) & set(values.index))
            if len(cells) < 3:
                continue
            delta = (values.loc[cells] - periodic.loc[cells]).to_numpy(dtype=float)
            draws = np.empty(10000)
            for i in range(len(draws)):
                draws[i] = rng.choice(delta, size=len(delta), replace=True).mean()
            records.append(
                {
                    "method": method,
                    "budget": budget,
                    "cells": len(cells),
                    "mean_delta_vs_periodic_pp": float(delta.mean()),
                    "bootstrap_95_low_pp": float(np.quantile(draws, 0.025)),
                    "bootstrap_95_high_pp": float(np.quantile(draws, 0.975)),
                }
            )
    return pd.DataFrame(records)


def _plot_outputs(metrics: pd.DataFrame, aggregate: pd.DataFrame, risk_cycles: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for budget in MAX_BUDGETS:
        group = metrics.loc[metrics["budget"].eq(budget)]
        point = group.groupby("method").agg(
            refs=("updates_used", "mean"), mae=("proxy_MAE_pp", "median")
        )
        ax.scatter(point["refs"], point["mae"], label=f"B≤{budget}", s=28)
        for name, row in point.iterrows():
            ax.annotate(name, (row["refs"], row["mae"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Mean completed RPT references per cell")
    ax.set_ylabel("Median later-life MAE vs SOC proxy (pp)")
    ax.set_title("Causal RPT head updates: proxy accuracy and reference cost")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "proxy_maintenance_pareto.png", dpi=180)
    plt.close(fig)

    selected = risk_cycles.loc[risk_cycles["cell_id"].eq("VAH01")]
    if not selected.empty:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(selected["charge_cycle_index"], selected["proxy_cycle_mae_pp"], linewidth=0.8)
        axes[0].set_ylabel("Proxy MAE (pp)")
        axes[1].plot(selected["charge_cycle_index"], selected["model_disagreement_pp"], color="#D55E00", linewidth=0.8)
        axes[1].set_ylabel("Model vs 3 Ah (pp)")
        axes[2].plot(selected["charge_cycle_index"], selected["voltage_residual_mV"], color="#0072B2", linewidth=0.8)
        axes[2].set_ylabel("Voltage residual (mV)")
        axes[2].set_xlabel("Sustained Ns=0 charge-block index (proxy)")
        axes[0].set_title("VAH01 offline proxy error and observable drift indicators")
        for ax in axes:
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURES / "proxy_drift_indicators.png", dpi=180)
        plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    segments = pd.read_csv(RESULTS / "segment_summary.csv")
    labels = pd.read_csv(RESULTS / "soc_proxy_phase_labels.csv")
    candidates = pd.read_csv(RESULTS / "capacity_test_candidates.csv")
    references_by_cell = _make_rpt_reference_table(segments, candidates)

    metric_rows: list[dict[str, object]] = []
    timeline_rows: list[dict[str, object]] = []
    risk_cycle_frames: list[pd.DataFrame] = []

    for cell_id, cell in labels.groupby("cell_id", sort=True):
        cell = cell.sort_values(["charge_cycle_index", "phase_Ns"]).copy()
        unique_cycles = sorted(cell["charge_cycle_index"].astype(int).unique().tolist())
        train_cycles = set(unique_cycles[:N20])
        train = cell.loc[cell["charge_cycle_index"].isin(train_cycles)].copy()
        test = cell.loc[~cell["charge_cycle_index"].isin(train_cycles)].copy()
        if test.empty or len(train) < 10:
            continue

        soc_model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        soc_model.fit(train[FEATURES], train["SOC_proxy"])
        voltage_model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        voltage_model.fit(train[VOLTAGE_FEATURES], train["Vmin_V"])

        for frame in (train, test):
            frame["base_prediction"] = np.clip(soc_model.predict(frame[FEATURES]), 0, 1)
            frame["voltage_prediction_V"] = voltage_model.predict(frame[VOLTAGE_FEATURES])
            frame["model_disagreement_pp"] = (
                frame["base_prediction"] - frame["SOC_nominal_3Ah"]
            ).abs() * 100
            frame["voltage_residual_mV"] = (
                frame["Vmin_V"] - frame["voltage_prediction_V"]
            ).abs() * 1000
            frame["proxy_abs_error_pp"] = (
                frame["base_prediction"] - frame["SOC_proxy"]
            ).abs() * 100

        test_cycles = test.groupby("charge_cycle_index", as_index=False).agg(
            proxy_cycle_mae_pp=("proxy_abs_error_pp", "mean"),
            model_disagreement_pp=("model_disagreement_pp", "mean"),
            voltage_residual_mV=("voltage_residual_mV", "mean"),
        )
        test_cycles["cell_id"] = cell_id
        risk_cycle_frames.append(test_cycles)

        train_cycles_signals = train.groupby("charge_cycle_index", as_index=False).agg(
            model_disagreement_pp=("model_disagreement_pp", "mean"),
            voltage_residual_mV=("voltage_residual_mV", "mean"),
        )
        refs = references_by_cell.get(cell_id, {})
        after_n20_refs = sorted(i for i in refs if i > max(train_cycles))
        test_index_list = sorted(test["charge_cycle_index"].astype(int).unique().tolist())

        # The reference cycle is omitted from mission scores. The update is
        # released only after its end and becomes effective on the next mission.
        strategies = [("Frozen", 0, "none", 0)]
        for budget in MAX_BUDGETS:
            strategies.append(("Periodic_K", budget, "periodic", 0))
            for seed in RANDOM_SEEDS:
                strategies.append((f"Random_seed{seed}", budget, "random", seed))
            strategies.extend(
                [
                    ("ModelDisagreement", budget, "threshold", 0),
                    ("VoltageResidual", budget, "threshold", 0),
                    ("PageHinkley_Voltage", budget, "page_hinkley", 0),
                    ("CUSUM_Voltage", budget, "cusum", 0),
                    ("ADWINlike_Voltage", budget, "adwin_like", 0),
                    ("Periodic_K_plus_Risk", budget, "hybrid", 0),
                ]
            )

        for method, budget, kind, seed in strategies:
            if method == "Frozen":
                selected = {}
                selection_signal = None
            elif kind in {"periodic", "random"}:
                selected = _timeline_selection(
                    test_index_list, after_n20_refs, {}, np.array([]), budget, kind, seed
                )
                selection_signal = None
            else:
                signal_name = "model_disagreement_pp" if method == "ModelDisagreement" else "voltage_residual_mV"
                score_map = dict(zip(test_cycles["charge_cycle_index"].astype(int), test_cycles[signal_name].astype(float)))
                train_scores = train_cycles_signals[signal_name].to_numpy(dtype=float)
                selected = _timeline_selection(
                    test_index_list, after_n20_refs, score_map, train_scores, budget, kind, seed
                )
                selection_signal = signal_name

            metric, updates, prediction = _simulate(
                cell_id, test, selected, refs, soc_model
            )
            if not metric:
                continue
            metric.update(
                method=method,
                budget=budget,
                seed=seed,
                reference_opportunities=int(len(after_n20_refs)),
                selection_signal=selection_signal or "planned schedule",
            )
            metric_rows.append(metric)
            for update in updates:
                update.update(method=method, budget=budget, seed=seed, selection_signal=selection_signal or "planned schedule")
                timeline_rows.append(update)
        print(f"Completed maintenance replay for {cell_id}: {len(strategies)} strategies", flush=True)

    metrics = pd.DataFrame(metric_rows)
    timelines = pd.DataFrame(timeline_rows)
    risk_cycles = pd.concat(risk_cycle_frames, ignore_index=True)
    correlations = pd.DataFrame(_corr_metrics(risk_cycles))
    onset = _error_onset(risk_cycles)
    onset_rows = []
    for (cell_id, method, budget, seed), group in timelines.groupby(["cell_id", "method", "budget", "seed"], sort=False):
        first = group.sort_values("reference_cycle_index").iloc[0]
        onset_idx = onset.get(cell_id)
        onset_rows.append(
            {
                "cell_id": cell_id,
                "method": method,
                "budget": budget,
                "seed": seed,
                "proxy_error_onset_cycle_index": onset_idx,
                "first_reference_cycle_index": int(first["reference_cycle_index"]),
                "first_effective_cycle_index": int(first["first_effective_cycle_index"]),
                "reference_minus_proxy_onset_cycles": (
                    int(first["reference_cycle_index"] - onset_idx) if onset_idx is not None else np.nan
                ),
                "pre_onset_reference": bool(onset_idx is not None and first["reference_cycle_index"] < onset_idx),
            }
        )

    metrics.to_csv(RESULTS / "proxy_maintenance_metrics.csv", index=False)
    timelines.to_csv(RESULTS / "proxy_maintenance_timeline.csv", index=False)
    risk_cycles.to_csv(RESULTS / "proxy_drift_cycle_indicators.csv", index=False)
    correlations.to_csv(RESULTS / "drift_indicator_correlations.csv", index=False)
    pd.DataFrame(onset_rows).to_csv(RESULTS / "proxy_detection_onset.csv", index=False)
    aggregate = _aggregate_metrics(metrics)
    aggregate.to_csv(RESULTS / "proxy_maintenance_group_summary.csv", index=False)
    _bootstrap_differences(metrics).to_csv(RESULTS / "proxy_maintenance_bootstrap.csv", index=False)
    _plot_outputs(metrics, aggregate, risk_cycles)
    (RESULTS / "proxy_maintenance_metadata.json").write_text(
        json.dumps(
            {
                "label_scope": "SOC_proxy only; not independent true SOC",
                "prediction_commit": "score each mission before any current-cycle reference is released",
                "request_reference_apply": "request after a mission; use the next selected completed candidate RPT; apply head from first later mission index",
                "reference_data": "three low-rate discharge phase endpoints Ns 3/5/7 with capacity measured by that completed RPT",
                "update": "bounded affine correction y=a*p+b, prior (a,b)=(1,0), prior strength 3; latest RPT replaces prior head",
                "indicators": {
                    "model_disagreement": "absolute difference between N20 phase Ridge prediction and 3 Ah coulomb-counter estimate",
                    "voltage_residual": "absolute residual of N20-trained phase minimum-voltage Ridge using phase, duration, current, Vmax and temperature; Vmin target is excluded from inputs; not an ECM time-series residual",
                    "page_hinkley": "Page-Hinkley-style cumulative standardized residual with running minimum; delta=0.5, threshold=5; fixed training center",
                    "cusum": "one-sided cumulative standardized residual; kappa=0.25, threshold=5",
                    "adwin_like": "fixed 20-vs-20 mean-shift check; not a full adaptive ADWIN implementation",
                    "threshold": "N20 median + 3 robust standard deviations; no test-time threshold search",
                },
                "budget": "B is a maximum number of completed RPT references; actual count is reported; periodic/random/hybrid use evenly spaced or sampled observed opportunities",
                "random_seeds": RANDOM_SEEDS,
                "cell_level_bootstrap_seed": 20261012,
                "configuration": CONFIG,
                "excluded": "FBG-based SRA from FiberResearch; eVTOL data contain no FBG signal",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(metrics)} maintenance evaluations across {metrics.cell_id.nunique()} cells")


if __name__ == "__main__":
    main()
