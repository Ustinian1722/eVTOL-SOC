"""Create derived audit tables and plots without modifying source CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parent
RESULTS = STUDY / "results"
FIGURES = STUDY / "figures"


def _greedy_matches(left: list[int], right: list[int], tolerance: int) -> int:
    """Count one-to-one near matches between sorted cycle-index sequences."""
    i = j = matched = 0
    while i < len(left) and j < len(right):
        delta = left[i] - right[j]
        if abs(delta) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return matched


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segments = pd.read_csv(RESULTS / "segment_summary.csv")
    base = segments.loc[
        segments["QDischarge_max_Ah"].ge(1.0)
        & segments["I_signed_mean_mA"].between(-1000, -300)
        & segments["V_min"].le(2.6)
    ].copy()

    sensitivity = []
    for duration in (1000, 3000):
        candidates = base.loc[base["duration_s"].ge(duration)].copy()
        candidates["duration_threshold_s"] = duration
        candidates["candidate_reason"] = (
            "low-rate discharge segment with cumulative QDischarge >=1 Ah, "
            "mean current -1.0 to -0.3 A, and Vmin <=2.6 V; protocol review required"
        )
        sensitivity.append(candidates)
    sensitivity_frame = pd.concat(sensitivity, ignore_index=True)
    sensitivity_frame.to_csv(RESULTS / "capacity_test_candidates_sensitivity.csv", index=False)

    strict = sensitivity_frame.loc[sensitivity_frame["duration_threshold_s"].eq(3000)]
    metrics = []
    for cell_id in sorted(segments["cell_id"].unique()):
        cell = strict.loc[strict["cell_id"].eq(cell_id)].sort_values("charge_cycle_index")
        row: dict[str, object] = {"cell_id": cell_id, "rpt_tail_candidates": int(len(cell))}
        if len(cell):
            first = cell.iloc[0]
            last = cell.iloc[-1]
            row.update(
                first_candidate_index=int(first["charge_cycle_index"]),
                last_candidate_index=int(last["charge_cycle_index"]),
                first_candidate_capacity_Ah=float(first["QDischarge_max_Ah"]),
                last_candidate_capacity_Ah=float(last["QDischarge_max_Ah"]),
                candidate_capacity_change_pct=float(
                    100 * (last["QDischarge_max_Ah"] / first["QDischarge_max_Ah"] - 1)
                ),
                positive_steps_over_0p02_Ah=int((cell["QDischarge_max_Ah"].diff() > 0.02).sum()),
            )
            candidate_indices = sorted(cell["charge_cycle_index"].dropna().astype(int).tolist())
        else:
            candidate_indices = []

        impedance_path = ROOT / f"{cell_id}_impedance.csv"
        if impedance_path.exists():
            impedance = pd.read_csv(impedance_path)
            cycles = pd.to_numeric(impedance["cycle numbers"], errors="coerce").dropna().astype(int)
            z = pd.to_numeric(impedance["60%_30_second"], errors="coerce").dropna()
            row.update(
                impedance_rows=int(len(impedance)),
                impedance_first_cycle=int(cycles.iloc[0]) if len(cycles) else None,
                impedance_last_cycle=int(cycles.iloc[-1]) if len(cycles) else None,
                impedance_60pct_30s_first_ohm=float(z.iloc[0]) if len(z) else None,
                impedance_60pct_30s_last_ohm=float(z.iloc[-1]) if len(z) else None,
                impedance_60pct_30s_change_pct=float(100 * (z.iloc[-1] / z.iloc[0] - 1))
                if len(z) > 1 and z.iloc[0] else None,
                exact_cycle_index_matches=int(len(set(candidate_indices) & set(cycles.tolist()))),
                near_matches_within_1_cycle=_greedy_matches(candidate_indices, sorted(cycles.tolist()), 1),
            )
        else:
            row.update(
                impedance_rows=0,
                impedance_first_cycle=None,
                impedance_last_cycle=None,
                impedance_60pct_30s_first_ohm=None,
                impedance_60pct_30s_last_ohm=None,
                impedance_60pct_30s_change_pct=None,
                exact_cycle_index_matches=0,
                near_matches_within_1_cycle=0,
            )
        metrics.append(row)

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(RESULTS / "capacity_impedance_metrics.csv", index=False)

    active = segments.loc[
        segments["Ns"].isin([4, 5, 6])
        & segments["I_signed_mean_mA"].le(-2000)
        & segments["duration_s"].ge(10)
    ].copy()
    phase = (
        active.groupby("Ns")
        .agg(
            segment_runs=("rows", "size"),
            cells=("cell_id", "nunique"),
            median_duration_s=("duration_s", "median"),
            median_mean_current_A=("I_signed_mean_mA", "median"),
            median_abs_peak_A=("I_abs_peak_mA", "median"),
            median_min_voltage_V=("V_min", "median"),
            median_temperature_C=("T_mean_C", "median"),
        )
        .reset_index()
    )
    phase["median_mean_current_A"] /= 1000
    phase["median_abs_peak_A"] /= 1000
    phase.to_csv(RESULTS / "active_phase_summary.csv", index=False)

    trend_rows = []
    for cell_id, group in active.loc[active["Ns"].eq(5)].groupby("cell_id"):
        group = group.dropna(subset=["charge_cycle_index", "V_min"]).sort_values("charge_cycle_index")
        if len(group) < 5:
            continue
        quarter = max(1, len(group) // 4)
        early = group.head(quarter)
        late = group.tail(quarter)
        trend_rows.append(
            {
                "cell_id": cell_id,
                "active_cruise_runs": int(len(group)),
                "first_charge_index": int(group["charge_cycle_index"].min()),
                "last_charge_index": int(group["charge_cycle_index"].max()),
                "early_quartile_median_Vmin_V": float(early["V_min"].median()),
                "late_quartile_median_Vmin_V": float(late["V_min"].median()),
                "late_minus_early_Vmin_V": float(late["V_min"].median() - early["V_min"].median()),
                "median_cruise_current_A": float(group["I_signed_mean_mA"].median() / 1000),
                "spearman_charge_index_vs_Vmin": float(
                    group["charge_cycle_index"].corr(group["V_min"], method="spearman")
                ),
            }
        )
    trend_frame = pd.DataFrame(trend_rows)
    trend_frame.to_csv(RESULTS / "cruise_voltage_trend.csv", index=False)
    return segments, active, metrics_frame


def plot_capacity_and_impedance() -> None:
    capacity = pd.read_csv(RESULTS / "capacity_test_candidates.csv")
    impedance_paths = sorted(ROOT.glob("VAH[0-9][0-9]_impedance.csv"))

    fig, axes = plt.subplots(5, 5, figsize=(15, 12), sharex=False, sharey=True)
    axes = list(axes.flat)
    for ax, (cell_id, group) in zip(axes, capacity.groupby("cell_id", sort=True)):
        group = group.sort_values("charge_cycle_index")
        ax.plot(group["charge_cycle_index"], group["QDischarge_max_Ah"], marker=".", linewidth=1)
        ax.set_title(cell_id, fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes[22:]:
        ax.axis("off")
    fig.supxlabel("Sustained Ns=0 charge-block index (proxy)")
    fig.supylabel("Candidate low-rate reference capacity (Ah)")
    fig.suptitle("Candidate RPT capacity by cell; low-rate tail signature, protocol review required")
    fig.tight_layout()
    fig.savefig(FIGURES / "capacity_fade_candidates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for path in impedance_paths:
        frame = pd.read_csv(path)
        cycle = pd.to_numeric(frame["cycle numbers"], errors="coerce")
        resistance = pd.to_numeric(frame["60%_30_second"], errors="coerce")
        ax.plot(cycle, resistance, marker=".", markersize=3, linewidth=1, label=path.stem.replace("_impedance", ""))
    ax.set_xlabel("Impedance table cycle number")
    ax.set_ylabel("60% 30-second resistance (Ω)")
    ax.set_title("Processed impedance measurements across cell life")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "impedance_lifecycle.png", dpi=180)
    plt.close(fig)


def plot_representative_missions(segments: pd.DataFrame) -> None:
    cell_id = "VAH01"
    cell = segments.loc[segments["cell_id"].eq(cell_id)]
    active = cell.loc[
        cell["Ns"].isin([4, 5, 6])
        & cell["I_signed_mean_mA"].le(-2000)
        & cell["duration_s"].ge(10)
    ]
    indices = sorted(active["charge_cycle_index"].dropna().astype(int).unique())
    if len(indices) < 3:
        return
    targets = [indices[0], indices[len(indices) // 2], indices[-1]]
    chosen = active.loc[active["charge_cycle_index"].isin(targets)]
    run_to_cycle = dict(zip(chosen["segment_run_id"].astype(int), chosen["charge_cycle_index"].astype(int)))
    wanted_runs = set(run_to_cycle)
    path = ROOT / f"{cell_id}.csv"
    previous_time = None
    previous_key = None
    run_id = 0
    extracted: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=300_000):
        cycle = chunk["cycleNumber"]
        ns = chunk["Ns"]
        changed = ~(cycle.eq(cycle.shift()) & ns.eq(ns.shift()))
        first = (None if pd.isna(cycle.iloc[0]) else float(cycle.iloc[0]), None if pd.isna(ns.iloc[0]) else float(ns.iloc[0]))
        if previous_key is None:
            changed.iloc[0] = True
        else:
            changed.iloc[0] = first != previous_key
        previous_key = (None if pd.isna(cycle.iloc[-1]) else float(cycle.iloc[-1]), None if pd.isna(ns.iloc[-1]) else float(ns.iloc[-1]))
        run = run_id + changed.astype(np.int64).cumsum()
        run_id += int(changed.sum())
        subset = chunk.loc[run.isin(wanted_runs), ["time_s", "Ecell_V", "I_mA", "Temperature__C", "Ns"]].copy()
        if len(subset):
            subset["segment_run_id"] = run.loc[subset.index].astype(int).to_numpy()
            subset["charge_cycle_index"] = subset["segment_run_id"].map(run_to_cycle)
            extracted.append(subset)
    if not extracted:
        return
    frame = pd.concat(extracted, ignore_index=True)
    frame.to_csv(RESULTS / "representative_mission_curves.csv", index=False)
    frame["mission_time_s"] = np.nan
    for index, group in frame.groupby("charge_cycle_index"):
        frame.loc[group.index, "mission_time_s"] = group["time_s"] - group["time_s"].min()

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    colors = {targets[0]: "#0072B2", targets[1]: "#E69F00", targets[2]: "#009E73"}
    names = {targets[0]: f"early (index {targets[0]})", targets[1]: f"mid (index {targets[1]})", targets[2]: f"late (index {targets[2]})"}
    for index, group in frame.groupby("charge_cycle_index"):
        group = group.sort_values("time_s")
        t = group["mission_time_s"] / 60
        axes[0].plot(t, group["Ecell_V"], color=colors[index], linewidth=1, label=names[index])
        axes[1].plot(t, group["I_mA"] / 1000, color=colors[index], linewidth=1)
        axes[2].plot(t, group["Temperature__C"], color=colors[index], linewidth=1)
    axes[0].set_ylabel("Voltage (V)")
    axes[1].set_ylabel("Current (A)")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("Time from first selected discharge sample (min)")
    axes[0].set_title("VAH01 representative high-power mission segments; line colors indicate life stage")
    axes[0].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "representative_mission_lifecycle.png", dpi=180)
    plt.close(fig)


def write_label_audit() -> None:
    metadata = {
        "independent_soc_ground_truth_present": False,
        "source_columns": [
            "time_s", "Ecell_V", "I_mA", "EnergyCharge_W_h", "QCharge_mA_h",
            "EnergyDischarge_W_h", "QDischarge_mA_h", "Temperature__C", "cycleNumber", "Ns",
        ],
        "offline_proxy_possible": True,
        "proxy_definition": (
            "Coulomb-counted SOC proxy may use measured QDischarge from the current mission and the latest "
            "completed earlier RPT capacity; use nominal 3 Ah only until the first valid RPT."
        ),
        "causal_rule": (
            "Do not use the current RPT result to label its own cycle or interpolate future capacity backward. "
            "An RPT capacity becomes available only after that test completes."
        ),
        "limitations": [
            "No independent electrochemical SOC or BMS SOC label is supplied.",
            "A Q-based target is a reference proxy, not ground truth; MAE against it is not true-SOC accuracy.",
            "Low-rate RPT capacity is not the same as high-rate mission-accessible capacity.",
            "A mission discharge endpoint is not SOC=0; mission profiles leave residual capacity.",
            "SOC during charging requires an estimated start SOC and charge-efficiency assumption.",
        ],
        "rpt_candidate_rule": (
            "A contiguous low-rate tail candidate has cumulative QDischarge >=1 Ah, mean current from -1.0 to -0.3 A, "
            "duration >=3000 s, and minimum voltage <=2.6 V. A >=1000 s sensitivity set is also exported. "
            "Candidate cycles require manual protocol review against README anomalies and impedance records."
        ),
        "cycle_index_note": (
            "charge_cycle_index counts sustained contiguous Ns=0 segments (>100 s) and is an ordering proxy; "
            "it is not the raw tester cycleNumber and may drift when phases are missing or malformed."
        ),
    }
    (RESULTS / "label_audit.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    segments, _, _ = build_tables()
    write_label_audit()
    plot_capacity_and_impedance()
    plot_representative_missions(segments)
    print("Wrote audit products to", RESULTS, "and", FIGURES)


if __name__ == "__main__":
    main()
