"""Audit *how* RPT references change a phase SOC_proxy model.

This is NOT true-SOC validation. SOC_proxy is constructed from current
QDischarge and the previous completed candidate RPT capacity, so directly
recomputing it with that same capacity is a tautological control.

Design: same target-cell N20 Ridge and *offline* B=4 periodic candidate
opportunities as the earlier study. Head policies share one reference schedule.
No test outcome or future RPT capacity is used to fit an update. The offline
schedule itself knows which future RPT opportunities exist and is NOT a
deployable decision policy.

Artifacts are diagnostic rather than a proposed validated SOC updater.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from proxy_drift_maintenance import (
    FEATURES,
    _calibrated_head,
    _evenly_spaced_positions,
    _make_rpt_reference_table,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N20 = 20
BUDGET = 4
POLICIES = {
    "frozen": ("full", 0.0),
    "legacy_affine": ("full", 1.0),
    "affine_shrink_010": ("full", 0.10),
    "affine_shrink_025": ("full", 0.25),
    "affine_shrink_050": ("full", 0.50),
    "bias_shrink_025": ("bias", 0.25),
    "current_support_gate": ("gated", 1.0),
}


def update_prediction(
    base: np.ndarray, scale: float, bias: float, kind: str, strength: float,
    eligible: bool = True,
) -> np.ndarray:
    """Apply a predeclared, bounded correction; never fit on test targets."""
    effective_strength = strength if (kind != "gated" or eligible) else 0.0
    a = scale if kind != "bias" else 1.0
    return np.clip(
        (1.0 + effective_strength * (a - 1.0)) * np.asarray(base)
        + effective_strength * bias, 0.0, 1.0,
    )


def current_support_eligible(train: pd.DataFrame, ref: pd.DataFrame) -> bool:
    """Conservative label-free domain gate: >=half RPT currents within
    central 90% range of the mission-training phase currents.
    A gate that accepts no updates must be reported as such.
    """
    lo, hi = np.quantile(train["I_mean_A"].to_numpy(dtype=float), [0.05, 0.95])
    values = ref["I_mean_A"].to_numpy(dtype=float)
    return bool((np.isfinite(values) & (values >= lo) & (values <= hi)).mean() >= 0.5)


def fixed_periodic_events(
    refs: dict[int, dict], cutoff: int, budget: int = BUDGET,
) -> list[int]:
    """Selected event indices: same known-opportunity offline B4 schedule."""
    candidates = sorted(int(i) for i in refs if int(i) > cutoff)
    return [candidates[p] for p in _evenly_spaced_positions(len(candidates), budget)]


def strictly_previous_index(sorted_events: list[int], task_cycle: int) -> int | None:
    """A reference at the same task index is NOT yet available."""
    pos = int(np.searchsorted(np.asarray(sorted_events, dtype=int), task_cycle, side="left") - 1)
    return sorted_events[pos] if pos >= 0 else None


def _metric(actual: np.ndarray, estimated: np.ndarray) -> tuple[float, float]:
    e = 100.0 * (np.asarray(estimated) - np.asarray(actual))
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e * e)))


def _head_event(
    model, train: pd.DataFrame, event_idx: int, event: dict,
) -> dict:
    ref = event["phases"]
    base = np.clip(model.predict(ref[FEATURES]), 0.0, 1.0)
    target = ref["SOC_reference_proxy"].to_numpy(dtype=float)
    scale, bias, _ = _calibrated_head(base, target)
    pre, _ = _metric(target, base)
    post, _ = _metric(target, update_prediction(base, scale, bias, "full", 1.0))
    currents = ref["I_mean_A"].to_numpy(dtype=float)
    mission_lo, mission_hi = np.quantile(train["I_mean_A"], [0.05, 0.95])
    return {
        "reference_cycle_index": event_idx,
        "scale": float(scale),
        "bias": float(bias),
        "rpt_ref_rows": len(ref),
        "rpt_before_mae_pp": pre,
        "rpt_after_mae_pp": post,
        "rpt_median_current_A": float(np.median(currents)),
        "mission_train_current_q05_A": float(mission_lo),
        "mission_train_current_q95_A": float(mission_hi),
        "current_support_eligible": current_support_eligible(train, ref),
        "rpt_phases": ";".join(map(str, ref["phase_Ns"].tolist())),
    }


def audit(labels: pd.DataFrame, references: dict[str, dict[int, dict]]):
    per_cell = []
    per_event = []
    proxy_controls = []
    all_predictions = []

    for cell_id, cell in labels.groupby("cell_id", sort=True):
        cell = cell.sort_values(["charge_cycle_index", "phase_Ns"]).copy()
        cycles = sorted(cell["charge_cycle_index"].astype(int).unique().tolist())
        if len(cycles) <= N20:
            continue
        cutoff = cycles[N20 - 1]
        train = cell.loc[cell["charge_cycle_index"].le(cutoff)].copy()
        test = cell.loc[cell["charge_cycle_index"].gt(cutoff)].copy()
        if train.empty or test.empty:
            continue
        assert train["charge_cycle_index"].max() < test["charge_cycle_index"].min()
        assert (test["capacity_anchor_index"] < test["charge_cycle_index"]).all()

        model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        model.fit(train[FEATURES], train["SOC_proxy"])
        test["base_pred"] = np.clip(model.predict(test[FEATURES]), 0.0, 1.0)
        test["SOC_proxy"] = test["SOC_proxy"].astype(float)
        refs = references.get(cell_id, {})
        events = fixed_periodic_events(refs, cutoff)
        event_params = {idx: _head_event(model, train, idx, refs[idx]) for idx in events}
        for idx, diagnostic in event_params.items():
            per_event.append({"cell_id": cell_id, **diagnostic})

        # One identical B=4 event schedule for every policy, with prediction
        # strictly later than its reference index; no test labels used in updates.
        last_event_by_cycle = {
            int(c): strictly_previous_index(events, int(c))
            for c in test["charge_cycle_index"].unique()
        }
        test["active_reference"] = test["charge_cycle_index"].map(last_event_by_cycle)
        actual = test["SOC_proxy"].to_numpy(dtype=float)
        frozen = test["base_pred"].to_numpy(dtype=float)
        cell_policy_preds = {}
        for name, (kind, strength) in POLICIES.items():
            preds = frozen.copy()
            for idx in events:
                positions = np.flatnonzero(test["active_reference"].eq(idx).to_numpy())
                if not positions.size:
                    continue
                info = event_params[idx]
                preds[positions] = update_prediction(
                    frozen[positions], info["scale"], info["bias"],
                    kind, strength, bool(info["current_support_eligible"]),
                )
            cell_policy_preds[name] = preds
            mae, rmse = _metric(actual, preds)
            late_cutoff = test["charge_cycle_index"].quantile(0.75)
            late = test["charge_cycle_index"].ge(late_cutoff).to_numpy()
            late_mae, _ = _metric(actual[late], preds[late])
            cycles_error = pd.DataFrame({
                "cycle": test["charge_cycle_index"].to_numpy(),
                "error": np.abs(actual - preds) * 100,
            }).groupby("cycle")["error"].mean()
            per_cell.append({
                "cell_id": cell_id, "policy": name,
                "events_completed": len(events),
                "events_eligible_for_current_gate": sum(
                    int(v["current_support_eligible"]) for v in event_params.values()
                ),
                "test_phase_rows": len(test),
                "test_missions": len(cycles) - N20,
                "proxy_MAE_pp": mae, "proxy_RMSE_pp": rmse,
                "late_proxy_MAE_pp": late_mae,
                "P95_cycle_MAE_pp": float(cycles_error.quantile(0.95)),
            })

        # Compare reference-domain fit with the next 50 mission cycles after
        # each RPT, before the next RPT. These scores are offline diagnostics
        # and are NEVER fed back into a selected policy or event threshold.
        for j, idx in enumerate(events):
            next_idx = events[j + 1] if j + 1 < len(events) else np.inf
            future_cycles = sorted(
                int(c) for c in test["charge_cycle_index"].unique()
                if idx < c < next_idx
            )[:50]
            if not future_cycles:
                continue
            mask = test["charge_cycle_index"].isin(future_cycles).to_numpy()
            before, _ = _metric(actual[mask], frozen[mask])
            after, _ = _metric(actual[mask], cell_policy_preds["legacy_affine"][mask])
            # Append as one event-level diagnostic; first record exists already.
            matching = next(v for v in per_event if v["cell_id"] == cell_id and v["reference_cycle_index"] == idx)
            matching.update({
                "next50_missions": len(future_cycles),
                "mission_frozen_mae_pp": before,
                "mission_after_affine_mae_pp": after,
                "mission_delta_mae_pp": after - before,
            })

        # IMPORTANT: this matches the label by construction and is only a
        # circularity diagnostic, NOT an independent SOC accuracy benchmark.
        capacity_aware = np.clip(
            1.0 - test["QDischarge_end_Ah"].to_numpy(dtype=float)
            / test["capacity_used_Ah"].to_numpy(dtype=float), 0.0, 1.0
        )
        nominal = test["SOC_nominal_3Ah"].to_numpy(dtype=float)
        oracle_proxy_mae, _ = _metric(actual, capacity_aware)
        nominal_proxy_mae, _ = _metric(actual, nominal)
        proxy_controls.append({
            "cell_id": cell_id,
            "same_capacity_as_label_identity_MAE_pp": oracle_proxy_mae,
            "nominal_3Ah_vs_proxy_MAE_pp": nominal_proxy_mae,
        })
        output = test[["cell_id", "charge_cycle_index", "phase_Ns", "SOC_proxy"]].copy()
        for name, predictions in cell_policy_preds.items():
            output[name] = predictions
        all_predictions.append(output)

    return (
        pd.DataFrame(per_cell), pd.DataFrame(per_event),
        pd.DataFrame(proxy_controls), pd.concat(all_predictions, ignore_index=True),
    )


def _report(cell: pd.DataFrame, events: pd.DataFrame, controls: pd.DataFrame) -> str:
    by_policy = cell.groupby("policy", sort=True).agg(
        physical_cells=("cell_id", "nunique"),
        median_MAE_pp=("proxy_MAE_pp", "median"),
        mean_MAE_pp=("proxy_MAE_pp", "mean"),
        median_late_MAE_pp=("late_proxy_MAE_pp", "median"),
        median_P95_cycle_MAE_pp=("P95_cycle_MAE_pp", "median"),
        completed_RPT_events=("events_completed", "sum"),
        current_gate_eligible_events=("events_eligible_for_current_gate", "sum"),
    ).reset_index()
    frozen = cell[cell.policy.eq("frozen")].set_index("cell_id")
    deltas = []
    for policy, subset in cell.groupby("policy", sort=True):
        joined = subset.set_index("cell_id")[["proxy_MAE_pp"]].join(
            frozen[["proxy_MAE_pp"]], rsuffix="_frozen"
        )
        paired = (joined["proxy_MAE_pp"] - joined["proxy_MAE_pp_frozen"]).to_numpy()
        rng = np.random.default_rng(20261012)
        draws = rng.choice(paired, size=(10000, len(paired)), replace=True).mean(axis=1)
        ci = np.quantile(draws, [0.025, 0.975])
        deltas.append((policy, float(paired.mean()), float(ci[0]), float(ci[1]), int((paired < 0).sum())))
    lines = [
        "# RPT-to-mission adaptation: matched offline B=4 audit",
        "",
        "## Scope (read before interpreting errors)",
        "",
        "- CMU laboratory VTC6 eVTOL *single-cell* data, not agricultural drone flight.",
        "- All SOC errors are in percentage points versus SOC_proxy, not independent true SOC.",
        "- SOC_proxy uses QDischarge and the latest completed prior candidate RPT capacity.",
        "- The same-capacity coulomb count control reproduces SOC_proxy by definition; it is NOT independent accuracy evidence.",
        "- We replay four RPT opportunities selected from the **known complete RPT schedule**. This offline schedule is not a deployable online policy.",
        "- Only previously completed RPTs affect later mission predictions; test SOC_proxy is never used to fit corrections, gates, or thresholds.",
        "- Fixed shrink strengths are ablations, NOT parameters selected on held-out test data.",
        "",
        "## Matched policy comparison",
        "",
        "| Policy | Cells | Median MAE (pp) | Mean MAE (pp) | Late MAE (pp) | P95-cycle MAE (pp) | Mean paired delta vs frozen (pp) | 95% cell-bootstrap CI (pp) | Cells improved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    delta_dict = {d[0]: d[1:] for d in deltas}
    for row in by_policy.itertuples():
        lines.append(
            f"| {row.policy} | {row.physical_cells} | {row.median_MAE_pp:.3f} | "
            f"{row.mean_MAE_pp:.3f} | {row.median_late_MAE_pp:.3f} | "
            f"{row.median_P95_cycle_MAE_pp:.3f} | {delta_dict[row.policy][0]:+.3f} | "
            f"[{delta_dict[row.policy][1]:+.3f}, {delta_dict[row.policy][2]:+.3f}] | "
            f"{delta_dict[row.policy][3]} |"
        )
    scored = events.dropna(subset=["mission_delta_mae_pp"])
    harmful = int(scored["mission_delta_mae_pp"].gt(0).sum()) if not scored.empty else 0
    n = len(scored)
    eligible = int(events["current_support_eligible"].sum()) if not events.empty else 0
    rpt_fit_improved = int((events["rpt_after_mae_pp"] < events["rpt_before_mae_pp"]).sum())
    paired_transfer_harm = int((
        scored["mission_delta_mae_pp"].gt(0)
        & scored["rpt_after_mae_pp"].lt(scored["rpt_before_mae_pp"])
    ).sum())
    lines += [
        "", "## RPT domain shift and update harm",
        "",
        f"- RPT events in B4 replay: {len(events)}; current-support-gate eligible: {eligible}.",
        f"- Paired next-50-mission windows with an observed task: {n}; legacy affine raises proxy MAE in {harmful}/{n}.",
        f"- RPT own-domain fit improved in {rpt_fit_improved}/{len(events)} selected references.",
        f"- RPT fit improves BUT next-50-mission transfer worsens in {paired_transfer_harm}/{n} evaluated windows.",
        f"- Median reference current: {events['rpt_median_current_A'].median():.3f} A.",
        f"- Mission-training current-support medians (q05–q95): {events['mission_train_current_q05_A'].median():.3f} to {events['mission_train_current_q95_A'].median():.3f} A.",
        "- RPT fit improvement with simultaneous mission deterioration is evidence of domain transfer risk, not proof of a particular electrochemical mechanism.",
        "- A gate which blocks all updates merely reproduces the frozen baseline; it is not a successful adaptation result.",
        "", "## Target construction identity audit",
        "",
        f"- Median same-capacity identity error: {controls['same_capacity_as_label_identity_MAE_pp'].median():.8f} pp.",
        f"- Median fixed-3Ah difference against SOC_proxy: {controls['nominal_3Ah_vs_proxy_MAE_pp'].median():.3f} pp.",
        "- Do not claim any method has improved real SOC without an independent SOC reference.",
        "", "## Decision",
        "",
        "- If unshrunk RPT Head correction is harmful, do not deploy it merely because a trigger detects drift.",
        "- Distinguish measured capacity updates from neural-network output calibration.",
        "- A follow-on method must prove improvement on an independent endpoint (e.g. next RPT capacity or future mission voltage) and ultimately independent SOC.",
        "- Capacity/RPT candidate validity and charge-voltage-dependent start SOC require further scrutiny.",
        "", "## Generated files",
        "",
        "- policy_by_cell.csv (one row per cell/policy)",
        "- reference_domain_and_next50.csv (every B4 RPT event, transfer diagnostic)",
        "- proxy_label_identity.csv (circularity check)",
        "- predictions_by_phase.csv (fully paired offline phase predictions)",
        "- run_metadata.json (selection and invariants)",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "active_rpt_audit_artifacts")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(RESULTS / "soc_proxy_phase_labels.csv")
    segments = pd.read_csv(RESULTS / "segment_summary.csv")
    candidates = pd.read_csv(RESULTS / "capacity_test_candidates.csv")
    refs = _make_rpt_reference_table(segments, candidates)
    cell, events, controls, predictions = audit(labels, refs)
    if cell["cell_id"].nunique() != 22:
        raise AssertionError("Expected all 22 cells in matched evaluation")
    if controls["same_capacity_as_label_identity_MAE_pp"].max() > 1e-6:
        raise AssertionError("SOC proxy identity control mismatch")
    if (cell.groupby("cell_id")["events_completed"].nunique() != 1).any():
        raise AssertionError("Reference budgets differ across head policies")
    cell.to_csv(out / "policy_by_cell.csv", index=False)
    events.to_csv(out / "reference_domain_and_next50.csv", index=False)
    controls.to_csv(out / "proxy_label_identity.csv", index=False)
    predictions.to_csv(out / "predictions_by_phase.csv", index=False)
    metadata = {
        "study": "offline RPT transfer diagnostic, not a deployed active update system",
        "commit_protocol": "target cell N20 Ridge; B=4 preknown periodic candidate opportunities",
        "reference_rule": "strictly prior completed candidate RPT only; all head policies share events",
        "labels": "SOC_proxy (coulomb-counted reference; no independent ground truth)",
        "test_label_used_for_head_or_gate": False,
        "offline_schedule_knows_future_rpt_opportunities": True,
        "model_selection_on_test": False,
        "policies": POLICIES,
        "physical_cells": int(cell.cell_id.nunique()),
        "reference_events": int(len(events)),
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report = _report(cell, events, controls)
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
