"""Causal capacity-state update benchmark on observed *next* candidate RPT.

Unlike SOC_proxy reconstruction, the target here is a later independently
recorded low-rate capacity measurement (not the same capacity used as input).
RPT candidates remain algorithmically detected and require protocol validation.

The B4 reference opportunities are selected using a known full historical
opportunity list: valid for retrospective equal-budget diagnostics ONLY.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
B = 4
SEED = 20261012


def observed_capacity(points: list[tuple[int, float]], target_cycle: int, policy: str) -> float:
    """Only prior *selected* RPTs are visible; never touch target capacity."""
    earlier = sorted((int(c), float(q)) for c, q in points if c < target_cycle)
    if not earlier:
        raise ValueError("Cannot predict capacity without a completed earlier reference")
    c0, q0 = earlier[0]
    cl, ql = earlier[-1]
    if policy == "freeze_first":
        return q0
    if policy == "last_reference":
        return ql
    if policy == "two_reference_rate_aware":
        if len(earlier) < 2:
            return ql
        cp, qp = earlier[-2]
        if cl <= cp:
            raise AssertionError("Non-increasing reference indices")
        # Predeclared nonpositive trend; no lookahead or fitted test settings.
        slope = min(0.0, (ql - qp) / (cl - cp))
        return float(np.clip(ql + slope * (target_cycle - cl), 0.5 * q0, ql))
    raise ValueError(policy)


def selected_rpt_indices(ids: list[int], budget: int = B) -> list[int]:
    """Retrospective B4 schedule excluding initial anchor and last (no follow-up)."""
    eligible = sorted(ids)[1:-1]
    if not eligible:
        return []
    loc = np.rint(np.linspace(0, len(eligible) - 1, min(budget, len(eligible)))).astype(int)
    return [eligible[i] for i in sorted(set(loc.tolist()))]


def evaluate(capacities: pd.DataFrame):
    expected = {"cell_id", "charge_cycle_index", "QDischarge_max_Ah"}
    if not expected.issubset(capacities.columns):
        raise ValueError(f"Missing capacity columns: {expected - set(capacities.columns)}")
    records = []
    refs_output = []
    for cell_id, group in capacities.groupby("cell_id", sort=True):
        # Some candidates can share charge_cycle_index; one candidate capacity
        # per index, exactly as used in the original SOC_proxy construction.
        samples = (
            group.groupby("charge_cycle_index")["QDischarge_max_Ah"]
            .max().sort_index()
        )
        ids = [int(x) for x in samples.index.tolist()]
        if len(ids) < 3:
            continue
        first = (ids[0], float(samples.iloc[0]))
        scheduled = selected_rpt_indices(ids)
        capacities_by_index = {int(k): float(v) for k, v in samples.items()}
        observed = [first]
        for cycle_id in ids[1:]:
            future_capacity = capacities_by_index[cycle_id]
            for policy in ["freeze_first", "last_reference", "two_reference_rate_aware"]:
                prediction = observed_capacity(observed, cycle_id, policy)
                records.append({
                    "cell_id": cell_id, "target_rpt_index": cycle_id,
                    "observed_rpt_capacity_Ah": future_capacity,
                    "prediction_capacity_Ah": prediction,
                    "policy": policy, "observed_refs_before_prediction": len(observed),
                    "abs_error_mAh": abs(prediction - future_capacity) * 1000.0,
                    "abs_error_SOH_pp": abs(prediction - future_capacity) / first[1] * 100.0,
                    "target_is_selected_RPT": cycle_id in scheduled,
                })
            if cycle_id in scheduled:
                # New capacity becomes known only AFTER predictions at this RPT.
                observed.append((cycle_id, future_capacity))
        refs_output.append({
            "cell_id": cell_id, "initial_rpt_cycle": first[0],
            "initial_capacity_Ah": first[1],
            "full_lifecycle_rpts": len(ids), "selected_extra_references": len(scheduled),
            "selected_rpt_cycles": ";".join(map(str, scheduled)),
        })
    return pd.DataFrame(records), pd.DataFrame(refs_output)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = predictions.groupby(["cell_id", "policy"], as_index=False).agg(
        RPT_targets=("target_rpt_index", "size"),
        MAE_mAh=("abs_error_mAh", "mean"),
        MAE_SOH_pp=("abs_error_SOH_pp", "mean"),
    )
    macro = cells.groupby("policy", as_index=False).agg(
        cells=("cell_id", "nunique"), median_MAE_mAh=("MAE_mAh", "median"),
        mean_MAE_mAh=("MAE_mAh", "mean"),
        median_MAE_SOH_pp=("MAE_SOH_pp", "median"),
    )
    return cells, macro


def report(cells: pd.DataFrame, macro: pd.DataFrame, events: pd.DataFrame) -> str:
    frozen = cells[cells.policy.eq("freeze_first")].set_index("cell_id")
    lines = [
        "# Causal capacity-state update: next-RPT forecasting",
        "",
        "Independent evaluation TARGET = NEXT recorded low-rate RPT candidate capacity; no future capacity is input.",
        "This is independent of the SOC_proxy equation, but RPT candidates still need manual verification.",
        "B=4 extra reference opportunities per cell where possible, selected from the known offline opportunity list.",
        "Such a full-future opportunity schedule is NOT a deployable active maintenance policy.",
        "",
        "| Capacity estimator | Cells | Median next-RPT MAE (mAh) | Median next-RPT MAE (SOH pp) | Mean paired difference vs frozen (mAh) | 95% cell-bootstrap CI (mAh) | Better cells |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in macro.itertuples():
        subset = cells[cells.policy.eq(row.policy)].set_index("cell_id")
        delta = (subset["MAE_mAh"] - frozen["MAE_mAh"]).to_numpy()
        rng = np.random.default_rng(SEED)
        draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        lo, hi = np.quantile(draws, [0.025, 0.975])
        lines.append(
            f"| {row.policy} | {row.cells} | {row.median_MAE_mAh:.2f} | "
            f"{row.median_MAE_SOH_pp:.3f} | {delta.mean():+.2f} | "
            f"[{lo:+.2f}, {hi:+.2f}] | {(delta < 0).sum()} |"
        )
    lines += [
        "", f"Reference opportunities selected across all cells: {events.selected_extra_references.sum()}.",
        "",
        "**Interpretation:** this isolates whether refreshing a slow physical capacity state improves forward measured capacity forecasting. It does NOT imply neural Head updating or real-time SOC updating is improved.",
        "RPT opportunities use a known retrospective schedule; a genuine When-to-Update trigger still requires online-only testing.",
        "Predictions at each capacity check are issued *before* that test's measured capacity is released.",
        "", "Artifacts: capacity_prediction_by_RPT.csv; capacity_update_by_cell.csv; capacity_update_schedules.csv; capacity_update_report.md",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(RESULTS / "capacity_test_candidates.csv")
    predictions, refs = evaluate(candidates)
    cells, macro = summarize(predictions)
    assert cells.cell_id.nunique() == 22
    assert (predictions["observed_refs_before_prediction"] >= 1).all()
    assert (refs["selected_extra_references"] <= B).all()
    predictions.to_csv(out / "capacity_prediction_by_RPT.csv", index=False)
    cells.to_csv(out / "capacity_update_by_cell.csv", index=False)
    refs.to_csv(out / "capacity_update_schedules.csv", index=False)
    text = report(cells, macro, refs)
    (out / "capacity_update_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
