"""Chunked structural audit for the CMU eVTOL battery dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_MAIN = [
    "time_s",
    "Ecell_V",
    "I_mA",
    "EnergyCharge_W_h",
    "QCharge_mA_h",
    "EnergyDischarge_W_h",
    "QDischarge_mA_h",
    "Temperature__C",
    "cycleNumber",
    "Ns",
]
CHUNK_ROWS = 250_000


def update_segment(store: dict, key: tuple, row: pd.Series) -> None:
    values = row.to_dict()
    if key not in store:
        values["segment_run_id"] = key[1]
        values["cycleNumber"] = key[2]
        values["Ns"] = key[3]
        values["cell_id"] = key[0]
        store[key] = values
        return
    target = store[key]
    target["rows"] += int(values["rows"])
    for name in ("time_start_s", "V_min", "T_min"):
        target[name] = min(target[name], values[name])
    for name in ("time_end_s", "V_max", "T_max", "QCharge_max_mAh", "QDischarge_max_mAh",
                 "EnergyCharge_max_Wh", "EnergyDischarge_max_Wh", "I_abs_peak_mA"):
        target[name] = max(target[name], values[name])
    for name in ("I_abs_sum_mA", "I_signed_sum_mA", "I_count", "T_sum_C", "T_count"):
        target[name] += values[name]


def summarize_cell(path: Path) -> tuple[dict, list[dict], list[dict]]:
    cell_id = path.stem
    header = list(pd.read_csv(path, nrows=0).columns)
    stats = {name: {"min": np.inf, "max": -np.inf, "missing": 0} for name in header}
    segments: dict[tuple, dict] = {}
    raw_cycles: set[float] = set()
    rows = 0
    nonpositive_dt = 0
    gaps_5m = 0
    gaps_1h = 0
    max_gap_s = 0.0
    previous_time = None
    previous_key = None
    run_id = 0
    segment_transitions = 0

    for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, on_bad_lines="error"):
        rows += len(chunk)
        for name in header:
            original = chunk[name]
            numeric = pd.to_numeric(original, errors="coerce")
            stats[name]["missing"] += int(numeric.isna().sum())
            if numeric.notna().any():
                stats[name]["min"] = min(stats[name]["min"], float(numeric.min()))
                stats[name]["max"] = max(stats[name]["max"], float(numeric.max()))
            chunk[name] = numeric

        times = chunk["time_s"]
        dt = times.diff()
        if previous_time is not None and len(times):
            dt.iloc[0] = float(times.iloc[0]) - previous_time
        valid_dt = dt.dropna()
        nonpositive_dt += int((valid_dt <= 0).sum())
        gaps_5m += int((valid_dt > 300).sum())
        gaps_1h += int((valid_dt > 3600).sum())
        if not valid_dt.empty:
            max_gap_s = max(max_gap_s, float(valid_dt.max()))
        if len(times):
            previous_time = float(times.iloc[-1])

        raw_cycles.update(float(x) for x in chunk["cycleNumber"].dropna().unique())
        cycle = chunk["cycleNumber"]
        ns = chunk["Ns"]
        cycle_prev, ns_prev = cycle.shift(), ns.shift()
        same_cycle = cycle.eq(cycle_prev) | (cycle.isna() & cycle_prev.isna())
        same_ns = ns.eq(ns_prev) | (ns.isna() & ns_prev.isna())
        changed = ~(same_cycle & same_ns)
        if len(chunk):
            first = (None if pd.isna(cycle.iloc[0]) else float(cycle.iloc[0]),
                     None if pd.isna(ns.iloc[0]) else float(ns.iloc[0]))
            if previous_key is None:
                changed.iloc[0] = True
            else:
                changed.iloc[0] = first != previous_key
            previous_key = (None if pd.isna(cycle.iloc[-1]) else float(cycle.iloc[-1]),
                            None if pd.isna(ns.iloc[-1]) else float(ns.iloc[-1]))
            segment_transitions += int(changed.sum())
            chunk["_run_id"] = run_id + changed.astype(np.int64).cumsum()
            run_id += int(changed.sum())

        chunk["_I_abs"] = chunk["I_mA"].abs()
        chunk["_T_valid"] = chunk["Temperature__C"].notna().astype(np.int64)
        chunk["_I_valid"] = chunk["I_mA"].notna().astype(np.int64)
        grouped = chunk.groupby(["_run_id", "cycleNumber", "Ns"], dropna=False, sort=False).agg(
            rows=("time_s", "size"),
            time_start_s=("time_s", "min"),
            time_end_s=("time_s", "max"),
            V_min=("Ecell_V", "min"),
            V_max=("Ecell_V", "max"),
            I_abs_sum_mA=("_I_abs", "sum"),
            I_signed_sum_mA=("I_mA", "sum"),
            I_count=("_I_valid", "sum"),
            I_abs_peak_mA=("_I_abs", "max"),
            T_sum_C=("Temperature__C", "sum"),
            T_count=("_T_valid", "sum"),
            T_min=("Temperature__C", "min"),
            T_max=("Temperature__C", "max"),
            QCharge_max_mAh=("QCharge_mA_h", "max"),
            QDischarge_max_mAh=("QDischarge_mA_h", "max"),
            EnergyCharge_max_Wh=("EnergyCharge_W_h", "max"),
            EnergyDischarge_max_Wh=("EnergyDischarge_W_h", "max"),
        )
        for (segment_id, cycle, ns), row in grouped.iterrows():
            cycle_key = None if pd.isna(cycle) else float(cycle)
            ns_key = None if pd.isna(ns) else float(ns)
            update_segment(segments, (cell_id, int(segment_id), cycle_key, ns_key), row)

    field_rows = []
    for name, item in stats.items():
        field_rows.append({
            "cell_id": cell_id,
            "field": name,
            "min": None if not np.isfinite(item["min"]) else item["min"],
            "max": None if not np.isfinite(item["max"]) else item["max"],
            "missing": item["missing"],
            "missing_fraction": item["missing"] / rows if rows else np.nan,
        })

    ordered_cycles = sorted(raw_cycles)
    cycle_gaps = sum(max(0, int(b - a - 1)) for a, b in zip(ordered_cycles[:-1], ordered_cycles[1:])
                     if float(b - a).is_integer())
    cell_row = {
        "cell_id": cell_id,
        "file_bytes": path.stat().st_size,
        "rows": rows,
        "header_matches_expected": header == EXPECTED_MAIN,
        "header": ",".join(header),
        "unique_cycle_labels": len(raw_cycles),
        "min_cycleNumber": min(raw_cycles) if raw_cycles else np.nan,
        "max_cycleNumber": max(raw_cycles) if raw_cycles else np.nan,
        "integer_cycle_label_gaps": cycle_gaps,
        "segment_key_count": len(segments),
        "segment_transitions": segment_transitions,
        "time_min_s": stats.get("time_s", {}).get("min"),
        "time_max_s": stats.get("time_s", {}).get("max"),
        "time_span_days": (stats["time_s"]["max"] - stats["time_s"]["min"]) / 86400
        if np.isfinite(stats["time_s"]["min"]) and np.isfinite(stats["time_s"]["max"]) else np.nan,
        "nonpositive_time_steps": nonpositive_dt,
        "gaps_over_5min": gaps_5m,
        "gaps_over_1h": gaps_1h,
        "max_time_gap_s": max_gap_s,
    }
    segment_rows = []
    for item in segments.values():
        item["duration_s"] = item["time_end_s"] - item["time_start_s"]
        count = item.pop("I_count")
        item["I_abs_mean_mA"] = item.pop("I_abs_sum_mA") / count if count else np.nan
        item["I_signed_mean_mA"] = item.pop("I_signed_sum_mA") / count if count else np.nan
        item["T_mean_C"] = item.pop("T_sum_C") / item.pop("T_count") if item.get("T_count") else np.nan
        item["QDischarge_max_Ah"] = item.pop("QDischarge_max_mAh") / 1000
        item["QCharge_max_Ah"] = item.pop("QCharge_max_mAh") / 1000
        segment_rows.append(item)
    return cell_row, segment_rows, field_rows


def summarize_impedance(path: Path) -> dict:
    frame = pd.read_csv(path)
    result = {
        "cell_id": path.stem.removesuffix("_impedance"),
        "file_bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": len(frame.columns),
        "header": ",".join(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
    }
    cycle_col = next((c for c in frame.columns if "cycle" in c.lower()), None)
    if cycle_col:
        result["cycle_min"] = pd.to_numeric(frame[cycle_col], errors="coerce").min()
        result["cycle_max"] = pd.to_numeric(frame[cycle_col], errors="coerce").max()
    return result


def main() -> None:
    study = Path(__file__).resolve().parents[1]
    data_root = study.parent
    out = study / "results"
    out.mkdir(parents=True, exist_ok=True)
    main_files = sorted(data_root.glob("VAH[0-9][0-9].csv"))
    impedance_files = sorted(data_root.glob("VAH[0-9][0-9]_impedance.csv"))
    cell_rows, segment_rows, field_rows = [], [], []
    for i, path in enumerate(main_files, 1):
        print(f"[{i}/{len(main_files)}] {path.name}", flush=True)
        cell, segments, fields = summarize_cell(path)
        cell_rows.append(cell)
        segment_rows.extend(segments)
        field_rows.extend(fields)
    pd.DataFrame(cell_rows).to_csv(out / "cell_summary.csv", index=False)
    segment_frame = pd.DataFrame(segment_rows).sort_values(["cell_id", "time_start_s"])
    segment_frame["charge_cycle_index"] = np.nan
    for cell_id, indices in segment_frame.groupby("cell_id").groups.items():
        order = segment_frame.loc[indices].sort_values("segment_run_id").index
        is_charge = segment_frame.loc[order, "Ns"].eq(0) & segment_frame.loc[order, "duration_s"].gt(100)
        segment_frame.loc[order, "charge_cycle_index"] = is_charge.astype(np.int64).cumsum().to_numpy()
    segment_frame.to_csv(out / "segment_summary.csv", index=False)
    pd.DataFrame(field_rows).to_csv(out / "field_summary.csv", index=False)
    impedance_rows = [summarize_impedance(path) for path in impedance_files]
    pd.DataFrame(impedance_rows).to_csv(out / "impedance_summary.csv", index=False)

    # The full-capacity test combines flight discharge with a low-rate tail.
    # Identify the tail by its long, low-current discharge and 2.5 V cutoff;
    # QDischarge is cumulative across earlier discharge phases in that test.
    candidates = segment_frame.loc[
        (segment_frame["QDischarge_max_Ah"] >= 1.0)
        & segment_frame["I_signed_mean_mA"].between(-1000, -300)
        & (segment_frame["duration_s"] >= 3000)
        & (segment_frame["V_min"] <= 2.6)
    ].copy()
    candidates["candidate_reason"] = (
        "low-rate discharge tail: -1.0 to -0.3 A mean, >=3000 s, Vmin <=2.6 V; verify against RPT records"
    )
    candidates.to_csv(out / "capacity_test_candidates.csv", index=False)
    cycle_summary = segment_frame.groupby("cell_id").agg(
        contiguous_phase_runs=("segment_run_id", "size"),
        charge_blocks_over_100s=("charge_cycle_index", "max"),
        one_row_phase_runs=("rows", lambda x: int((x == 1).sum())),
    ).reset_index()
    rpt_counts = candidates.groupby("cell_id").size().rename("rpt_tail_candidates").reset_index()
    cycle_summary = cycle_summary.merge(rpt_counts, on="cell_id", how="left").fillna({"rpt_tail_candidates": 0})
    cycle_summary.to_csv(out / "study_cycle_summary.csv", index=False)

    metadata = {
        "source_readme": "README.txt",
        "main_files": len(main_files),
        "impedance_files": len(impedance_files),
        "expected_main_columns": EXPECTED_MAIN,
        "chunk_rows": CHUNK_ROWS,
        "capacity_test_candidate_rule": "low-rate, long-duration discharge tail ending near 2.5 V",
        "capacity_candidates_require_manual_protocol_review": True,
        "reported_dataset_cycle_count": 21392,
        "cycle_count_proxy": "contiguous Ns=0 charge phase lasting >100 s",
    }
    (out / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote audit tables to {out}", flush=True)


if __name__ == "__main__":
    main()
