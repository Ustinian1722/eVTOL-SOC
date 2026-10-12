"""Stream the source files and retain only anomalous rows for audit review."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "soc_drift_study"
RESULTS = STUDY / "results"
CHUNK_ROWS = 250_000


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    expected_numeric = [
        "time_s", "Ecell_V", "I_mA", "EnergyCharge_W_h", "QCharge_mA_h",
        "EnergyDischarge_W_h", "QDischarge_mA_h", "Temperature__C", "cycleNumber", "Ns",
    ]
    output_rows: list[pd.DataFrame] = []
    counts: dict[str, int] = {
        "rows": 0,
        "nonpositive_time_steps": 0,
        "gaps_over_5min": 0,
        "gaps_over_1h": 0,
        "voltage_over_4p25V": 0,
        "temperature_over_50C": 0,
        "rows_with_missing_or_nonnumeric_fields": 0,
    }

    for path in sorted(ROOT.glob("VAH[0-9][0-9].csv")):
        print("Scanning", path.name, flush=True)
        previous_time: float | None = None
        offset = 0
        for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, on_bad_lines="error"):
            numeric = chunk[expected_numeric].apply(pd.to_numeric, errors="coerce")
            counts["rows"] += len(chunk)
            bad_numeric = numeric.isna().any(axis=1)
            time = numeric["time_s"]
            dt = time.diff()
            if previous_time is not None and len(dt):
                dt.iloc[0] = float(time.iloc[0]) - previous_time
            nonpositive = dt.le(0).fillna(False)
            gap5 = dt.gt(300).fillna(False)
            gap1h = dt.gt(3600).fillna(False)
            high_v = numeric["Ecell_V"].gt(4.25).fillna(False)
            high_t = numeric["Temperature__C"].gt(50).fillna(False)
            counts["nonpositive_time_steps"] += int(nonpositive.sum())
            counts["gaps_over_5min"] += int(gap5.sum())
            counts["gaps_over_1h"] += int(gap1h.sum())
            counts["voltage_over_4p25V"] += int(high_v.sum())
            counts["temperature_over_50C"] += int(high_t.sum())
            counts["rows_with_missing_or_nonnumeric_fields"] += int(bad_numeric.sum())

            flagged = nonpositive | gap5 | high_v | high_t | bad_numeric
            if flagged.any():
                rows = numeric.loc[flagged].copy()
                rows.insert(0, "cell_id", path.stem)
                rows.insert(1, "source_row_0based", offset + rows.index.to_numpy())
                rows["dt_from_previous_s"] = dt.loc[flagged].to_numpy()
                flags = []
                for idx in rows.index:
                    labels = []
                    if bool(nonpositive.loc[idx]):
                        labels.append("nonpositive_dt")
                    if bool(gap5.loc[idx]):
                        labels.append("gap_gt_300s")
                    if bool(high_v.loc[idx]):
                        labels.append("voltage_gt_4.25V")
                    if bool(high_t.loc[idx]):
                        labels.append("temperature_gt_50C")
                    if bool(bad_numeric.loc[idx]):
                        labels.append("missing_or_nonnumeric")
                    flags.append(";".join(labels))
                rows["flags"] = flags
                output_rows.append(rows.reset_index(drop=True))
            if len(time):
                previous_time = float(time.iloc[-1])
            offset += len(chunk)

    anomaly_frame = pd.concat(output_rows, ignore_index=True) if output_rows else pd.DataFrame()
    anomaly_frame.to_csv(RESULTS / "anomaly_rows.csv", index=False)
    (RESULTS / "anomaly_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print("Anomaly counts:", json.dumps(counts, sort_keys=True), flush=True)
    print("Wrote", len(anomaly_frame), "flagged rows", flush=True)


if __name__ == "__main__":
    main()
