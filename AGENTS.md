# Repository Guidelines

## Project Structure & Data

This repository is a flat data collection: `README.txt` documents the eVTOL Battery Dataset, and `VAHxx.csv` files contain life-cycle measurements for individual cells. Matching `VAHxx_impedance.csv` files hold impedance readings where available. Keep the original CSVs at the repository root and treat them as source data; put analysis scripts and derived results in clearly named subdirectories (for example, `analysis/` and `derived/`) rather than replacing source files. Consult the README for each cell’s test protocol and known anomalies.

## Build, Test & Development

There is no application code, package manifest, or configured build/test runner. No build command is currently applicable. For data changes, inspect the CSV header and confirm every row has the same number of fields; compare updated files with the source and record the tool or script used to produce derived outputs.

## Data Conventions

Preserve the existing `VAH` identifier and lowercase `_impedance` suffix. Keep column names and units intact (such as `time_s`, `Ecell_V`, and `Temperature__C`). Do not infer a continuous cycle sequence from `cycleNumber`: the README notes these are raw tester values and files may combine multiple tester outputs. Flag documented anomalies in analysis rather than silently correcting measurements.

## Testing & Review

There is no test framework or coverage target. When adding analysis code, include a small reproducible check for expected columns, numeric fields, and missing or malformed rows. Pull requests should explain the dataset or analysis change, list affected cell IDs, describe validation, and include plots or screenshots when they clarify a result. No Git history is present here to establish a commit style; use a short imperative subject, for example, `Document CSV validation workflow`.