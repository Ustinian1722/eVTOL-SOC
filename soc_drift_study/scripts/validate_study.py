"""Validate key dataset, causality, model, report, and artifact invariants."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parent
RESULTS = STUDY / "results"
REPORTS = STUDY / "reports"
MODELS = STUDY / "models"
FIGURES = STUDY / "figures"
LOGS = STUDY / "logs"


def main() -> None:
    checks: list[dict[str, object]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    audit = json.loads((RESULTS / "audit_metadata.json").read_text(encoding="utf-8"))
    anomaly = json.loads((RESULTS / "anomaly_summary.json").read_text(encoding="utf-8"))
    label_audit = json.loads((RESULTS / "label_audit.json").read_text(encoding="utf-8"))
    labels = pd.read_csv(RESULTS / "soc_proxy_phase_labels.csv")
    record("dataset_file_counts", audit["main_files"] == 22 and audit["impedance_files"] == 21,
           f"main={audit['main_files']}, impedance={audit['impedance_files']}")
    record("source_file_inventory", len(list(ROOT.glob("VAH*.csv"))) == 43,
           f"VAH*.csv count={len(list(ROOT.glob('VAH*.csv')))}")
    record("dataset_row_audit", anomaly["rows"] == 15309317 and anomaly["rows_with_missing_or_nonnumeric_fields"] == 0,
           f"rows={anomaly['rows']}, malformed={anomaly['rows_with_missing_or_nonnumeric_fields']}")
    record("independent_soc_absent_is_documented", label_audit["independent_soc_ground_truth_present"] is False,
           "label audit explicitly records no independent SOC truth")
    anchors_causal = bool((labels["capacity_anchor_index"] < labels["charge_cycle_index"]).all())
    proxy_bounded = bool(labels["SOC_proxy"].between(0, 1).all())
    record("proxy_anchor_no_future_leakage", anchors_causal,
           f"rows={len(labels)}, all capacity_anchor_index < charge_cycle_index")
    record("proxy_range", proxy_bounded, "all SOC_proxy labels lie in [0,1]")
    record("proxy_label_count", len(labels) == 62563 and labels.cell_id.nunique() == 22,
           f"rows={len(labels)}, cells={labels.cell_id.nunique()}")

    baseline = pd.read_csv(RESULTS / "soc_proxy_baseline_metrics.csv")
    record("ridge_cell_evaluations", int((baseline.model == "target_cell_N20_ridge").sum()) == 22,
           f"target N20 Ridge cells={(baseline.model == 'target_cell_N20_ridge').sum()}")
    record("ridge_weights_exist", (MODELS / "phase_ridge_models.joblib").is_file(),
           "phase_ridge_models.joblib present")

    gru_metrics = pd.read_csv(RESULTS / "gru_proxy_crosscell_metrics.csv")
    gru_cell_count = gru_metrics.groupby("method").cell_id.nunique().to_dict()
    gru_weights = list(MODELS.glob("phase_gru_fold*_seed*.weights.h5"))
    gru_scalers = list(MODELS.glob("phase_gru_fold*_seed*_scaler.json"))
    fold_configs = [json.loads(path.read_text(encoding="utf-8")) for path in gru_scalers]
    fold_disjoint = all(set(cfg["source_cells"]).isdisjoint(cfg["heldout_cells"]) for cfg in fold_configs)
    record("gru_cell_heldout_evaluations", len(gru_metrics) == 132 and all(n == 22 for n in gru_cell_count.values()),
           f"rows={len(gru_metrics)}, cells_per_method={gru_cell_count}")
    record("gru_fold_artifacts", len(gru_weights) == 15 and len(gru_scalers) == 15,
           f"weights={len(gru_weights)}, scalers={len(gru_scalers)}")
    record("gru_source_target_cell_disjoint", fold_disjoint,
           "each saved fold has disjoint source and held-out cells")
    full_ridge = pd.read_csv(RESULTS / "source_full_ridge_crosscell_metrics.csv")
    record("matched_full_source_ridge_control", len(full_ridge) == 44 and full_ridge.cell_id.nunique() == 22,
           f"rows={len(full_ridge)}, cells={full_ridge.cell_id.nunique()}")

    maintenance = pd.read_csv(RESULTS / "proxy_maintenance_metrics.csv")
    method_counts = maintenance.groupby(["method", "budget"]).cell_id.nunique()
    complete_groups = bool((method_counts == 22).all())
    budget_ok = bool((maintenance["updates_used"] <= maintenance["budget"]).all())
    record("maintenance_group_completeness", len(maintenance) == 814 and complete_groups,
           f"rows={len(maintenance)}, method-budget groups={len(method_counts)}")
    record("maintenance_budget_caps", budget_ok, "updates_used never exceeds configured budget")
    timeline = pd.read_csv(RESULTS / "proxy_maintenance_timeline.csv")
    request_causal = bool((timeline["request_cycle_index"] < timeline["reference_cycle_index"]).all())
    effective = timeline["first_effective_cycle_index"].dropna()
    effective_causal = bool((effective > timeline.loc[effective.index, "reference_cycle_index"]).all())
    record("maintenance_request_before_reference", request_causal,
           f"timeline rows={len(timeline)}; request cycle precedes selected RPT")
    record("head_effect_after_reference", effective_causal,
           f"effective updates={len(effective)}; each begins after RPT completion")

    config = json.loads((STUDY / "configs" / "proxy_maintenance.json").read_text(encoding="utf-8"))
    record("voltage_target_excluded_from_features", "Vmin_V" not in config["voltage_residual_features"],
           f"features={config['voltage_residual_features']}")
    oracle = pd.read_csv(RESULTS / "offline_oracle_metrics.csv")
    oracle_caps = bool((oracle.updates_used <= oracle.budget).all())
    oracle_meta = json.loads((RESULTS / "offline_oracle_metadata.json").read_text(encoding="utf-8"))
    record("oracle_scope_is_hindsight", "hindsight" in oracle_meta["meaning"].lower() and oracle_caps,
           f"rows={len(oracle)}, labels={oracle_meta['selection_labels']}")
    replay = pd.read_csv(RESULTS / "rpt_head_replay_metrics.csv")
    record("head_replay_sensitivity", len(replay) == 132 and replay.cell_id.nunique() == 22,
           f"rows={len(replay)}, cells={replay.cell_id.nunique()}")

    reports = sorted(REPORTS.glob("*.md"))
    report_names = {path.name for path in reports}
    required_reports = {f"{i:02d}_{name}.md" for i, name in [
        (1, "数据审计"), (2, "SOC标签与基线"), (3, "生命周期漂移分析"),
        (4, "漂移检测实验"), (5, "主动维护实验"), (6, "模型更新与成本分析"),
        (7, "完整实验结果"), (8, "硕士论文研究框架"), (9, "最终研究决策"),
    ]}
    missing_reports = sorted(required_reports - report_names)
    record("nine_reports_present", not missing_reports, f"missing={missing_reports}")
    required_figures = [
        "capacity_fade_candidates.png", "impedance_lifecycle.png", "representative_mission_lifecycle.png",
        "soc_proxy_baseline_by_model.png", "soc_proxy_error_lifecycle.png", "proxy_drift_indicators.png",
        "proxy_maintenance_pareto.png", "maintenance_strategy_errors.png",
        "maintenance_timeline_VAH01_B4.png", "rpt_head_replay_sensitivity.png",
        "gru_proxy_crosscell.png", "gru_ridge_proxy_comparison.png",
    ]
    missing_figures = [name for name in required_figures if not (FIGURES / name).is_file()]
    record("figures_present", not missing_figures, f"missing={missing_figures}")
    record("experiment_log_present", (LOGS / "experiment_log.md").is_file() and (LOGS / "environment.json").is_file(),
           "experiment log and environment record present")

    result = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_directory": str(STUDY),
        "passed": all(bool(item["passed"]) for item in checks),
        "check_count": len(checks),
        "checks": checks,
        "scope_note": "SOC metrics are against SOC_proxy; Oracle uses hindsight labels and is not deployable.",
    }
    output = RESULTS / "pipeline_validation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = [item for item in checks if not item["passed"]]
    print(f"Validation: {len(checks) - len(failed)}/{len(checks)} checks passed; wrote {output}")
    if failed:
        for item in failed:
            print(f"FAIL {item['check']}: {item['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
