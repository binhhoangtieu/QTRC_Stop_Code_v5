from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REQUIRED = [
    "cp0_inventory/selected/inventory.json",
    "cp1_events/events.csv",
    "cp2_features/features.csv.gz",
    "cp3_gates/gate_report.json",
    "cp5_policies/loss_matrix.json",
    "cp5_policies/calibration_argmax_audit.csv",
    "cp5_policies/fixed_risk_matched_grid.csv",
    "cp6_results/fold_metrics.csv",
    "cp6_results/event_decisions.csv.gz",
    "cp6_results/summary_metrics.csv",
    "cp6_results/species_metrics.csv",
    "cp6_results/subject_metrics.csv",
    "cp6_results/policy_audit.csv",
    "cp6_results/certification_summary.csv",
    "cp6_results/calibration_argmax_audit.csv",
    "cp6_results/calibration_safeguard.json",
    "cp6_results/confusion_matrices.csv",
    "cp8_reports/final_report.json",
]

SUBMISSION_REQUIRED = [
    "cp7_robustness/status.json",
    "cp7_robustness/robustness_summary.csv",
    "cp7_robustness/robustness_seed_summary.csv",
    "cp7_robustness/feature_drift_summary.csv",
    "cp7_robustness/species_sensitivity_summary.csv",
    "cp7_robustness/prediction_distribution_seed.csv",
    "cp7_robustness/awgn_monotonicity_diagnostic.json",
    "cp7_robustness/paper_sensitivity_table_e0.050.csv",
    "cp7_robustness/paper_sensitivity_rows_e0.050.tex",
    "cp8_reports/runtime/runtime_summary.csv",
    "cp8_reports/runtime/runtime_report.json",
    "cp8_reports/runtime/hardware_manifest.json",
    "cp8_reports/runtime/runtime_manuscript_sentence.txt",
    "cp8_reports/manuscript_inputs.json",
    "cp8_reports/manuscript_update_snippet.tex",
    "cp8_reports/figures/fixed_duration_curve.png",
    "cp8_reports/figures/risk_gap_vs_time.png",
    "cp8_reports/figures/confusion_matrix_qtrc_e005.png",
    "cp8_reports/figures/awgn_sensitivity_curve.png",
    "cp8_reports/figures/feature_drift_vs_snr.png",
    "cp8_reports/tables/confusion_matrix_qtrc_e005_counts.csv",
    "cp8_reports/tables/confusion_matrix_qtrc_e005_row_normalized.csv",
]


def verify_run(run_dir: str | Path, submission_ready: bool = False) -> tuple[bool, dict]:
    root = Path(run_dir)
    required = REQUIRED + (SUBMISSION_REQUIRED if submission_ready else [])
    report = {
        "run_dir": str(root.resolve()),
        "submission_ready_requested": bool(submission_ready),
        "missing": [],
        "failed_stages": [],
        "warnings": [],
    }
    for rel in required:
        if not (root / rel).exists():
            report["missing"].append(rel)

    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        report["failed_stages"] = [
            k for k, v in state.get("stages", {}).items() if v.get("status") == "failed"
        ]
        report["stages"] = {k: v.get("status") for k, v in state.get("stages", {}).items()}
    else:
        report["missing"].append("state.json")

    gate_path = root / "cp3_gates" / "gate_report.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        report["cp3_pass"] = bool(gate.get("pass", False))
        report["causal_filter"] = bool(gate.get("causal_filter", False))
        report["chronological_event_selection"] = bool(gate.get("chronological_event_selection", False))
        if not report["cp3_pass"]:
            report["warnings"].append("CP3 gate is not PASS")

    summary_path = root / "cp6_results" / "summary_metrics.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        report["summary_rows"] = int(len(summary))
        report["methods"] = sorted(summary["method"].dropna().unique().tolist())
        required_metrics = {
            "accuracy", "macro_f1", "balanced_accuracy", "worst_recall", "p90_stop_sec",
        }
        report["pooled_metric_columns_present"] = required_metrics.issubset(summary.columns)
        report["risk_matched_fixed_present"] = "fixed_risk_matched" in set(summary["method"])
        if summary.empty:
            report["warnings"].append("Summary table is empty")

    safeguard_path = root / "cp6_results" / "calibration_safeguard.json"
    if safeguard_path.exists():
        safeguard = json.loads(safeguard_path.read_text(encoding="utf-8"))
        report["calibration_argmax_preserved"] = bool(safeguard.get("pass", False))
        report["calibration_argmax_changes"] = int(safeguard.get("total_argmax_changes", -1))
        if not report["calibration_argmax_preserved"]:
            report["warnings"].append("Calibration changed one or more predicted classes")

    if submission_ready:
        cp7_status_path = root / "cp7_robustness" / "status.json"
        if cp7_status_path.exists():
            cp7 = json.loads(cp7_status_path.read_text(encoding="utf-8"))
            report["cp7_status"] = cp7.get("status")
            report["cp7_complete"] = cp7.get("status") == "complete" and bool(cp7.get("enabled", False))
        else:
            report["cp7_complete"] = False
        runtime_path = root / "cp8_reports" / "runtime" / "runtime_report.json"
        if runtime_path.exists():
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            report["runtime_status"] = runtime.get("status")
            report["runtime_complete"] = runtime.get("status") == "complete" and bool(runtime.get("enabled", False))
        else:
            report["runtime_complete"] = False
        table_path = root / "cp7_robustness" / "paper_sensitivity_table_e0.050.csv"
        if table_path.exists():
            table = pd.read_csv(table_path)
            report["paper_sensitivity_rows"] = int(len(table))
            report["paper_sensitivity_shape_ok"] = len(table) == 5 and {
                "accuracy_mean", "accuracy_std", "accuracy_delta_mean",
                "mean_stop_sec_mean", "subject_mean_gap_mean", "seeds"
            }.issubset(table.columns)
            report["cp7_multiseed_min"] = int(table["seeds"].min()) if "seeds" in table else 0
        else:
            report["paper_sensitivity_shape_ok"] = False
            report["cp7_multiseed_min"] = 0
        drift_path = root / "cp7_robustness" / "feature_drift_summary.csv"
        report["feature_drift_present"] = drift_path.exists()
        awgn_diag_path = root / "cp7_robustness" / "awgn_monotonicity_diagnostic.json"
        if awgn_diag_path.exists():
            awgn_diag = json.loads(awgn_diag_path.read_text(encoding="utf-8"))
            report["awgn_accuracy_non_decreasing_with_snr"] = bool(awgn_diag.get("accuracy_non_decreasing_with_snr", False))
            if not report["awgn_accuracy_non_decreasing_with_snr"]:
                report["warnings"].append("AWGN accuracy is non-monotonic across SNR; report this as a sensitivity limitation, not robustness.")

    ok = (
        not report["missing"]
        and not report["failed_stages"]
        and report.get("cp3_pass", False)
        and report.get("summary_rows", 0) > 0
        and report.get("pooled_metric_columns_present", False)
        and report.get("risk_matched_fixed_present", False)
        and report.get("calibration_argmax_preserved", False)
    )
    if submission_ready:
        ok = ok and report.get("cp7_complete", False) and report.get("runtime_complete", False)
        ok = ok and report.get("paper_sensitivity_shape_ok", False)
        ok = ok and report.get("cp7_multiseed_min", 0) >= 3 and report.get("feature_drift_present", False)
    report["ok"] = bool(ok)
    return bool(ok), report
