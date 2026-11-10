from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from .baselines import (
    apply_score_stopping,
    confidence_policy,
    fit_two_stage_reliability,
    fixed_duration_decisions,
    select_fixed_duration_split_certified,
    select_score_threshold,
)
from .calibration import apply_temperature, argmax_change_count, fit_temperature
from .checkpoint import RunState, atomic_dataframe_csv, atomic_write_json, task_lock
from .config import config_fingerprint, save_resolved_config
from .events import detect_events, events_to_frame
from .features import extract_event_feature_rows, feature_columns
from .io import ensure_dataset, inventory_dataset, read_signal_csv, select_inventory, write_inventory_reports
from .logging_utils import setup_logging
from .metrics import (
    add_subject_bootstrap_ci,
    attach_decision_losses,
    brier_multiclass,
    classwise_ece,
    decision_metrics,
    expected_calibration_error,
    pooled_decision_metrics,
    species_metrics,
)
from .models import HierarchicalClassifier, subject_balanced_weights
from .plots import (
    plot_accuracy_time,
    plot_calibration,
    plot_certification,
    plot_fixed_duration,
    plot_risk_time,
    plot_confusion_matrix,
    plot_awgn_sensitivity,
    plot_feature_drift_vs_snr,
)
from .robustness import make_common_perturbation_draws, perturb_preprocessed_signal_common
from .runtime_benchmark import hardware_manifest, stratified_event_sample, summarize_runtime_samples
from .signal import complex_from_amplitude_phase, preprocess_complex
from .stopping import (
    Policy,
    apply_policy,
    evaluate_policy_grid,
    select_policy_split_certified,
    split_calibration_subjects,
)
from .taxonomy import SPECIES, label_indices, taxonomic_loss_matrix


STAGES = ["cp0", "cp1", "cp2", "cp3", "cp4", "cp5", "cp6", "cp7", "cp8"]
BOMBUS_CLASSES = ["bombus_lapidarius", "bombus_muscorum", "bombus_terrestris"]


class Pipeline:
    def __init__(self, cfg: dict[str, Any], run_dir: str | Path, dataset_override: str | None = None):
        self.cfg = cfg
        if dataset_override:
            self.cfg["dataset"]["path"] = str(Path(dataset_override).expanduser().resolve())
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state = RunState(self.run_dir)
        self.logger = setup_logging(self.run_dir, cfg["runtime"]["log_level"])
        fingerprint = config_fingerprint(self.cfg)
        meta_path = self.run_dir / "run_meta.json"
        if meta_path.exists():
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
            if previous.get("config_fingerprint") != fingerprint:
                raise RuntimeError(
                    "The run directory contains checkpoints from a different configuration. "
                    "Use a new run directory or restore the original configuration."
                )
        save_resolved_config(self.cfg, self.run_dir / "resolved_config.yaml")
        atomic_write_json(meta_path, {
            "config_fingerprint": fingerprint,
            "pid": os.getpid(),
            "pipeline_version": "5.0.0",
            "causal_preprocessing": bool(self.cfg["signal"].get("causal", True)),
        })

    def run(self, start_stage: str | None = None, end_stage: str | None = None):
        start = STAGES.index(start_stage) if start_stage else 0
        end = STAGES.index(end_stage) if end_stage else len(STAGES) - 1
        for stage in STAGES[start:end + 1]:
            method = getattr(self, stage)
            self.logger.info("=== %s START ===", stage.upper())
            self.state.update_stage(stage, "running")
            try:
                method()
                self.state.update_stage(stage, "done")
                self.logger.info("=== %s DONE ===", stage.upper())
            except Exception as exc:
                self.state.update_stage(stage, "failed", error=str(exc), traceback=traceback.format_exc())
                self.logger.exception("Stage %s failed", stage)
                raise

    @property
    def stale_minutes(self) -> int:
        return int(self.cfg["runtime"]["stale_lock_minutes"])

    @property
    def fs(self) -> float:
        return float(self.cfg["dataset"]["sampling_rate_hz"])

    def cp0(self):
        out = self.run_dir / "cp0_inventory"
        dataset_root = ensure_dataset(self.cfg["dataset"]["path"], self.run_dir / "data_cache" / "raw")
        df = inventory_dataset(dataset_root)
        write_inventory_reports(df, out / "all")
        expected = int(self.cfg["dataset"].get("expected_files", 0))
        if self.cfg["dataset"].get("strict_inventory", True) and expected and len(df) != expected:
            raise RuntimeError(f"Expected {expected} files, found {len(df)}")
        selected = select_inventory(
            df,
            self.cfg["dataset"].get("max_subjects_per_species"),
            self.cfg["dataset"].get("max_recordings_per_subject"),
        )
        write_inventory_reports(selected, out / "selected")
        self.logger.info("Selected %d recordings from %d subjects", len(selected), selected.subject_id.nunique())

    def cp1(self):
        inv = pd.read_csv(self.run_dir / "cp0_inventory" / "selected" / "recordings.csv")
        parts = self.run_dir / "cp1_events" / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        for row in inv.itertuples(index=False):
            artifact = parts / f"{row.recording_id}.csv"
            task = f"cp1::{row.recording_id}"
            if self.state.task_done(task, artifact):
                continue
            with task_lock(artifact.with_suffix(".lock"), self.stale_minutes):
                amp, phase = read_signal_csv(row.path)
                z = complex_from_amplitude_phase(amp, phase)
                z_hp = preprocess_complex(z, self.fs, self.cfg["signal"])
                events = detect_events(z_hp, self.fs, self.cfg["events"])
                frame = events_to_frame(pd.Series(row._asdict()), events)
                atomic_dataframe_csv(frame, artifact)
                self.state.update_task(task, "done", events=len(frame), artifact=str(artifact))
        frames = [pd.read_csv(p) for p in sorted(parts.glob("*.csv")) if p.stat().st_size > 0]
        events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        atomic_dataframe_csv(events, self.run_dir / "cp1_events" / "events.csv")
        if events.empty:
            raise RuntimeError("No events detected")
        self.logger.info("Detected %d chronological events; fallback fraction %.3f", len(events), events["fallback_event"].mean())

    def cp2(self):
        events = pd.read_csv(self.run_dir / "cp1_events" / "events.csv")
        parts = self.run_dir / "cp2_features" / "parts"
        parts.mkdir(parents=True, exist_ok=True)
        for recording_id, group in events.groupby("recording_id", sort=True):
            artifact = parts / f"{recording_id}.csv.gz"
            task = f"cp2::{recording_id}"
            if self.state.task_done(task, artifact):
                continue
            with task_lock(artifact.with_suffix(".lock"), self.stale_minutes):
                amp, phase = read_signal_csv(group.iloc[0]["path"])
                z = complex_from_amplitude_phase(amp, phase)
                z_hp = preprocess_complex(z, self.fs, self.cfg["signal"])
                rows: list[dict[str, Any]] = []
                for _, event in group.iterrows():
                    rows.extend(extract_event_feature_rows(event, z_hp, self.fs, self.cfg["features"]))
                frame = pd.DataFrame(rows)
                atomic_dataframe_csv(frame, artifact, compression="gzip")
                self.state.update_task(task, "done", rows=len(frame), artifact=str(artifact))
        frames = [pd.read_csv(p) for p in sorted(parts.glob("*.csv.gz"))]
        features = pd.concat(frames, ignore_index=True)
        artifact = self.run_dir / "cp2_features" / "features.csv.gz"
        atomic_dataframe_csv(features, artifact, compression="gzip")
        self.logger.info("Extracted %d prefix rows with %d numerical features", len(features), len(feature_columns(features)))

    def cp3(self):
        df = pd.read_csv(self.run_dir / "cp2_features" / "features.csv.gz")
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        report: dict[str, Any] = {
            "rows": int(len(df)),
            "events": int(df.event_id.nunique()),
            "subjects": int(df.subject_id.nunique()),
            "species": df.groupby("species")["subject_id"].nunique().astype(int).to_dict(),
            "nan_fraction": float(df[feature_columns(df)].replace([np.inf, -np.inf], np.nan).isna().mean().mean()),
            "causal_filter": bool(self.cfg["signal"].get("causal", True)),
            "record_normalization": self.cfg["signal"].get("normalize", "none"),
            "prefix_normalization": self.cfg["features"].get("prefix_normalize", "robust_prefix"),
            "chronological_event_selection": bool(self.cfg["events"].get("causal", True)),
            "first_prefix_stability_unavailable": bool(np.isinf(df.loc[np.isclose(df.duration_sec, min(durations)), "f0_delta_hz"]).all()),
        }
        duration_counts = df.groupby("event_id")["duration_sec"].nunique()
        report["complete_event_fraction"] = float(np.mean(duration_counts == len(durations)))
        report["fallback_event_fraction"] = float(pd.read_csv(self.run_dir / "cp1_events" / "events.csv")["fallback_event"].mean())
        gates = self.cfg.get("gates", {})
        expected_species = gates.get("expected_subjects_per_species")
        expected_ok = True
        if expected_species:
            expected_ok = all(report["species"].get(k, 0) >= int(v) for k, v in expected_species.items())
        report["pass"] = bool(
            report["subjects"] >= int(gates.get("min_subjects_total", 10))
            and len(report["species"]) >= int(gates.get("min_species", 5))
            and report["nan_fraction"] <= float(gates.get("nan_fraction_max", 0.05))
            and report["complete_event_fraction"] >= float(gates.get("complete_event_fraction_min", 0.95))
            and report["fallback_event_fraction"] <= float(gates.get("fallback_event_fraction_max", 1.0))
            and report["causal_filter"]
            and report["chronological_event_selection"]
            and report["first_prefix_stability_unavailable"]
            and expected_ok
        )
        report["gate_config"] = gates
        atomic_write_json(self.run_dir / "cp3_gates" / "gate_report.json", report)
        if not report["pass"] and self.cfg["runtime"].get("fail_on_gate", True):
            raise RuntimeError(f"CP3 gate failed: {report}")

    def _outer_splits(self, df: pd.DataFrame):
        subject_df = df[["subject_id", "species"]].drop_duplicates().sort_values("subject_id")
        x = np.zeros((len(subject_df), 1))
        y = subject_df["species"].to_numpy()
        groups = subject_df["subject_id"].to_numpy()
        protocol = self.cfg["cv"]["outer_protocol"]
        if protocol == "loso":
            splits = list(LeaveOneGroupOut().split(x, y, groups))
        else:
            splitter = StratifiedGroupKFold(
                n_splits=int(self.cfg["cv"]["outer_splits"]),
                shuffle=True,
                random_state=int(self.cfg["seed"]),
            )
            splits = list(splitter.split(x, y, groups))
        max_folds = self.cfg["cv"].get("max_outer_folds")
        if max_folds is not None:
            splits = splits[: int(max_folds)]
        for fold, (tr, te) in enumerate(splits):
            yield fold, set(groups[tr]), set(groups[te])

    @staticmethod
    def _prepare_xy(df: pd.DataFrame, fcols: list[str]):
        x = df[fcols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        y = df["species"].to_numpy()
        w = subject_balanced_weights(df["subject_id"].to_numpy(), y)
        return x, y, w

    @staticmethod
    def _component_columns() -> list[str]:
        return [
            "p_family_apidae", "p_family_vespidae",
            "p_genus_apis", "p_genus_bombus",
            "p_cond_bombus_lapidarius", "p_cond_bombus_muscorum", "p_cond_bombus_terrestris",
        ]

    @staticmethod
    def _attach_probabilities(pred: pd.DataFrame, components: dict[str, np.ndarray]) -> pd.DataFrame:
        pred = pred.copy()
        pred["p_family_apidae"] = components["family"][:, 0]
        pred["p_family_vespidae"] = components["family"][:, 1]
        pred["p_genus_apis"] = components["genus"][:, 0]
        pred["p_genus_bombus"] = components["genus"][:, 1]
        for k, species in enumerate(BOMBUS_CLASSES):
            pred[f"p_cond_{species}"] = components["bombus"][:, k]
        species_probs = HierarchicalClassifier.combine_components(
            components["family"], components["genus"], components["bombus"]
        )
        for k, species in enumerate(SPECIES):
            pred[f"p_{species}"] = species_probs[:, k]
        return pred

    def _predict_task(self, train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int, artifact: Path, model_path: Path | None):
        fcols = feature_columns(train_df)
        x_train, y_train, w = self._prepare_xy(train_df, fcols)
        x_val, _, _ = self._prepare_xy(val_df, fcols)
        model = HierarchicalClassifier(self.cfg["model"], seed).fit(x_train, y_train, fcols, w)
        components = model.predict_components(x_val)
        quality_cols = ["harmonic_ratio", "harmonic_energy_ratio", "sbr_db", "f0_delta_hz", "stability_available"]
        pred = val_df[["event_id", "recording_id", "subject_id", "species", "duration_sec", *quality_cols]].copy()
        pred = self._attach_probabilities(pred, components)
        atomic_dataframe_csv(pred, artifact, compression="gzip")
        if model_path is not None:
            model.save(model_path)

    def cp4(self):
        df = pd.read_csv(self.run_dir / "cp2_features" / "features.csv.gz")
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        root = self.run_dir / "cp4_predictions"
        for outer_fold, train_subjects, test_subjects in self._outer_splits(df):
            fold_dir = root / f"outer_{outer_fold:03d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            outer_train = df[df.subject_id.isin(train_subjects)].copy()
            outer_test = df[df.subject_id.isin(test_subjects)].copy()
            atomic_write_json(fold_dir / "split.json", {
                "train_subjects": sorted(train_subjects),
                "test_subjects": sorted(test_subjects),
            })
            subject_df = outer_train[["subject_id", "species"]].drop_duplicates().sort_values("subject_id")
            min_subjects = int(subject_df.groupby("species")["subject_id"].nunique().min())
            n_inner = max(2, min(int(self.cfg["cv"]["inner_splits"]), min_subjects))
            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=int(self.cfg["seed"]) + outer_fold)
            xs = np.zeros((len(subject_df), 1))
            ys = subject_df.species.to_numpy()
            gs = subject_df.subject_id.to_numpy()
            inner_parts: list[Path] = []
            for inner_fold, (itr, iva) in enumerate(inner.split(xs, ys, gs)):
                inner_train_subjects = set(gs[itr])
                inner_val_subjects = set(gs[iva])
                for duration in durations:
                    artifact = fold_dir / "oof_parts" / f"inner_{inner_fold:02d}_t{duration:.1f}.csv.gz"
                    task = f"cp4::outer{outer_fold}::inner{inner_fold}::t{duration:.1f}"
                    if not self.state.task_done(task, artifact):
                        with task_lock(artifact.with_suffix(".lock"), self.stale_minutes):
                            tr = outer_train[outer_train.subject_id.isin(inner_train_subjects) & np.isclose(outer_train.duration_sec, duration)]
                            va = outer_train[outer_train.subject_id.isin(inner_val_subjects) & np.isclose(outer_train.duration_sec, duration)]
                            self._predict_task(tr, va, int(self.cfg["seed"]) + outer_fold * 1000 + inner_fold * 100 + int(duration * 10), artifact, None)
                            self.state.update_task(task, "done", artifact=str(artifact))
                    inner_parts.append(artifact)
            oof = pd.concat([pd.read_csv(p) for p in sorted(set(inner_parts))], ignore_index=True)
            atomic_dataframe_csv(oof, fold_dir / "oof_predictions.csv.gz", compression="gzip")

            test_parts: list[Path] = []
            for duration in durations:
                artifact = fold_dir / "test_parts" / f"t{duration:.1f}.csv.gz"
                model_path = fold_dir / "models" / f"t{duration:.1f}.joblib"
                task = f"cp4::outer{outer_fold}::final::t{duration:.1f}"
                if not self.state.task_done(task, artifact):
                    with task_lock(artifact.with_suffix(".lock"), self.stale_minutes):
                        tr = outer_train[np.isclose(outer_train.duration_sec, duration)]
                        te = outer_test[np.isclose(outer_test.duration_sec, duration)]
                        self._predict_task(tr, te, int(self.cfg["seed"]) + outer_fold * 1000 + 900 + int(duration * 10), artifact, model_path)
                        self.state.update_task(task, "done", artifact=str(artifact), model=str(model_path))
                test_parts.append(artifact)
            test = pd.concat([pd.read_csv(p) for p in test_parts], ignore_index=True)
            atomic_dataframe_csv(test, fold_dir / "test_predictions.csv.gz", compression="gzip")

    @staticmethod
    def _apply_species_temperature_to_frame(
        frame: pd.DataFrame, temperature_payload: dict[str, Any] | float
    ) -> tuple[pd.DataFrame, int]:
        """Calibrate the final species posterior with one scalar temperature.

        A scalar temperature preserves the species argmax. Hierarchical component
        probabilities remain available for diagnostics but are not independently
        rescaled because separate temperatures can change the final class.
        """
        out = frame.copy()
        temperature = (
            float(temperature_payload.get("species", 1.0))
            if isinstance(temperature_payload, dict)
            else float(temperature_payload)
        )
        cols = [f"p_{species}" for species in SPECIES]
        raw = out[cols].to_numpy(dtype=float)
        calibrated = apply_temperature(raw, temperature)
        changes = argmax_change_count(raw, calibrated)
        if changes != 0:
            raise RuntimeError(
                f"Class-preserving calibration failed: {changes} argmax predictions changed"
            )
        out.loc[:, cols] = calibrated
        return out, changes

    def _calibrate_fold(
        self, fold_dir: Path
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
        oof = pd.read_csv(fold_dir / "oof_predictions.csv.gz")
        test = pd.read_csv(fold_dir / "test_predictions.csv.gz")
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        bounds = self.cfg["calibration"]["temperature_bounds"]
        temperatures: dict[str, dict[str, Any]] = {}
        calibrated_oof_parts: list[pd.DataFrame] = []
        calibrated_test_parts: list[pd.DataFrame] = []
        diagnostics: list[dict[str, Any]] = []
        cols = [f"p_{species}" for species in SPECIES]
        for duration in durations:
            o = oof[np.isclose(oof.duration_sec, duration)].copy()
            t = test[np.isclose(test.duration_sec, duration)].copy()
            probs = o[cols].to_numpy(dtype=float)
            y_idx = label_indices(o["species"])
            weights = subject_balanced_weights(
                o["subject_id"].to_numpy(), o["species"].to_numpy()
            )
            temperature = fit_temperature(probs, y_idx, weights, bounds)
            payload = {
                "species": float(temperature),
                "method": "species_scalar_temperature",
                "argmax_preserving": True,
            }
            calibrated_oof, oof_changes = self._apply_species_temperature_to_frame(o, payload)
            calibrated_test, test_changes = self._apply_species_temperature_to_frame(t, payload)
            payload.update({
                "oof_argmax_changes": int(oof_changes),
                "test_argmax_changes": int(test_changes),
            })
            temperatures[f"{duration:.1f}"] = payload
            diagnostics.append({
                "duration_sec": float(duration),
                "temperature": float(temperature),
                "oof_events": int(len(o)),
                "test_events": int(len(t)),
                "oof_argmax_changes": int(oof_changes),
                "test_argmax_changes": int(test_changes),
                "argmax_preserving": bool(oof_changes == 0 and test_changes == 0),
            })
            calibrated_oof_parts.append(calibrated_oof)
            calibrated_test_parts.append(calibrated_test)
        return (
            pd.concat(calibrated_oof_parts, ignore_index=True),
            pd.concat(calibrated_test_parts, ignore_index=True),
            temperatures,
            pd.DataFrame(diagnostics),
        )

    def _fold_loss_matrix(self, fold_dir: Path, features: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
        split = json.loads((fold_dir / "split.json").read_text(encoding="utf-8"))
        train_subjects = set(split["train_subjects"])
        counts = (
            features[features.subject_id.isin(train_subjects)][["subject_id", "species"]]
            .drop_duplicates().groupby("species")["subject_id"].nunique().astype(int).to_dict()
        )
        matrix = taxonomic_loss_matrix(
            counts,
            float(self.cfg["policy"]["beta_family"]),
            float(self.cfg["policy"]["beta_genus"]),
            bool(self.cfg["policy"]["inverse_subject_class_weight"]),
        )
        return matrix, counts

    def _policy_variants(self) -> dict[str, dict[str, Any]]:
        base = dict(self.cfg["policy"])
        variants = {"qtrc_stop": base}
        if bool(self.cfg.get("ablation", {}).get("enabled", True)):
            no_quality = dict(base)
            no_quality.update({
                "harmonic_ratio_grid": [0.0],
                "harmonic_energy_grid": [0.0],
                "sbr_grid_db": [-1e9],
            })
            variants["confidence_stability"] = no_quality
            no_sbr = dict(base)
            no_sbr.update({"sbr_grid_db": [-1e9]})
            variants["qtrc_no_sbr"] = no_sbr
        confidence_only = dict(base)
        confidence_only.update({
            "js_grid": [1.0],
            "harmonic_ratio_grid": [0.0],
            "harmonic_energy_grid": [0.0],
            "sbr_grid_db": [-1e9],
            "f0_delta_grid_hz": [1e9],
            "persistence_grid": [1],
            "require_stability_available": False,
        })
        variants["confidence_only"] = confidence_only
        return variants

    def cp5(self):
        features = pd.read_csv(self.run_dir / "cp2_features" / "features.csv.gz")
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        fold_index: list[dict[str, Any]] = []
        calibration_audit_parts: list[pd.DataFrame] = []
        fixed_grid_parts: list[pd.DataFrame] = []
        for fold_dir in sorted((self.run_dir / "cp4_predictions").glob("outer_*")):
            fold = int(fold_dir.name.split("_")[1])
            loss_matrix, counts = self._fold_loss_matrix(fold_dir, features)
            loss_path = fold_dir / "loss_matrix.json"
            atomic_write_json(loss_path, {
                "species": SPECIES,
                "matrix": loss_matrix.tolist(),
                "outer_training_subject_counts": counts,
                "leakage_control": "computed from outer-training subjects only",
            })
            fold_index.append({
                "outer_fold": fold,
                "path": str(loss_path.relative_to(self.run_dir)),
                "subject_counts": counts,
            })

            calibrated_oof = fold_dir / "calibrated_oof.csv.gz"
            calibrated_test = fold_dir / "calibrated_test.csv.gz"
            temp_path = fold_dir / "temperatures.json"
            calibration_audit_path = fold_dir / "calibration_argmax_audit.csv"
            task = f"cp5v3::outer{fold}::species_temperature"
            if not self.state.task_done(task, calibrated_test):
                oof, test, temps, audit = self._calibrate_fold(fold_dir)
                audit["outer_fold"] = fold
                atomic_dataframe_csv(oof, calibrated_oof, compression="gzip")
                atomic_dataframe_csv(test, calibrated_test, compression="gzip")
                atomic_write_json(temp_path, temps)
                atomic_dataframe_csv(audit, calibration_audit_path)
                self.state.update_task(task, "done", artifact=str(calibrated_test))
            else:
                oof = pd.read_csv(calibrated_oof)
                audit = pd.read_csv(calibration_audit_path)
            calibration_audit_parts.append(audit)

            selection_subjects, certification_subjects = split_calibration_subjects(
                oof, int(self.cfg["seed"]) + fold,
                float(self.cfg["policy"].get("certification_fraction", 0.5)),
            )
            atomic_write_json(fold_dir / "calibration_split.json", {
                "selection_subjects": sorted(selection_subjects),
                "certification_subjects": sorted(certification_subjects),
                "independent_certification": True,
            })

            fixed_grid_path = fold_dir / "policy_grids" / "fixed_risk_matched.csv"
            for epsilon in map(float, self.cfg["policy"]["epsilons"]):
                fixed_policy_path = fold_dir / "policies" / "fixed_risk_matched" / f"epsilon_{epsilon:.3f}.json"
                fixed_cert_path = fold_dir / "policies" / "fixed_risk_matched" / f"epsilon_{epsilon:.3f}_certification.json"
                fixed_task = f"cp5v3::outer{fold}::fixed_risk_matched::eps{epsilon:.3f}"
                if not self.state.task_done(fixed_task, fixed_policy_path):
                    fixed_policy, fixed_grid, fixed_cert = select_fixed_duration_split_certified(
                        oof, durations, epsilon, loss_matrix, self.cfg["policy"],
                        int(self.cfg["seed"]) + 13000 + fold,
                        selection_subjects, certification_subjects,
                    )
                    atomic_write_json(fixed_policy_path, fixed_policy)
                    atomic_write_json(fixed_cert_path, fixed_cert)
                    self.state.update_task(fixed_task, "done", artifact=str(fixed_policy_path))
                    if not fixed_grid_path.exists():
                        atomic_dataframe_csv(fixed_grid, fixed_grid_path)
            fixed_grid = pd.read_csv(fixed_grid_path)
            fixed_grid["outer_fold"] = fold
            fixed_grid_parts.append(fixed_grid)

            for method, policy_cfg in self._policy_variants().items():
                grid_path = fold_dir / "policy_grids" / f"{method}.csv.gz"
                grid_task = f"cp5v3::outer{fold}::grid::{method}"
                if not self.state.task_done(grid_task, grid_path):
                    with task_lock(grid_path.with_suffix(".lock"), self.stale_minutes):
                        grid = evaluate_policy_grid(
                            oof, durations, loss_matrix, policy_cfg,
                            int(self.cfg["seed"]) + fold,
                            subjects=selection_subjects,
                        )
                        atomic_dataframe_csv(grid, grid_path, compression="gzip")
                        self.state.update_task(grid_task, "done", artifact=str(grid_path))
                else:
                    grid = pd.read_csv(grid_path)

                for epsilon in map(float, self.cfg["policy"]["epsilons"]):
                    policy_path = fold_dir / "policies" / method / f"epsilon_{epsilon:.3f}.json"
                    cert_path = fold_dir / "policies" / method / f"epsilon_{epsilon:.3f}_certification.json"
                    policy_task = f"cp5v3::outer{fold}::{method}::eps{epsilon:.3f}"
                    if self.state.task_done(policy_task, policy_path):
                        continue
                    policy, _, cert = select_policy_split_certified(
                        oof, durations, epsilon, loss_matrix, policy_cfg,
                        int(self.cfg["seed"]) + fold,
                        selection_grid=grid,
                        selection_subjects=selection_subjects,
                        certification_subjects=certification_subjects,
                    )
                    atomic_write_json(policy_path, asdict(policy))
                    atomic_write_json(cert_path, cert)
                    self.state.update_task(policy_task, "done", artifact=str(policy_path))

        cp5_out = self.run_dir / "cp5_policies"
        atomic_write_json(cp5_out / "loss_matrix.json", {
            "usage": "Each policy and outer-test fold uses a matrix computed only from its outer-training subjects.",
            "folds": fold_index,
            "class_weight_formula": "inverse subject count normalized to sum one",
        })
        calibration_audit = pd.concat(calibration_audit_parts, ignore_index=True)
        atomic_dataframe_csv(calibration_audit, cp5_out / "calibration_argmax_audit.csv")
        if int(calibration_audit["oof_argmax_changes"].sum() + calibration_audit["test_argmax_changes"].sum()) != 0:
            raise RuntimeError("CP5 class-preserving calibration gate failed")
        atomic_dataframe_csv(pd.concat(fixed_grid_parts, ignore_index=True), cp5_out / "fixed_risk_matched_grid.csv")

    @staticmethod
    def _load_policy(path: Path) -> Policy:
        return Policy(**json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _select_fixed_duration(oof: pd.DataFrame, durations: list[float], loss_matrix: np.ndarray) -> tuple[float, pd.DataFrame]:
        rows = []
        for duration in durations:
            decisions = attach_decision_losses(fixed_duration_decisions(oof, duration), loss_matrix)
            subject_loss = decisions.groupby("subject_id")["taxonomic_loss"].mean()
            rows.append({
                "duration_sec": float(duration),
                "subject_mean_taxonomic_loss": float(subject_loss.mean()),
                "mean_accuracy": float(np.mean(decisions["species"] == decisions["pred_species"])),
            })
        grid = pd.DataFrame(rows)
        chosen = grid.sort_values(["subject_mean_taxonomic_loss", "duration_sec"], ascending=[True, True]).iloc[0]
        return float(chosen.duration_sec), grid

    @staticmethod
    def _subject_rows(decisions: pd.DataFrame, method: str, epsilon: float) -> pd.DataFrame:
        rows = []
        for subject, group in decisions.groupby("subject_id", sort=True):
            active = group[~group["abstain"]]
            rows.append({
                "subject_id": subject,
                "species": str(group["species"].iloc[0]),
                "method": method,
                "epsilon": epsilon,
                "n_events": int(len(group)),
                "accuracy": float(np.mean(active["species"] == active["pred_species"])) if len(active) else np.nan,
                "coverage": float(len(active) / len(group)),
                "mean_stop_sec": float(group["stop_duration_sec"].mean()),
                "mean_taxonomic_gap": float(group["taxonomic_gap"].mean()),
            })
        return pd.DataFrame(rows)

    def cp6(self):
        zero_one = np.ones((len(SPECIES), len(SPECIES)), dtype=float) - np.eye(len(SPECIES))
        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        max_duration = max(durations)
        fold_metric_rows: list[dict[str, Any]] = []
        decision_parts: list[pd.DataFrame] = []
        fixed_selection_rows: list[pd.DataFrame] = []
        calibration_prediction_parts: list[pd.DataFrame] = []
        policy_audit_rows: list[dict[str, Any]] = []

        def record_decisions(
            fold: int,
            method: str,
            epsilon: float,
            decisions: pd.DataFrame,
            loss_matrix: np.ndarray,
            extra: dict[str, Any] | None = None,
        ) -> None:
            enriched = attach_decision_losses(decisions, loss_matrix)
            enriched["outer_fold"] = fold
            enriched["method"] = method
            enriched["epsilon"] = epsilon
            if extra:
                for key, value in extra.items():
                    enriched[key] = value
            decision_parts.append(enriched)
            metrics = decision_metrics(decisions, loss_matrix, max_duration)
            risk_violation = np.nan
            if np.isfinite(epsilon):
                risk_violation = float(metrics["subject_mean_gap"] > epsilon)
            fold_metric_rows.append({
                "outer_fold": fold,
                "method": method,
                "epsilon": epsilon,
                "risk_violation": risk_violation,
                **(extra or {}),
                **metrics,
            })

        def qtrc_audit_row(
            fold: int,
            method: str,
            epsilon: float,
            policy: Policy,
            cert: dict[str, Any],
        ) -> dict[str, Any]:
            candidate = cert.get("selected_candidate") or {}
            return {
                "outer_fold": fold,
                "method": method,
                "epsilon": float(epsilon),
                "status": str(cert.get("status", policy.fallback_reason)),
                "certification_passed": bool(policy.certification_passed),
                "fallback_full_duration": bool(policy.fallback_full_duration),
                "selected_full_duration": bool(
                    policy.fallback_full_duration
                    or policy.calibration_mean_time >= max_duration - 1e-12
                ),
                "fallback_reason": str(policy.fallback_reason),
                "selection_subjects": int(policy.selection_subjects),
                "certification_subjects": int(policy.certification_subjects),
                "certification_mean_gap": float(cert.get("certification_mean_gap", policy.calibration_mean_gap)),
                "certification_ucb_gap": float(cert.get("certification_ucb_gap", policy.calibration_ucb_gap)),
                "candidate_mean_time": float(candidate.get("mean_time", policy.calibration_mean_time)),
                "final_policy_mean_time": float(policy.calibration_mean_time),
                "confidence": float(candidate.get("confidence", policy.confidence)),
                "js_max": float(candidate.get("js_max", policy.js_max)),
                "harmonic_ratio_min": float(candidate.get("harmonic_ratio_min", policy.harmonic_ratio_min)),
                "harmonic_energy_min": float(candidate.get("harmonic_energy_min", policy.harmonic_energy_min)),
                "sbr_min_db": float(candidate.get("sbr_min_db", policy.sbr_min_db)),
                "f0_delta_max_hz": float(candidate.get("f0_delta_max_hz", policy.f0_delta_max_hz)),
                "persistence": int(candidate.get("persistence", policy.persistence)),
                "bound_method": str(policy.bound_method),
                "risk_protocol": str(policy.risk_protocol),
            }

        for fold_dir in sorted((self.run_dir / "cp4_predictions").glob("outer_*")):
            fold = int(fold_dir.name.split("_")[1])
            loss_payload = json.loads((fold_dir / "loss_matrix.json").read_text(encoding="utf-8"))
            loss_matrix = np.asarray(loss_payload["matrix"], dtype=float)
            raw_test = pd.read_csv(fold_dir / "test_predictions.csv.gz")
            test = pd.read_csv(fold_dir / "calibrated_test.csv.gz")
            oof = pd.read_csv(fold_dir / "calibrated_oof.csv.gz")

            for status, frame in [("raw", raw_test), ("calibrated", test)]:
                cp = frame.copy()
                cp["outer_fold"] = fold
                cp["calibration_status"] = status
                calibration_prediction_parts.append(cp)

            for duration in durations:
                record_decisions(
                    fold,
                    f"fixed_{duration:.1f}",
                    np.nan,
                    fixed_duration_decisions(test, duration),
                    loss_matrix,
                )

            fixed_grid = pd.read_csv(fold_dir / "policy_grids" / "fixed_risk_matched.csv")
            fixed_grid["outer_fold"] = fold
            fixed_selection_rows.append(fixed_grid)
            for epsilon in map(float, self.cfg["policy"]["epsilons"]):
                fixed_payload = json.loads(
                    (fold_dir / "policies" / "fixed_risk_matched" / f"epsilon_{epsilon:.3f}.json").read_text(encoding="utf-8")
                )
                fixed_cert = json.loads(
                    (fold_dir / "policies" / "fixed_risk_matched" / f"epsilon_{epsilon:.3f}_certification.json").read_text(encoding="utf-8")
                )
                fixed_extra = {
                    "certification_ucb_gap": float(fixed_payload["certification_ucb_gap"]),
                    "certification_passed": bool(fixed_payload["certification_passed"]),
                    "fallback_full_duration": bool(fixed_payload["fallback_full_duration"]),
                    "selected_full_duration": bool(fixed_payload["selected_full_duration"]),
                    "selected_duration_sec": float(fixed_payload["duration_sec"]),
                    "candidate_duration_sec": float(fixed_payload["candidate_duration_sec"]),
                }
                record_decisions(
                    fold,
                    "fixed_risk_matched",
                    epsilon,
                    fixed_duration_decisions(test, float(fixed_payload["duration_sec"])),
                    loss_matrix,
                    fixed_extra,
                )
                policy_audit_rows.append({
                    "outer_fold": fold,
                    "method": "fixed_risk_matched",
                    "epsilon": epsilon,
                    "status": str(fixed_payload["status"]),
                    "certification_passed": bool(fixed_payload["certification_passed"]),
                    "fallback_full_duration": bool(fixed_payload["fallback_full_duration"]),
                    "selected_full_duration": bool(fixed_payload["selected_full_duration"]),
                    "fallback_reason": "certification_failed" if fixed_payload["fallback_full_duration"] else "none",
                    "selection_subjects": int(fixed_payload["selection_subjects"]),
                    "certification_subjects": int(fixed_payload["certification_subjects"]),
                    "certification_mean_gap": float(fixed_payload["certification_mean_gap"]),
                    "certification_ucb_gap": float(fixed_payload["certification_ucb_gap"]),
                    "candidate_mean_time": float(fixed_payload["candidate_duration_sec"]),
                    "final_policy_mean_time": float(fixed_payload["duration_sec"]),
                    "confidence": np.nan,
                    "js_max": np.nan,
                    "harmonic_ratio_min": np.nan,
                    "harmonic_energy_min": np.nan,
                    "sbr_min_db": np.nan,
                    "f0_delta_max_hz": np.nan,
                    "persistence": np.nan,
                    "bound_method": str(fixed_payload["bound_method"]),
                    "risk_protocol": str(fixed_payload["risk_protocol"]),
                    "selection_status": str(fixed_cert.get("selection_status", "selected")),
                })

            for status, frame in [("raw", raw_test), ("calibrated", test)]:
                for threshold in [0.70, 0.80, 0.90, 0.95]:
                    record_decisions(
                        fold,
                        f"{status}_confidence_{threshold:.2f}",
                        np.nan,
                        apply_policy(frame, durations, confidence_policy(threshold), selective_mode=False),
                        loss_matrix,
                    )

            rel_oof, rel_test = fit_two_stage_reliability(
                oof, test, durations, int(self.cfg["seed"]) + 20000 + fold
            )
            for epsilon in map(float, self.cfg["policy"]["epsilons"]):
                threshold, _ = select_score_threshold(
                    rel_oof,
                    durations,
                    epsilon,
                    zero_one,
                    self.cfg["policy"],
                    int(self.cfg["seed"]) + 21000 + fold,
                )
                record_decisions(
                    fold,
                    "reliability_style",
                    epsilon,
                    apply_score_stopping(rel_test, durations, threshold),
                    loss_matrix,
                    {"reliability_threshold": threshold},
                )

                for method in self._policy_variants():
                    policy_path = fold_dir / "policies" / method / f"epsilon_{epsilon:.3f}.json"
                    cert_path = fold_dir / "policies" / method / f"epsilon_{epsilon:.3f}_certification.json"
                    policy = self._load_policy(policy_path)
                    cert = json.loads(cert_path.read_text(encoding="utf-8"))
                    audit = qtrc_audit_row(fold, method, epsilon, policy, cert)
                    policy_audit_rows.append(audit)
                    record_decisions(
                        fold,
                        method,
                        epsilon,
                        apply_policy(
                            test,
                            durations,
                            policy,
                            selective_mode=bool(self.cfg["policy"]["selective_mode"]),
                        ),
                        loss_matrix,
                        {
                            "calibration_ucb_gap": audit["certification_ucb_gap"],
                            "certification_passed": audit["certification_passed"],
                            "fallback_full_duration": audit["fallback_full_duration"],
                            "selected_full_duration": audit["selected_full_duration"],
                        },
                    )

        metrics_df = pd.DataFrame(fold_metric_rows)
        decisions_df = pd.concat(decision_parts, ignore_index=True)
        policy_audit = pd.DataFrame(policy_audit_rows)
        out = self.run_dir / "cp6_results"
        atomic_dataframe_csv(metrics_df, out / "fold_metrics.csv")
        atomic_dataframe_csv(decisions_df, out / "event_decisions.csv.gz", compression="gzip")
        atomic_dataframe_csv(pd.concat(fixed_selection_rows, ignore_index=True), out / "fixed_duration_selection.csv")
        atomic_dataframe_csv(policy_audit, out / "policy_audit.csv")

        certification_rows: list[dict[str, Any]] = []
        for (method, epsilon), group in policy_audit.groupby(["method", "epsilon"], sort=True):
            certification_rows.append({
                "method": method,
                "epsilon": float(epsilon),
                "outer_folds": int(len(group)),
                "certification_pass_rate": float(group["certification_passed"].mean()),
                "fallback_fold_rate": float(group["fallback_full_duration"].mean()),
                "selected_full_duration_rate": float(group["selected_full_duration"].mean()),
                "mean_certification_ucb_gap": float(group["certification_ucb_gap"].mean()),
                "mean_candidate_time_sec": float(group["candidate_mean_time"].mean()),
                "mean_final_policy_time_sec": float(group["final_policy_mean_time"].mean()),
            })
        certification_df = pd.DataFrame(certification_rows)
        atomic_dataframe_csv(certification_df, out / "certification_summary.csv")

        audit_lookup = {
            (str(row.method), float(row.epsilon)): row._asdict()
            for row in certification_df.itertuples(index=False)
        }
        summary_rows = []
        species_parts = []
        subject_parts = []
        confusion_rows: list[dict[str, Any]] = []
        repeats = int(self.cfg.get("reporting", {}).get("bootstrap_repeats", 2000))
        for group_index, ((method, epsilon), group) in enumerate(
            decisions_df.groupby(["method", "epsilon"], dropna=False, sort=True)
        ):
            eps_value = float(epsilon) if pd.notna(epsilon) else np.nan
            base = pooled_decision_metrics(
                group, max_duration, eps_value if np.isfinite(eps_value) else None
            )
            base = add_subject_bootstrap_ci(
                base, group, repeats, int(self.cfg["seed"]) + group_index * 1009
            )
            base.update({"method": method, "epsilon": eps_value})
            if np.isfinite(eps_value) and (str(method), eps_value) in audit_lookup:
                cert_summary = audit_lookup[(str(method), eps_value)]
                for key in [
                    "certification_pass_rate",
                    "fallback_fold_rate",
                    "selected_full_duration_rate",
                    "mean_certification_ucb_gap",
                    "mean_candidate_time_sec",
                    "mean_final_policy_time_sec",
                ]:
                    base[key] = cert_summary[key]
            summary_rows.append(base)
            sm = species_metrics(group)
            sm["method"] = method
            sm["epsilon"] = eps_value
            species_parts.append(sm)
            subject_parts.append(self._subject_rows(group, method, eps_value))
            active = group[~group["abstain"]]
            cm = pd.crosstab(
                pd.Categorical(active["species"], categories=SPECIES),
                pd.Categorical(active["pred_species"], categories=SPECIES),
                dropna=False,
            )
            for true_species in SPECIES:
                for pred_species in SPECIES:
                    confusion_rows.append({
                        "method": method,
                        "epsilon": eps_value,
                        "true_species": true_species,
                        "pred_species": pred_species,
                        "count": int(cm.loc[true_species, pred_species]),
                    })
        summary_df = pd.DataFrame(summary_rows)
        atomic_dataframe_csv(summary_df, out / "summary_metrics.csv")
        atomic_dataframe_csv(pd.concat(species_parts, ignore_index=True), out / "species_metrics.csv")
        atomic_dataframe_csv(pd.concat(subject_parts, ignore_index=True), out / "subject_metrics.csv")
        atomic_dataframe_csv(pd.DataFrame(confusion_rows), out / "confusion_matrices.csv")

        calibration_df = pd.concat(calibration_prediction_parts, ignore_index=True)
        calibration_rows = []
        class_rows = []
        for (status, duration), group in calibration_df.groupby(
            ["calibration_status", "duration_sec"], sort=True
        ):
            probs = group[[f"p_{s}" for s in SPECIES]].to_numpy(dtype=float)
            y_idx = label_indices(group["species"])
            class_values = classwise_ece(probs, y_idx)
            calibration_rows.append({
                "status": status,
                "duration_sec": float(duration),
                "events": int(len(group)),
                "subjects": int(group.subject_id.nunique()),
                "ece": expected_calibration_error(probs, y_idx),
                "nll": float(log_loss(y_idx, probs, labels=list(range(len(SPECIES))))),
                "brier": brier_multiclass(probs, y_idx),
                "mean_classwise_ece": float(np.mean(list(class_values.values()))),
            })
            for species, value in class_values.items():
                class_rows.append({
                    "status": status,
                    "duration_sec": float(duration),
                    "species": species,
                    "classwise_ece": value,
                })
        calibration_metrics = pd.DataFrame(calibration_rows)
        atomic_dataframe_csv(calibration_metrics, out / "calibration_metrics.csv")
        atomic_dataframe_csv(pd.DataFrame(class_rows), out / "classwise_calibration.csv")

        raw = calibration_df[calibration_df["calibration_status"] == "raw"].copy()
        calibrated = calibration_df[calibration_df["calibration_status"] == "calibrated"].copy()
        join_cols = ["outer_fold", "event_id", "duration_sec"]
        pcols = [f"p_{s}" for s in SPECIES]
        merged = raw[join_cols + pcols].merge(
            calibrated[join_cols + pcols],
            on=join_cols,
            suffixes=("_raw", "_calibrated"),
            validate="one_to_one",
        )
        raw_probs = merged[[f"{c}_raw" for c in pcols]].to_numpy(dtype=float)
        calibrated_probs = merged[[f"{c}_calibrated" for c in pcols]].to_numpy(dtype=float)
        merged["argmax_changed"] = (
            np.argmax(raw_probs, axis=1) != np.argmax(calibrated_probs, axis=1)
        )
        argmax_audit = (
            merged.groupby("duration_sec", sort=True)["argmax_changed"]
            .agg(events="size", argmax_changes="sum", argmax_change_rate="mean")
            .reset_index()
        )
        atomic_dataframe_csv(argmax_audit, out / "calibration_argmax_audit.csv")
        total_changes = int(argmax_audit["argmax_changes"].sum())
        atomic_write_json(out / "calibration_safeguard.json", {
            "method": "species_scalar_temperature",
            "argmax_preserving_required": True,
            "total_argmax_changes": total_changes,
            "pass": bool(total_changes == 0),
        })
        if total_changes != 0:
            raise RuntimeError("CP6 calibration safeguard failed: argmax changed")

    def cp7(self):
        """Controlled multi-seed sensitivity using common random numbers.

        For a fixed outer fold, recording, and robustness seed, the same AWGN
        vector is reused across all SNR values; the same Gaussian phase process
        is reused across phase-noise standard deviations; and the same sinusoid
        phase is reused across gain-ripple depths. Models and stopping policies
        remain the nominal CP4/CP5 artifacts and are not re-fit or re-certified.
        """
        out = self.run_dir / "cp7_robustness"
        enabled = bool(self.cfg.get("robustness", {}).get("enabled", False))
        if not enabled:
            atomic_write_json(out / "status.json", {"enabled": False, "status": "skipped"})
            return

        robust_cfg = self.cfg.get("robustness", {})
        protocol = str(robust_cfg.get("protocol", "common_random_numbers_multiseed"))
        if protocol != "common_random_numbers_multiseed":
            raise ValueError(f"Unsupported robustness protocol: {protocol}")
        robust_seeds = [int(v) for v in robust_cfg.get("seeds", [0, 1, 2, 3, 4])]
        if len(robust_seeds) < 3:
            raise ValueError("CP7 multi-seed sensitivity requires at least three robustness seeds")

        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        all_events = pd.read_csv(self.run_dir / "cp1_events" / "events.csv")
        all_features = pd.read_csv(self.run_dir / "cp2_features" / "features.csv.gz")
        conditions: list[dict[str, Any]] = [{"name": "nominal", "family": "nominal"}]
        for snr in robust_cfg.get("snr_db", []):
            conditions.append({"name": f"snr_{snr}dB", "family": "awgn", "snr_db": snr})
        for std in robust_cfg.get("phase_noise_std_rad", []):
            conditions.append({"name": f"phase_noise_{std}", "family": "phase_noise", "phase_noise_std_rad": std})
        for depth in robust_cfg.get("gain_ripple_depth", []):
            conditions.append({
                "name": f"gain_ripple_{depth}", "family": "gain_ripple",
                "gain_ripple_depth": depth,
                "gain_ripple_rate_hz": float(robust_cfg.get("gain_ripple_rate_hz", 1.0)),
            })
        atomic_write_json(out / "conditions.json", {
            "protocol": protocol,
            "robustness_seeds": robust_seeds,
            "common_random_numbers": {
                "awgn": "same complex unit-noise vector across SNR values for each recording and seed",
                "phase_noise": "same standard-normal phase process across phase-noise magnitudes",
                "gain_ripple": "same sinusoid phase across ripple depths",
            },
            "conditions": conditions,
        })

        decision_parts: list[pd.DataFrame] = []
        diagnostic_rows: list[dict[str, Any]] = []
        drift_parts: list[pd.DataFrame] = []
        pcols = [f"p_{species}" for species in SPECIES]

        for fold_dir in sorted((self.run_dir / "cp4_predictions").glob("outer_*")):
            fold = int(fold_dir.name.split("_")[1])
            loss_matrix = np.asarray(json.loads((fold_dir / "loss_matrix.json").read_text(encoding="utf-8"))["matrix"], dtype=float)
            split = json.loads((fold_dir / "split.json").read_text(encoding="utf-8"))
            test_subjects = set(split["test_subjects"])
            train_subjects = set(split["train_subjects"])
            fold_events = all_events[all_events.subject_id.isin(test_subjects)].copy()
            fold_nominal_features = all_features[all_features.subject_id.isin(test_subjects)].copy()
            fold_train_features = all_features[all_features.subject_id.isin(train_subjects)].copy()
            temperatures = json.loads((fold_dir / "temperatures.json").read_text(encoding="utf-8"))

            for robust_seed in robust_seeds:
                nominal_pred_ref: pd.DataFrame | None = None
                for condition in conditions:
                    name = str(condition["name"]).replace("/", "_")
                    seed_dir = out / f"outer_{fold:03d}" / f"seed_{robust_seed:03d}" / name
                    pred_path = seed_dir / "predictions.csv.gz"
                    drift_path = seed_dir / "feature_drift.csv.gz"
                    diag_path = seed_dir / "diagnostics.csv"
                    task = f"cp7v5::outer{fold}::seed{robust_seed}::{name}"
                    if not self.state.task_done(task, pred_path) or not drift_path.exists():
                        with task_lock(pred_path.with_suffix(".lock"), self.stale_minutes):
                            feature_parts: list[pd.DataFrame] = []
                            local_diag: list[dict[str, Any]] = []
                            for recording_id, group in fold_events.groupby("recording_id", sort=True):
                                amp, phase = read_signal_csv(group.iloc[0]["path"])
                                z = complex_from_amplitude_phase(amp, phase)
                                z_hp = preprocess_complex(z, self.fs, self.cfg["signal"])
                                key = f"{self.cfg['seed']}::{fold}::{robust_seed}::{recording_id}".encode("utf-8")
                                common_seed = int(hashlib.sha256(key).hexdigest()[:8], 16)
                                draws = make_common_perturbation_draws(len(z_hp), common_seed)
                                degraded, diagnostics = perturb_preprocessed_signal_common(
                                    z_hp, condition, draws, self.fs
                                )
                                local_diag.append({
                                    "outer_fold": fold,
                                    "robust_seed": robust_seed,
                                    "condition": name,
                                    "condition_family": condition.get("family", "unknown"),
                                    "recording_id": recording_id,
                                    **diagnostics,
                                })
                                rows: list[dict[str, Any]] = []
                                for _, event in group.iterrows():
                                    rows.extend(extract_event_feature_rows(event, degraded, self.fs, self.cfg["features"]))
                                feature_parts.append(pd.DataFrame(rows))
                            degraded_features = pd.concat(feature_parts, ignore_index=True)

                            pred_parts: list[pd.DataFrame] = []
                            drift_rows: list[pd.DataFrame] = []
                            for duration in durations:
                                sub = degraded_features[np.isclose(degraded_features.duration_sec, duration)].copy()
                                nom = fold_nominal_features[np.isclose(fold_nominal_features.duration_sec, duration)].copy()
                                model = HierarchicalClassifier.load(fold_dir / "models" / f"t{duration:.1f}.joblib")
                                x = sub[model.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                                components = model.predict_components(x)
                                quality_cols = ["harmonic_ratio", "harmonic_energy_ratio", "sbr_db", "f0_hz", "f0_delta_hz", "stability_available"]
                                pred = sub[["event_id", "recording_id", "subject_id", "species", "duration_sec", *quality_cols]].copy()
                                pred = self._attach_probabilities(pred, components)
                                pred, changes = self._apply_species_temperature_to_frame(
                                    pred, temperatures[f"{duration:.1f}"]
                                )
                                if changes != 0:
                                    raise RuntimeError("CP7 calibration changed the predicted class")
                                pred_parts.append(pred)

                                merged = sub[["event_id", "duration_sec", *model.feature_names]].merge(
                                    nom[["event_id", "duration_sec", *model.feature_names]],
                                    on=["event_id", "duration_sec"], suffixes=("_degraded", "_nominal"), validate="one_to_one"
                                )
                                train = fold_train_features[np.isclose(fold_train_features.duration_sec, duration)]
                                train_values = train[model.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                                scale = train_values.std(axis=0, ddof=0).to_numpy(dtype=float)
                                scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
                                deg = merged[[f"{c}_degraded" for c in model.feature_names]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                                base = merged[[f"{c}_nominal" for c in model.feature_names]].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                                dz = (deg - base) / scale[None, :]
                                drift = merged[["event_id", "duration_sec"]].copy()
                                drift["outer_fold"] = fold
                                drift["robust_seed"] = robust_seed
                                drift["condition"] = name
                                drift["condition_family"] = condition.get("family", "unknown")
                                drift["feature_rms_z"] = np.sqrt(np.mean(dz ** 2, axis=1))
                                drift["feature_mean_abs_z"] = np.mean(np.abs(dz), axis=1)
                                for q in ["f0_hz", "harmonic_ratio", "harmonic_energy_ratio", "sbr_db"]:
                                    if q in model.feature_names:
                                        drift[f"{q}_abs_delta"] = np.abs(
                                            merged[f"{q}_degraded"].to_numpy(dtype=float)
                                            - merged[f"{q}_nominal"].to_numpy(dtype=float)
                                        )
                                drift_rows.append(drift)

                            pred_df = pd.concat(pred_parts, ignore_index=True)
                            drift_df = pd.concat(drift_rows, ignore_index=True)
                            atomic_dataframe_csv(pred_df, pred_path, compression="gzip")
                            atomic_dataframe_csv(drift_df, drift_path, compression="gzip")
                            atomic_dataframe_csv(pd.DataFrame(local_diag), diag_path)
                            self.state.update_task(task, "done", artifact=str(pred_path), drift=str(drift_path))
                    else:
                        pred_df = pd.read_csv(pred_path)
                        drift_df = pd.read_csv(drift_path)

                    if diag_path.exists():
                        diagnostic_rows.extend(pd.read_csv(diag_path).to_dict("records"))

                    # Posterior and decision drift are paired to the nominal condition
                    # generated with the same fold/robustness seed.
                    if name == "nominal":
                        nominal_pred_ref = pred_df.copy()
                        drift_df["posterior_js_bits"] = 0.0
                        drift_df["prefix_prediction_changed"] = 0.0
                    else:
                        if nominal_pred_ref is None:
                            nominal_path = out / f"outer_{fold:03d}" / f"seed_{robust_seed:03d}" / "nominal" / "predictions.csv.gz"
                            nominal_pred_ref = pd.read_csv(nominal_path)
                        pair = pred_df[["event_id", "duration_sec", *pcols]].merge(
                            nominal_pred_ref[["event_id", "duration_sec", *pcols]],
                            on=["event_id", "duration_sec"], suffixes=("_degraded", "_nominal"), validate="one_to_one"
                        )
                        p = pair[[f"{c}_degraded" for c in pcols]].to_numpy(dtype=float)
                        q = pair[[f"{c}_nominal" for c in pcols]].to_numpy(dtype=float)
                        p = np.maximum(p, 1e-12); p /= p.sum(axis=1, keepdims=True)
                        q = np.maximum(q, 1e-12); q /= q.sum(axis=1, keepdims=True)
                        m = 0.5 * (p + q)
                        js = 0.5 * np.sum(p * np.log2(p / m), axis=1) + 0.5 * np.sum(q * np.log2(q / m), axis=1)
                        pred_change = np.argmax(p, axis=1) != np.argmax(q, axis=1)
                        drift_df = drift_df.merge(
                            pair[["event_id", "duration_sec"]].assign(
                                posterior_js_bits=js,
                                prefix_prediction_changed=pred_change.astype(float),
                            ), on=["event_id", "duration_sec"], how="left", validate="one_to_one"
                        )
                    drift_parts.append(drift_df)

                    for epsilon in map(float, self.cfg["policy"]["epsilons"]):
                        policy = self._load_policy(fold_dir / "policies" / "qtrc_stop" / f"epsilon_{epsilon:.3f}.json")
                        decisions = attach_decision_losses(
                            apply_policy(pred_df, durations, policy, selective_mode=bool(self.cfg["policy"]["selective_mode"])),
                            loss_matrix,
                        )
                        decisions["outer_fold"] = fold
                        decisions["condition"] = name
                        decisions["condition_family"] = condition.get("family", "unknown")
                        decisions["robust_seed"] = robust_seed
                        decisions["epsilon"] = epsilon
                        decisions["policy_certification_passed"] = bool(policy.certification_passed)
                        decisions["policy_fallback_full_duration"] = bool(policy.fallback_full_duration)
                        decision_parts.append(decisions)

        decisions_df = pd.concat(decision_parts, ignore_index=True)
        atomic_dataframe_csv(decisions_df, out / "robustness_event_decisions.csv.gz", compression="gzip")
        diagnostic_df = pd.DataFrame(diagnostic_rows)
        atomic_dataframe_csv(diagnostic_df, out / "perturbation_diagnostics.csv")
        drift_events = pd.concat(drift_parts, ignore_index=True)
        atomic_dataframe_csv(drift_events, out / "feature_drift_event.csv.gz", compression="gzip")

        # Per-seed metrics preserve the independent perturbation replicate.
        seed_rows: list[dict[str, Any]] = []
        for (robust_seed, condition, epsilon), group in decisions_df.groupby(["robust_seed", "condition", "epsilon"], sort=True):
            row = pooled_decision_metrics(group, max(durations), float(epsilon))
            row["full_duration_accuracy"] = float(np.mean(group["full_pred_species"].astype(str) == group["species"].astype(str)))
            fold_status = group[["outer_fold", "policy_certification_passed", "policy_fallback_full_duration"]].drop_duplicates("outer_fold")
            row.update({
                "robust_seed": int(robust_seed),
                "condition": condition,
                "epsilon": float(epsilon),
                "nominal_policy_certification_pass_rate": float(fold_status["policy_certification_passed"].mean()),
                "nominal_policy_fallback_rate": float(fold_status["policy_fallback_full_duration"].mean()),
                "outer_folds": int(fold_status["outer_fold"].nunique()),
            })
            seed_rows.append(row)
        seed_summary = pd.DataFrame(seed_rows)
        nominal_seed = seed_summary[seed_summary.condition.eq("nominal")][
            ["robust_seed", "epsilon", "accuracy", "balanced_accuracy", "macro_f1", "mean_stop_sec", "subject_mean_gap", "full_duration_accuracy"]
        ].rename(columns={
            "accuracy": "nominal_accuracy",
            "balanced_accuracy": "nominal_balanced_accuracy",
            "macro_f1": "nominal_macro_f1",
            "mean_stop_sec": "nominal_mean_stop_sec",
            "subject_mean_gap": "nominal_subject_mean_gap",
            "full_duration_accuracy": "nominal_full_duration_accuracy",
        })
        seed_summary = seed_summary.merge(nominal_seed, on=["robust_seed", "epsilon"], how="left", validate="many_to_one")
        seed_summary["accuracy_delta"] = seed_summary["accuracy"] - seed_summary["nominal_accuracy"]
        seed_summary["balanced_accuracy_delta"] = seed_summary["balanced_accuracy"] - seed_summary["nominal_balanced_accuracy"]
        seed_summary["macro_f1_delta"] = seed_summary["macro_f1"] - seed_summary["nominal_macro_f1"]
        seed_summary["mean_stop_delta_sec"] = seed_summary["mean_stop_sec"] - seed_summary["nominal_mean_stop_sec"]
        seed_summary["subject_mean_gap_delta"] = seed_summary["subject_mean_gap"] - seed_summary["nominal_subject_mean_gap"]
        seed_summary["full_duration_accuracy_delta"] = seed_summary["full_duration_accuracy"] - seed_summary["nominal_full_duration_accuracy"]
        atomic_dataframe_csv(seed_summary, out / "robustness_seed_summary.csv")

        metric_cols = [
            "accuracy", "balanced_accuracy", "macro_f1", "worst_recall", "mean_stop_sec",
            "subject_mean_gap", "full_duration_accuracy", "accuracy_delta", "balanced_accuracy_delta", "macro_f1_delta",
            "mean_stop_delta_sec", "subject_mean_gap_delta", "full_duration_accuracy_delta",
        ]
        agg_rows: list[dict[str, Any]] = []
        for (condition, epsilon), group in seed_summary.groupby(["condition", "epsilon"], sort=True):
            row: dict[str, Any] = {
                "condition": condition,
                "epsilon": float(epsilon),
                "seeds": int(group.robust_seed.nunique()),
                "events_per_seed": int(group.events.iloc[0]),
                "subjects": int(group.subjects.iloc[0]),
                "nominal_policy_certification_pass_rate": float(group.nominal_policy_certification_pass_rate.iloc[0]),
                "nominal_policy_fallback_rate": float(group.nominal_policy_fallback_rate.iloc[0]),
            }
            for col in metric_cols:
                vals = group[col].to_numpy(dtype=float)
                row[f"{col}_mean"] = float(np.mean(vals))
                row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row[f"{col}_min"] = float(np.min(vals))
                row[f"{col}_max"] = float(np.max(vals))
            agg_rows.append(row)
        summary = pd.DataFrame(agg_rows)
        atomic_dataframe_csv(summary, out / "robustness_summary.csv")

        species_seed_parts: list[pd.DataFrame] = []
        pred_dist_rows: list[dict[str, Any]] = []
        for (robust_seed, condition, epsilon), group in decisions_df.groupby(["robust_seed", "condition", "epsilon"], sort=True):
            sp = species_metrics(group)
            sp["robust_seed"] = int(robust_seed)
            sp["condition"] = condition
            sp["epsilon"] = float(epsilon)
            species_seed_parts.append(sp)
            for label in SPECIES:
                pred_dist_rows.append({
                    "robust_seed": int(robust_seed), "condition": condition, "epsilon": float(epsilon),
                    "pred_species": label,
                    "qtrc_prediction_fraction": float(np.mean(group["pred_species"].astype(str) == label)),
                    "full_duration_prediction_fraction": float(np.mean(group["full_pred_species"].astype(str) == label)),
                })
        species_seed = pd.concat(species_seed_parts, ignore_index=True)
        atomic_dataframe_csv(species_seed, out / "species_sensitivity_seed.csv")
        species_agg = (
            species_seed.groupby(["condition", "epsilon", "species"], sort=True)
            .agg(
                seeds=("robust_seed", "nunique"),
                recall_total_mean=("recall_total", "mean"),
                recall_total_std=("recall_total", "std"),
                fnr_total_mean=("fnr_total", "mean"),
                mean_stop_sec_mean=("mean_stop_sec", "mean"),
                mean_stop_sec_std=("mean_stop_sec", "std"),
            ).reset_index()
        )
        atomic_dataframe_csv(species_agg, out / "species_sensitivity_summary.csv")
        atomic_dataframe_csv(pd.DataFrame(pred_dist_rows), out / "prediction_distribution_seed.csv")

        drift_summary = (
            drift_events.groupby(["condition", "robust_seed", "duration_sec"], sort=True)
            .agg(
                events=("event_id", "size"),
                feature_rms_z_mean=("feature_rms_z", "mean"),
                feature_rms_z_median=("feature_rms_z", "median"),
                feature_mean_abs_z_mean=("feature_mean_abs_z", "mean"),
                posterior_js_bits_mean=("posterior_js_bits", "mean"),
                prefix_prediction_change_rate=("prefix_prediction_changed", "mean"),
                f0_hz_abs_delta_mean=("f0_hz_abs_delta", "mean"),
                harmonic_ratio_abs_delta_mean=("harmonic_ratio_abs_delta", "mean"),
                harmonic_energy_ratio_abs_delta_mean=("harmonic_energy_ratio_abs_delta", "mean"),
                sbr_db_abs_delta_mean=("sbr_db_abs_delta", "mean"),
            ).reset_index()
        )
        atomic_dataframe_csv(drift_summary, out / "feature_drift_seed_summary.csv")
        drift_aggregate = (
            drift_summary.groupby(["condition", "duration_sec"], sort=True)
            .agg(
                seeds=("robust_seed", "nunique"),
                feature_rms_z_mean=("feature_rms_z_mean", "mean"),
                feature_rms_z_std=("feature_rms_z_mean", "std"),
                posterior_js_bits_mean=("posterior_js_bits_mean", "mean"),
                posterior_js_bits_std=("posterior_js_bits_mean", "std"),
                prefix_prediction_change_rate_mean=("prefix_prediction_change_rate", "mean"),
                prefix_prediction_change_rate_std=("prefix_prediction_change_rate", "std"),
                f0_hz_abs_delta_mean=("f0_hz_abs_delta_mean", "mean"),
                harmonic_ratio_abs_delta_mean=("harmonic_ratio_abs_delta_mean", "mean"),
                harmonic_energy_ratio_abs_delta_mean=("harmonic_energy_ratio_abs_delta_mean", "mean"),
                sbr_db_abs_delta_mean=("sbr_db_abs_delta_mean", "mean"),
            ).reset_index()
        )
        atomic_dataframe_csv(drift_aggregate, out / "feature_drift_summary.csv")

        paper_epsilon = float(robust_cfg.get("paper_epsilon", 0.05))
        phase_std = float(robust_cfg.get("paper_phase_noise_std_rad", 0.10))
        gain_depth = float(robust_cfg.get("paper_gain_ripple_depth", 0.20))
        requested = [(f"AWGN {float(v):g} dB", f"snr_{v}dB") for v in robust_cfg.get("paper_snr_db", [0, 10, 20])]
        requested.extend([
            (f"Phase noise ({phase_std:g} rad)", f"phase_noise_{phase_std}"),
            (f"Gain ripple ({gain_depth:g})", f"gain_ripple_{gain_depth}"),
        ])
        paper_rows: list[dict[str, Any]] = []
        for label, condition_name in requested:
            match = summary[summary.condition.eq(condition_name) & np.isclose(summary.epsilon.astype(float), paper_epsilon)]
            if match.empty:
                raise RuntimeError(f"Missing CP7 paper condition {condition_name} at epsilon={paper_epsilon}")
            row = match.iloc[0]
            paper_rows.append({
                "condition": label,
                "condition_key": condition_name,
                "epsilon": paper_epsilon,
                "seeds": int(row["seeds"]),
                "accuracy_mean": float(row["accuracy_mean"]),
                "accuracy_std": float(row["accuracy_std"]),
                "accuracy_delta_mean": float(row["accuracy_delta_mean"]),
                "accuracy_delta_std": float(row["accuracy_delta_std"]),
                "mean_stop_sec_mean": float(row["mean_stop_sec_mean"]),
                "mean_stop_sec_std": float(row["mean_stop_sec_std"]),
                "subject_mean_gap_mean": float(row["subject_mean_gap_mean"]),
                "subject_mean_gap_std": float(row["subject_mean_gap_std"]),
                "full_duration_accuracy_mean": float(row["full_duration_accuracy_mean"]),
                "full_duration_accuracy_std": float(row["full_duration_accuracy_std"]),
                "nominal_policy_certification_pass_rate": float(row["nominal_policy_certification_pass_rate"]),
                "nominal_policy_fallback_rate": float(row["nominal_policy_fallback_rate"]),
            })
        paper_table = pd.DataFrame(paper_rows)
        paper_csv = out / f"paper_sensitivity_table_e{paper_epsilon:.3f}.csv"
        atomic_dataframe_csv(paper_table, paper_csv)
        # Backward-compatible alias for earlier manuscript tooling.
        atomic_dataframe_csv(paper_table, out / f"paper_robustness_table_e{paper_epsilon:.3f}.csv")

        tex_lines: list[str] = []
        for r in paper_table.itertuples(index=False):
            tex_lines.append(
                f"{r.condition} & {r.accuracy_mean:.4f}$\\pm${r.accuracy_std:.4f} & "
                f"{r.accuracy_delta_mean:+.4f}$\\pm${r.accuracy_delta_std:.4f} & "
                f"{r.mean_stop_sec_mean:.3f}$\\pm${r.mean_stop_sec_std:.3f} & "
                f"{r.subject_mean_gap_mean:.4f}$\\pm${r.subject_mean_gap_std:.4f} \\\\"
            )
        tex_path = out / f"paper_sensitivity_rows_e{paper_epsilon:.3f}.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
        (out / f"paper_robustness_rows_e{paper_epsilon:.3f}.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

        # Compact AWGN monotonicity diagnostic. Non-monotonicity is reported, not hidden.
        awgn = paper_table[paper_table.condition_key.str.startswith("snr_")].copy()
        awgn["snr_db"] = awgn.condition_key.str.extract(r"snr_([-+]?\d+(?:\.\d+)?)dB")[0].astype(float)
        awgn = awgn.sort_values("snr_db")
        accuracy_non_decreasing = bool(np.all(np.diff(awgn.accuracy_mean.to_numpy(dtype=float)) >= -1e-12)) if len(awgn) > 1 else True
        atomic_write_json(out / "awgn_monotonicity_diagnostic.json", {
            "snr_db": awgn.snr_db.tolist(),
            "accuracy_mean": awgn.accuracy_mean.tolist(),
            "accuracy_non_decreasing_with_snr": accuracy_non_decreasing,
            "interpretation": (
                "If false, treat AWGN as an identified sensitivity limitation and use feature_drift_summary.csv to diagnose the mechanism; do not relabel the result as robustness."
            ),
        })

        atomic_write_json(out / "status.json", {
            "enabled": True,
            "status": "complete",
            "protocol": protocol,
            "robustness_seeds": robust_seeds,
            "conditions": len(conditions),
            "paper_epsilon": paper_epsilon,
            "paper_table": str(paper_csv),
            "feature_drift_summary": str(out / "feature_drift_summary.csv"),
            "certification_note": "Certification/fallback status is inherited from nominal policy selection; CP7 does not re-fit or re-certify under degradation.",
            "snr_definition": "AWGN power is referenced to the causal high-pass-filtered complex signal used by feature extraction.",
            "common_noise_note": "For each recording and robustness seed, one unit complex noise vector is scaled across all SNR values.",
            "note": "Synthetic perturbations are controlled sensitivity analyses and do not establish field-domain robustness.",
        })

    def _run_receiver_runtime_benchmark(self) -> dict[str, Any]:
        """Measure receiver-side compute on saved outer-fold models.

        Acquisition time, disk I/O, and model loading are excluded. The total is
        the sum of causal filtering, the current overlapping-prefix feature
        extractor, classifier inference at all seven checkpoints, and the
        QTRC policy decision. This definition is conservative for deployment
        because the current extractor recomputes overlapping prefixes.
        """
        cfg = self.cfg.get("runtime_benchmark", {})
        out = self.run_dir / "cp8_reports" / "runtime"
        out.mkdir(parents=True, exist_ok=True)
        if not bool(cfg.get("enabled", False)):
            payload = {"enabled": False, "status": "skipped"}
            atomic_write_json(out / "status.json", payload)
            return payload

        durations = list(map(float, self.cfg["features"]["prefix_durations_sec"]))
        max_samples = int(round(max(durations) * self.fs))
        epsilon = float(cfg.get("primary_epsilon", 0.05))
        repeats = max(1, int(cfg.get("repeats", 3)))
        n_events = max(1, int(cfg.get("events", 100)))
        warmup_events = max(0, int(cfg.get("warmup_events", 5)))
        include_filtering = bool(cfg.get("include_filtering", True))

        events = pd.read_csv(self.run_dir / "cp1_events" / "events.csv")
        events["subject_id"] = events["subject_id"].astype(str)
        subject_to_fold: dict[str, Path] = {}
        fold_resources: dict[int, dict[str, Any]] = {}
        for fold_dir in sorted((self.run_dir / "cp4_predictions").glob("outer_*")):
            fold = int(fold_dir.name.split("_")[1])
            split = json.loads((fold_dir / "split.json").read_text(encoding="utf-8"))
            for subject in split["test_subjects"]:
                subject_to_fold[str(subject)] = fold_dir
            models = {
                duration: HierarchicalClassifier.load(fold_dir / "models" / f"t{duration:.1f}.joblib")
                for duration in durations
            }
            fold_resources[fold] = {
                "fold_dir": fold_dir,
                "models": models,
                "temperatures": json.loads((fold_dir / "temperatures.json").read_text(encoding="utf-8")),
                "policy": self._load_policy(
                    fold_dir / "policies" / "qtrc_stop" / f"epsilon_{epsilon:.3f}.json"
                ),
            }

        events = events[events["subject_id"].isin(subject_to_fold)].copy()
        sample = stratified_event_sample(events, n_events, int(self.cfg["seed"]) + 404)
        if sample.empty:
            raise RuntimeError("No events available for runtime benchmark")

        contexts: list[dict[str, Any]] = []
        for recording_id, group in sample.groupby("recording_id", sort=True):
            amp, phase = read_signal_csv(group.iloc[0]["path"])
            z = complex_from_amplitude_phase(amp, phase)
            z_hp = preprocess_complex(z, self.fs, self.cfg["signal"])
            for _, event in group.iterrows():
                start = int(event["start_sample"])
                if start + max_samples > len(z_hp):
                    continue
                fold_dir = subject_to_fold[str(event["subject_id"])]
                fold = int(fold_dir.name.split("_")[1])
                raw_segment = z[start:start + max_samples].copy()
                contexts.append({
                    "event": event.copy(),
                    "z_hp": z_hp,
                    "raw_segment": raw_segment,
                    "fold": fold,
                })
        if not contexts:
            raise RuntimeError("Runtime benchmark sample has no complete 2-s events")

        quality_cols = [
            "harmonic_ratio", "harmonic_energy_ratio", "sbr_db",
            "f0_delta_hz", "stability_available",
        ]

        def run_one(context: dict[str, Any], measure: bool) -> dict[str, float]:
            event = context["event"]
            resources = fold_resources[int(context["fold"])]

            t0 = time.perf_counter_ns()
            if include_filtering:
                _ = preprocess_complex(context["raw_segment"], self.fs, self.cfg["signal"])
            t1 = time.perf_counter_ns()

            feature_rows = extract_event_feature_rows(
                event, context["z_hp"], self.fs, self.cfg["features"]
            )
            feature_frame = pd.DataFrame(feature_rows)
            t2 = time.perf_counter_ns()

            pred_parts: list[pd.DataFrame] = []
            for duration in durations:
                sub = feature_frame[np.isclose(feature_frame.duration_sec, duration)].copy()
                if sub.empty:
                    raise RuntimeError(f"Missing runtime prefix duration {duration}")
                model = resources["models"][duration]
                x = (
                    sub[model.feature_names]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                components = model.predict_components(x)
                pred = sub[[
                    "event_id", "recording_id", "subject_id", "species",
                    "duration_sec", *quality_cols,
                ]].copy()
                pred = self._attach_probabilities(pred, components)
                pred, changes = self._apply_species_temperature_to_frame(
                    pred, resources["temperatures"][f"{duration:.1f}"]
                )
                if changes != 0:
                    raise RuntimeError("Runtime calibration changed predicted class")
                pred_parts.append(pred)
            pred_df = pd.concat(pred_parts, ignore_index=True)
            t3 = time.perf_counter_ns()

            decision = apply_policy(
                pred_df, durations, resources["policy"],
                selective_mode=bool(self.cfg["policy"]["selective_mode"]),
            )
            if len(decision) != 1:
                raise RuntimeError("Runtime policy benchmark must return one decision")
            t4 = time.perf_counter_ns()

            scale = 1e-6
            filtering_ms = (t1 - t0) * scale if include_filtering else 0.0
            feature_ms = (t2 - t1) * scale
            inference_ms = (t3 - t2) * scale
            policy_ms = (t4 - t3) * scale
            return {
                "filtering_ms": filtering_ms,
                "feature_extraction_ms": feature_ms,
                "classifier_inference_ms": inference_ms,
                "policy_decision_ms": policy_ms,
                "total_receiver_compute_ms": filtering_ms + feature_ms + inference_ms + policy_ms,
                "stop_duration_sec": float(decision.iloc[0]["stop_duration_sec"]),
            }

        for context in contexts[: min(warmup_events, len(contexts))]:
            run_one(context, measure=False)

        rows: list[dict[str, Any]] = []
        for repeat in range(repeats):
            for context in contexts:
                measured = run_one(context, measure=True)
                rows.append({
                    "repeat": repeat,
                    "event_id": str(context["event"]["event_id"]),
                    "subject_id": str(context["event"]["subject_id"]),
                    "species": str(context["event"]["species"]),
                    "outer_fold": int(context["fold"]),
                    **measured,
                })
        samples = pd.DataFrame(rows)
        atomic_dataframe_csv(samples, out / "runtime_event_samples.csv.gz", compression="gzip")
        components = [
            "filtering_ms",
            "feature_extraction_ms",
            "classifier_inference_ms",
            "policy_decision_ms",
            "total_receiver_compute_ms",
        ]
        summary = summarize_runtime_samples(samples, components)
        atomic_dataframe_csv(summary, out / "runtime_summary.csv")

        fold_model_sizes = []
        for fold_dir in sorted((self.run_dir / "cp4_predictions").glob("outer_*")):
            fold_bytes = sum(path.stat().st_size for path in (fold_dir / "models").glob("*.joblib"))
            if fold_bytes > 0:
                fold_model_sizes.append(float(fold_bytes / (1024 ** 2)))
        model_total_mib = float(np.sum(fold_model_sizes)) if fold_model_sizes else 0.0
        hardware = hardware_manifest()
        hardware.update({
            "model_artifact_all_loso_folds_mib": model_total_mib,
            "deployment_model_mib_mean_per_fold": float(np.mean(fold_model_sizes)) if fold_model_sizes else None,
            "deployment_model_mib_min_per_fold": float(np.min(fold_model_sizes)) if fold_model_sizes else None,
            "deployment_model_mib_max_per_fold": float(np.max(fold_model_sizes)) if fold_model_sizes else None,
            "deployment_model_note": "The all-fold value is an evaluation archive size. A deployed receiver uses one trained model set, represented by the per-fold statistics.",
            "benchmark_events": int(len(contexts)),
            "benchmark_repeats": repeats,
            "primary_epsilon": epsilon,
        })
        atomic_write_json(out / "hardware_manifest.json", hardware)

        by_component = summary.set_index("component")
        payload = {
            "enabled": True,
            "status": "complete",
            "definition": (
                "Per-event receiver compute excludes signal acquisition, disk I/O, and model loading; "
                "it includes causal filtering, overlapping-prefix feature extraction, seven saved-model "
                "inferences, calibration, and one QTRC policy decision."
            ),
            "events": int(len(contexts)),
            "repeats": repeats,
            "primary_epsilon": epsilon,
            "median_ms": {
                component: float(by_component.loc[component, "median_ms"])
                for component in by_component.index
            },
            "p90_ms": {
                component: float(by_component.loc[component, "p90_ms"])
                for component in by_component.index
            },
            "hardware_manifest": "hardware_manifest.json",
            "runtime_summary": "runtime_summary.csv",
        }
        atomic_write_json(out / "runtime_report.json", payload)

        runtime_sentence = (
            "Receiver-side compute was measured on "
            f"{hardware['cpu_model']} using {hardware['python_implementation']} "
            f"{hardware['python']}. Median per-event times were "
            f"{by_component.loc['feature_extraction', 'median_ms']:.2f} ms for feature extraction, "
            f"{by_component.loc['classifier_inference', 'median_ms']:.2f} ms for classifier inference, "
            f"{by_component.loc['policy_decision', 'median_ms']:.2f} ms for the stopping decision, and "
            f"{by_component.loc['total_receiver_compute', 'median_ms']:.2f} ms in total; the total "
            f"90th percentile was {by_component.loc['total_receiver_compute', 'p90_ms']:.2f} ms."
        )
        (out / "runtime_manuscript_sentence.txt").write_text(runtime_sentence + "\n", encoding="utf-8")
        return payload

    def cp8(self):
        summary = pd.read_csv(self.run_dir / "cp6_results" / "summary_metrics.csv")
        certification = pd.read_csv(self.run_dir / "cp6_results" / "certification_summary.csv")
        calibration = pd.read_csv(self.run_dir / "cp6_results" / "calibration_metrics.csv")
        confusion = pd.read_csv(self.run_dir / "cp6_results" / "confusion_matrices.csv")
        safeguard = json.loads(
            (self.run_dir / "cp6_results" / "calibration_safeguard.json").read_text(encoding="utf-8")
        )
        report_dir = self.run_dir / "cp8_reports"
        fig_dir = report_dir / "figures"
        table_dir = report_dir / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)

        plot_fixed_duration(summary, fig_dir / "fixed_duration_curve.png")
        plot_risk_time(summary, fig_dir / "risk_gap_vs_time.png")
        plot_accuracy_time(summary, fig_dir / "balanced_accuracy_vs_time.png")
        plot_certification(certification, fig_dir / "certification_rate.png")
        plot_calibration(calibration, fig_dir / "calibration_ece.png")
        cm_counts = plot_confusion_matrix(
            confusion,
            fig_dir / "confusion_matrix_qtrc_e005.png",
            method="qtrc_stop",
            epsilon=0.05,
            normalize=True,
        )
        atomic_dataframe_csv(cm_counts, table_dir / "confusion_matrix_qtrc_e005_counts.csv")
        cm_norm = cm_counts.copy()
        value_cols = [c for c in cm_norm.columns if c != "true_species"]
        denom = cm_norm[value_cols].sum(axis=1).replace(0, 1)
        cm_norm.loc[:, value_cols] = cm_norm[value_cols].div(denom, axis=0)
        atomic_dataframe_csv(cm_norm, table_dir / "confusion_matrix_qtrc_e005_row_normalized.csv")

        runtime_report = self._run_receiver_runtime_benchmark()

        robust_cfg = self.cfg.get("robustness", {})
        paper_epsilon = float(robust_cfg.get("paper_epsilon", 0.05))
        cp7_table = self.run_dir / "cp7_robustness" / f"paper_sensitivity_table_e{paper_epsilon:.3f}.csv"
        cp7_rows = self.run_dir / "cp7_robustness" / f"paper_sensitivity_rows_e{paper_epsilon:.3f}.tex"
        cp7_summary_path = self.run_dir / "cp7_robustness" / "robustness_summary.csv"
        cp7_drift_path = self.run_dir / "cp7_robustness" / "feature_drift_summary.csv"
        cp7_status_path = self.run_dir / "cp7_robustness" / "status.json"
        cp7_status = (
            json.loads(cp7_status_path.read_text(encoding="utf-8"))
            if cp7_status_path.exists()
            else {"status": "missing"}
        )
        if cp7_table.exists():
            atomic_dataframe_csv(pd.read_csv(cp7_table), table_dir / cp7_table.name)
        if cp7_rows.exists():
            (table_dir / cp7_rows.name).write_text(cp7_rows.read_text(encoding="utf-8"), encoding="utf-8")
        if cp7_summary_path.exists():
            cp7_summary = pd.read_csv(cp7_summary_path)
            plot_awgn_sensitivity(cp7_summary, fig_dir / "awgn_sensitivity_curve.png", epsilon=paper_epsilon)
            atomic_dataframe_csv(cp7_summary, table_dir / "robustness_summary.csv")
        if cp7_drift_path.exists():
            cp7_drift = pd.read_csv(cp7_drift_path)
            plot_feature_drift_vs_snr(cp7_drift, fig_dir / "feature_drift_vs_snr.png", duration_sec=max(map(float, self.cfg["features"]["prefix_durations_sec"])))
            atomic_dataframe_csv(cp7_drift, table_dir / "feature_drift_summary.csv")

        primary = summary[
            summary["method"].eq("qtrc_stop")
            & summary["epsilon"].notna()
            & np.isclose(summary["epsilon"].astype(float), 0.05)
        ]
        if primary.empty:
            raise RuntimeError("Missing QTRC primary operating point epsilon=0.05")
        primary_row = primary.iloc[0]

        manuscript_inputs = {
            "pipeline_version": "5.0.0",
            "primary_epsilon": 0.05,
            "qtrc_primary": {
                key: float(primary_row[key])
                for key in [
                    "accuracy", "macro_f1", "balanced_accuracy", "worst_recall",
                    "mean_stop_sec", "p90_stop_sec", "subject_mean_gap",
                ]
                if key in primary_row and pd.notna(primary_row[key])
            },
            "cp7_status": cp7_status,
            "cp7_paper_table": str(cp7_table) if cp7_table.exists() else None,
            "runtime": runtime_report,
            "figures": {
                "fixed_duration_curve": "figures/fixed_duration_curve.png",
                "risk_gap_vs_time": "figures/risk_gap_vs_time.png",
                "confusion_matrix_qtrc_e005": "figures/confusion_matrix_qtrc_e005.png",
            },
        }
        atomic_write_json(report_dir / "manuscript_inputs.json", manuscript_inputs)

        snippet: list[str] = [
            "% Auto-generated by QTRC-Stop v5. Do not edit numerical values manually.",
            "% CP7 rows: Condition & Accuracy mean+-std & Delta accuracy mean+-std & Mean time mean+-std & Gap mean+-std",
        ]
        if cp7_rows.exists():
            snippet.append(cp7_rows.read_text(encoding="utf-8").rstrip())
        else:
            snippet.append(r"\FACT{Run CP7 to generate the five controlled-degradation rows.}")
        runtime_sentence_path = report_dir / "runtime" / "runtime_manuscript_sentence.txt"
        snippet.append("% Receiver-side runtime sentence")
        if runtime_sentence_path.exists():
            snippet.append(runtime_sentence_path.read_text(encoding="utf-8").rstrip())
        else:
            snippet.append(r"\FACT{Run the runtime benchmark on the target receiver.}")
        (report_dir / "manuscript_update_snippet.tex").write_text(
            "\n".join(snippet) + "\n", encoding="utf-8"
        )

        report = {
            "status": "complete",
            "config_fingerprint": config_fingerprint(self.cfg),
            "pipeline_version": "5.0.0",
            "calibration": "one species-level scalar temperature per prefix; argmax preserved",
            "calibration_safeguard_pass": bool(safeguard.get("pass", False)),
            "metric_aggregation": "pooled outer-test event decisions; stratified subject bootstrap confidence intervals",
            "risk_selection": "subject-disjoint selection and certification split inside each outer-training fold",
            "fixed_baseline": "shortest globally fixed duration selected and certified with the same epsilon protocol",
            "rows_in_summary": int(len(summary)),
            "cp7_status": cp7_status.get("status", "missing"),
            "runtime_status": runtime_report.get("status", "missing"),
            "required_outputs": [
                "cp0_inventory/selected/inventory.json",
                "cp1_events/events.csv",
                "cp2_features/features.csv.gz",
                "cp3_gates/gate_report.json",
                "cp4_predictions/outer_*/test_predictions.csv.gz",
                "cp4_predictions/outer_*/loss_matrix.json",
                "cp5_policies/calibration_argmax_audit.csv",
                "cp5_policies/fixed_risk_matched_grid.csv",
                "cp6_results/event_decisions.csv.gz",
                "cp6_results/summary_metrics.csv",
                "cp6_results/species_metrics.csv",
                "cp6_results/subject_metrics.csv",
                "cp6_results/policy_audit.csv",
                "cp6_results/certification_summary.csv",
                "cp6_results/calibration_argmax_audit.csv",
                "cp6_results/confusion_matrices.csv",
                "cp7_robustness/paper_sensitivity_table_e0.050.csv",
                "cp7_robustness/robustness_seed_summary.csv",
                "cp7_robustness/feature_drift_summary.csv",
                "cp7_robustness/awgn_monotonicity_diagnostic.json",
                "cp8_reports/runtime/runtime_summary.csv",
                "cp8_reports/runtime/hardware_manifest.json",
                "cp8_reports/figures/fixed_duration_curve.png",
                "cp8_reports/figures/risk_gap_vs_time.png",
                "cp8_reports/figures/confusion_matrix_qtrc_e005.png",
                "cp8_reports/figures/awgn_sensitivity_curve.png",
                "cp8_reports/figures/feature_drift_vs_snr.png",
            ],
            "claim_limits": [
                "bootstrap_subject is an empirical independent-split upper estimate, not a distribution-free guarantee",
                "reliability_style and confidence_only are internal baselines, not exact reproductions of TEASER or AAGC",
                "CP7 is a common-random-number multi-seed synthetic sensitivity analysis and does not re-certify the policy under degradation",
                "Non-monotonic AWGN behavior, if reproduced across common-noise seeds, must be reported as an identified sensitivity limitation rather than robustness",
                "quality-gate contribution requires a measurable ablation difference",
                "runtime excludes signal acquisition, disk I/O, and model loading",
            ],
        }
        atomic_write_json(report_dir / "final_report.json", report)
