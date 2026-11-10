from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "seed": 20260731,
    "dataset": {
        "path": "data/Polliradar dataset.zip",
        "sampling_rate_hz": 1000,
        "expected_files": 167,
        "strict_inventory": True,
        "max_subjects_per_species": None,
        "max_recordings_per_subject": None,
    },
    "signal": {
        "highpass_cutoff_hz": 50.0,
        "highpass_order": 2,
        "cheby_ripple_db": 0.5,
        "causal": True,
        "normalize": "none",
    },
    "events": {
        "detector_window_sec": 0.4,
        "detector_step_sec": 0.1,
        "harmonic_ratio_threshold": 0.35,
        "min_event_gap_sec": 2.2,
        "max_events_per_recording": 12,
        "fallback_top_k": 0,
        "max_prefix_sec": 2.0,
        "causal": True,
    },
    "gates": {
        "min_subjects_total": 20,
        "min_species": 5,
        "complete_event_fraction_min": 0.95,
        "nan_fraction_max": 0.05,
        "fallback_event_fraction_max": 0.25,
        "expected_subjects_per_species": None,
    },
    "features": {
        "prefix_durations_sec": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0],
        "mfcc_coefficients": 8,
        "mel_filters": 20,
        "n_fft": 256,
        "band_edges_hz": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        "prefix_normalize": "robust_prefix",
    },
    "model": {
        "backend": "paper_hybrid",
        "catboost_iterations": 250,
        "catboost_depth": 6,
        "catboost_learning_rate": 0.05,
        "extra_trees_estimators": 400,
        "min_samples_leaf": 2,
        "n_jobs": -1,
    },
    "cv": {
        "outer_protocol": "loso",
        "outer_splits": 4,
        "inner_splits": 3,
        "max_outer_folds": None,
    },
    "calibration": {
        "temperature_bounds": [0.25, 8.0],
        "subject_balanced": True,
    },
    "policy": {
        "epsilons": [0.02, 0.05, 0.10, 0.15],
        "delta": 0.10,
        "bound_method": "bootstrap_subject",
        "bootstrap_repeats": 2000,
        "confidence_grid": [0.60, 0.70, 0.80, 0.90, 0.95],
        "js_grid": [0.05, 0.10, 0.20, 1.0],
        "harmonic_ratio_grid": [0.0, 0.30, 0.50],
        "harmonic_energy_grid": [0.0, 0.15, 0.30],
        "sbr_grid_db": [-20.0, 0.0, 5.0],
        "f0_delta_grid_hz": [50.0, 100.0, 1000000.0],
        "persistence_grid": [1, 2],
        "beta_family": 0.4,
        "beta_genus": 0.2,
        "inverse_subject_class_weight": True,
        "selective_mode": False,
        "certification_fraction": 0.5,
        "selection_epsilon_fraction": 1.0,
        "require_stability_available": True,
    },
    "robustness": {
        "enabled": False,
        "protocol": "common_random_numbers_multiseed",
        "seeds": [0, 1, 2, 3, 4],
        "snr_db": [0, 5, 10, 15, 20],
        "phase_noise_std_rad": [0.05, 0.10],
        "gain_ripple_depth": [0.10, 0.20],
        "gain_ripple_rate_hz": 1.0,
        "paper_epsilon": 0.05,
        "paper_snr_db": [0, 10, 20],
        "paper_phase_noise_std_rad": 0.10,
        "paper_gain_ripple_depth": 0.20,
    },
    "runtime_benchmark": {
        "enabled": False,
        "primary_epsilon": 0.05,
        "events": 100,
        "repeats": 3,
        "warmup_events": 5,
        "include_filtering": True,
    },
    "ablation": {
        "enabled": True,
    },
    "reporting": {
        "bootstrap_repeats": 2000,
    },
    "runtime": {
        "resume": True,
        "fail_on_gate": True,
        "stale_lock_minutes": 180,
        "log_level": "INFO",
    },
}


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    with Path(path).open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    _deep_update(cfg, user_cfg)
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


def save_resolved_config(cfg: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    tmp.replace(path)


def config_fingerprint(cfg: dict[str, Any]) -> str:
    import hashlib

    # Runtime logging, the physical dataset path, and optional synthetic
    # robustness conditions do not change CP0-CP6 scientific artifacts.
    semantic = copy.deepcopy(cfg)
    semantic.pop("runtime", None)
    semantic.pop("robustness", None)
    semantic.pop("runtime_benchmark", None)
    if isinstance(semantic.get("dataset"), dict):
        semantic["dataset"].pop("path", None)
    payload = json.dumps(semantic, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
