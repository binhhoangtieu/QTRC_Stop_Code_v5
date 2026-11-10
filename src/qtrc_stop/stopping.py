from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from .taxonomy import SPECIES, label_indices


@dataclass(frozen=True)
class Policy:
    confidence: float
    js_max: float
    harmonic_ratio_min: float
    harmonic_energy_min: float
    sbr_min_db: float
    f0_delta_max_hz: float
    persistence: int
    epsilon: float
    bound_method: str
    calibration_mean_gap: float
    calibration_ucb_gap: float
    calibration_mean_time: float
    selection_subjects: int = 0
    certification_subjects: int = 0
    certification_passed: bool = False
    risk_protocol: str = "split_selection_certification"
    require_stability_available: bool = True
    fallback_full_duration: bool = False
    fallback_reason: str = "none"


def _pivot_probabilities(pred_df: pd.DataFrame, durations: list[float]) -> tuple[list[str], np.ndarray, pd.DataFrame]:
    events = sorted(pred_df["event_id"].unique())
    meta = pred_df.drop_duplicates("event_id").set_index("event_id").loc[events]
    probs = np.zeros((len(events), len(durations), len(SPECIES)), dtype=float)
    for j, duration in enumerate(durations):
        sub = pred_df[np.isclose(pred_df["duration_sec"], duration)].set_index("event_id").reindex(events)
        if sub[[f"p_{s}" for s in SPECIES]].isna().any().any():
            raise ValueError(f"Missing prefix predictions at duration {duration}")
        for k, species in enumerate(SPECIES):
            probs[:, j, k] = sub[f"p_{species}"].to_numpy()
    return events, probs, meta


def _quality_arrays(pred_df: pd.DataFrame, events: list[str], durations: list[float]) -> dict[str, np.ndarray]:
    defaults = {
        "harmonic_ratio": 0.0,
        "harmonic_energy_ratio": 0.0,
        "sbr_db": -np.inf,
        "f0_delta_hz": np.inf,
        "stability_available": 0.0,
    }
    out = {k: np.full((len(events), len(durations)), v, dtype=float) for k, v in defaults.items()}
    for j, duration in enumerate(durations):
        sub = pred_df[np.isclose(pred_df["duration_sec"], duration)].set_index("event_id").reindex(events)
        for key, default in defaults.items():
            if key in sub.columns:
                out[key][:, j] = sub[key].fillna(default).to_numpy(dtype=float)
    return out


def _js_array(probs: np.ndarray) -> np.ndarray:
    n, t, _ = probs.shape
    # No stability statement is available at the first prefix.
    js = np.full((n, t), np.inf, dtype=float)
    for j in range(1, t):
        js[:, j] = np.asarray([
            jensenshannon(probs[i, j], probs[i, j - 1], base=2.0) ** 2
            for i in range(n)
        ])
    return js


def prepare_policy_data(pred_df: pd.DataFrame, durations: list[float]) -> dict[str, Any]:
    events, probs, meta = _pivot_probabilities(pred_df, durations)
    quality = _quality_arrays(pred_df, events, durations)
    labels = probs.argmax(axis=2)
    persistence = {
        1: np.ones(labels.shape, dtype=bool),
        2: np.concatenate([
            np.zeros((len(events), 1), dtype=bool),
            labels[:, 1:] == labels[:, :-1],
        ], axis=1),
    }
    return {
        "events": events,
        "probs": probs,
        "meta": meta,
        "labels": labels,
        "confidence": probs.max(axis=2),
        "js": _js_array(probs),
        "quality": quality,
        "persistence": persistence,
        "durations": np.asarray(durations, dtype=float),
        "true_idx": label_indices(meta["species"].to_numpy()),
        "subject_ids": meta["subject_id"].to_numpy(),
    }


def subset_policy_data(data: dict[str, Any], subjects: set[str] | list[str] | np.ndarray) -> dict[str, Any]:
    subject_set = set(map(str, subjects))
    mask = np.asarray([str(s) in subject_set for s in data["subject_ids"]], dtype=bool)
    idx = np.flatnonzero(mask)
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key == "meta":
            out[key] = value.iloc[idx].copy()
        elif key == "events":
            out[key] = [value[i] for i in idx]
        elif key == "durations":
            out[key] = value.copy()
        elif key == "quality":
            out[key] = {k: v[idx].copy() for k, v in value.items()}
        elif key == "persistence":
            out[key] = {k: v[idx].copy() for k, v in value.items()}
        elif isinstance(value, np.ndarray) and value.shape[0] == len(mask):
            out[key] = value[idx].copy()
        else:
            out[key] = value
    return out


def _candidate_indices(data: dict[str, Any], policy: Policy) -> tuple[np.ndarray, np.ndarray]:
    p_mask = data["persistence"].get(policy.persistence)
    if p_mask is None:
        labels = data["labels"]
        p_mask = np.zeros(labels.shape, dtype=bool)
        for j in range(policy.persistence - 1, labels.shape[1]):
            window = labels[:, j - policy.persistence + 1:j + 1]
            p_mask[:, j] = np.all(window == window[:, [-1]], axis=1)
    if policy.require_stability_available:
        stability_mask = (
            (data["js"] <= policy.js_max)
            & (data["quality"]["f0_delta_hz"] <= policy.f0_delta_max_hz)
            & (data["quality"]["stability_available"] >= 0.5)
        )
    else:
        stability_mask = np.ones(data["confidence"].shape, dtype=bool)
    eligible = (
        (data["confidence"] >= policy.confidence)
        & stability_mask
        & (data["quality"]["harmonic_ratio"] >= policy.harmonic_ratio_min)
        & (data["quality"]["harmonic_energy_ratio"] >= policy.harmonic_energy_min)
        & (data["quality"]["sbr_db"] >= policy.sbr_min_db)
        & p_mask
    )
    has = eligible.any(axis=1)
    idx = np.where(has, eligible.argmax(axis=1), eligible.shape[1] - 1)
    return idx.astype(int), has


def _decisions_from_data(data: dict[str, Any], policy: Policy, selective_mode: bool) -> pd.DataFrame:
    idx, has = _candidate_indices(data, policy)
    row = np.arange(len(idx))
    pred_idx = data["labels"][row, idx]
    full_idx = data["labels"][:, -1]
    abstain = ~has if selective_mode else np.zeros(len(idx), dtype=bool)
    meta = data["meta"]
    return pd.DataFrame(
        {
            "event_id": data["events"],
            "subject_id": meta["subject_id"].to_numpy(),
            "species": meta["species"].to_numpy(),
            "pred_species": [SPECIES[i] for i in pred_idx],
            "stop_duration_sec": data["durations"][idx],
            "confidence": data["confidence"][row, idx],
            "abstain": abstain,
            "full_pred_species": [SPECIES[i] for i in full_idx],
        }
    )


def apply_policy(pred_df: pd.DataFrame, durations: list[float], policy: Policy, selective_mode: bool = False) -> pd.DataFrame:
    return _decisions_from_data(prepare_policy_data(pred_df, durations), policy, selective_mode)


def _subject_gap_from_arrays(data: dict[str, Any], idx: np.ndarray, loss_matrix: np.ndarray) -> np.ndarray:
    row = np.arange(len(idx))
    early_idx = data["labels"][row, idx]
    full_idx = data["labels"][:, -1]
    true_idx = data["true_idx"]
    gaps = np.maximum(loss_matrix[true_idx, early_idx] - loss_matrix[true_idx, full_idx], 0.0)
    subject_ids, inv = np.unique(data["subject_ids"], return_inverse=True)
    sums = np.bincount(inv, weights=gaps, minlength=len(subject_ids))
    counts = np.bincount(inv, minlength=len(subject_ids))
    return sums / np.maximum(counts, 1)


def risk_ucb(subject_losses: np.ndarray, delta: float, method: str, repeats: int, seed: int) -> float:
    x = np.asarray(subject_losses, dtype=float)
    if len(x) == 0:
        return 1.0
    mean = float(np.mean(x))
    if method == "empirical":
        return mean
    if method == "hoeffding_subject":
        return min(1.0, mean + np.sqrt(np.log(1.0 / delta) / (2.0 * len(x))))
    if method == "empirical_bernstein_subject":
        var = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
        term = np.sqrt(2.0 * var * np.log(3.0 / delta) / max(len(x), 1)) + 3.0 * np.log(3.0 / delta) / max(len(x), 1)
        return min(1.0, mean + term)
    if method == "bootstrap_subject":
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(x), size=(int(repeats), len(x)))
        means = x[indices].mean(axis=1)
        return float(np.quantile(means, 1.0 - delta))
    raise ValueError(f"Unknown bound method: {method}")


def policy_grid(cfg: dict) -> list[tuple[float, float, float, float, float, float, int]]:
    return list(
        itertools.product(
            cfg["confidence_grid"],
            cfg["js_grid"],
            cfg["harmonic_ratio_grid"],
            cfg["harmonic_energy_grid"],
            cfg.get("sbr_grid_db", [-1e9]),
            cfg["f0_delta_grid_hz"],
            cfg["persistence_grid"],
        )
    )


def _candidate_policy(values: tuple, cfg: dict) -> Policy:
    conf, js_max, hr_min, he_min, sbr_min, f0_delta, persistence = values
    return Policy(
        float(conf), float(js_max), float(hr_min), float(he_min), float(sbr_min),
        float(f0_delta), int(persistence), 0.0, str(cfg["bound_method"]),
        0.0, 0.0, 0.0,
        require_stability_available=bool(cfg.get("require_stability_available", True)),
    )


def evaluate_policy_grid_data(data: dict[str, Any], loss_matrix: np.ndarray, cfg: dict, seed: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for values in policy_grid(cfg):
        candidate = _candidate_policy(values, cfg)
        idx, _ = _candidate_indices(data, candidate)
        subject_losses = _subject_gap_from_arrays(data, idx, loss_matrix)
        records.append(
            {
                "confidence": candidate.confidence,
                "js_max": candidate.js_max,
                "harmonic_ratio_min": candidate.harmonic_ratio_min,
                "harmonic_energy_min": candidate.harmonic_energy_min,
                "sbr_min_db": candidate.sbr_min_db,
                "f0_delta_max_hz": candidate.f0_delta_max_hz,
                "persistence": candidate.persistence,
                "mean_gap": float(np.mean(subject_losses)),
                "ucb_gap": risk_ucb(subject_losses, float(cfg["delta"]), str(cfg["bound_method"]), int(cfg["bootstrap_repeats"]), seed),
                "mean_time": float(np.mean(data["durations"][idx])),
                "subjects": int(len(np.unique(data["subject_ids"]))),
            }
        )
    return pd.DataFrame(records)


def evaluate_policy_grid(pred_df: pd.DataFrame, durations: list[float], loss_matrix: np.ndarray, cfg: dict, seed: int, subjects: set[str] | None = None) -> pd.DataFrame:
    data = prepare_policy_data(pred_df, durations)
    if subjects is not None:
        data = subset_policy_data(data, subjects)
    return evaluate_policy_grid_data(data, loss_matrix, cfg, seed)


def _row_to_policy(row: pd.Series, epsilon: float, cfg: dict, selection_subjects: int, certification_subjects: int, cert_mean: float, cert_ucb: float, passed: bool) -> Policy:
    return Policy(
        float(row.confidence), float(row.js_max), float(row.harmonic_ratio_min),
        float(row.harmonic_energy_min), float(row.sbr_min_db),
        float(row.f0_delta_max_hz), int(row.persistence), float(epsilon),
        str(cfg["bound_method"]), float(cert_mean), float(cert_ucb),
        float(row.mean_time), int(selection_subjects), int(certification_subjects),
        bool(passed), "split_selection_certification",
        bool(cfg.get("require_stability_available", True)),
        False, "none",
    )


def full_duration_policy(
    durations: list[float],
    epsilon: float,
    cfg: dict,
    selection_subjects: int = 0,
    certification_subjects: int = 0,
    certification_passed: bool = False,
    certification_mean_gap: float = 0.0,
    certification_ucb_gap: float = 0.0,
    fallback_reason: str = "full_duration",
) -> Policy:
    return Policy(
        1.1, 0.0, 1.1, 1.1, 1e9, 0.0, 99, float(epsilon),
        str(cfg["bound_method"]), float(certification_mean_gap),
        float(certification_ucb_gap), float(max(durations)),
        int(selection_subjects), int(certification_subjects), bool(certification_passed),
        "split_selection_certification",
        bool(cfg.get("require_stability_available", True)),
        True, str(fallback_reason),
    )


def split_calibration_subjects(pred_df: pd.DataFrame, seed: int, certification_fraction: float = 0.5) -> tuple[set[str], set[str]]:
    subject_df = pred_df[["subject_id", "species"]].drop_duplicates().sort_values(["species", "subject_id"])
    rng = np.random.default_rng(seed)
    selection: set[str] = set()
    certification: set[str] = set()
    for _, group in subject_df.groupby("species", sort=True):
        subjects = group["subject_id"].astype(str).to_numpy().copy()
        rng.shuffle(subjects)
        if len(subjects) == 1:
            selection.add(str(subjects[0]))
            continue
        n_cert = int(round(len(subjects) * float(certification_fraction)))
        n_cert = min(max(n_cert, 1), len(subjects) - 1)
        certification.update(map(str, subjects[:n_cert]))
        selection.update(map(str, subjects[n_cert:]))
    if not certification:
        ordered = sorted(selection)
        certification.add(ordered[-1])
        selection.remove(ordered[-1])
    return selection, certification


def select_policy_split_certified(
    pred_df: pd.DataFrame,
    durations: list[float],
    epsilon: float,
    loss_matrix: np.ndarray,
    cfg: dict,
    seed: int,
    selection_grid: pd.DataFrame | None = None,
    selection_subjects: set[str] | None = None,
    certification_subjects: set[str] | None = None,
) -> tuple[Policy, pd.DataFrame, dict[str, Any]]:
    if selection_subjects is None or certification_subjects is None:
        selection_subjects, certification_subjects = split_calibration_subjects(
            pred_df, seed, float(cfg.get("certification_fraction", 0.5))
        )
    data = prepare_policy_data(pred_df, durations)
    selection_data = subset_policy_data(data, selection_subjects)
    certification_data = subset_policy_data(data, certification_subjects)
    if selection_grid is None:
        selection_grid = evaluate_policy_grid_data(selection_data, loss_matrix, cfg, seed)

    safety = float(cfg.get("selection_epsilon_fraction", 1.0))
    feasible = selection_grid[selection_grid["mean_gap"] <= float(epsilon) * safety].copy()
    if feasible.empty:
        policy = full_duration_policy(
            durations, epsilon, cfg, len(selection_subjects), len(certification_subjects),
            False, 0.0, 0.0, "no_selection_candidate",
        )
        return policy, selection_grid, {
            "status": "no_selection_candidate",
            "selection_subjects": sorted(selection_subjects),
            "certification_subjects": sorted(certification_subjects),
            "certification_mean_gap": 0.0,
            "certification_ucb_gap": 0.0,
            "fallback_full_duration": True,
        }

    row = feasible.sort_values(["mean_time", "mean_gap", "confidence"], ascending=[True, True, False]).iloc[0]
    provisional = _row_to_policy(row, epsilon, cfg, len(selection_subjects), len(certification_subjects), 0.0, 1.0, False)
    idx, _ = _candidate_indices(certification_data, provisional)
    cert_losses = _subject_gap_from_arrays(certification_data, idx, loss_matrix)
    cert_mean = float(np.mean(cert_losses))
    cert_ucb = risk_ucb(
        cert_losses, float(cfg["delta"]), str(cfg["bound_method"]),
        int(cfg["bootstrap_repeats"]), seed + 991,
    )
    passed = bool(cert_ucb <= float(epsilon))
    if passed:
        policy = _row_to_policy(row, epsilon, cfg, len(selection_subjects), len(certification_subjects), cert_mean, cert_ucb, True)
    else:
        policy = full_duration_policy(
            durations, epsilon, cfg, len(selection_subjects), len(certification_subjects),
            False, cert_mean, cert_ucb, "certification_failed",
        )
    return policy, selection_grid, {
        "status": "certified" if passed else "certification_failed",
        "selection_subjects": sorted(selection_subjects),
        "certification_subjects": sorted(certification_subjects),
        "selected_candidate": row.to_dict(),
        "certification_mean_gap": cert_mean,
        "certification_ucb_gap": cert_ucb,
        "fallback_full_duration": not passed,
        "bound_interpretation": (
            "distribution-free" if str(cfg["bound_method"]).startswith(("hoeffding", "empirical_bernstein"))
            else "independent-split empirical bootstrap"
        ),
    }


def select_policy_from_grid(grid_df: pd.DataFrame, durations: list[float], epsilon: float, cfg: dict) -> Policy:
    """Legacy single-pool selector retained for diagnostic baselines only."""
    feasible = grid_df[grid_df["ucb_gap"] <= float(epsilon)].copy()
    if feasible.empty:
        return full_duration_policy(durations, epsilon, cfg)
    row = feasible.sort_values(["mean_time", "ucb_gap", "confidence"], ascending=[True, True, False]).iloc[0]
    return _row_to_policy(row, epsilon, cfg, int(row.get("subjects", 0)), 0, float(row.mean_gap), float(row.ucb_gap), True)


def select_policy(pred_df: pd.DataFrame, durations: list[float], epsilon: float, loss_matrix: np.ndarray, cfg: dict, seed: int) -> tuple[Policy, pd.DataFrame]:
    policy, grid, _ = select_policy_split_certified(pred_df, durations, epsilon, loss_matrix, cfg, seed)
    return policy, grid
