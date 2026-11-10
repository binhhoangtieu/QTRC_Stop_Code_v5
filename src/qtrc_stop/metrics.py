from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

from .taxonomy import SPECIES, label_indices


def expected_calibration_error(probs: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_idx).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(correct[mask]) - np.mean(conf[mask]))
    return float(ece)


def classwise_ece(probs: np.ndarray, y_idx: np.ndarray, bins: int = 15) -> dict[str, float]:
    out: dict[str, float] = {}
    edges = np.linspace(0.0, 1.0, bins + 1)
    for k, species in enumerate(SPECIES):
        conf = probs[:, k]
        target = (y_idx == k).astype(float)
        value = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi)
            if np.any(mask):
                value += np.mean(mask) * abs(np.mean(target[mask]) - np.mean(conf[mask]))
        out[species] = float(value)
    return out


def brier_multiclass(probs: np.ndarray, y_idx: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y_idx]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def attach_decision_losses(decisions: pd.DataFrame, loss_matrix: np.ndarray) -> pd.DataFrame:
    out = decisions.copy()
    true_idx = label_indices(out["species"])
    pred_idx = label_indices(out["pred_species"])
    full_idx = label_indices(out["full_pred_species"])
    out["taxonomic_loss"] = loss_matrix[true_idx, pred_idx]
    out["full_taxonomic_loss"] = loss_matrix[true_idx, full_idx]
    out["taxonomic_gap"] = np.maximum(out["taxonomic_loss"] - out["full_taxonomic_loss"], 0.0)
    return out


def _base_time_metrics(decisions: pd.DataFrame, max_duration: float) -> dict[str, float]:
    return {
        "coverage": float(np.mean(~decisions["abstain"].astype(bool))) if len(decisions) else np.nan,
        "mean_stop_sec": float(decisions["stop_duration_sec"].mean()) if len(decisions) else np.nan,
        "median_stop_sec": float(decisions["stop_duration_sec"].median()) if len(decisions) else np.nan,
        "p90_stop_sec": float(decisions["stop_duration_sec"].quantile(0.90)) if len(decisions) else np.nan,
        "earliness": float(1.0 - decisions["stop_duration_sec"].mean() / max_duration) if len(decisions) else np.nan,
    }


def decision_metrics(decisions: pd.DataFrame, loss_matrix: np.ndarray, max_duration: float) -> dict[str, float]:
    """Metrics for one fold.

    Multiclass metrics are returned as NaN when the fold does not contain all
    taxonomic classes. This prevents LOSO single-class folds from producing a
    misleading Macro-F1, balanced accuracy, or worst-class recall.
    """
    enriched = attach_decision_losses(decisions, loss_matrix)
    active = enriched[~enriched["abstain"]].copy()
    base = _base_time_metrics(enriched, max_duration)
    if active.empty:
        return {
            **base, "accuracy": np.nan, "macro_f1": np.nan,
            "balanced_accuracy": np.nan, "worst_recall": np.nan,
            "mean_taxonomic_loss": np.nan, "subject_mean_gap": np.nan,
            "subject_max_gap": np.nan,
        }
    y_true = active["species"].to_numpy()
    y_pred = active["pred_species"].to_numpy()
    present = set(map(str, np.unique(y_true)))
    multiclass_valid = present == set(SPECIES)
    subject_gap = active.groupby("subject_id")["taxonomic_gap"].mean().to_numpy()
    return {
        **base,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=SPECIES, average="macro", zero_division=0)) if multiclass_valid else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if multiclass_valid else np.nan,
        "worst_recall": float(np.min(recall_score(y_true, y_pred, labels=SPECIES, average=None, zero_division=0))) if multiclass_valid else np.nan,
        "mean_taxonomic_loss": float(active["taxonomic_loss"].mean()),
        "subject_mean_gap": float(np.mean(subject_gap)),
        "subject_max_gap": float(np.max(subject_gap)),
    }


def pooled_decision_metrics(decisions: pd.DataFrame, max_duration: float, epsilon: float | None = None) -> dict[str, float]:
    """Compute paper-level metrics after pooling all outer-test decisions."""
    active = decisions[~decisions["abstain"]].copy()
    base = _base_time_metrics(decisions, max_duration)
    if active.empty:
        return {
            **base, "events": int(len(decisions)), "subjects": int(decisions.subject_id.nunique()),
            "accuracy": np.nan, "macro_f1": np.nan, "balanced_accuracy": np.nan,
            "worst_recall": np.nan, "mean_taxonomic_loss": np.nan,
            "subject_mean_gap": np.nan, "subject_max_gap": np.nan,
            "risk_violation_rate": np.nan,
        }
    required = {"taxonomic_loss", "taxonomic_gap"}
    missing = required.difference(active.columns)
    if missing:
        raise ValueError(f"Pooled decisions are missing loss columns: {sorted(missing)}")
    y_true = active["species"].to_numpy()
    y_pred = active["pred_species"].to_numpy()
    recalls = recall_score(y_true, y_pred, labels=SPECIES, average=None, zero_division=0)
    subject_gap = active.groupby("subject_id")["taxonomic_gap"].mean()
    violation = np.nan if epsilon is None or not np.isfinite(epsilon) else float(np.mean(subject_gap.to_numpy() > float(epsilon)))
    return {
        **base,
        "events": int(len(decisions)),
        "subjects": int(decisions.subject_id.nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=SPECIES, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "worst_recall": float(np.min(recalls)),
        "mean_taxonomic_loss": float(active["taxonomic_loss"].mean()),
        "subject_mean_gap": float(subject_gap.mean()),
        "subject_max_gap": float(subject_gap.max()),
        "risk_violation_rate": violation,
    }


def _subject_bootstrap_values(decisions: pd.DataFrame, repeats: int, seed: int, metric: str) -> np.ndarray:
    subjects = np.asarray(sorted(decisions["subject_id"].astype(str).unique()))
    lookup = {species: i for i, species in enumerate(SPECIES)}
    n_subjects = len(subjects)
    total_events = np.zeros(n_subjects, dtype=float)
    active_events = np.zeros(n_subjects, dtype=float)
    correct = np.zeros(n_subjects, dtype=float)
    stop_sum = np.zeros(n_subjects, dtype=float)
    gap_mean = np.zeros(n_subjects, dtype=float)
    confusion = np.zeros((n_subjects, len(SPECIES), len(SPECIES)), dtype=float)
    subject_species = np.empty(n_subjects, dtype=object)
    sid = decisions["subject_id"].astype(str)
    for i, subject in enumerate(subjects):
        group = decisions[sid == subject]
        subject_species[i] = str(group["species"].iloc[0])
        active = group[~group["abstain"]]
        total_events[i] = len(group)
        active_events[i] = len(active)
        correct[i] = float(np.sum(active["species"].to_numpy() == active["pred_species"].to_numpy()))
        stop_sum[i] = float(group["stop_duration_sec"].sum())
        gap_mean[i] = float(group["taxonomic_gap"].mean())
        for true_label, pred_label in zip(active["species"], active["pred_species"]):
            confusion[i, lookup[str(true_label)], lookup[str(pred_label)]] += 1.0

    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(subject_species == species) for species in SPECIES]
    values = np.empty(int(repeats), dtype=float)
    for b in range(int(repeats)):
        sampled_parts = [rng.choice(idx, size=len(idx), replace=True) for idx in strata if len(idx)]
        indices = np.concatenate(sampled_parts)
        if metric == "subject_mean_gap":
            values[b] = float(np.mean(gap_mean[indices]))
        elif metric == "accuracy":
            values[b] = float(np.sum(correct[indices]) / max(np.sum(active_events[indices]), 1.0))
        elif metric == "mean_stop_sec":
            values[b] = float(np.sum(stop_sum[indices]) / max(np.sum(total_events[indices]), 1.0))
        elif metric in {"balanced_accuracy", "macro_f1"}:
            cm = confusion[indices].sum(axis=0)
            tp = np.diag(cm)
            true_count = cm.sum(axis=1)
            pred_count = cm.sum(axis=0)
            if metric == "balanced_accuracy":
                recalls = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
                values[b] = float(np.mean(recalls))
            else:
                denom = 2.0 * tp + (pred_count - tp) + (true_count - tp)
                f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
                values[b] = float(np.mean(f1))
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
    return values[np.isfinite(values)]

def add_subject_bootstrap_ci(summary: dict[str, float], decisions: pd.DataFrame, repeats: int, seed: int) -> dict[str, float]:
    out = dict(summary)
    for idx, metric in enumerate(["accuracy", "balanced_accuracy", "macro_f1", "mean_stop_sec", "subject_mean_gap"]):
        values = _subject_bootstrap_values(decisions, repeats, seed + idx * 101, metric)
        out[f"{metric}_ci_low"] = float(np.quantile(values, 0.025)) if len(values) else np.nan
        out[f"{metric}_ci_high"] = float(np.quantile(values, 0.975)) if len(values) else np.nan
    return out


def species_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for species in SPECIES:
        sub = decisions[decisions["species"] == species]
        if sub.empty:
            continue
        active = sub[~sub["abstain"]]
        recall_covered = float(np.mean(active["pred_species"] == species)) if len(active) else 0.0
        recall_total = float(np.mean((sub["pred_species"] == species) & (~sub["abstain"])))
        rows.append(
            {
                "species": species,
                "n_events": int(len(sub)),
                "n_subjects": int(sub["subject_id"].nunique()),
                "coverage": float(len(active) / len(sub)),
                "recall_covered": recall_covered,
                "recall_total": recall_total,
                "fnr_total": float(1.0 - recall_total),
                "mean_stop_sec": float(sub["stop_duration_sec"].mean()),
            }
        )
    return pd.DataFrame(rows)
