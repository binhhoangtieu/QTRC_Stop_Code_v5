from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .stopping import Policy, apply_policy, prepare_policy_data, risk_ucb
from .taxonomy import SPECIES, label_indices


def fixed_duration_decisions(pred_df: pd.DataFrame, duration: float) -> pd.DataFrame:
    sub = pred_df[np.isclose(pred_df["duration_sec"], duration)].copy()
    probs = sub[[f"p_{s}" for s in SPECIES]].to_numpy()
    pred = probs.argmax(axis=1)
    max_duration = float(pred_df["duration_sec"].max())
    full = pred_df[np.isclose(pred_df["duration_sec"], max_duration)].set_index("event_id")
    full_probs = full[[f"p_{s}" for s in SPECIES]].to_numpy()
    full_labels = full_probs.argmax(axis=1)
    full_map = dict(zip(full.index, [SPECIES[i] for i in full_labels]))
    return pd.DataFrame(
        {
            "event_id": sub["event_id"].to_numpy(),
            "subject_id": sub["subject_id"].to_numpy(),
            "species": sub["species"].to_numpy(),
            "pred_species": [SPECIES[i] for i in pred],
            "stop_duration_sec": float(duration),
            "confidence": probs.max(axis=1),
            "abstain": False,
            "full_pred_species": [full_map[e] for e in sub["event_id"]],
        }
    )


def confidence_policy(threshold: float, epsilon: float = 1.0) -> Policy:
    return Policy(
        confidence=float(threshold),
        js_max=1.0,
        harmonic_ratio_min=0.0,
        harmonic_energy_min=0.0,
        sbr_min_db=-1e9,
        f0_delta_max_hz=1e9,
        persistence=1,
        epsilon=epsilon,
        bound_method="none",
        calibration_mean_gap=0.0,
        calibration_ucb_gap=0.0,
        calibration_mean_time=0.0,
        require_stability_available=False,
    )


class _ConstantReliability:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full(len(x), self.value, dtype=float)
        return np.column_stack([1.0 - p, p])


def _reliability_features(data: dict, duration_index: int) -> np.ndarray:
    probs = data["probs"][:, duration_index, :]
    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1) / np.log(probs.shape[1])
    duration_norm = np.full(len(probs), data["durations"][duration_index] / data["durations"][-1])
    return np.column_stack(
        [
            data["confidence"][:, duration_index],
            margin,
            entropy,
            data["js"][:, duration_index],
            data["quality"]["harmonic_ratio"][:, duration_index],
            data["quality"]["harmonic_energy_ratio"][:, duration_index],
            np.where(
                np.isfinite(data["quality"]["f0_delta_hz"][:, duration_index]),
                np.log1p(data["quality"]["f0_delta_hz"][:, duration_index]),
                0.0,
            ),
            data["quality"]["stability_available"][:, duration_index],
            duration_norm,
        ]
    )


def fit_two_stage_reliability(
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    durations: list[float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a TEASER-style reliability layer on subject-disjoint OOF predictions.

    This is an internal two-stage reliability baseline, not an exact reproduction
    of the original TEASER implementation.
    """
    oof_data = prepare_policy_data(oof_df, durations)
    test_data = prepare_policy_data(test_df, durations)
    oof_scores = np.zeros((len(oof_data["events"]), len(durations)), dtype=float)
    test_scores = np.zeros((len(test_data["events"]), len(durations)), dtype=float)

    subjects, counts = np.unique(oof_data["subject_ids"], return_counts=True)
    inv_count = {s: 1.0 / c for s, c in zip(subjects, counts)}
    weights = np.asarray([inv_count[s] for s in oof_data["subject_ids"]], dtype=float)
    weights /= np.mean(weights)

    for j, _ in enumerate(durations):
        x_train = _reliability_features(oof_data, j)
        target = (oof_data["labels"][:, j] == oof_data["true_idx"]).astype(int)
        if len(np.unique(target)) == 1:
            model = _ConstantReliability(float(target[0]))
        else:
            model = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed + j,
                solver="lbfgs",
            )
            model.fit(x_train, target, sample_weight=weights)
        oof_scores[:, j] = model.predict_proba(x_train)[:, 1]
        test_scores[:, j] = model.predict_proba(_reliability_features(test_data, j))[:, 1]

    def attach(frame: pd.DataFrame, data: dict, scores: np.ndarray) -> pd.DataFrame:
        mapping = {
            (event, float(duration)): float(scores[i, j])
            for i, event in enumerate(data["events"])
            for j, duration in enumerate(durations)
        }
        out = frame.copy()
        out["reliability_score"] = [mapping[(e, float(t))] for e, t in zip(out["event_id"], out["duration_sec"])]
        return out

    return attach(oof_df, oof_data, oof_scores), attach(test_df, test_data, test_scores)


def apply_score_stopping(pred_df: pd.DataFrame, durations: list[float], threshold: float, score_col: str = "reliability_score") -> pd.DataFrame:
    rows = []
    for event_id, group in pred_df.groupby("event_id", sort=True):
        group = group.sort_values("duration_sec")
        eligible = group[group[score_col] >= float(threshold)]
        chosen = eligible.iloc[0] if not eligible.empty else group.iloc[-1]
        full = group.iloc[-1]
        p_chosen = chosen[[f"p_{s}" for s in SPECIES]].to_numpy(dtype=float)
        p_full = full[[f"p_{s}" for s in SPECIES]].to_numpy(dtype=float)
        rows.append(
            {
                "event_id": event_id,
                "subject_id": chosen["subject_id"],
                "species": chosen["species"],
                "pred_species": SPECIES[int(np.argmax(p_chosen))],
                "stop_duration_sec": float(chosen["duration_sec"]),
                "confidence": float(chosen[score_col]),
                "abstain": False,
                "full_pred_species": SPECIES[int(np.argmax(p_full))],
            }
        )
    return pd.DataFrame(rows)


def _subject_gap(decisions: pd.DataFrame, loss_matrix: np.ndarray) -> np.ndarray:
    true_idx = label_indices(decisions["species"])
    pred_idx = label_indices(decisions["pred_species"])
    full_idx = label_indices(decisions["full_pred_species"])
    gaps = np.maximum(loss_matrix[true_idx, pred_idx] - loss_matrix[true_idx, full_idx], 0.0)
    tmp = decisions[["subject_id"]].copy()
    tmp["gap"] = gaps
    return tmp.groupby("subject_id")["gap"].mean().to_numpy()


def select_score_threshold(
    oof_scored: pd.DataFrame,
    durations: list[float],
    epsilon: float,
    loss_matrix: np.ndarray,
    policy_cfg: dict,
    seed: int,
    thresholds: list[float] | None = None,
) -> tuple[float, pd.DataFrame]:
    thresholds = thresholds or [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
    rows = []
    for threshold in thresholds:
        decisions = apply_score_stopping(oof_scored, durations, threshold)
        losses = _subject_gap(decisions, loss_matrix)
        rows.append(
            {
                "threshold": threshold,
                "mean_gap": float(np.mean(losses)),
                "ucb_gap": risk_ucb(
                    losses,
                    float(policy_cfg["delta"]),
                    str(policy_cfg["bound_method"]),
                    int(policy_cfg["bootstrap_repeats"]),
                    seed,
                ),
                "mean_time": float(decisions["stop_duration_sec"].mean()),
            }
        )
    grid = pd.DataFrame(rows)
    feasible = grid[grid["ucb_gap"] <= float(epsilon)]
    if feasible.empty:
        return 1.1, grid
    best = feasible.sort_values(["mean_time", "ucb_gap", "threshold"], ascending=[True, True, False]).iloc[0]
    return float(best.threshold), grid


def evaluate_fixed_duration_grid(
    pred_df: pd.DataFrame,
    durations: list[float],
    loss_matrix: np.ndarray,
    subjects: set[str] | list[str] | np.ndarray,
) -> pd.DataFrame:
    """Evaluate global fixed durations by subject-level taxonomic gap.

    The reference decision is the prediction at the maximum duration. Only the
    supplied subjects are used, so the grid can be used on the selection split.
    """
    subject_set = set(map(str, subjects))
    frame = pred_df[pred_df["subject_id"].astype(str).isin(subject_set)].copy()
    rows: list[dict] = []
    for duration in sorted(map(float, durations)):
        decisions = fixed_duration_decisions(frame, duration)
        gaps = _subject_gap(decisions, loss_matrix)
        rows.append(
            {
                "duration_sec": float(duration),
                "mean_gap": float(np.mean(gaps)) if len(gaps) else np.nan,
                "mean_time": float(duration),
                "subjects": int(len(np.unique(decisions["subject_id"].astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def select_fixed_duration_split_certified(
    pred_df: pd.DataFrame,
    durations: list[float],
    epsilon: float,
    loss_matrix: np.ndarray,
    policy_cfg: dict,
    seed: int,
    selection_subjects: set[str] | list[str] | np.ndarray,
    certification_subjects: set[str] | list[str] | np.ndarray,
) -> tuple[dict, pd.DataFrame, dict]:
    """Select the shortest global duration and certify it on disjoint subjects.

    This baseline uses the same selection/certification protocol as QTRC-Stop.
    If the selected duration fails certification, the final duration is forced
    to the maximum observation time.
    """
    selection_subjects = set(map(str, selection_subjects))
    certification_subjects = set(map(str, certification_subjects))
    durations = sorted(map(float, durations))
    max_duration = float(max(durations))
    grid = evaluate_fixed_duration_grid(pred_df, durations, loss_matrix, selection_subjects)
    safety = float(policy_cfg.get("selection_epsilon_fraction", 1.0))
    feasible = grid[grid["mean_gap"] <= float(epsilon) * safety].copy()
    if feasible.empty:
        candidate_duration = max_duration
        selection_mean_gap = 0.0
        selection_status = "no_selection_candidate"
    else:
        selected = feasible.sort_values(["duration_sec", "mean_gap"], ascending=[True, True]).iloc[0]
        candidate_duration = float(selected.duration_sec)
        selection_mean_gap = float(selected.mean_gap)
        selection_status = "selected"

    cert_frame = pred_df[pred_df["subject_id"].astype(str).isin(certification_subjects)].copy()
    cert_decisions = fixed_duration_decisions(cert_frame, candidate_duration)
    cert_losses = _subject_gap(cert_decisions, loss_matrix)
    cert_mean = float(np.mean(cert_losses)) if len(cert_losses) else 1.0
    cert_ucb = risk_ucb(
        cert_losses,
        float(policy_cfg["delta"]),
        str(policy_cfg["bound_method"]),
        int(policy_cfg["bootstrap_repeats"]),
        seed + 1771,
    )
    passed = bool(cert_ucb <= float(epsilon))
    final_duration = candidate_duration if passed else max_duration
    fallback = bool(not passed and candidate_duration < max_duration)
    payload = {
        "method": "fixed_risk_matched",
        "epsilon": float(epsilon),
        "candidate_duration_sec": float(candidate_duration),
        "duration_sec": float(final_duration),
        "selection_mean_gap": float(selection_mean_gap),
        "selection_subjects": int(len(selection_subjects)),
        "certification_subjects": int(len(certification_subjects)),
        "certification_mean_gap": float(cert_mean),
        "certification_ucb_gap": float(cert_ucb),
        "certification_passed": bool(passed),
        "fallback_full_duration": fallback,
        "selected_full_duration": bool(final_duration >= max_duration - 1e-12),
        "status": "certified" if passed else "certification_failed",
        "selection_status": selection_status,
        "bound_method": str(policy_cfg["bound_method"]),
        "risk_protocol": "split_selection_certification",
    }
    cert = {
        **payload,
        "selection_subject_ids": sorted(selection_subjects),
        "certification_subject_ids": sorted(certification_subjects),
        "bound_interpretation": (
            "distribution-free"
            if str(policy_cfg["bound_method"]).startswith(("hoeffding", "empirical_bernstein"))
            else "independent-split empirical bootstrap"
        ),
    }
    return payload, grid, cert
