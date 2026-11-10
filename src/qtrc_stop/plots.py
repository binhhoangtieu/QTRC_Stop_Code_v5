from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_fixed_duration(summary: pd.DataFrame, out_path: Path) -> None:
    sub = summary[summary["method"].str.match(r"^fixed_\d")].copy()
    if sub.empty:
        return
    sub["duration"] = sub["method"].str.replace("fixed_", "", regex=False).astype(float)
    sub = sub.sort_values("duration")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(sub["duration"], sub["accuracy"], marker="o", label="Accuracy")
    ax.plot(sub["duration"], sub["macro_f1"], marker="s", label="Macro-F1")
    ax.plot(sub["duration"], sub["balanced_accuracy"], marker="^", label="Balanced accuracy")
    ax.set_xlabel("Observation duration (s)")
    ax.set_ylabel("Pooled outer-test score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out_path)


def plot_risk_time(summary: pd.DataFrame, out_path: Path) -> None:
    sub = summary[summary["method"].isin(["qtrc_stop", "fixed_risk_matched"])].copy()
    sub = sub[sub["epsilon"].notna()].sort_values(["method", "epsilon"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method, group in sub.groupby("method", sort=True):
        ax.plot(group["mean_stop_sec"], group["subject_mean_gap"], marker="o", label=method)
        points = group.groupby(["mean_stop_sec", "subject_mean_gap"], sort=False)["epsilon"].apply(list)
        for (x, y), eps_values in points.items():
            label = ",".join(f"{e:.2f}" for e in eps_values)
            ax.annotate(f"eps={label}", (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean observation duration (s)")
    ax.set_ylabel("Empirical subject-mean taxonomic gap")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out_path)


def plot_accuracy_time(summary: pd.DataFrame, out_path: Path) -> None:
    sub = summary[summary["method"].isin(["qtrc_stop", "fixed_risk_matched"])].copy()
    sub = sub[sub["epsilon"].notna()].sort_values(["method", "epsilon"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method, group in sub.groupby("method", sort=True):
        ax.plot(group["mean_stop_sec"], group["balanced_accuracy"], marker="o", label=method)
    ax.set_xlabel("Mean observation duration (s)")
    ax.set_ylabel("Pooled balanced accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out_path)


def plot_certification(certification: pd.DataFrame, out_path: Path) -> None:
    sub = certification[certification["method"].isin(["qtrc_stop", "fixed_risk_matched"])].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method, group in sub.groupby("method", sort=True):
        group = group.sort_values("epsilon")
        ax.plot(group["epsilon"], group["certification_pass_rate"], marker="o", label=method)
    ax.set_xlabel("Risk tolerance epsilon")
    ax.set_ylabel("Certified outer-fold fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out_path)


def plot_calibration(calibration: pd.DataFrame, out_path: Path) -> None:
    if calibration.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for status, group in calibration.groupby("status", sort=True):
        group = group.sort_values("duration_sec")
        ax.plot(group["duration_sec"], group["ece"], marker="o", label=f"ECE: {status}")
    ax.set_xlabel("Observation duration (s)")
    ax.set_ylabel("Expected calibration error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out_path)


def plot_confusion_matrix(
    confusion: pd.DataFrame,
    out_path: Path,
    method: str = "qtrc_stop",
    epsilon: float = 0.05,
    normalize: bool = True,
) -> pd.DataFrame:
    """Plot and return a species confusion matrix for one operating point."""
    from .taxonomy import SPECIES

    sub = confusion[confusion["method"].eq(method)].copy()
    if "epsilon" in sub.columns:
        sub = sub[sub["epsilon"].notna() & (sub["epsilon"].astype(float) - float(epsilon)).abs().lt(1e-9)]
    if sub.empty:
        raise ValueError(f"No confusion matrix found for {method} at epsilon={epsilon}")
    matrix = (
        sub.pivot_table(index="true_species", columns="pred_species", values="count", aggfunc="sum", fill_value=0)
        .reindex(index=SPECIES, columns=SPECIES, fill_value=0)
    )
    counts = matrix.to_numpy(dtype=float)
    if normalize:
        denom = counts.sum(axis=1, keepdims=True)
        display = counts / np.maximum(denom, 1.0)
    else:
        display = counts
    labels = [
        r"A. mellifera",
        r"B. lapidarius",
        r"B. muscorum",
        r"B. terrestris",
        r"V. vulgaris",
    ]
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    image = ax.imshow(display, vmin=0.0, vmax=1.0 if normalize else None, aspect="equal")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species")
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            text = f"{display[i, j]:.2f}\n({int(counts[i, j])})" if normalize else str(int(counts[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Row-normalized fraction" if normalize else "Event count")
    _save(fig, out_path)
    result = matrix.reset_index().rename(columns={"true_species": "true_species"})
    return result


def plot_awgn_sensitivity(robustness_summary: pd.DataFrame, out_path: Path, epsilon: float = 0.05) -> None:
    """Plot multi-seed AWGN accuracy and observation time versus SNR."""
    sub = robustness_summary[
        robustness_summary["condition"].astype(str).str.startswith("snr_")
        & np.isclose(robustness_summary["epsilon"].astype(float), float(epsilon))
    ].copy()
    if sub.empty:
        return
    sub["snr_db"] = sub["condition"].str.extract(r"snr_([-+]?\d+(?:\.\d+)?)dB")[0].astype(float)
    sub = sub.sort_values("snr_db")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(sub["snr_db"], sub["accuracy_mean"], yerr=sub["accuracy_std"], marker="o", capsize=3)
    ax.set_xlabel("Injected AWGN SNR (dB)")
    ax.set_ylabel("Pooled accuracy")
    ax.grid(True, alpha=0.3)
    _save(fig, out_path)


def plot_feature_drift_vs_snr(feature_drift_summary: pd.DataFrame, out_path: Path, duration_sec: float = 2.0) -> None:
    """Plot standardized feature drift and posterior JS divergence versus SNR."""
    sub = feature_drift_summary[
        feature_drift_summary["condition"].astype(str).str.startswith("snr_")
        & np.isclose(feature_drift_summary["duration_sec"].astype(float), float(duration_sec))
    ].copy()
    if sub.empty:
        return
    sub["snr_db"] = sub["condition"].str.extract(r"snr_([-+]?\d+(?:\.\d+)?)dB")[0].astype(float)
    sub = sub.sort_values("snr_db")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.errorbar(
        sub["snr_db"], sub["feature_rms_z_mean"],
        yerr=sub["feature_rms_z_std"].fillna(0.0), marker="o", capsize=3,
        label="Feature RMS drift"
    )
    ax.set_xlabel("Injected AWGN SNR (dB)")
    ax.set_ylabel("Feature drift (training-SD units)")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.errorbar(
        sub["snr_db"], sub["posterior_js_bits_mean"],
        yerr=sub["posterior_js_bits_std"].fillna(0.0), marker="s", capsize=3,
        label="Posterior JS"
    )
    ax2.set_ylabel("Posterior JS divergence (bits)")
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="best")
    _save(fig, out_path)
