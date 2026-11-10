from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import fft, signal, stats

from .signal import autocorrelation_harmonic_ratio, combined_psd, robust_normalize


def _safe_stat(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _spectral_features(freqs: np.ndarray, psd: np.ndarray) -> dict[str, float]:
    total = float(np.sum(psd))
    if total <= 1e-18:
        return {"spectral_centroid_hz": 0.0, "spectral_bandwidth_hz": 0.0, "spectral_entropy": 0.0, "spectral_flatness": 0.0, "rolloff_85_hz": 0.0}
    p = psd / total
    centroid = float(np.sum(freqs * p))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))
    entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-18))) / np.log(len(p)))
    flatness = float(np.exp(np.mean(np.log(np.maximum(psd, 1e-18)))) / np.mean(psd))
    cdf = np.cumsum(p)
    idx = int(np.searchsorted(cdf, 0.85, side="left"))
    rolloff = float(freqs[min(idx, len(freqs) - 1)])
    return {
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_entropy": entropy,
        "spectral_flatness": flatness,
        "rolloff_85_hz": rolloff,
    }


def _mel_filterbank(n_filters: int, n_fft: int, fs: float, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    fmax = fs / 2.0 if fmax is None else min(fmax, fs / 2.0)
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / fs).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1), dtype=float)
    for m in range(1, n_filters + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(right - center, 1)
    return fb


def _mfcc_summary(x: np.ndarray, fs: float, n_fft: int, n_filters: int, n_coeff: int) -> dict[str, float]:
    frame_len = min(max(int(round(0.128 * fs)), 64), max(len(x), 64))
    hop = max(frame_len // 2, 1)
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    frames = []
    for start in range(0, len(x) - frame_len + 1, hop):
        frame = x[start:start + frame_len] * np.hanning(frame_len)
        spec = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
        frames.append(spec)
    if not frames:
        frames = [np.zeros(n_fft // 2 + 1)]
    power = np.vstack(frames)
    fb = _mel_filterbank(n_filters, n_fft, fs, fmin=0.0, fmax=min(500.0, fs / 2.0))
    energies = np.maximum(power @ fb.T, 1e-18)
    coeff = fft.dct(np.log(energies), type=2, axis=1, norm="ortho")[:, :n_coeff]
    delta = np.diff(coeff, axis=0) if len(coeff) > 1 else np.zeros_like(coeff)
    delta2 = np.diff(delta, axis=0) if len(delta) > 1 else np.zeros_like(delta)
    out: dict[str, float] = {}
    for i in range(n_coeff):
        out[f"mfcc_{i}_mean"] = _safe_stat(np.mean(coeff[:, i]))
        out[f"mfcc_{i}_std"] = _safe_stat(np.std(coeff[:, i]))
        out[f"mfcc_delta_{i}_mean"] = _safe_stat(np.mean(delta[:, i]))
        out[f"mfcc_delta2_{i}_mean"] = _safe_stat(np.mean(delta2[:, i]))
    return out


def extract_prefix_features(z: np.ndarray, fs: float, cfg: dict) -> dict[str, float]:
    if len(z) < 8:
        raise ValueError("Prefix is too short")
    mode = cfg.get("prefix_normalize", "robust_prefix")
    if mode == "robust_prefix":
        z = robust_normalize(z)
    elif mode == "zscore_prefix":
        scale = float(np.std(z))
        z = (z - np.mean(z)) / (scale if scale > 1e-12 else 1.0)
    elif mode not in {"none", None}:
        raise ValueError(f"Unknown prefix normalization mode: {mode}")
    x = np.real(z)
    q = np.imag(z)
    amp = np.abs(z)
    phase = np.unwrap(np.angle(z))
    dphase = np.diff(phase, prepend=phase[0]) * fs / (2.0 * np.pi)

    features: dict[str, float] = {}
    for name, arr in {"i": x, "q": q, "amp": amp, "doppler": dphase}.items():
        features[f"{name}_mean"] = _safe_stat(np.mean(arr))
        features[f"{name}_std"] = _safe_stat(np.std(arr))
        features[f"{name}_rms"] = _safe_stat(np.sqrt(np.mean(arr ** 2)))
        features[f"{name}_median"] = _safe_stat(np.median(arr))
        features[f"{name}_iqr"] = _safe_stat(np.percentile(arr, 75) - np.percentile(arr, 25))
        if len(arr) > 4 and np.std(arr) > 1e-12:
            features[f"{name}_skew"] = _safe_stat(stats.skew(arr, bias=False))
            features[f"{name}_kurtosis"] = _safe_stat(stats.kurtosis(arr, fisher=True, bias=False))
        else:
            features[f"{name}_skew"] = 0.0
            features[f"{name}_kurtosis"] = 0.0

    features["zero_crossing_rate_i"] = float(np.mean(np.diff(np.signbit(x)).astype(float))) if len(x) > 1 else 0.0
    features["crest_factor"] = _safe_stat(np.max(np.abs(z)) / max(np.sqrt(np.mean(np.abs(z) ** 2)), 1e-12))

    hr, f0 = autocorrelation_harmonic_ratio(x, fs)
    freqs, psd = combined_psd(z, fs, int(cfg["n_fft"]))
    features["harmonic_ratio"] = hr
    features["f0_hz"] = _safe_stat(f0)
    features.update(_spectral_features(freqs, psd))

    total_power = _band_power(freqs, psd, 0.0, fs / 2.0)
    edges = list(map(float, cfg["band_edges_hz"]))
    for lo, hi in zip(edges[:-1], edges[1:]):
        bp = _band_power(freqs, psd, lo, hi)
        features[f"bandpower_{int(lo)}_{int(hi)}"] = bp / max(total_power, 1e-18)

    harmonic_energy = 0.0
    harmonic_peaks: list[float] = []
    if np.isfinite(f0) and f0 > 0:
        width = max(5.0, 0.05 * f0)
        for h in range(1, 6):
            center = h * f0
            if center >= fs / 2.0:
                harmonic_peaks.append(0.0)
                continue
            energy = _band_power(freqs, psd, max(0.0, center - width), min(fs / 2.0, center + width))
            harmonic_energy += energy
            harmonic_peaks.append(energy / max(total_power, 1e-18))
    else:
        harmonic_peaks = [0.0] * 5
    for idx, value in enumerate(harmonic_peaks, start=1):
        features[f"harmonic_{idx}_energy_ratio"] = float(value)
    features["harmonic_energy_ratio"] = float(harmonic_energy / max(total_power, 1e-18))
    background = np.median(psd[(freqs >= 50) & (freqs <= min(500, fs / 2.0))]) if np.any((freqs >= 50) & (freqs <= min(500, fs / 2.0))) else np.median(psd)
    peak = max(harmonic_peaks) * max(total_power, 1e-18)
    features["sbr_db"] = float(10.0 * np.log10(max(peak, 1e-18) / max(background, 1e-18)))

    features.update(
        _mfcc_summary(
            x,
            fs,
            n_fft=int(cfg["n_fft"]),
            n_filters=int(cfg["mel_filters"]),
            n_coeff=int(cfg["mfcc_coefficients"]),
        )
    )
    return features


def extract_event_feature_rows(event: pd.Series, z_hp: np.ndarray, fs: float, cfg: dict) -> list[dict]:
    rows: list[dict] = []
    start = int(event.start_sample)
    durations = list(map(float, cfg["prefix_durations_sec"]))
    previous_f0: float | None = None
    for duration in durations:
        n = int(round(duration * fs))
        segment = z_hp[start:start + n]
        if len(segment) < n:
            continue
        feats = extract_prefix_features(segment, fs, cfg)
        f0 = float(feats.get("f0_hz", 0.0))
        if previous_f0 is None:
            feats["f0_delta_hz"] = float("inf")
            feats["stability_available"] = 0.0
        else:
            feats["f0_delta_hz"] = abs(f0 - previous_f0)
            feats["stability_available"] = 1.0
        previous_f0 = f0
        rows.append(
            {
                "event_id": event.event_id,
                "recording_id": event.recording_id,
                "subject_id": event.subject_id,
                "species": event.species,
                "path": event.path,
                "start_sample": start,
                "duration_sec": duration,
                **feats,
            }
        )
    return rows


def metadata_columns() -> list[str]:
    return ["event_id", "recording_id", "subject_id", "species", "path", "start_sample", "duration_sec"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in metadata_columns()]
