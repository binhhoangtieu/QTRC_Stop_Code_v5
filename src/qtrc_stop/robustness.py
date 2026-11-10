from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CommonPerturbationDraws:
    """Reusable random draws for controlled degradation.

    The same draw is reused across all magnitudes in one perturbation family for
    a fixed recording and robustness seed. This common-random-number design
    isolates the effect of SNR/phase/gain magnitude from realization changes.
    """

    awgn_unit: np.ndarray
    phase_unit: np.ndarray
    gain_phase_rad: float
    realization_seed: int


def make_common_perturbation_draws(length: int, seed: int) -> CommonPerturbationDraws:
    if length <= 0:
        raise ValueError("length must be positive")
    rng = np.random.default_rng(int(seed))
    awgn = rng.normal(size=length) + 1j * rng.normal(size=length)
    awgn_power = float(np.mean(np.abs(awgn) ** 2))
    if awgn_power <= 1e-18:
        raise RuntimeError("Degenerate AWGN draw")
    awgn = awgn / np.sqrt(awgn_power)
    phase = rng.normal(size=length)
    gain_phase = float(rng.uniform(0.0, 2.0 * np.pi))
    return CommonPerturbationDraws(
        awgn_unit=np.asarray(awgn, dtype=np.complex128),
        phase_unit=np.asarray(phase, dtype=float),
        gain_phase_rad=gain_phase,
        realization_seed=int(seed),
    )


def _apply_awgn_common(z: np.ndarray, snr_db: float, unit_noise: np.ndarray) -> tuple[np.ndarray, float, float]:
    signal_power = float(np.mean(np.abs(z) ** 2))
    target_noise_power = signal_power / max(10.0 ** (float(snr_db) / 10.0), 1e-18)
    noise = np.sqrt(max(target_noise_power, 0.0)) * unit_noise
    measured_noise_power = float(np.mean(np.abs(noise) ** 2))
    measured_snr = 10.0 * np.log10(max(signal_power, 1e-18) / max(measured_noise_power, 1e-18))
    return z + noise, float(measured_snr), float(measured_noise_power)


def perturb_preprocessed_signal_common(
    z_hp: np.ndarray,
    condition: dict,
    draws: CommonPerturbationDraws,
    fs: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply one controlled perturbation using common random numbers."""
    out = np.asarray(z_hp, dtype=np.complex128).copy()
    if len(out) != len(draws.awgn_unit) or len(out) != len(draws.phase_unit):
        raise ValueError("Common perturbation draws must match signal length")
    diagnostics: dict[str, float | int] = {
        "realization_seed": int(draws.realization_seed),
        "nominal_signal_power": float(np.mean(np.abs(out) ** 2)),
    }

    if "gain_ripple_depth" in condition:
        depth = float(condition["gain_ripple_depth"])
        rate_hz = float(condition.get("gain_ripple_rate_hz", 1.0))
        t = np.arange(len(out), dtype=float) / float(fs)
        gain = 1.0 + depth * np.sin(2.0 * np.pi * rate_hz * t + draws.gain_phase_rad)
        out *= gain
        diagnostics["gain_ripple_depth"] = depth
        diagnostics["gain_ripple_phase_rad"] = float(draws.gain_phase_rad)

    if "phase_noise_std_rad" in condition:
        std = float(condition["phase_noise_std_rad"])
        phase_noise = std * draws.phase_unit
        out *= np.exp(1j * phase_noise)
        diagnostics["phase_noise_std_rad"] = std
        diagnostics["realized_phase_std_rad"] = float(np.std(phase_noise))

    if "snr_db" in condition:
        target = float(condition["snr_db"])
        out, measured, noise_power = _apply_awgn_common(out, target, draws.awgn_unit)
        diagnostics["target_snr_db"] = target
        diagnostics["measured_snr_db"] = measured
        diagnostics["added_noise_power"] = noise_power

    return out, diagnostics


def perturb_preprocessed_signal(z_hp: np.ndarray, condition: dict, seed: int, fs: float) -> tuple[np.ndarray, dict[str, float | int]]:
    """Compatibility wrapper using one deterministic common draw."""
    draws = make_common_perturbation_draws(len(z_hp), seed)
    return perturb_preprocessed_signal_common(z_hp, condition, draws, fs)


def perturb_signal(z: np.ndarray, condition: dict, seed: int, fs: float = 1000.0) -> np.ndarray:
    """Compatibility wrapper returning only the perturbed signal."""
    out, _ = perturb_preprocessed_signal(z, condition, seed, fs)
    return out
