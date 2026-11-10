from __future__ import annotations

import numpy as np

from qtrc_stop.robustness import make_common_perturbation_draws, perturb_preprocessed_signal_common


def test_common_awgn_uses_same_direction_across_snr():
    rng = np.random.default_rng(3)
    z = rng.normal(size=2048) + 1j * rng.normal(size=2048)
    draws = make_common_perturbation_draws(len(z), 123)
    y0, d0 = perturb_preprocessed_signal_common(z, {"snr_db": 0.0}, draws, 1000.0)
    y20, d20 = perturb_preprocessed_signal_common(z, {"snr_db": 20.0}, draws, 1000.0)
    n0 = y0 - z
    n20 = y20 - z
    ratio = n0 / np.where(np.abs(n20) > 1e-15, n20, 1.0)
    valid = np.abs(n20) > 1e-15
    assert np.std(ratio[valid]) < 1e-10
    assert np.isclose(np.mean(np.real(ratio[valid])), 10.0, rtol=1e-8)
    assert abs(float(d0["measured_snr_db"]) - 0.0) < 1e-10
    assert abs(float(d20["measured_snr_db"]) - 20.0) < 1e-10


def test_common_phase_process_scales_with_std():
    z = np.ones(4096, dtype=np.complex128)
    draws = make_common_perturbation_draws(len(z), 9)
    y05, _ = perturb_preprocessed_signal_common(z, {"phase_noise_std_rad": 0.05}, draws, 1000.0)
    y10, _ = perturb_preprocessed_signal_common(z, {"phase_noise_std_rad": 0.10}, draws, 1000.0)
    p05 = np.unwrap(np.angle(y05))
    p10 = np.unwrap(np.angle(y10))
    assert np.allclose(p10, 2.0 * p05, atol=1e-10)


def test_common_draws_are_deterministic():
    a = make_common_perturbation_draws(100, 77)
    b = make_common_perturbation_draws(100, 77)
    assert np.allclose(a.awgn_unit, b.awgn_unit)
    assert np.allclose(a.phase_unit, b.phase_unit)
    assert a.gain_phase_rad == b.gain_phase_rad
