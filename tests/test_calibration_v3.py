import numpy as np

from qtrc_stop.calibration import apply_temperature, argmax_change_count


def test_species_temperature_preserves_argmax():
    probs = np.asarray(
        [
            [0.60, 0.20, 0.10, 0.05, 0.05],
            [0.05, 0.10, 0.15, 0.50, 0.20],
            [0.12, 0.13, 0.14, 0.15, 0.46],
        ],
        dtype=float,
    )
    for temperature in [0.25, 0.7, 1.0, 2.5, 8.0]:
        calibrated = apply_temperature(probs, temperature)
        assert argmax_change_count(probs, calibrated) == 0
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
