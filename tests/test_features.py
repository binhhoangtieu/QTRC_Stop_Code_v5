import numpy as np
from qtrc_stop.features import extract_prefix_features


def test_harmonic_feature_detects_periodicity():
    fs = 1000
    t = np.arange(1000) / fs
    z = np.sin(2 * np.pi * 150 * t) + 1j * np.cos(2 * np.pi * 150 * t)
    cfg = {"n_fft": 256, "mel_filters": 20, "mfcc_coefficients": 8, "band_edges_hz": [50,100,150,200,250,300,350,400,450,500]}
    f = extract_prefix_features(z, fs, cfg)
    assert f["harmonic_ratio"] > 0.5
    assert 100 <= f["f0_hz"] <= 200
