import numpy as np

from qtrc_stop.signal import preprocess_complex


def test_causal_prefix_is_unchanged_by_future_samples():
    rng = np.random.default_rng(3)
    prefix = rng.normal(size=400) + 1j * rng.normal(size=400)
    future_a = rng.normal(size=600) + 1j * rng.normal(size=600)
    future_b = 20.0 * (rng.normal(size=600) + 1j * rng.normal(size=600))
    cfg = {
        "causal": True,
        "normalize": "none",
        "highpass_cutoff_hz": 50.0,
        "highpass_order": 2,
        "cheby_ripple_db": 0.5,
    }
    out_a = preprocess_complex(np.concatenate([prefix, future_a]), 1000.0, cfg)
    out_b = preprocess_complex(np.concatenate([prefix, future_b]), 1000.0, cfg)
    assert np.allclose(out_a[: len(prefix)], out_b[: len(prefix)])
