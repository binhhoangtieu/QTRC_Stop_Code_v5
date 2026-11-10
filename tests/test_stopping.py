import pandas as pd

from qtrc_stop.stopping import Policy, apply_policy
from qtrc_stop.taxonomy import SPECIES


def test_policy_stops_after_stability_is_available():
    rows = []
    for d, p, stable in [(0.1, 0.6, 0.0), (0.2, 0.95, 1.0), (2.0, 0.99, 1.0)]:
        row = {
            "event_id": "e1",
            "subject_id": "s1",
            "species": "apis_mellifera",
            "duration_sec": d,
            "harmonic_ratio": 0.8,
            "harmonic_energy_ratio": 0.5,
            "sbr_db": 10.0,
            "f0_delta_hz": 1.0,
            "stability_available": stable,
        }
        for species in SPECIES:
            row[f"p_{species}"] = p if species == "apis_mellifera" else (1 - p) / 4
        rows.append(row)
    policy = Policy(
        confidence=0.9,
        js_max=1.0,
        harmonic_ratio_min=0.0,
        harmonic_energy_min=0.0,
        sbr_min_db=-20.0,
        f0_delta_max_hz=1e9,
        persistence=1,
        epsilon=0.1,
        bound_method="empirical",
        calibration_mean_gap=0.0,
        calibration_ucb_gap=0.0,
        calibration_mean_time=0.0,
    )
    out = apply_policy(pd.DataFrame(rows), [0.1, 0.2, 2.0], policy)
    assert out.iloc[0].stop_duration_sec == 0.2
