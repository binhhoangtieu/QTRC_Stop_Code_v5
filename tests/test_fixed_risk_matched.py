import numpy as np
import pandas as pd

from qtrc_stop.baselines import select_fixed_duration_split_certified
from qtrc_stop.taxonomy import SPECIES


def _row(event, subject, species, duration, pred):
    row = {
        "event_id": event,
        "subject_id": subject,
        "species": species,
        "duration_sec": duration,
    }
    for label in SPECIES:
        row[f"p_{label}"] = 0.9 if label == pred else 0.1 / (len(SPECIES) - 1)
    return row


def test_risk_matched_fixed_selects_shortest_certified_duration():
    rows = []
    subjects = ["s1", "s2", "s3", "s4"]
    for i, subject in enumerate(subjects):
        species = SPECIES[i]
        rows.append(_row(f"e{i}", subject, species, 0.5, species))
        rows.append(_row(f"e{i}", subject, species, 2.0, species))
    frame = pd.DataFrame(rows)
    loss = np.ones((len(SPECIES), len(SPECIES))) - np.eye(len(SPECIES))
    cfg = {
        "delta": 0.1,
        "bound_method": "empirical",
        "bootstrap_repeats": 100,
        "selection_epsilon_fraction": 1.0,
    }
    policy, grid, cert = select_fixed_duration_split_certified(
        frame,
        [0.5, 2.0],
        0.05,
        loss,
        cfg,
        7,
        {"s1", "s2"},
        {"s3", "s4"},
    )
    assert len(grid) == 2
    assert policy["duration_sec"] == 0.5
    assert policy["certification_passed"] is True
    assert cert["fallback_full_duration"] is False
