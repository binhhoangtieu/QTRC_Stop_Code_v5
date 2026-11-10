import numpy as np
import pandas as pd

from qtrc_stop.metrics import attach_decision_losses, decision_metrics, pooled_decision_metrics
from qtrc_stop.taxonomy import SPECIES


def test_single_class_fold_does_not_report_multiclass_metrics():
    decisions = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "subject_id": ["s1", "s1"],
        "species": [SPECIES[0], SPECIES[0]],
        "pred_species": [SPECIES[0], SPECIES[1]],
        "full_pred_species": [SPECIES[0], SPECIES[0]],
        "stop_duration_sec": [0.2, 0.2],
        "abstain": [False, False],
    })
    loss = np.ones((5, 5)) - np.eye(5)
    metrics = decision_metrics(decisions, loss, 2.0)
    assert np.isnan(metrics["macro_f1"])
    assert np.isnan(metrics["balanced_accuracy"])
    assert np.isnan(metrics["worst_recall"])


def test_pooled_metrics_use_all_species():
    rows = []
    for i, species in enumerate(SPECIES):
        rows.append({
            "event_id": f"e{i}",
            "subject_id": f"subject{i}",
            "species": species,
            "pred_species": species,
            "full_pred_species": species,
            "stop_duration_sec": 1.0,
            "abstain": False,
        })
    loss = np.ones((5, 5)) - np.eye(5)
    decisions = attach_decision_losses(pd.DataFrame(rows), loss)
    metrics = pooled_decision_metrics(decisions, 2.0)
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["worst_recall"] == 1.0
