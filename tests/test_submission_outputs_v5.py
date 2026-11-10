from __future__ import annotations

import pandas as pd

from qtrc_stop.plots import plot_confusion_matrix
from qtrc_stop.runtime_benchmark import stratified_event_sample, summarize_runtime_samples
from qtrc_stop.taxonomy import SPECIES


def test_runtime_sample_and_summary_are_deterministic():
    rows = []
    for species_idx, species in enumerate(SPECIES):
        for subject_idx in range(2):
            for event_idx in range(3):
                rows.append({
                    "event_id": f"e{species_idx}_{subject_idx}_{event_idx}",
                    "species": species,
                    "subject_id": f"u{species_idx}_{subject_idx}",
                })
    events = pd.DataFrame(rows)
    a = stratified_event_sample(events, 12, 7)
    b = stratified_event_sample(events, 12, 7)
    assert a["event_id"].tolist() == b["event_id"].tolist()
    assert len(a) == 12
    samples = pd.DataFrame({
        "feature_extraction_ms": [1.0, 2.0, 3.0],
        "total_receiver_compute_ms": [2.0, 4.0, 6.0],
    })
    summary = summarize_runtime_samples(
        samples, ["feature_extraction_ms", "total_receiver_compute_ms"]
    )
    assert set(summary["component"]) == {"feature_extraction", "total_receiver_compute"}
    assert float(summary.loc[summary.component.eq("feature_extraction"), "median_ms"].iloc[0]) == 2.0


def test_confusion_matrix_figure_and_table(tmp_path):
    rows = []
    for i, true_species in enumerate(SPECIES):
        for j, pred_species in enumerate(SPECIES):
            rows.append({
                "method": "qtrc_stop",
                "epsilon": 0.05,
                "true_species": true_species,
                "pred_species": pred_species,
                "count": 10 if i == j else 1,
            })
    frame = pd.DataFrame(rows)
    out = tmp_path / "cm.png"
    table = plot_confusion_matrix(frame, out, method="qtrc_stop", epsilon=0.05)
    assert out.exists() and out.stat().st_size > 0
    assert table.shape == (5, 6)
