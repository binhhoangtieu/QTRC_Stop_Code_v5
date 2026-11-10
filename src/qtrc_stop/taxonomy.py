from __future__ import annotations

import numpy as np

SPECIES = [
    "apis_mellifera",
    "bombus_lapidarius",
    "bombus_muscorum",
    "bombus_terrestris",
    "vespula_vulgaris",
]

FAMILY = {
    "apis_mellifera": "apidae",
    "bombus_lapidarius": "apidae",
    "bombus_muscorum": "apidae",
    "bombus_terrestris": "apidae",
    "vespula_vulgaris": "vespidae",
}

GENUS = {
    "apis_mellifera": "apis",
    "bombus_lapidarius": "bombus",
    "bombus_muscorum": "bombus",
    "bombus_terrestris": "bombus",
    "vespula_vulgaris": "vespula",
}

ALIASES = {
    "bombus_lapidaries": "bombus_lapidarius",
}


def canonical_species(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    return ALIASES.get(key, key)


def taxonomic_loss_matrix(
    subject_counts: dict[str, int],
    beta_family: float = 0.4,
    beta_genus: float = 0.2,
    inverse_subject_class_weight: bool = True,
) -> np.ndarray:
    if inverse_subject_class_weight:
        raw = {s: 1.0 / max(subject_counts.get(s, 1), 1) for s in SPECIES}
        total_w = float(np.sum(list(raw.values())))
        weights = {s: raw[s] / max(total_w, 1e-12) for s in SPECIES}
    else:
        weights = {s: 1.0 for s in SPECIES}

    mat = np.zeros((len(SPECIES), len(SPECIES)), dtype=float)
    max_loss = 0.0
    for i, true_s in enumerate(SPECIES):
        for j, pred_s in enumerate(SPECIES):
            if true_s == pred_s:
                loss = 0.0
            else:
                loss = weights[true_s]
                loss += beta_genus * float(GENUS[true_s] != GENUS[pred_s])
                loss += beta_family * float(FAMILY[true_s] != FAMILY[pred_s])
            mat[i, j] = loss
            max_loss = max(max_loss, loss)
    if max_loss > 0:
        mat /= max_loss
    return mat


def label_indices(labels) -> np.ndarray:
    lookup = {s: i for i, s in enumerate(SPECIES)}
    return np.asarray([lookup[str(x)] for x in labels], dtype=int)
