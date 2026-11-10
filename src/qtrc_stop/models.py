from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from .taxonomy import FAMILY, GENUS, SPECIES


class ConstantClassifier:
    def __init__(self, classes: list[str], constant: str):
        self.classes_ = np.asarray(classes)
        self.constant = constant

    def predict_proba(self, x):
        out = np.zeros((len(x), len(self.classes_)), dtype=float)
        idx = int(np.where(self.classes_ == self.constant)[0][0])
        out[:, idx] = 1.0
        return out


@dataclass
class HierarchicalClassifier:
    cfg: dict[str, Any]
    seed: int

    def __post_init__(self):
        self.family_model = None
        self.genus_model = None
        self.bombus_model = None
        self.feature_names: list[str] = []

    def _binary_model(self, classes: np.ndarray, seed_offset: int):
        backend = self.cfg.get("backend", "paper_hybrid")
        if backend == "paper_hybrid":
            return CatBoostClassifier(
                iterations=int(self.cfg["catboost_iterations"]),
                depth=int(self.cfg["catboost_depth"]),
                learning_rate=float(self.cfg["catboost_learning_rate"]),
                loss_function="Logloss",
                random_seed=self.seed + seed_offset,
                verbose=False,
                allow_writing_files=False,
                thread_count=int(self.cfg.get("n_jobs", -1)),
                auto_class_weights="Balanced",
            )
        return ExtraTreesClassifier(
            n_estimators=int(self.cfg["extra_trees_estimators"]),
            min_samples_leaf=int(self.cfg.get("min_samples_leaf", 2)),
            class_weight="balanced",
            random_state=self.seed + seed_offset,
            n_jobs=int(self.cfg.get("n_jobs", -1)),
        )

    def _multiclass_model(self, seed_offset: int):
        return ExtraTreesClassifier(
            n_estimators=int(self.cfg["extra_trees_estimators"]),
            min_samples_leaf=int(self.cfg.get("min_samples_leaf", 2)),
            class_weight="balanced",
            random_state=self.seed + seed_offset,
            n_jobs=int(self.cfg.get("n_jobs", -1)),
        )

    @staticmethod
    def _fit_or_constant(model, x, y, expected_classes: list[str], sample_weight=None):
        unique = np.unique(y)
        if len(unique) == 1:
            return ConstantClassifier(expected_classes, str(unique[0]))
        model.fit(x, y, sample_weight=sample_weight)
        return model

    def fit(self, x: np.ndarray, y_species: np.ndarray, feature_names: list[str], sample_weight: np.ndarray | None = None):
        self.feature_names = list(feature_names)
        y_family = np.asarray([FAMILY[str(y)] for y in y_species])
        self.family_model = self._fit_or_constant(
            self._binary_model(np.unique(y_family), 11), x, y_family, ["apidae", "vespidae"], sample_weight
        )

        apidae = y_family == "apidae"
        y_genus = np.asarray([GENUS[str(y)] for y in y_species[apidae]])
        sw_genus = sample_weight[apidae] if sample_weight is not None else None
        self.genus_model = self._fit_or_constant(
            self._binary_model(np.unique(y_genus), 23), x[apidae], y_genus, ["apis", "bombus"], sw_genus
        )

        bombus = np.asarray([GENUS[str(y)] == "bombus" for y in y_species])
        bombus_classes = ["bombus_lapidarius", "bombus_muscorum", "bombus_terrestris"]
        sw_bombus = sample_weight[bombus] if sample_weight is not None else None
        self.bombus_model = self._fit_or_constant(
            self._multiclass_model(37), x[bombus], y_species[bombus], bombus_classes, sw_bombus
        )
        return self

    @staticmethod
    def _proba_for(model, x: np.ndarray, ordered_classes: list[str]) -> np.ndarray:
        p = model.predict_proba(x)
        out = np.zeros((len(x), len(ordered_classes)), dtype=float)
        for src_idx, cls in enumerate(model.classes_):
            if str(cls) in ordered_classes:
                out[:, ordered_classes.index(str(cls))] = p[:, src_idx]
        row_sum = out.sum(axis=1, keepdims=True)
        return out / np.maximum(row_sum, 1e-12)

    @staticmethod
    def combine_components(p_family: np.ndarray, p_genus: np.ndarray, p_bombus: np.ndarray) -> np.ndarray:
        out = np.zeros((len(p_family), len(SPECIES)), dtype=float)
        out[:, SPECIES.index("vespula_vulgaris")] = p_family[:, 1]
        out[:, SPECIES.index("apis_mellifera")] = p_family[:, 0] * p_genus[:, 0]
        for idx, species in enumerate(["bombus_lapidarius", "bombus_muscorum", "bombus_terrestris"]):
            out[:, SPECIES.index(species)] = p_family[:, 0] * p_genus[:, 1] * p_bombus[:, idx]
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)

    def predict_components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        p_family = self._proba_for(self.family_model, x, ["apidae", "vespidae"])
        p_genus = self._proba_for(self.genus_model, x, ["apis", "bombus"])
        p_bombus = self._proba_for(
            self.bombus_model, x,
            ["bombus_lapidarius", "bombus_muscorum", "bombus_terrestris"],
        )
        return {"family": p_family, "genus": p_genus, "bombus": p_bombus}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        parts = self.predict_components(x)
        return self.combine_components(parts["family"], parts["genus"], parts["bombus"])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        joblib.dump(self, tmp, compress=3)
        tmp.replace(path)

    @staticmethod
    def load(path: str | Path) -> "HierarchicalClassifier":
        return joblib.load(path)


def subject_balanced_weights(subject_ids, species_labels) -> np.ndarray:
    subject_ids = np.asarray(subject_ids)
    species_labels = np.asarray(species_labels)
    unique_subjects, subject_counts = np.unique(subject_ids, return_counts=True)
    inv_subject = {s: 1.0 / c for s, c in zip(unique_subjects, subject_counts)}
    unique_species, species_counts = np.unique(species_labels, return_counts=True)
    inv_species = {s: len(species_labels) / (len(unique_species) * c) for s, c in zip(unique_species, species_counts)}
    weights = np.asarray([inv_subject[sid] * inv_species[sp] for sid, sp in zip(subject_ids, species_labels)], dtype=float)
    return weights / np.mean(weights)
