from __future__ import annotations

import csv
import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .checkpoint import atomic_dataframe_csv, atomic_write_json
from .taxonomy import SPECIES, canonical_species


FILENAME_RE = re.compile(r"^(?P<species>.+)_(?P<subject>\d+)_(?P<recording>\d+)\.csv$", re.IGNORECASE)


def ensure_dataset(dataset_path: str | Path, target_dir: str | Path) -> Path:
    dataset_path = Path(dataset_path).expanduser().resolve()
    target_dir = Path(target_dir).resolve()
    if dataset_path.is_dir():
        return dataset_path
    if dataset_path.suffix.lower() != ".zip":
        raise ValueError(f"Dataset must be a directory or ZIP file: {dataset_path}")
    marker = target_dir / ".complete"
    if marker.exists():
        candidates = list(target_dir.rglob("*.csv"))
        if candidates:
            return common_csv_root(candidates)
    tmp = target_dir.with_name(target_dir.name + ".extracting")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dataset_path, "r") as zf:
        zf.extractall(tmp)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp.replace(target_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return common_csv_root(list(target_dir.rglob("*.csv")))


def common_csv_root(files: list[Path]) -> Path:
    if not files:
        raise FileNotFoundError("No CSV files found in dataset")
    parents = [str(p.parent) for p in files]
    return Path(Path(*Path(Path(parents[0])).parts)) if len(set(parents)) == 1 else Path(__import__('os').path.commonpath(parents))


def parse_recording_filename(path: Path) -> dict[str, str | int]:
    m = FILENAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Unsupported filename: {path.name}")
    species = canonical_species(m.group("species"))
    if species not in SPECIES:
        raise ValueError(f"Unknown species token '{species}' in {path.name}")
    subject_num = int(m.group("subject"))
    rec_num = int(m.group("recording"))
    subject_id = f"{species}__s{subject_num:02d}"
    recording_id = f"{subject_id}__r{rec_num:02d}"
    return {
        "species": species,
        "subject_num": subject_num,
        "recording_num": rec_num,
        "subject_id": subject_id,
        "recording_id": recording_id,
    }


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as f:
        count = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
    return max(count - 1, 0)


def inventory_dataset(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*.csv")):
        meta = parse_recording_filename(path)
        rows.append({**meta, "path": str(path.resolve()), "n_samples": count_csv_rows(path)})
    return pd.DataFrame(rows)


def select_inventory(df: pd.DataFrame, max_subjects_per_species: int | None, max_recordings_per_subject: int | None) -> pd.DataFrame:
    out = df.sort_values(["species", "subject_num", "recording_num"]).copy()
    if max_subjects_per_species is not None:
        selected_subjects: list[str] = []
        for _, g in out[["species", "subject_id", "subject_num"]].drop_duplicates().groupby("species", sort=True):
            selected_subjects.extend(g.sort_values("subject_num").head(int(max_subjects_per_species))["subject_id"].tolist())
        out = out[out["subject_id"].isin(selected_subjects)]
    if max_recordings_per_subject is not None:
        out = out.groupby("subject_id", group_keys=False, sort=False).head(int(max_recordings_per_subject))
    return out.reset_index(drop=True)


def read_signal_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path, usecols=["amplitude", "phase"], dtype={"amplitude": "float64", "phase": "float64"})
    return df["amplitude"].to_numpy(), df["phase"].to_numpy()


def write_inventory_reports(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_dataframe_csv(df, out_dir / "recordings.csv")
    summary = {
        "files": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "species": df.groupby("species")["recording_id"].count().astype(int).to_dict(),
        "subjects_per_species": df.groupby("species")["subject_id"].nunique().astype(int).to_dict(),
        "sample_lengths": df["n_samples"].value_counts().sort_index().astype(int).to_dict(),
        "total_samples": int(df["n_samples"].sum()),
    }
    atomic_write_json(out_dir / "inventory.json", summary)
