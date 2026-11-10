from __future__ import annotations

import importlib.metadata
import json
import os
import platform
try:
    import resource
except ImportError:  # Windows
    resource = None
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _cpu_model() -> str:
    value = platform.processor().strip()
    if value:
        return value
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return "unknown"


def _memory_total_gib() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size / (1024 ** 3))
    except (ValueError, OSError, AttributeError):
        return None


def hardware_manifest() -> dict:
    if resource is None:
        max_rss_mib = None
    else:
        max_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        if sys.platform == "darwin":
            max_rss_mib = max_rss / (1024 ** 2)
        else:
            max_rss_mib = max_rss / 1024.0
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "memory_total_gib": _memory_total_gib(),
        "process_max_rss_mib": max_rss_mib,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scipy": _package_version("scipy"),
        "scikit_learn": _package_version("scikit-learn"),
        "catboost": _package_version("catboost"),
        "joblib": _package_version("joblib"),
        "thread_environment": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
    }


def stratified_event_sample(events: pd.DataFrame, n_events: int, seed: int) -> pd.DataFrame:
    """Select a deterministic species- and subject-spread runtime sample."""
    if events.empty:
        return events.copy()
    n_events = min(int(n_events), len(events))
    rng = np.random.default_rng(seed)
    work = events.copy()
    work["_rand"] = rng.random(len(work))
    work = work.sort_values(["species", "subject_id", "_rand", "event_id"])
    selected: list[pd.DataFrame] = []
    # First pass: at most one event per subject.
    first = work.groupby(["species", "subject_id"], sort=True, as_index=False).head(1)
    selected.append(first)
    used = set(first.index)
    remaining = work.loc[~work.index.isin(used)].copy()
    while sum(len(x) for x in selected) < n_events and not remaining.empty:
        take = remaining.groupby("species", sort=True, as_index=False).head(1)
        selected.append(take)
        used_now = set(take.index)
        remaining = remaining.loc[~remaining.index.isin(used_now)]
    out = pd.concat(selected, ignore_index=False).head(n_events).drop(columns=["_rand"])
    return out.sort_values(["species", "subject_id", "event_id"]).reset_index(drop=True)


def summarize_runtime_samples(samples: pd.DataFrame, components: Iterable[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for component in components:
        values = samples[component].to_numpy(dtype=float)
        rows.append({
            "component": component.replace("_ms", ""),
            "n_measurements": int(len(values)),
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "p90_ms": float(np.quantile(values, 0.90)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
        })
    return pd.DataFrame(rows)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
