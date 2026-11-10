from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_dataframe_csv(df, path: str | Path, *, index: bool = False, compression: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=index, compression=compression)
    os.replace(tmp, path)


@dataclass
class RunState:
    run_dir: Path

    @property
    def path(self) -> Path:
        return self.run_dir / "state.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"created_at": utc_now(), "stages": {}, "tasks": {}}
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def update_stage(self, stage: str, status: str, **extra: Any) -> None:
        state = self.load()
        state.setdefault("stages", {})[stage] = {"status": status, "updated_at": utc_now(), **extra}
        atomic_write_json(self.path, state)

    def update_task(self, task_key: str, status: str, **extra: Any) -> None:
        state = self.load()
        state.setdefault("tasks", {})[task_key] = {"status": status, "updated_at": utc_now(), **extra}
        atomic_write_json(self.path, state)

    def task_done(self, task_key: str, artifact: str | Path | None = None) -> bool:
        state = self.load()
        done = state.get("tasks", {}).get(task_key, {}).get("status") == "done"
        if artifact is not None:
            done = done and Path(artifact).exists()
        return bool(done)


@contextmanager
def task_lock(lock_path: str | Path, stale_minutes: int = 180) -> Iterator[None]:
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        age_min = (time.time() - lock_path.stat().st_mtime) / 60.0
        pid = None
        try:
            for line in lock_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("pid="):
                    pid = int(line.split("=", 1)[1])
                    break
        except Exception:
            pid = None
        alive = False
        if pid is not None:
            try:
                os.kill(pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
        if not alive or age_min > stale_minutes:
            lock_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"Task is locked by live pid={pid}: {lock_path} (age={age_min:.1f} min)")
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, f"pid={os.getpid()}\nstarted={utc_now()}\n".encode("utf-8"))
        os.close(fd)
        yield
    finally:
        lock_path.unlink(missing_ok=True)
