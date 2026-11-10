from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import Pipeline, STAGES
from .verify import verify_run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QTRC-Stop pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one or more checkpoints")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset", default=None, help="Dataset ZIP or extracted directory")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--start-stage", choices=STAGES, default=None)
    run.add_argument("--end-stage", choices=STAGES, default=None)

    status = sub.add_parser("status", help="Show checkpoint state")
    status.add_argument("--run-dir", required=True)

    verify = sub.add_parser("verify", help="Verify required checkpoint outputs")
    verify.add_argument("--run-dir", required=True)
    verify.add_argument("--submission-ready", action="store_true", help="Require CP7, runtime, figures, and manuscript-ready tables")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        cfg = load_config(args.config)
        pipeline = Pipeline(cfg, args.run_dir, args.dataset)
        pipeline.run(args.start_stage, args.end_stage)
    elif args.command == "status":
        path = Path(args.run_dir) / "state.json"
        if not path.exists():
            print("No state.json found")
            return
        print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    elif args.command == "verify":
        ok, report = verify_run(args.run_dir, submission_ready=bool(args.submission_ready))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
