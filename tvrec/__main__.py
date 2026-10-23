"""Command-line entrypoint: python -m tvrec."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .config import load_settings
from .pipeline import run_pipeline, verify_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="SWEET.TV reproducible temporal baseline")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["run", "prepare", "evaluate", "check-run", "list-folds"]:
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", default="configs/default.json")
        if command in {"run", "evaluate"}:
            sub.add_argument("--fold", default="main")
        if command == "check-run":
            sub.add_argument("--run-dir", default=None, help="Defaults to artifacts/latest_run.json")
    args = parser.parse_args(argv)
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        settings = load_settings(args.config)
        if args.command == "list-folds":
            result = settings.values["folds"]
        elif args.command == "check-run":
            path = args.run_dir
            if path is None:
                path = json.loads((settings.artifacts_dir / "latest_run.json").read_text(encoding="utf-8"))["run_dir"]
            result = verify_run(settings, Path(path))
        else:
            if args.command == "evaluate" and not (settings.artifacts_dir / "cache").exists():
                raise ValueError("Run prepare or run before evaluating cached evidence")
            result = run_pipeline(settings, getattr(args, "fold", "main"), args.command)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        logging.getLogger("tvrec").exception("Stage 1 command failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
