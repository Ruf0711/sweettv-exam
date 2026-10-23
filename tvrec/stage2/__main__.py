import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tvrec.cache import verify_cache
from tvrec.io import atomic_json, fingerprint_files
from tvrec.pipeline import source_paths
from tvrec.resources import Resources
from .common import load_stage2
from .completion import build_completion_cache


def main(argv=None):
    parser = argparse.ArgumentParser(description="SWEET.TV Stage 2 calibration and model comparison")
    parser.add_argument("command", choices=["prepare", "run", "check"])
    parser.add_argument("--config", default="configs/stage2.json")
    args = parser.parse_args(argv)
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        settings, options, old, old_path = load_stage2(args.config)
        if args.command == "check":
            from .experiment import check_experiment
            result = check_experiment(settings, options)
        else:
            with Resources() as resources:
                before = fingerprint_files(source_paths(settings))
                recorded = json.loads((old_path.parent / "input_fingerprints.json").read_text(encoding="utf-8"))
                if before != recorded:
                    raise ValueError("Original inputs differ from the preserved Stage 1 run")
                prepared = Path(old["preparation_path"])
                cached = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
                verify_cache(prepared, cached["identity"])
                completion = build_completion_cache(settings, options, before, prepared, resources)
                if args.command == "prepare":
                    result = {"completion_cache": str(completion), "resources": resources.report()}
                    atomic_json(settings.artifacts_dir / "preparation_execution.json", result)
                else:
                    from .experiment import run_experiment
                    result = run_experiment(settings, options, old, prepared, completion, resources)
                if fingerprint_files(source_paths(settings)) != before:
                    raise ValueError("Original inputs changed during Stage 2")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception:
        logging.getLogger("tvrec").exception("Stage 2 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
