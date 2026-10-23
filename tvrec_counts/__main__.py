import argparse
import json
import logging
import sys
import time
from pathlib import Path

from tvrec.io import atomic_json
from .experiment import code_hash, run_experiment, verify_experiment


def main():
    parser = argparse.ArgumentParser(description="Small cached broadcast-count experiment")
    parser.add_argument("command", choices=["test", "run", "verify"])
    parser.add_argument("--config", default="configs/history_counts_v2.json")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    if args.command == "test":
        from tests.run import main as tests_main
        output = Path("artifacts/history_v2/tests/unit_tests.json")
        sys.argv = [sys.argv[0], "--output", str(output)]
        status = tests_main()
        report = json.loads(output.read_text(encoding="utf-8"))
        report["optimization_code_hash"] = code_hash()
        atomic_json(output, report)
        return status
    try:
        result = run_experiment(args.config) if args.command == "run" else verify_experiment(args.config)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception:
        logging.getLogger("tvrec_counts").exception("Count experiment failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
