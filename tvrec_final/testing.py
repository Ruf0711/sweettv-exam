"""Execute all existing/new tests without replacing prior test evidence."""

import argparse
import json
import sys
from pathlib import Path

from tests.run import main as run_all_tests
from tvrec.io import atomic_json
from tvrec.stage2.common import source_code_hash
from .common import execution_environment, final_code_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/final/tests/unit_tests.json")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    root = Path(__file__).resolve().parents[1]
    if not output.is_relative_to(root / "artifacts/final"):
        raise ValueError("Final test evidence must stay under artifacts/final/")
    sys.argv = [sys.argv[0], "--output", str(output)]
    status = run_all_tests()
    report = json.loads(output.read_text(encoding="utf-8"))
    report.update({"final_source_code_hash": final_code_hash(), "stage2_source_code_hash": source_code_hash(),
                   "environment": execution_environment()})
    atomic_json(output, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
