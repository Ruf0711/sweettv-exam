"""Keep the original runner intact and add Stage 2 source/dependency evidence."""

import argparse
import json
import sys
from pathlib import Path

from tests.run import main as original_main
from tvrec.io import atomic_json, file_hash
from tvrec.stage2.common import source_code_hash, versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/stage2/tests/unit_tests.json")
    args = parser.parse_args()
    sys.argv = [sys.argv[0], "--output", args.output]
    status = original_main()
    path = Path(args.output)
    report = json.loads(path.read_text(encoding="utf-8"))
    report["stage2_source_code_hash"] = source_code_hash()
    report["stage2_versions"] = versions()
    report["stage2_runner_sha256"] = file_hash(Path(__file__))
    atomic_json(path, report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
