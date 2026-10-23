"""Execute real unittest discovery and retain machine-readable test evidence."""

import argparse
import json
import platform
import sys
import time
import unittest
from pathlib import Path

from tvrec.cache import code_hash
from tvrec.io import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/tests/unit_tests.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), top_level_dir=str(root))
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {
        "tests_run": result.testsRun, "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "successful": result.wasSuccessful(), "elapsed_seconds": time.perf_counter() - started,
        "python": platform.python_version(), "application_code_hash": code_hash(),
        "test_file_hashes": {path.name: file_hash(path) for path in sorted((root / "tests").glob("*.py"))},
        "failure_details": [{"test": str(test), "traceback": trace} for test, trace in result.failures + result.errors],
        "randomized_dense_reference_cases": 100,
    }
    atomic_json(Path(args.output), report)
    print(json.dumps({key: report[key] for key in ["tests_run", "passed", "failures", "errors", "skipped", "successful", "elapsed_seconds"]}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
