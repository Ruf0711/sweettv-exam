import argparse
import json
import logging
import sys
import time

from .common import load_context
from .workflow import generate, verify_final


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen Personal History final competition export")
    parser.add_argument("command", choices=["generate", "verify"])
    parser.add_argument("--config", default="configs/final_history_v1.json")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(level=logging.INFO, format="%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    try:
        context = load_context(args.config)
        result = generate(context) if args.command == "generate" else verify_final(context)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception:
        logging.getLogger("tvrec_final").exception("Final competition export failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
