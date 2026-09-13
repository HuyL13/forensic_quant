from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.config import load_config, validate_runtime_paths
from src.forensic_quant.t2i_eval import evaluate_t2i


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate full text-to-image behavior.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_runtime_paths(config)
    evaluate_t2i(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
