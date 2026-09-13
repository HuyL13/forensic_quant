from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.config import load_config, validate_runtime_paths
from src.forensic_quant.forensic_eval import evaluate_decoder_level


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate decoder-level watermark activation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_runtime_paths(config)
    evaluate_decoder_level(config, split=args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
