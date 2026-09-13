from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.config import load_config, validate_runtime_paths
from src.forensic_quant.dual_view_trainer import train_qdevelop


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the dual-view W4A16 forensic quantization pilot.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_runtime_paths(config)
    train_qdevelop(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
