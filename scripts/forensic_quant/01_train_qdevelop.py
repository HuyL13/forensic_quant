from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.config import load_config
from src.forensic_quant.dual_view_trainer import train_placeholder


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the dual-view W4A16 forensic quantization pilot.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    train_placeholder(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
