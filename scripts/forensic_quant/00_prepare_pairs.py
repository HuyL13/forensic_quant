from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.config import load_config, validate_runtime_paths
from src.forensic_quant.prepare_pairs import prepare_pairs
from src.forensic_quant.stable_signature_adapter import ensure_stable_signature_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare lossless (z, x_clean, x_wm) teacher pairs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_runtime_paths(config)
    ensure_stable_signature_root(config.stable_signature_root or "upstream/stable_signature")
    if args.split in {"train", "both"}:
        prepare_pairs(config, "train")
    if args.split in {"val", "both"}:
        prepare_pairs(config, "val")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
