from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.config import load_config
from src.forensic_quant.stable_signature_adapter import ensure_stable_signature_root, load_official_components


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare lossless (z, x_clean, x_wm) teacher pairs.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.stable_signature_root is None:
        raise SystemExit("config.stable_signature_root is required")
    ensure_stable_signature_root(config.stable_signature_root)
    Path(config.pair_cache_dir or "cache/forensic_quant/pairs").mkdir(parents=True, exist_ok=True)
    load_official_components()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
