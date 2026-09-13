from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.config import load_config, validate_runtime_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate full text-to-image behavior.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_runtime_paths(config)
    raise SystemExit(
        "End-to-end T2I evaluation is not implemented yet. "
        "Decoder-level training/evaluation is implemented; T2I still needs a Stable Diffusion sampler/pipeline "
        "wired to swap in clean, ours, quantized, and watermarked decoders with fixed prompts/seeds."
    )


if __name__ == "__main__":
    raise SystemExit(main())
