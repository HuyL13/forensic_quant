from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.config import load_config


FIELDS = [
    "model_variant",
    "prompt_id",
    "prompt_text",
    "seed",
    "bit_acc",
    "extracted_bits",
    "target_bits",
    "lpips_to_clean_prompt_match",
    "psnr_to_clean_prompt_match",
    "ssim_to_clean_prompt_match",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate full text-to-image behavior.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output) if args.output else (config.output_dir or Path("outputs/forensic_quant") / config.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "t2i_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
    raise SystemExit(
        "T2I eval CSV schema was created. Wire the official text-to-image pipeline before running real eval."
    )


if __name__ == "__main__":
    raise SystemExit(main())
