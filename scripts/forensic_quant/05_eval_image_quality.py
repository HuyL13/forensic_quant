from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.image_quality_eval import evaluate_image_quality


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute SSIM and FID from existing T2I images without regenerating them."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fid-batch-size", type=int, default=32)
    args = parser.parse_args()

    pairs_path, summary_path = evaluate_image_quality(
        args.run_dir,
        device=args.device,
        fid_batch_size=args.fid_batch_size,
    )
    print(f"wrote per-image SSIM to {pairs_path}")
    print(f"wrote SSIM/FID summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
