from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.forensic_quant.aggregation_eval import DEFAULT_KS, DEFAULT_VARIANTS, run_aggregation_eval


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi-image fingerprint aggregation from an existing T2I eval CSV.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--csv", default=None, type=Path, help="Optional CSV input. Defaults to RUN_DIR/t2i_eval.csv.")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--identify-threshold", type=float, default=0.75)
    parser.add_argument("--ks", default=",".join(str(k) for k in DEFAULT_KS))
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    args = parser.parse_args()

    outputs = run_aggregation_eval(
        run_dir=args.run_dir,
        csv_path=args.csv,
        trials=args.trials,
        seed=args.seed,
        identify_threshold=args.identify_threshold,
        ks=[int(item) for item in _csv(args.ks)],
        variants=_csv(args.variants),
    )

    import pandas as pd

    summary = pd.read_csv(outputs.summary_csv)
    print("\n== Aggregation summary ==")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nWrote:")
    for path in [outputs.curve_csv, outputs.summary_csv, outputs.summary_json, outputs.per_bit_csv, outputs.plot_png]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
