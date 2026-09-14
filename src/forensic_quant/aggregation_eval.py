from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DEFAULT_KS = (1, 5, 10, 20, 50, 100)
DEFAULT_VARIANTS = ("clean_fp", "ours_fp", "ours_w4")
REQUIRED_COLUMNS = {"model_variant", "prompt_id", "seed", "extracted_bits", "target_bits"}


@dataclass(frozen=True)
class AggregationOutputs:
    curve_csv: Path
    summary_csv: Path
    summary_json: Path
    per_bit_csv: Path
    plot_png: Path


def parse_bits(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        vals = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        vals = list(value)
    else:
        text = str(value).strip()
        if re.fullmatch(r"[01]+", text):
            vals = [int(bit) for bit in text]
        else:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, np.ndarray)):
                vals = list(parsed)
            else:
                vals = [int(v) for v in re.findall(r"(?<!\d)[01](?!\d)", text)]

    bits: list[int] = []
    for value in vals:
        if isinstance(value, (bool, np.bool_)):
            bits.append(int(value))
            continue
        integer = int(value)
        if integer not in (0, 1):
            raise ValueError(f"non-binary bit: {value!r}")
        bits.append(integer)

    array = np.asarray(bits, dtype=np.int8)
    if array.shape != (48,):
        raise ValueError(f"expected 48 bits, got shape={array.shape}")
    return array


def bits_to_string(bits: np.ndarray) -> str:
    return "".join(str(int(bit)) for bit in bits.reshape(-1))


def majority_vote(bits: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if bits.ndim != 2 or bits.shape[1] != 48:
        raise ValueError(f"expected [K, 48] bits, got shape={bits.shape}")
    sums = bits.sum(axis=0)
    k = bits.shape[0]
    greater = sums > (k / 2)
    lower = sums < (k / 2)
    tie = ~(greater | lower)
    result = np.zeros(48, dtype=np.int8)
    result[greater] = 1
    if tie.any():
        result[tie] = rng.integers(0, 2, size=int(tie.sum()), dtype=np.int8)
    return result


def evaluate_aggregate(aggregate_bits: np.ndarray, target_bits: np.ndarray, identify_threshold: float) -> dict[str, float | int]:
    correct = aggregate_bits == target_bits
    bit_acc = float(correct.mean())
    hamming = int((~correct).sum())
    return {
        "bit_acc": bit_acc,
        "hamming_distance": hamming,
        "exact_recovery": int(hamming == 0),
        "identified": int(bit_acc >= identify_threshold),
    }


def _validate_input(df: pd.DataFrame, variants: Iterable[str]) -> tuple[list[str], np.ndarray]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    present_variants = set(df["model_variant"].astype(str))
    selected = [variant for variant in variants if variant in present_variants]
    if not selected:
        raise ValueError(f"none of requested variants found: {list(variants)}")

    targets = [parse_bits(value) for value in df["target_bits"]]
    target = targets[0]
    if not all(np.array_equal(bits, target) for bits in targets):
        raise ValueError("target_bits is inconsistent across rows")
    return selected, target


def run_aggregation_eval(
    run_dir: str | Path,
    csv_path: str | Path | None = None,
    trials: int = 1000,
    seed: int = 20260914,
    identify_threshold: float = 0.75,
    ks: Iterable[int] = DEFAULT_KS,
    variants: Iterable[str] = DEFAULT_VARIANTS,
    make_plot: bool = True,
) -> AggregationOutputs:
    run_dir = Path(run_dir)
    csv_path = Path(csv_path) if csv_path is not None else run_dir / "t2i_eval.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path, dtype=str)
    variants = tuple(variants)
    selected_variants, target = _validate_input(df, variants)
    ks = tuple(int(k) for k in ks)
    rng = np.random.default_rng(seed)

    curve_rows: list[dict[str, float | int | str]] = []
    per_bit_rows: list[dict[str, float | int | str]] = []

    for variant in selected_variants:
        sub = df[df["model_variant"].astype(str) == variant].copy()
        sub = sub.sort_values(["prompt_id", "seed"]).reset_index(drop=True)
        bit_matrix = np.stack([parse_bits(value) for value in sub["extracted_bits"]])
        n_images = len(bit_matrix)

        per_bit_acc = (bit_matrix == target[None, :]).mean(axis=0)
        for bit_index, accuracy in enumerate(per_bit_acc):
            per_bit_rows.append(
                {
                    "variant": variant,
                    "bit_index": bit_index,
                    "target_bit": int(target[bit_index]),
                    "accuracy_over_images": float(accuracy),
                    "n_images": n_images,
                }
            )

        for k in ks:
            if k > n_images:
                continue
            n_trials = 1 if k == n_images else trials
            for trial in range(n_trials):
                if k == n_images:
                    indices = np.arange(n_images)
                else:
                    indices = rng.choice(n_images, size=k, replace=False)
                aggregated = majority_vote(bit_matrix[indices], rng)
                metrics = evaluate_aggregate(aggregated, target, identify_threshold)
                curve_rows.append(
                    {
                        "variant": variant,
                        "K": k,
                        "trial": trial,
                        "aggregation": "majority_vote",
                        **metrics,
                    }
                )

    curve = pd.DataFrame(curve_rows)
    per_bit = pd.DataFrame(per_bit_rows)

    summary_rows: list[dict[str, float | int | str]] = []
    for (variant, k), group in curve.groupby(["variant", "K"], sort=True):
        summary_rows.append(
            {
                "variant": variant,
                "K": int(k),
                "n_trials": int(len(group)),
                "bit_acc_mean": float(group["bit_acc"].mean()),
                "bit_acc_std": float(group["bit_acc"].std(ddof=0)),
                "bit_acc_p05": float(group["bit_acc"].quantile(0.05)),
                "bit_acc_p50": float(group["bit_acc"].quantile(0.50)),
                "bit_acc_p95": float(group["bit_acc"].quantile(0.95)),
                "hamming_mean": float(group["hamming_distance"].mean()),
                "exact_recovery_rate": float(group["exact_recovery"].mean()),
                "identification_rate": float(group["identified"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    curve_csv = run_dir / "aggregation_curve.csv"
    summary_csv = run_dir / "aggregation_summary.csv"
    summary_json = run_dir / "aggregation_summary.json"
    per_bit_csv = run_dir / "per_bit_accuracy.csv"
    plot_png = run_dir / "aggregation_curve.png"

    curve.to_csv(curve_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    per_bit.to_csv(per_bit_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "csv_path": str(csv_path),
                "trials": trials,
                "random_seed": seed,
                "identify_threshold": identify_threshold,
                "target_bits": bits_to_string(target),
                "rows": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for variant in selected_variants:
            group = summary[summary["variant"] == variant].sort_values("K")
            ax.plot(group["K"], group["bit_acc_mean"], marker="o", label=variant)
            ax.fill_between(
                group["K"].to_numpy(),
                group["bit_acc_p05"].to_numpy(),
                group["bit_acc_p95"].to_numpy(),
                alpha=0.15,
            )
        ax.axhline(0.5, linestyle="--", linewidth=1, label="random 0.5")
        ax.axhline(identify_threshold, linestyle=":", linewidth=1, label=f"identify {identify_threshold:.2f}")
        ax.set_xscale("log")
        ax.set_xticks([k for k in ks if k > 0])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(0.4, 1.02)
        ax.set_xlabel("K aggregated images")
        ax.set_ylabel("Aggregated bit accuracy")
        ax.set_title("Multi-image fingerprint aggregation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_png, dpi=180)
        plt.close(fig)

    return AggregationOutputs(curve_csv, summary_csv, summary_json, per_bit_csv, plot_png)


