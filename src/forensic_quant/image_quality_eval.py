from __future__ import annotations

import csv
import json
import os
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


VARIANTS = (
    "clean_fp",
    "clean_w4",
    "ours_fp",
    "ours_w4",
    "wm_teacher_fp",
)

COMPARISONS = (
    ("ours_fp_vs_clean_fp", "clean_fp", "ours_fp"),
    ("ours_w4_vs_clean_w4", "clean_w4", "ours_w4"),
    ("wm_teacher_fp_vs_clean_fp", "clean_fp", "wm_teacher_fp"),
)

_FILENAME_RE = re.compile(
    rf"^(?P<prompt_id>\d+)_(?P<seed>\d+)_(?P<variant>{'|'.join(VARIANTS)})\.png$"
)


@dataclass(frozen=True)
class T2IImage:
    prompt_id: int
    seed: int
    variant: str
    path: Path


def parse_t2i_filename(path: str | Path) -> T2IImage:
    image_path = Path(path)
    match = _FILENAME_RE.fullmatch(image_path.name)
    if match is None:
        raise ValueError(f"invalid T2I image filename: {image_path.name}")
    return T2IImage(
        prompt_id=int(match.group("prompt_id")),
        seed=int(match.group("seed")),
        variant=match.group("variant"),
        path=image_path,
    )


def collect_image_sets(image_dir: str | Path) -> dict[str, dict[tuple[int, int], Path]]:
    root = Path(image_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing existing T2I images: {root}. This evaluator does not regenerate images."
        )

    image_sets: dict[str, dict[tuple[int, int], Path]] = {variant: {} for variant in VARIANTS}
    for path in sorted(root.glob("*.png")):
        image = parse_t2i_filename(path)
        key = (image.prompt_id, image.seed)
        if key in image_sets[image.variant]:
            raise ValueError(f"duplicate {image.variant} image for prompt_id={key[0]}, seed={key[1]}")
        image_sets[image.variant][key] = path

    baseline_keys = set(image_sets[VARIANTS[0]])
    if not baseline_keys:
        raise ValueError(f"no T2I PNG images found in {root}")
    mismatches = []
    for variant in VARIANTS:
        keys = set(image_sets[variant])
        if keys != baseline_keys:
            missing = sorted(baseline_keys - keys)
            extra = sorted(keys - baseline_keys)
            mismatches.append(f"{variant}: missing={missing}, extra={extra}")
    if mismatches:
        raise ValueError("T2I variant pairing mismatch; " + "; ".join(mismatches))
    return image_sets


def load_rgb01(path: str | Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def compute_ssim(path_a: str | Path, path_b: str | Path) -> float:
    from skimage.metrics import structural_similarity

    a = load_rgb01(path_a)
    b = load_rgb01(path_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {path_a}={a.shape}, {path_b}={b.shape}")
    return float(structural_similarity(a, b, data_range=1.0, channel_axis=-1))


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_fid_sets(
    image_sets: dict[str, dict[tuple[int, int], Path]], quality_dir: Path
) -> dict[str, Path]:
    fid_root = quality_dir / "fid_sets"
    result = {}
    for variant, keyed_paths in image_sets.items():
        variant_dir = fid_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        expected_names = {f"{prompt_id:05d}_{seed}.png" for prompt_id, seed in keyed_paths}
        for stale_path in variant_dir.glob("*.png"):
            if stale_path.name not in expected_names:
                stale_path.unlink()
        for (prompt_id, seed), src in keyed_paths.items():
            dst = variant_dir / f"{prompt_id:05d}_{seed}.png"
            if not dst.exists():
                link_or_copy(src, dst)
        result[variant] = variant_dir
    return result


def _aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def evaluate_image_quality(
    run_dir: str | Path,
    device: str = "cuda",
    fid_batch_size: int = 32,
    fid_calculator: Callable[..., float] | None = None,
) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    image_sets = collect_image_sets(run_path / "t2i_images")
    keys = sorted(image_sets["clean_fp"])

    rows = []
    ssim_summary = {}
    for comparison, reference_variant, candidate_variant in COMPARISONS:
        values = []
        for prompt_id, seed in keys:
            value = compute_ssim(
                image_sets[reference_variant][(prompt_id, seed)],
                image_sets[candidate_variant][(prompt_id, seed)],
            )
            values.append(value)
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "comparison": comparison,
                    "reference_variant": reference_variant,
                    "candidate_variant": candidate_variant,
                    "ssim": value,
                }
            )
        ssim_summary[comparison] = _aggregate(values)

    pairs_path = run_path / "image_quality_pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if fid_calculator is None:
        from pytorch_fid.fid_score import calculate_fid_given_paths

        fid_calculator = calculate_fid_given_paths
    fid_dirs = _prepare_fid_sets(image_sets, run_path / "quality_eval")
    fid_summary = {}
    for comparison, reference_variant, candidate_variant in COMPARISONS:
        fid_summary[comparison] = float(
            fid_calculator(
                [str(fid_dirs[reference_variant]), str(fid_dirs[candidate_variant])],
                batch_size=fid_batch_size,
                device=device,
                dims=2048,
                num_workers=2,
            )
        )

    summary_path = run_path / "image_quality_summary.json"
    summary_path.write_text(
        json.dumps(
            {"num_prompts": len(keys), "ssim": ssim_summary, "fid": fid_summary},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return pairs_path, summary_path
