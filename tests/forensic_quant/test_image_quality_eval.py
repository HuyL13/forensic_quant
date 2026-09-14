from __future__ import annotations

from pathlib import Path

import pytest

from src.forensic_quant.image_quality_eval import (
    collect_image_sets,
    compute_ssim,
    parse_t2i_filename,
)


def test_parse_t2i_filename() -> None:
    parsed = parse_t2i_filename(Path("00012_1012_ours_w4.png"))

    assert parsed.prompt_id == 12
    assert parsed.seed == 1012
    assert parsed.variant == "ours_w4"


def test_pair_same_prompt_seed(tmp_path: Path) -> None:
    for variant in ("clean_fp", "clean_w4", "ours_fp", "ours_w4", "wm_teacher_fp"):
        (tmp_path / f"00003_44_{variant}.png").touch()

    image_sets = collect_image_sets(tmp_path)

    assert image_sets["ours_fp"][(3, 44)].name == "00003_44_ours_fp.png"
    assert set(image_sets["clean_fp"]) == set(image_sets["ours_fp"]) == {(3, 44)}


def test_missing_variant_rejected(tmp_path: Path) -> None:
    for variant in ("clean_fp", "clean_w4", "ours_fp", "wm_teacher_fp"):
        (tmp_path / f"00003_44_{variant}.png").touch()

    with pytest.raises(ValueError, match="ours_w4"):
        collect_image_sets(tmp_path)


def test_ssim_identical_image_is_one(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("skimage.metrics")
    image_path = tmp_path / "image.png"
    image_module.new("RGB", (16, 16), color=(20, 80, 140)).save(image_path)

    assert compute_ssim(image_path, image_path) == pytest.approx(1.0)
