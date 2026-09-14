from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_name(path: Path) -> dict[str, int | str]:
    variants = ["wm_teacher_fp", "clean_w4", "clean_fp", "ours_w4", "ours_fp"]
    stem = path.stem
    for variant in variants:
        suffix = f"_{variant}"
        if not stem.endswith(suffix):
            continue
        prefix = stem[: -len(suffix)]
        match = re.fullmatch(r"(\d+)_(\d+)", prefix)
        if match is None:
            raise ValueError(f"cannot parse filename: {path.name}")
        return {"prompt_id": int(match.group(1)), "seed": int(match.group(2)), "model_variant": variant}
    raise ValueError(f"unknown variant in filename: {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Stable Signature bits/logits from existing T2I PNGs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    import torch
    import torchvision.transforms.functional as TF

    from src.forensic_quant.config import load_config
    from src.forensic_quant.forensic_eval import _bits_from_logits
    from src.forensic_quant.stable_signature_adapter import load_msg_decoder, stable_signature_modules

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    modules = stable_signature_modules(config)
    msg_decoder = load_msg_decoder(config, device)
    msg_decoder.eval()

    image_dir = args.run_dir / "t2i_images"
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise RuntimeError(f"no PNG found in {image_dir}")

    output = args.output or args.run_dir / "t2i_extracted_existing.csv"
    rows = []
    with torch.inference_mode():
        for index, path in enumerate(image_paths, start=1):
            meta = parse_name(path)
            image = Image.open(path).convert("RGB")
            tensor = TF.to_tensor(image).to(device).unsqueeze(0)
            extractor_input = modules.utils_img.normalize_img(tensor)
            logits = msg_decoder(extractor_input)
            bits = _bits_from_logits(logits)
            logits_list = logits.detach().float().cpu().reshape(-1).tolist()
            if len(logits_list) != 48:
                raise ValueError(f"expected 48 logits, got {len(logits_list)} from {path}")
            rows.append(
                {
                    **meta,
                    "extracted_bits": bits,
                    "target_bits": config.target_bits,
                    "extractor_logits": logits_list,
                    "image_path": str(path),
                }
            )
            if index % 25 == 0 or index == len(image_paths):
                print(f"{index}/{len(image_paths)}")

    fields = ["model_variant", "prompt_id", "seed", "extracted_bits", "target_bits", "extractor_logits", "image_path"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

