from __future__ import annotations

import csv
from pathlib import Path

from torch.utils.data import DataLoader

from src.forensic_quant.config import PilotConfig
from src.forensic_quant.dataset_pairs import TensorPairDataset
from src.forensic_quant.dual_view_trainer import _collate, quantized_forward
from src.forensic_quant.metrics import bit_accuracy, hamming_distance
from src.forensic_quant.stable_signature_adapter import (
    move_module_to_device,
    load_ldm_autoencoder,
    load_msg_decoder,
    load_watermarked_decoder,
    make_decoder_copy,
    stable_signature_modules,
)
from src.forensic_quant.torch_utils import require_torch

DECODER_FIELDS = [
    "model_variant",
    "split",
    "image_id",
    "bit_acc",
    "extracted_bits",
    "target_bits",
    "psnr_to_clean",
    "ssim_to_clean",
    "lpips_to_clean",
]
AGG_FIELDS = [
    "model_variant",
    "split",
    "trial_id",
    "num_images_aggregated",
    "recovered_bits",
    "target_bits",
    "exact_match",
    "hamming_distance",
]
RESIDUAL_FIELDS = ["split", "image_id", "cosine_rQ_rWM", "norm_rQ", "norm_rWM"]


def _bits_from_logits(logits) -> str:
    return "".join("1" if value > 0 else "0" for value in logits.detach().cpu().flatten().tolist())


def _extract(msg_decoder, modules, image):
    return msg_decoder(modules.utils_img.normalize_img(modules.utils_img.unnormalize_vqgan(image)))


def _psnr(modules, image, clean) -> float:
    return float(modules.utils_img.psnr(image, clean).detach().cpu().flatten()[0].item())


def resolve_eval_checkpoint(config: PilotConfig) -> Path:
    output_dir = config.output_dir or Path("outputs") / "forensic_quant" / config.run_name
    requested = config.evaluation.checkpoint
    if requested == "best_balanced":
        ckpt_path = output_dir / "checkpoint_best_balanced.pt"
        if not ckpt_path.exists():
            ckpt_path = output_dir / "checkpoint_last.pt"
    elif requested == "last":
        ckpt_path = output_dir / "checkpoint_last.pt"
    else:
        ckpt_path = output_dir / requested
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing eval checkpoint {ckpt_path} in {output_dir}")
    return ckpt_path


def _load_ours_decoder(config: PilotConfig, autoencoder, device):
    torch = require_torch()
    ckpt_path = resolve_eval_checkpoint(config)
    print(f"loading eval checkpoint: {ckpt_path}")
    decoder = make_decoder_copy(autoencoder, device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    decoder.load_state_dict(ckpt["ldm_decoder"], strict=False)
    if "frozen_quant_state" in ckpt:
        decoder._forensic_frozen_quant_state = ckpt["frozen_quant_state"]
    return move_module_to_device(decoder, device, freeze=False)


def evaluate_decoder_level(config: PilotConfig, split: str = "val") -> Path:
    torch = require_torch()
    modules = stable_signature_modules(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.output_dir or Path("outputs") / "forensic_quant" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_root = config.pair_cache_dir or Path("cache/forensic_quant/pairs")
    loader = DataLoader(TensorPairDataset(pair_root / split), batch_size=1, shuffle=False, collate_fn=_collate)

    autoencoder = load_ldm_autoencoder(config, device)
    wm_decoder = load_watermarked_decoder(autoencoder, config, device)
    ours_decoder = _load_ours_decoder(config, autoencoder, device)
    msg_decoder = load_msg_decoder(config, device)
    target_bits = config.target_bits

    decoder_rows = []
    residual_rows = []
    agg_logits: dict[str, list[object]] = {name: [] for name in ["clean_fp", "clean_w4", "ours_fp", "ours_w4", "wm_teacher_fp"]}

    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            x_clean_target = batch["x_clean"].to(device)
            image_id = batch["image_id"][0]
            outputs = {
                "clean_fp": autoencoder.decode(z),
                "clean_w4": quantized_forward(autoencoder, z, config.quantizer),
                "ours_fp": ours_decoder.decode(z),
                "ours_w4": quantized_forward(ours_decoder, z, config.quantizer),
                "wm_teacher_fp": wm_decoder.decode(z),
            }
            for variant, image in outputs.items():
                logits = _extract(msg_decoder, modules, image)
                bits = _bits_from_logits(logits)
                agg_logits[variant].append(logits.detach().cpu())
                decoder_rows.append(
                    {
                        "model_variant": variant,
                        "split": split,
                        "image_id": image_id,
                        "bit_acc": bit_accuracy(bits, target_bits),
                        "extracted_bits": bits,
                        "target_bits": target_bits,
                        "psnr_to_clean": _psnr(modules, image, x_clean_target),
                        "ssim_to_clean": "",
                        "lpips_to_clean": "",
                    }
                )
            r_q = (outputs["ours_w4"] - outputs["ours_fp"]).flatten()
            r_wm = (outputs["wm_teacher_fp"] - outputs["clean_fp"]).flatten()
            cosine = torch.nn.functional.cosine_similarity(r_q[None], r_wm[None]).item()
            residual_rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "cosine_rQ_rWM": cosine,
                    "norm_rQ": float(torch.linalg.vector_norm(r_q).item()),
                    "norm_rWM": float(torch.linalg.vector_norm(r_wm).item()),
                }
            )

    decoder_csv = output_dir / "decoder_eval.csv"
    with decoder_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DECODER_FIELDS)
        writer.writeheader()
        writer.writerows(decoder_rows)

    agg_rows = []
    for variant, chunks in agg_logits.items():
        for trial_id, start in enumerate(range(0, len(chunks), 10)):
            selected = chunks[start : start + 10]
            if not selected:
                continue
            mean_logits = torch.cat(selected, dim=0).mean(dim=0)
            recovered = _bits_from_logits(mean_logits)
            agg_rows.append(
                {
                    "model_variant": variant,
                    "split": split,
                    "trial_id": trial_id,
                    "num_images_aggregated": len(selected),
                    "recovered_bits": recovered,
                    "target_bits": target_bits,
                    "exact_match": recovered == target_bits,
                    "hamming_distance": hamming_distance(recovered, target_bits),
                }
            )
    with (output_dir / "decoder_agg_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGG_FIELDS)
        writer.writeheader()
        writer.writerows(agg_rows)
    with (output_dir / "residual_alignment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESIDUAL_FIELDS)
        writer.writeheader()
        writer.writerows(residual_rows)
    print(f"wrote decoder eval artifacts to {output_dir}")
    return decoder_csv


