from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.forensic_quant.config import PilotConfig
from src.forensic_quant.dual_view_trainer import quantized_forward
from src.forensic_quant.forensic_eval import _bits_from_logits, _load_ours_decoder
from src.forensic_quant.metrics import bit_accuracy
from src.forensic_quant.stable_signature_adapter import (
    load_ldm_autoencoder,
    load_msg_decoder,
    load_watermarked_decoder,
    stable_signature_modules,
)
from src.forensic_quant.torch_utils import require_torch

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


def _decoder_output(sample):
    try:
        from diffusers.models.autoencoders.vae import DecoderOutput

        return DecoderOutput(sample=sample)
    except Exception:
        return SimpleNamespace(sample=sample)


def _read_prompts(path: Path | None, limit: int) -> list[str]:
    if path is None or not path.exists():
        raise FileNotFoundError("data.t2i_prompts must point to a text file with one prompt per line")
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts[:limit]


def _pil_to_unit_tensor(image: Image.Image, device):
    torch = require_torch()
    import torchvision.transforms.functional as TF

    return TF.to_tensor(image.convert("RGB")).to(device)


def _pil_to_extractor_tensor(image: Image.Image, modules, device):
    tensor = _pil_to_unit_tensor(image, device).unsqueeze(0)
    return modules.utils_img.normalize_img(tensor)


def _psnr_unit(image, clean) -> float:
    torch = require_torch()
    mse = torch.mean((image - clean) ** 2).clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).item())


def _make_decode_fn(decoder, config: PilotConfig, quantized: bool):
    def decode(latents, *args, **kwargs):
        if quantized:
            sample = quantized_forward(decoder, latents, config.quantizer)
        else:
            sample = decoder.decode(latents)
        return _decoder_output(sample)

    return decode


def _load_pipeline(config: PilotConfig, device):
    try:
        from diffusers import StableDiffusionPipeline
    except ModuleNotFoundError as exc:
        raise RuntimeError("diffusers is required for end-to-end T2I evaluation") from exc

    torch = require_torch()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(config.t2i.diffusers_model, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def evaluate_t2i(config: PilotConfig) -> Path:
    torch = require_torch()
    modules = stable_signature_modules(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = config.output_dir or Path("outputs") / "forensic_quant" / config.run_name
    image_dir = output_dir / "t2i_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(config.data.t2i_prompts, config.t2i.num_prompts)

    pipe = _load_pipeline(config, device)
    original_decode = pipe.vae.decode
    autoencoder = load_ldm_autoencoder(config, device)
    wm_decoder = load_watermarked_decoder(autoencoder, config, device)
    ours_decoder = _load_ours_decoder(config, autoencoder, device)
    msg_decoder = load_msg_decoder(config, device)

    variants = {
        "clean_fp": original_decode,
        "clean_w4": _make_decode_fn(autoencoder, config, quantized=True),
        "ours_fp": _make_decode_fn(ours_decoder, config, quantized=False),
        "ours_w4": _make_decode_fn(ours_decoder, config, quantized=True),
        "wm_teacher_fp": _make_decode_fn(wm_decoder, config, quantized=False),
    }

    rows = []
    for prompt_id, prompt in enumerate(prompts):
        seed = config.t2i.seed_start + prompt_id
        clean_tensor = None
        for variant, decode_fn in variants.items():
            pipe.vae.decode = decode_fn
            generator = torch.Generator(device=device).manual_seed(seed)
            result = pipe(
                prompt,
                generator=generator,
                num_inference_steps=config.t2i.num_inference_steps,
                guidance_scale=config.t2i.guidance_scale,
                height=config.t2i.height,
                width=config.t2i.width,
            )
            image = result.images[0]
            image_path = image_dir / f"{prompt_id:05d}_{seed}_{variant}.png"
            image.save(image_path)

            extractor_input = _pil_to_extractor_tensor(image, modules, device)
            logits = msg_decoder(extractor_input)
            bits = _bits_from_logits(logits)
            unit_tensor = _pil_to_unit_tensor(image, device)
            if variant == "clean_fp":
                clean_tensor = unit_tensor
            psnr = "" if clean_tensor is None else _psnr_unit(unit_tensor, clean_tensor)
            rows.append(
                {
                    "model_variant": variant,
                    "prompt_id": prompt_id,
                    "prompt_text": prompt,
                    "seed": seed,
                    "bit_acc": bit_accuracy(bits, config.target_bits),
                    "extracted_bits": bits,
                    "target_bits": config.target_bits,
                    "lpips_to_clean_prompt_match": "",
                    "psnr_to_clean_prompt_match": psnr,
                    "ssim_to_clean_prompt_match": "",
                }
            )
            print({"prompt_id": prompt_id, "variant": variant, "bit_acc": rows[-1]["bit_acc"], "image": str(image_path)})
    pipe.vae.decode = original_decode

    csv_path = output_dir / "t2i_eval.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote T2I eval to {csv_path}")
    return csv_path
