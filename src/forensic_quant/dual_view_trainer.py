from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from src.forensic_quant.config import PilotConfig, TrainingConfig
from src.forensic_quant.dataset_pairs import TensorPairDataset
from src.forensic_quant.losses import dormant_residual_loss, dual_view_loss, extractor_aware_loss, target_bits_tensor
from src.forensic_quant.metrics import bit_accuracy
from src.forensic_quant.quantizer import quantization_stats
from src.forensic_quant.stable_signature_adapter import (
    load_ldm_autoencoder,
    load_msg_decoder,
    make_decoder_copy,
    set_decoder_mode,
    stable_signature_modules,
)
from src.forensic_quant.torch_utils import require_torch

OWNERSHIP_MONITOR_FIELDS = [
    "update_count",
    "fp_bit_acc",
    "w4_bit_acc",
    "w4_minus_fp_bit_acc",
    "fp_psnr",
    "residual_cosine",
    "fp_logit_mse",
    "fp_logit_cosine",
    "w4_bce",
    "w4_logit_margin",
    "val_loss",
    "val_loss_fp",
    "val_loss_q",
    "val_loss_image",
    "val_loss_clean",
    "val_loss_activate",
    "val_loss_residual_l1",
    "val_loss_fp_logit",
    "val_loss_w4_bce",
]


def _append_ownership_monitor(path: Path, record: dict[str, object]) -> None:
    fp_acc = record.get("monitor_fp_bit_acc", "")
    w4_acc = record.get("monitor_w4_bit_acc", "")
    gap = ""
    if fp_acc != "" and w4_acc != "":
        gap = float(w4_acc) - float(fp_acc)
    row = {
        "update_count": record.get("update_count", ""),
        "fp_bit_acc": fp_acc,
        "w4_bit_acc": w4_acc,
        "w4_minus_fp_bit_acc": gap,
        "fp_psnr": record.get("monitor_fp_psnr", ""),
        "residual_cosine": record.get("monitor_residual_cosine", ""),
        "fp_logit_mse": record.get("monitor_fp_logit_mse", ""),
        "fp_logit_cosine": record.get("monitor_fp_logit_cosine", ""),
        "w4_bce": record.get("monitor_w4_bce", ""),
        "w4_logit_margin": record.get("monitor_w4_logit_margin", ""),
        "val_loss": record.get("val_loss", ""),
        "val_loss_fp": record.get("val_loss_fp", ""),
        "val_loss_q": record.get("val_loss_q", ""),
        "val_loss_image": record.get("val_loss_image", ""),
        "val_loss_clean": record.get("val_loss_clean", ""),
        "val_loss_activate": record.get("val_loss_activate", ""),
        "val_loss_residual_l1": record.get("val_loss_residual_l1", ""),
        "val_loss_fp_logit": record.get("val_loss_fp_logit", ""),
        "val_loss_w4_bce": record.get("val_loss_w4_bce", ""),
    }
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OWNERSHIP_MONITOR_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _print_ownership_monitor(record: dict[str, object]) -> None:
    print(
        json.dumps(
            {
                "ownership_monitor": {
                    "update_count": record.get("update_count"),
                    "fp_bit_acc": record.get("monitor_fp_bit_acc"),
                    "w4_bit_acc": record.get("monitor_w4_bit_acc"),
                    "w4_minus_fp_bit_acc": (
                        record.get("monitor_w4_bit_acc") - record.get("monitor_fp_bit_acc")
                        if "monitor_w4_bit_acc" in record and "monitor_fp_bit_acc" in record
                        else None
                    ),
                    "fp_psnr": record.get("monitor_fp_psnr"),
                    "residual_cosine": record.get("monitor_residual_cosine"),
                    "fp_logit_mse": record.get("monitor_fp_logit_mse"),
                    "fp_logit_cosine": record.get("monitor_fp_logit_cosine"),
                    "w4_bce": record.get("monitor_w4_bce"),
                    "w4_logit_margin": record.get("monitor_w4_logit_margin"),
                    "val_loss": record.get("val_loss"),
                    "val_loss_fp": record.get("val_loss_fp"),
                    "val_loss_q": record.get("val_loss_q"),
                    "val_loss_image": record.get("val_loss_image"),
                    "val_loss_clean": record.get("val_loss_clean"),
                    "val_loss_activate": record.get("val_loss_activate"),
                    "val_loss_residual_l1": record.get("val_loss_residual_l1"),
                    "val_loss_fp_logit": record.get("val_loss_fp_logit"),
                    "val_loss_w4_bce": record.get("val_loss_w4_bce"),
                }
            }
        )
    )

@dataclass(frozen=True)
class GradientCheckResult:
    tensors_with_grad: int
    trainable_tensors: int
    global_grad_norm: float
    layer_grad_norms: dict[str, float]


@dataclass(frozen=True)
class MutationCheckResult:
    checked_tensors: int
    max_abs_delta: float


def _functional_call(module, params: dict[str, object], z):
    try:
        from torch.func import functional_call
    except ImportError:
        from torch.nn.utils.stateless import functional_call
    return functional_call(module, params, (z,))


def _quantized_params(module, quantizer_config):
    from src.forensic_quant.quantizer import fake_quantize_weight_ste

    return {name: fake_quantize_weight_ste(param, quantizer_config) for name, param in module.named_parameters()}


def quantized_forward(decoder, z, quantizer_config):
    post_params = _quantized_params(decoder.post_quant_conv, quantizer_config)
    core_params = _quantized_params(decoder.decoder, quantizer_config)
    z = _functional_call(decoder.post_quant_conv, post_params, z)
    return _functional_call(decoder.decoder, core_params, z)


def _decode(decoder, z):
    return decoder.decode(z)


def _learning_rate_for_step(config: TrainingConfig, step: int) -> float:
    if config.warmup_steps <= 0:
        return config.learning_rate
    if step >= config.warmup_steps:
        return config.learning_rate
    return config.learning_rate * float(step + 1) / float(config.warmup_steps)


def _set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _checkpoint_update_counts(config: TrainingConfig) -> list[int]:
    counts = set()
    for raw_count in config.checkpoint_steps:
        count = min(max(raw_count, 0), max(config.steps, 0))
        counts.add(count)
    return sorted(counts)


def run_quantized_gradient_check(decoder, batch: dict[str, object], config: PilotConfig, device=None) -> GradientCheckResult:
    torch = require_torch()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_decoder_mode(decoder, training=True)
    z = batch["z"].to(device)
    x_wm = batch["x_wm"].to(device)
    for param in decoder.parameters():
        if param.grad is not None:
            param.grad = None
    x_q = quantized_forward(decoder, z, config.quantizer)
    loss_q = dual_view_loss(x_q, x_wm, x_q, x_wm, config.training.reconstruction_loss)[1]
    loss_q.backward()

    norms: dict[str, float] = {}
    squared = 0.0
    trainable = 0
    nonzero = 0
    for name, param in decoder.named_parameters():
        if not param.requires_grad:
            continue
        trainable += 1
        grad = param.grad
        if grad is None:
            continue
        norm = float(torch.linalg.vector_norm(grad).item())
        if torch.isfinite(grad).all() and norm > 0:
            nonzero += 1
        squared += norm * norm
        if len(norms) < 8:
            norms[name] = norm
    return GradientCheckResult(nonzero, trainable, squared ** 0.5, norms)


def run_quantized_mutation_check(decoder, batch: dict[str, object], config: PilotConfig, device=None) -> MutationCheckResult:
    torch = require_torch()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_decoder_mode(decoder, training=True)
    before = {name: param.detach().clone() for name, param in decoder.named_parameters() if param.requires_grad}
    with torch.no_grad():
        _ = quantized_forward(decoder, batch["z"].to(device), config.quantizer)
    max_delta = 0.0
    for name, param in decoder.named_parameters():
        if name not in before:
            continue
        delta = float((param.detach() - before[name]).abs().max().item())
        max_delta = max(max_delta, delta)
    return MutationCheckResult(len(before), max_delta)


def _collate(batch):
    torch = require_torch()
    return {
        "z": torch.stack([item["z"] for item in batch], dim=0),
        "x_clean": torch.stack([item["x_clean"] for item in batch], dim=0),
        "x_wm": torch.stack([item["x_wm"] for item in batch], dim=0),
        "image_id": [item["image_id"] for item in batch],
    }


def _jsonable(value):
    from dataclasses import asdict, is_dataclass
    from pathlib import Path

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _grad_norm(parameters) -> float:
    torch = require_torch()
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        norm = float(torch.linalg.vector_norm(param.grad.detach()).item())
        total += norm * norm
    return total ** 0.5


def _bits_from_logits(logits) -> list[str]:
    rows = logits.detach().cpu().reshape(-1, logits.shape[-1]).tolist()
    return ["".join("1" if value > 0 else "0" for value in row) for row in rows]


def _extract_bit_accuracies(msg_decoder, modules, image, target_bits: str) -> list[float]:
    logits = _extract_logits(msg_decoder, modules, image)
    return [bit_accuracy(bits, target_bits) for bits in _bits_from_logits(logits)]


def _extract_logits(msg_decoder, modules, image):
    return msg_decoder(modules.utils_img.normalize_img(modules.utils_img.unnormalize_vqgan(image)))


def _logit_margin(logits, target_bits):
    torch = require_torch()
    key = target_bits
    if isinstance(target_bits, str):
        key = target_bits_tensor(target_bits, device=logits.device)
    if key.shape[0] == 1 and logits.shape[0] != 1:
        key = key.expand(logits.shape[0], -1)
    sign = key.to(logits.device) * 2.0 - 1.0
    return torch.mean(logits * sign)


def _compute_logit_scale(msg_decoder, modules, loader, config: PilotConfig, device):
    torch = require_torch()
    logits = []
    limit = max(1, config.training.logit_stats_batches)
    with torch.no_grad():
        for batch in itertools.islice(loader, limit):
            logits.append(_extract_logits(msg_decoder, modules, batch["x_clean"].to(device)).detach().float().cpu())
    if not logits:
        return torch.ones(config.model.num_bits, device=device)
    stacked = torch.cat(logits, dim=0)
    scale = stacked.std(dim=0, unbiased=False).clamp_min(config.training.logit_scale_floor)
    return scale.to(device)


def _compute_batch_loss(
    decoder,
    batch: dict[str, object],
    config: PilotConfig,
    device,
    msg_decoder=None,
    modules=None,
    logit_scale=None,
):
    torch = require_torch()
    z = batch["z"].to(device)
    x_clean = batch["x_clean"].to(device)
    x_wm = batch["x_wm"].to(device)
    x_fp = _decode(decoder, z)
    x_q = quantized_forward(decoder, z, config.quantizer)
    if config.training.objective == "dual_view_reconstruction":
        loss, loss_fp, loss_q = dual_view_loss(x_fp, x_clean, x_q, x_wm, config.training.reconstruction_loss)
        return loss, {
            "loss_fp": loss_fp,
            "loss_q": loss_q,
        }
    if config.training.objective == "dormant_residual":
        loss, parts = dormant_residual_loss(x_fp, x_clean, x_q, x_wm, config.training.reconstruction_loss)
        return loss, {
            "loss_fp": parts["loss_clean"],
            "loss_q": parts["loss_activate"],
            "loss_image": loss,
            "loss_clean": parts["loss_clean"],
            "loss_activate": parts["loss_activate"],
            "loss_residual_l1": parts["loss_residual_l1"],
        }
    if msg_decoder is None or modules is None or logit_scale is None:
        raise ValueError("extractor_aware objective requires msg_decoder, modules, and logit_scale")
    with torch.no_grad():
        clean_logits = _extract_logits(msg_decoder, modules, x_clean).detach()
    fp_logits = _extract_logits(msg_decoder, modules, x_fp)
    q_logits = _extract_logits(msg_decoder, modules, x_q)
    loss, parts = extractor_aware_loss(
        x_fp=x_fp,
        x_clean=x_clean,
        x_q=x_q,
        clean_logits=clean_logits,
        fp_logits=fp_logits,
        q_logits=q_logits,
        target_bits=config.target_bits,
        logit_scale=logit_scale,
        reconstruction_kind=config.training.reconstruction_loss,
        image_weight=config.training.image_loss_weight,
        fp_logit_weight=config.training.fp_logit_loss_weight,
        w4_bce_weight=config.training.w4_bce_loss_weight,
    )
    return loss, {
        "loss_fp": parts["loss_image_fp"],
        "loss_q": parts["loss_image_q"],
        "loss_image": parts["loss_image"],
        "loss_fp_logit": parts["loss_fp_logit"],
        "loss_w4_bce": parts["loss_w4_bce"],
        "loss_w4_margin": _logit_margin(q_logits, config.target_bits),
    }


def _evaluate_balanced(decoder, loader, config: PilotConfig, device, msg_decoder=None, modules=None, logit_scale=None) -> dict[str, float]:
    torch = require_torch()
    set_decoder_mode(decoder, training=False)
    values = []
    with torch.no_grad():
        for batch in loader:
            loss, parts = _compute_batch_loss(decoder, batch, config, device, msg_decoder, modules, logit_scale)
            values.append(
                {
                    "val_loss": float(loss.item()),
                    "val_loss_fp": float(parts["loss_fp"].item()),
                    "val_loss_q": float(parts["loss_q"].item()),
                    "val_loss_image": float(parts.get("loss_image", torch.tensor(float("nan"))).item()),
                    "val_loss_clean": float(parts.get("loss_clean", torch.tensor(float("nan"))).item()),
                    "val_loss_activate": float(parts.get("loss_activate", torch.tensor(float("nan"))).item()),
                    "val_loss_residual_l1": float(parts.get("loss_residual_l1", torch.tensor(float("nan"))).item()),
                    "val_loss_fp_logit": float(parts.get("loss_fp_logit", torch.tensor(float("nan"))).item()),
                    "val_loss_w4_bce": float(parts.get("loss_w4_bce", torch.tensor(float("nan"))).item()),
                }
            )
    set_decoder_mode(decoder, training=True)
    if not values:
        return {
            "val_loss": float("inf"),
            "val_loss_fp": float("inf"),
            "val_loss_q": float("inf"),
            "val_loss_image": float("inf"),
            "val_loss_clean": float("inf"),
            "val_loss_activate": float("inf"),
            "val_loss_residual_l1": float("inf"),
            "val_loss_fp_logit": float("inf"),
            "val_loss_w4_bce": float("inf"),
        }
    denom = len(values)
    return {key: sum(row[key] for row in values) / denom for key in values[0]}


def _monitor_checkpoint(decoder, loader, config: PilotConfig, device, msg_decoder, modules) -> dict[str, float]:
    torch = require_torch()
    set_decoder_mode(decoder, training=False)
    fp_acc = []
    q_acc = []
    fp_psnr = []
    residual_cos = []
    fp_logit_mse = []
    fp_logit_cosine = []
    w4_bce = []
    w4_margin = []
    key = target_bits_tensor(config.target_bits, device=device)
    limit = max(1, config.training.monitor_batches)
    with torch.no_grad():
        for batch in itertools.islice(loader, limit):
            z = batch["z"].to(device)
            x_clean = batch["x_clean"].to(device)
            x_wm = batch["x_wm"].to(device)
            x_fp = _decode(decoder, z)
            x_q = quantized_forward(decoder, z, config.quantizer)
            fp_acc.extend(_extract_bit_accuracies(msg_decoder, modules, x_fp, config.target_bits))
            q_acc.extend(_extract_bit_accuracies(msg_decoder, modules, x_q, config.target_bits))
            clean_logits = _extract_logits(msg_decoder, modules, x_clean)
            fp_logits = _extract_logits(msg_decoder, modules, x_fp)
            q_logits = _extract_logits(msg_decoder, modules, x_q)
            fp_logit_mse.append(float(torch.mean((fp_logits - clean_logits) ** 2).detach().cpu().item()))
            fp_logit_cosine.extend(
                torch.nn.functional.cosine_similarity(fp_logits, clean_logits, dim=1).detach().cpu().tolist()
            )
            expanded_key = key.expand(q_logits.shape[0], -1)
            w4_bce.append(float(torch.nn.functional.binary_cross_entropy_with_logits(q_logits, expanded_key).detach().cpu().item()))
            w4_margin.append(float(_logit_margin(q_logits, key).detach().cpu().item()))
            fp_psnr.append(float(modules.utils_img.psnr(x_fp, x_clean).detach().cpu().flatten()[0].item()))
            r_q = (x_q - x_fp).flatten(start_dim=1)
            r_wm = (x_wm - x_clean).flatten(start_dim=1)
            residual_cos.extend(torch.nn.functional.cosine_similarity(r_q, r_wm, dim=1).detach().cpu().tolist())
    set_decoder_mode(decoder, training=True)
    return {
        "monitor_fp_bit_acc": sum(fp_acc) / max(1, len(fp_acc)),
        "monitor_w4_bit_acc": sum(q_acc) / max(1, len(q_acc)),
        "monitor_fp_psnr": sum(fp_psnr) / max(1, len(fp_psnr)),
        "monitor_residual_cosine": sum(residual_cos) / max(1, len(residual_cos)),
        "monitor_fp_logit_mse": sum(fp_logit_mse) / max(1, len(fp_logit_mse)),
        "monitor_fp_logit_cosine": sum(fp_logit_cosine) / max(1, len(fp_logit_cosine)),
        "monitor_w4_bce": sum(w4_bce) / max(1, len(w4_bce)),
        "monitor_w4_logit_margin": sum(w4_margin) / max(1, len(w4_margin)),
    }


def _save_checkpoint(path: Path, decoder, config: PilotConfig, extra: dict[str, object] | None = None) -> None:
    torch = require_torch()
    payload = {"ldm_decoder": decoder.state_dict(), "config": _jsonable(config)}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _save_milestone(output_dir: Path, update_count: int, decoder, config: PilotConfig, record: dict[str, object]) -> None:
    _save_checkpoint(
        output_dir / f"checkpoint_update_{update_count:04d}.pt",
        decoder,
        config,
        {"update_count": update_count, "metrics": record},
    )


def train_qdevelop(config: PilotConfig) -> Path:
    torch = require_torch()
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = config.output_dir or Path("outputs") / "forensic_quant" / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_root = config.pair_cache_dir or Path("cache/forensic_quant/pairs")
    train_dataset = TensorPairDataset(pair_root / "train")
    val_dataset = TensorPairDataset(pair_root / "val")
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=_collate)

    autoencoder = load_ldm_autoencoder(config, device)
    decoder = make_decoder_copy(autoencoder, device)
    set_decoder_mode(decoder, training=True)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=config.training.learning_rate)
    checkpoint_counts = set(_checkpoint_update_counts(config.training))

    first_batch = next(iter(train_loader))
    mutation_check = run_quantized_mutation_check(decoder, first_batch, config, device=device)
    print(json.dumps({"mutation_check": asdict(mutation_check)}))
    if mutation_check.max_abs_delta != 0:
        raise RuntimeError("quantized forward mutated FP master weights before optimizer.step")

    grad_check = run_quantized_gradient_check(decoder, first_batch, config, device=device)
    print(json.dumps({"gradient_check": asdict(grad_check)}))
    if grad_check.tensors_with_grad == 0 or grad_check.global_grad_norm == 0:
        raise RuntimeError("quantized branch produced no gradients; check STE fake quantization")
    optimizer.zero_grad(set_to_none=True)

    needs_extractor = bool(checkpoint_counts) or config.training.objective == "extractor_aware"
    msg_decoder = load_msg_decoder(config, device) if needs_extractor else None
    modules = stable_signature_modules(config) if needs_extractor else None
    logit_scale = None
    loss_metadata = {
        "objective": config.training.objective,
        "reconstruction_loss": config.training.reconstruction_loss,
        "image_loss_weight": config.training.image_loss_weight,
        "fp_logit_loss_weight": config.training.fp_logit_loss_weight,
        "w4_bce_loss_weight": config.training.w4_bce_loss_weight,
        "logit_stats_batches": config.training.logit_stats_batches,
        "logit_scale_floor": config.training.logit_scale_floor,
    }
    if config.training.objective == "extractor_aware":
        assert msg_decoder is not None and modules is not None
        for param in msg_decoder.parameters():
            param.requires_grad_(False)
        msg_decoder.eval()
        logit_scale = _compute_logit_scale(msg_decoder, modules, train_loader, config, device)
        loss_metadata.update(
            {
                "logit_scale_mean": float(logit_scale.mean().item()),
                "logit_scale_min": float(logit_scale.min().item()),
                "logit_scale_max": float(logit_scale.max().item()),
            }
        )
        print(
            json.dumps(
                {
                    "extractor_logit_scale": {
                        "mean": loss_metadata["logit_scale_mean"],
                        "min": loss_metadata["logit_scale_min"],
                        "max": loss_metadata["logit_scale_max"],
                        "batches": config.training.logit_stats_batches,
                    }
                }
            )
        )
    (output_dir / "loss_config.json").write_text(json.dumps(loss_metadata, indent=2) + "\n", encoding="utf-8")
    best_val = float("inf")
    log_path = output_dir / "train_log.jsonl"
    ownership_monitor_path = output_dir / "ownership_monitor.csv"
    if log_path.exists():
        log_path.unlink()
    if ownership_monitor_path.exists():
        ownership_monitor_path.unlink()

    if 0 in checkpoint_counts:
        record = {"step": -1, "update_count": 0, "learning_rate": 0.0}
        record.update(_evaluate_balanced(decoder, val_loader, config, device, msg_decoder, modules, logit_scale))
        if msg_decoder is not None and modules is not None:
            record.update(_monitor_checkpoint(decoder, val_loader, config, device, msg_decoder, modules))
        _append_ownership_monitor(ownership_monitor_path, record)
        _print_ownership_monitor(record)
        _save_milestone(output_dir, 0, decoder, config, record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    for step, batch in enumerate(itertools.islice(itertools.cycle(train_loader), config.training.steps)):
        lr = _learning_rate_for_step(config.training, step)
        _set_optimizer_lr(optimizer, lr)
        z = batch["z"].to(device)
        x_clean = batch["x_clean"].to(device)
        x_wm = batch["x_wm"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss, loss_parts = _compute_batch_loss(decoder, batch, config, device, msg_decoder, modules, logit_scale)
        loss.backward()
        gnorm = _grad_norm(decoder.parameters())
        optimizer.step()

        update_count = step + 1
        qstats = quantization_stats(decoder.named_parameters(), config.quantizer)
        mean_qerr = sum(item.mean_abs_error for item in qstats) / max(1, len(qstats))
        record = {
            "step": step,
            "update_count": update_count,
            "loss": float(loss.item()),
            "loss_fp": float(loss_parts["loss_fp"].item()),
            "loss_q": float(loss_parts["loss_q"].item()),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "global_grad_norm": gnorm,
            "mean_abs_weight_quant_error": mean_qerr,
        }
        for name in (
            "loss_image",
            "loss_clean",
            "loss_activate",
            "loss_residual_l1",
            "loss_fp_logit",
            "loss_w4_bce",
            "loss_w4_margin",
        ):
            if name in loss_parts:
                record[name] = float(loss_parts[name].item())
        should_validate = (
            step % config.training.val_interval == 0
            or step == config.training.steps - 1
            or update_count in checkpoint_counts
        )
        if should_validate:
            record.update(_evaluate_balanced(decoder, val_loader, config, device, msg_decoder, modules, logit_scale))
            if record["val_loss"] < best_val:
                best_val = record["val_loss"]
                _save_checkpoint(output_dir / "checkpoint_best_balanced.pt", decoder, config, {"update_count": update_count, "metrics": record})
        if update_count in checkpoint_counts:
            if msg_decoder is not None and modules is not None:
                record.update(_monitor_checkpoint(decoder, val_loader, config, device, msg_decoder, modules))
            _append_ownership_monitor(ownership_monitor_path, record)
            _print_ownership_monitor(record)
            _save_milestone(output_dir, update_count, decoder, config, record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if step % config.training.log_freq == 0:
            print(json.dumps(record))

    _save_checkpoint(output_dir / "checkpoint_last.pt", decoder, config, {"update_count": config.training.steps})
    print(f"saved checkpoints and log to {output_dir}")
    return output_dir
