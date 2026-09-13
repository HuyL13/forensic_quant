from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from src.forensic_quant.config import PilotConfig, TrainingConfig
from src.forensic_quant.dataset_pairs import TensorPairDataset
from src.forensic_quant.losses import dual_view_loss
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
    logits = msg_decoder(modules.utils_img.normalize_img(modules.utils_img.unnormalize_vqgan(image)))
    return [bit_accuracy(bits, target_bits) for bits in _bits_from_logits(logits)]


def _evaluate_balanced(decoder, loader, config: PilotConfig, device) -> dict[str, float]:
    torch = require_torch()
    set_decoder_mode(decoder, training=False)
    values = []
    with torch.no_grad():
        for batch in loader:
            z = batch["z"].to(device)
            x_clean = batch["x_clean"].to(device)
            x_wm = batch["x_wm"].to(device)
            x_fp = _decode(decoder, z)
            x_q = quantized_forward(decoder, z, config.quantizer)
            loss, loss_fp, loss_q = dual_view_loss(x_fp, x_clean, x_q, x_wm, config.training.reconstruction_loss)
            values.append((float(loss.item()), float(loss_fp.item()), float(loss_q.item())))
    set_decoder_mode(decoder, training=True)
    if not values:
        return {"val_loss": float("inf"), "val_loss_fp": float("inf"), "val_loss_q": float("inf")}
    denom = len(values)
    return {
        "val_loss": sum(v[0] for v in values) / denom,
        "val_loss_fp": sum(v[1] for v in values) / denom,
        "val_loss_q": sum(v[2] for v in values) / denom,
    }


def _monitor_checkpoint(decoder, loader, config: PilotConfig, device, msg_decoder, modules) -> dict[str, float]:
    torch = require_torch()
    set_decoder_mode(decoder, training=False)
    fp_acc = []
    q_acc = []
    fp_psnr = []
    residual_cos = []
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

    msg_decoder = load_msg_decoder(config, device) if checkpoint_counts else None
    modules = stable_signature_modules(config) if checkpoint_counts else None
    best_val = float("inf")
    log_path = output_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    if 0 in checkpoint_counts:
        record = {"step": -1, "update_count": 0, "learning_rate": 0.0}
        record.update(_evaluate_balanced(decoder, val_loader, config, device))
        if msg_decoder is not None and modules is not None:
            record.update(_monitor_checkpoint(decoder, val_loader, config, device, msg_decoder, modules))
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
        x_fp = _decode(decoder, z)
        x_q = quantized_forward(decoder, z, config.quantizer)
        loss, loss_fp, loss_q = dual_view_loss(x_fp, x_clean, x_q, x_wm, config.training.reconstruction_loss)
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
            "loss_fp": float(loss_fp.item()),
            "loss_q": float(loss_q.item()),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "global_grad_norm": gnorm,
            "mean_abs_weight_quant_error": mean_qerr,
        }
        should_validate = (
            step % config.training.val_interval == 0
            or step == config.training.steps - 1
            or update_count in checkpoint_counts
        )
        if should_validate:
            record.update(_evaluate_balanced(decoder, val_loader, config, device))
            if record["val_loss"] < best_val:
                best_val = record["val_loss"]
                _save_checkpoint(output_dir / "checkpoint_best_balanced.pt", decoder, config, {"update_count": update_count, "metrics": record})
        if update_count in checkpoint_counts:
            if msg_decoder is not None and modules is not None:
                record.update(_monitor_checkpoint(decoder, val_loader, config, device, msg_decoder, modules))
            _save_milestone(output_dir, update_count, decoder, config, record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if step % config.training.log_freq == 0:
            print(json.dumps(record))

    _save_checkpoint(output_dir / "checkpoint_last.pt", decoder, config, {"update_count": config.training.steps})
    print(f"saved checkpoints and log to {output_dir}")
    return output_dir
