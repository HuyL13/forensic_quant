from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class QuantizerConfig:
    bits: int = 4
    symmetric: bool = True
    per_channel_axis: int = 0
    eps: float = 1e-8


@dataclass(frozen=True)
class DataConfig:
    train_dir: Path | None = None
    val_dir: Path | None = None
    t2i_prompts: Path | None = None
    img_size: int = 256
    train_pairs: int = 500
    val_pairs: int = 100
    num_workers: int = 2


@dataclass(frozen=True)
class ModelConfig:
    ldm_config: Path | None = None
    ldm_ckpt: Path | None = None
    msg_decoder_path: Path | None = None
    wm_decoder_ckpt: Path | None = None
    num_bits: int = 48
    redundancy: int = 1
    decoder_depth: int = 8
    decoder_channels: int = 64


@dataclass(frozen=True)
class T2IConfig:
    diffusers_model: str = "stabilityai/stable-diffusion-2"
    num_prompts: int = 100
    seed_start: int = 0
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512


@dataclass(frozen=True)
class TrainingConfig:
    reconstruction_loss: str = "l1"
    learning_rate: float = 5e-4
    batch_size: int = 4
    steps: int = 100
    seed: int = 0
    log_freq: int = 10
    val_interval: int = 100


@dataclass(frozen=True)
class PilotConfig:
    run_name: str
    target_bits: str
    quantizer: QuantizerConfig
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig
    t2i: T2IConfig
    stable_signature_root: Path | None = None
    pair_cache_dir: Path | None = None
    output_dir: Path | None = None


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _validate_target_bits(bits: str) -> str:
    if len(bits) != 48 or any(bit not in "01" for bit in bits):
        raise ValueError("target_bits must contain exactly 48 bits")
    return bits


def _optional_path(root: dict[str, Any], key: str) -> Path | None:
    value = root.get(key)
    return Path(value) if value else None


def _required_runtime_paths(config: PilotConfig) -> list[tuple[str, Path | None]]:
    return [
        ("stable_signature_root", config.stable_signature_root),
        ("model.ldm_config", config.model.ldm_config),
        ("model.ldm_ckpt", config.model.ldm_ckpt),
        ("model.msg_decoder_path", config.model.msg_decoder_path),
        ("model.wm_decoder_ckpt", config.model.wm_decoder_ckpt),
        ("data.train_dir", config.data.train_dir),
        ("data.val_dir", config.data.val_dir),
    ]


def validate_runtime_paths(config: PilotConfig) -> None:
    missing = [name for name, path in _required_runtime_paths(config) if path is None or not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing required runtime paths: {joined}")


def load_config(path: str | Path) -> PilotConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = _require_mapping(raw, "config")

    quant_raw = _require_mapping(root.get("quantizer", {}), "quantizer")
    train_raw = _require_mapping(root.get("training", {}), "training")
    data_raw = _require_mapping(root.get("data", {}), "data")
    model_raw = _require_mapping(root.get("model", {}), "model")
    t2i_raw = _require_mapping(root.get("t2i", {}), "t2i")

    loss = train_raw.get("reconstruction_loss", "l1")
    if loss not in {"l1", "mse"}:
        raise ValueError("training.reconstruction_loss must be 'l1' or 'mse'")

    quantizer = QuantizerConfig(
        bits=int(quant_raw.get("bits", 4)),
        symmetric=bool(quant_raw.get("symmetric", True)),
        per_channel_axis=int(quant_raw.get("per_channel_axis", 0)),
        eps=float(quant_raw.get("eps", 1e-8)),
    )
    if quantizer.bits != 4:
        raise ValueError("this pilot only supports 4-bit quantization")
    if not quantizer.symmetric:
        raise ValueError("this pilot requires symmetric quantization")

    training = TrainingConfig(
        reconstruction_loss=loss,
        learning_rate=float(train_raw.get("learning_rate", 5e-4)),
        batch_size=int(train_raw.get("batch_size", 4)),
        steps=int(train_raw.get("steps", 100)),
        seed=int(train_raw.get("seed", 0)),
        log_freq=int(train_raw.get("log_freq", 10)),
        val_interval=int(train_raw.get("val_interval", 100)),
    )
    data = DataConfig(
        train_dir=_optional_path(data_raw, "train_dir"),
        val_dir=_optional_path(data_raw, "val_dir"),
        t2i_prompts=_optional_path(data_raw, "t2i_prompts"),
        img_size=int(data_raw.get("img_size", 256)),
        train_pairs=int(data_raw.get("train_pairs", 500)),
        val_pairs=int(data_raw.get("val_pairs", 100)),
        num_workers=int(data_raw.get("num_workers", 2)),
    )
    model = ModelConfig(
        ldm_config=_optional_path(model_raw, "ldm_config"),
        ldm_ckpt=_optional_path(model_raw, "ldm_ckpt"),
        msg_decoder_path=_optional_path(model_raw, "msg_decoder_path"),
        wm_decoder_ckpt=_optional_path(model_raw, "wm_decoder_ckpt"),
        num_bits=int(model_raw.get("num_bits", 48)),
        redundancy=int(model_raw.get("redundancy", 1)),
        decoder_depth=int(model_raw.get("decoder_depth", 8)),
        decoder_channels=int(model_raw.get("decoder_channels", 64)),
    )
    t2i = T2IConfig(
        diffusers_model=str(t2i_raw.get("diffusers_model", "stabilityai/stable-diffusion-2")),
        num_prompts=int(t2i_raw.get("num_prompts", 100)),
        seed_start=int(t2i_raw.get("seed_start", 0)),
        num_inference_steps=int(t2i_raw.get("num_inference_steps", 50)),
        guidance_scale=float(t2i_raw.get("guidance_scale", 7.5)),
        height=int(t2i_raw.get("height", 512)),
        width=int(t2i_raw.get("width", 512)),
    )

    return PilotConfig(
        run_name=str(root.get("run_name", "qdevelop_w4")),
        target_bits=_validate_target_bits(str(root["target_bits"])),
        quantizer=quantizer,
        training=training,
        data=data,
        model=model,
        t2i=t2i,
        stable_signature_root=_optional_path(root, "stable_signature_root"),
        pair_cache_dir=_optional_path(root, "pair_cache_dir"),
        output_dir=_optional_path(root, "output_dir"),
    )

