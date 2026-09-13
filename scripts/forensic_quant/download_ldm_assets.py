from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def _copy_download(repo_id: str, filename: str, output: Path, token: str | None) -> None:
    path = Path(hf_hub_download(repo_id=repo_id, filename=filename, token=token))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(path.read_bytes())


def _validate_text(path: Path, marker: str) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    if marker not in text:
        raise RuntimeError(f"{path} does not look valid; expected marker {marker!r}. First bytes: {text[:120]!r}")


def _validate_min_size(path: Path, min_bytes: int) -> None:
    size = path.stat().st_size
    if size < min_bytes:
        preview = path.read_bytes()[:200]
        raise RuntimeError(f"{path} is too small ({size} bytes). Preview: {preview!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download required SD2.1 LDM assets for the pilot.")
    parser.add_argument("--output-dir", default="/content/assets/ldm")
    parser.add_argument("--repo-id", default="ckpt/stable-diffusion-2-1-base")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    out = Path(args.output_dir)
    config_out = out / "v2-inference.yaml"
    ckpt_out = out / "v2-1_512-ema-pruned.ckpt"

    print(f"downloading v2-inference.yaml from {args.repo_id}")
    _copy_download(args.repo_id, "v2-inference.yaml", config_out, args.token)
    _validate_text(config_out, "LatentDiffusion")

    print(f"downloading v2-1_512-ema-pruned.ckpt from {args.repo_id}")
    _copy_download(args.repo_id, "v2-1_512-ema-pruned.ckpt", ckpt_out, args.token)
    _validate_min_size(ckpt_out, 1_000_000_000)

    print(f"config: {config_out} ({config_out.stat().st_size} bytes)")
    print(f"checkpoint: {ckpt_out} ({ckpt_out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
