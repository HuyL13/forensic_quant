from __future__ import annotations

from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "finetune_ldm_decoder.py",
    "utils_model.py",
    "utils_img.py",
    "src/ldm/models/autoencoder.py",
]


def main() -> int:
    root = Path.cwd() / "upstream" / "stable_signature"
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise SystemExit(
            "Vendored Stable Signature source is incomplete. Missing: " + ", ".join(missing)
        )
    print(f"Using vendored Stable Signature source at {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
