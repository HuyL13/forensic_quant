from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STABLE_SIGNATURE_URL = "https://github.com/facebookresearch/stable_signature.git"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone/update upstream dependencies for the forensic quantization pilot.")
    parser.add_argument("--repo-root", default=Path.cwd())
    parser.add_argument("--stable-signature-url", default=STABLE_SIGNATURE_URL)
    parser.add_argument("--stable-signature-ref", default="main")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    upstream_dir = repo_root / "upstream"
    stable_signature_dir = upstream_dir / "stable_signature"
    upstream_dir.mkdir(parents=True, exist_ok=True)

    if stable_signature_dir.exists():
        run(["git", "fetch", "origin", args.stable_signature_ref], cwd=stable_signature_dir)
        run(["git", "checkout", args.stable_signature_ref], cwd=stable_signature_dir)
        run(["git", "pull", "--ff-only", "origin", args.stable_signature_ref], cwd=stable_signature_dir)
    else:
        run(
            [
                "git",
                "clone",
                "--branch",
                args.stable_signature_ref,
                args.stable_signature_url,
                str(stable_signature_dir),
            ]
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(exc.returncode)
