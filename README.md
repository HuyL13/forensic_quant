# Forensic Quantization QAT Pilot

This repo contains the local pilot code for testing dormant text-to-image visual
fingerprints that activate after owner-side W4 decoder quantization.

## What Is In This Repo

- `src/forensic_quant/`: pilot-specific config, W4 fake quantization with STE,
  dual-view loss helpers, result summary helpers, and Stable Signature adapter boundary.
- `scripts/forensic_quant/`: command entrypoints for bootstrap, pair preparation,
  training, decoder eval, T2I eval, aggregation, and a Colab runner script.
- `configs/forensic_quant/qdevelop_w4.yaml`: default W4/L1 smoke config.

Runtime state is intentionally not committed: `upstream/`, `cache/`, and `outputs/`
are ignored. Colab clones this repo and then bootstraps `facebookresearch/stable_signature`
into `upstream/stable_signature`.

## Current Status

Local pilot scaffolding is committed and syntax-checked. The full research run still
requires wiring `src/forensic_quant/stable_signature_adapter.py` to the exact Stable
Signature loader/extractor calls and providing external SD/COCO assets. Without that,
`00_prepare_pairs.py` stops intentionally instead of silently running a fake training job.

Required external assets for the real full run:

- Stable Diffusion/LDM config YAML.
- Stable Diffusion/LDM checkpoint.
- COCO train/validation image subsets.
- Stable Signature extractor and watermarked decoder weights.

## Colab Flow

1. Prepare a Colab runtime with the packages in `requirements-colab.txt` installed.
2. Clone/pull this repo.
3. Run `bash scripts/forensic_quant/run_colab_job.sh`.

The runner order is:

1. check required imports (`yaml`, `torch`),
2. clone/update Stable Signature,
3. download Stable Signature extractor/watermarked decoder assets,
4. prepare teacher pairs,
5. train dual-view QAT,
6. run decoder eval,
7. run T2I eval,
8. aggregate summary.

## Guardrails

- Keep one trainable FP master decoder.
- Use the same quantizer implementation for training and evaluation.
- Keep L1 as the default objective; MSE is only a configured ablation.
- Do not add BCE or LPIPS to the first training objective.
- Do not save teacher targets through JPEG or other lossy formats.
