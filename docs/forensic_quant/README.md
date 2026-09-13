# Forensic Quantization QAT Pilot

This module is a small wrapper around the official Stable Signature codebase for
testing whether a full-precision text-to-image decoder can stay fingerprint
dormant while its owner-quantized W4 decoder becomes fingerprint active.

The local code intentionally implements only the new pilot-specific pieces:

- W4 symmetric per-output-channel fake quantization with STE.
- Dual-view reconstruction losses: FP to clean teacher, W4 to watermarked teacher.
- Config/output schemas and required summary artifacts.

The watermark extractor, decoder loading, image preprocessing, and text-to-image
pipeline should come from `facebookresearch/stable_signature`. Wire those calls in
`src/forensic_quant/stable_signature_adapter.py` after inspecting the cloned repo.

## Flow

1. From this repo root, run `python scripts/forensic_quant/bootstrap_upstream.py`
   to clone `facebookresearch/stable_signature` into the ignored
   `upstream/stable_signature` dependency directory.
2. Prepare lossless teacher pairs with `scripts/forensic_quant/00_prepare_pairs.py`.
3. Train the dual-view decoder with `scripts/forensic_quant/01_train_qdevelop.py`.
4. Run decoder-level evaluation with `scripts/forensic_quant/02_eval_decoder_level.py`.
5. Run text-to-image evaluation with `scripts/forensic_quant/03_eval_t2i.py`.
6. Aggregate CSVs and answer Q1-Q7 with `scripts/forensic_quant/04_aggregate_results.py`.

## Repository Boundary

Push only this repository to GitHub. The `upstream/`, `cache/`, and `outputs/`
directories are ignored local/runtime state. Colab should clone this repo, then
run `scripts/forensic_quant/bootstrap_upstream.py` to fetch Stable Signature.

## Guardrails

- Keep one trainable FP master decoder.
- Use the same quantizer implementation for training and evaluation.
- Keep L1 as the default objective; MSE is only a configured ablation.
- Do not add BCE or LPIPS to the first training objective.
- Do not save teacher targets through JPEG or other lossy formats.
