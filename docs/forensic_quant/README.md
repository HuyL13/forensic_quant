# Forensic Quantization QAT Pilot

This repo contains the local pilot code for testing dormant text-to-image visual
fingerprints that activate after owner-side W4 decoder quantization.

## Implemented

- Stable Signature bootstrap into ignored `upstream/stable_signature`.
- LDM autoencoder loading through Stable Signature's `utils_model.load_model_from_config`.
- Lossless teacher-pair preparation: `(z, x_clean, x_wm)` cached as `.pt` tensors.
- W4 symmetric per-output-channel fake quantization with STE on decoder weights.
- Dual-view QAT training from cached pairs with one FP master decoder.
- Quantized-branch gradient sanity check before training.
- Decoder-level eval for `clean_fp`, `clean_w4`, `ours_fp`, `ours_w4`, `wm_teacher_fp`.
- Aggregated 10-image decoding and residual alignment CSVs.

## Not Yet Implemented

- End-to-end text-to-image generation/eval with fixed prompts and seeds. `03_eval_t2i.py`
  fails clearly instead of writing fake metrics. This still needs a Stable Diffusion sampler
  or Diffusers pipeline wired to swap decoder variants.

## Required External Assets

Place these paths to match `configs/forensic_quant/qdevelop_w4.yaml`, or edit the YAML:

- `assets/ldm/v2-inference.yaml`
- `assets/ldm/v2-1_512-ema-pruned.ckpt`
- `data/coco/train/` with training images
- `data/coco/val/` with held-out images
- `upstream/stable_signature/models/dec_48b_whit.torchscript.pt` downloaded by runner
- `upstream/stable_signature/models/sd2_decoder.pth` downloaded by runner

Runtime state is intentionally not committed: `upstream/`, `cache/`, `outputs/`, `assets/`,
and `data/` are ignored/local state.

## Colab Flow

1. Prepare a Colab runtime with the packages in `requirements-colab.txt` installed.
2. Clone/pull this repo.
3. Put SD config/checkpoint and COCO subsets at the configured paths.
4. Run `bash scripts/forensic_quant/run_colab_job.sh`.

The runner order is:

1. check required imports (`yaml`, `torch`),
2. clone/update Stable Signature,
3. download Stable Signature extractor/watermarked decoder assets,
4. prepare teacher pairs,
5. train dual-view QAT,
6. run decoder eval,
7. attempt T2I eval and stop clearly because it is not implemented,
8. aggregate summary only if earlier stages complete.

## Guardrails

- Keep one trainable FP master decoder.
- Use the same quantizer implementation for training and evaluation.
- Keep L1 as the default objective; MSE is only a configured ablation.
- Do not add BCE or LPIPS to the first training objective.
- Do not save teacher targets through JPEG or other lossy formats.
