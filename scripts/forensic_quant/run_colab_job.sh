#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/forensic_quant/qdevelop_w4.yaml}

python - <<'PY'
import importlib.util
missing = [name for name in ["yaml", "torch"] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required Python packages: " + ", ".join(missing))
PY

python scripts/forensic_quant/bootstrap_upstream.py

mkdir -p upstream/stable_signature/models
if [ ! -f upstream/stable_signature/models/dec_48b_whit.torchscript.pt ]; then
  curl -L https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt \
    -o upstream/stable_signature/models/dec_48b_whit.torchscript.pt
fi
if [ ! -f upstream/stable_signature/models/sd2_decoder.pth ]; then
  curl -L https://dl.fbaipublicfiles.com/ssl_watermarking/sd2_decoder.pth \
    -o upstream/stable_signature/models/sd2_decoder.pth
fi

python scripts/forensic_quant/00_prepare_pairs.py --config "$CONFIG"
python scripts/forensic_quant/01_train_qdevelop.py --config "$CONFIG"
python scripts/forensic_quant/02_eval_decoder_level.py --config "$CONFIG"
python scripts/forensic_quant/03_eval_t2i.py --config "$CONFIG"
python scripts/forensic_quant/04_aggregate_results.py --config "$CONFIG"
