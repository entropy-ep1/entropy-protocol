#!/bin/bash
# Runs on a fresh evaluation machine (Deep Learning AMI, PyTorch 2.7): pinned libraries, caches on the local NVMe.
# Usage (from the Mac): ssh ubuntu@IP 'bash -s' < bootstrap.sh [vllm]
set -euo pipefail
NV=/opt/dlami/nvme; [ -d "$NV" ] && sudo chown -R ubuntu "$NV" || NV=$HOME
mkdir -p "$NV/hf" "$HOME/ep/runs" "$HOME/ep/data"
cat > "$HOME/ep.env" <<EOF
export HF_HOME=$NV/hf
export EP_RUNS=\$HOME/ep/runs EP_DATA=\$HOME/ep/data
export PYTHONPATH=\$HOME/ep/src
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
source /opt/pytorch/bin/activate
EOF
source "$HOME/ep.env"
pip install -q "transformers==4.53.2" "accelerate>=1.2" "datasets>=3.0" "cryptography>=42" "cbor2>=5.6" \
    "markdown-it-py>=3" sentencepiece protobuf numpy
python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
EOF
if [ "${1:-}" = "vllm" ]; then                  # separate environment: vLLM pins its own torch
  python3 -m venv "$HOME/vllm"
  "$HOME/vllm/bin/pip" install -q vllm "cryptography>=42" "cbor2>=5.6" numpy     # + what common.py imports
  "$HOME/vllm/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
fi
echo "bootstrap done"
