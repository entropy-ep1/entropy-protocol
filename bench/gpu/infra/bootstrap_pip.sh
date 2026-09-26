#!/bin/bash
# Runs on a fresh GCP or Azure evaluation machine that already has an NVIDIA driver (not the AWS Deep Learning AMI,
# which has its own bootstrap.sh): a venv with the pinned stack, caches on the home disk, the same ep.env layout the
# phase scripts source. torch 2.7.1 (CUDA 12.8 wheel) and transformers 4.53.2 match phase A where numbers depend on
# them; every machine writes its exact versions into logs/B_versions.log (run_phase_b.sh) for the report.
# Usage (from the Mac): ssh ubuntu@IP 'bash -s' < bootstrap_pip.sh
set -euo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip rsync >/dev/null
python3 -m venv "$HOME/venv"
# shellcheck disable=SC1091
source "$HOME/venv/bin/activate"
pip install -q --upgrade pip
pip install -q "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install -q "transformers==4.53.2" "accelerate>=1.2" "datasets>=3.0" "cryptography>=42" "cbor2>=5.6" \
    "markdown-it-py>=3" sentencepiece protobuf numpy "pytest>=8" huggingface_hub
mkdir -p "$HOME/hf" "$HOME/ep/runs" "$HOME/ep/data"
cat > "$HOME/ep.env" <<'EOF'
export HF_HOME=$HOME/hf
export EP_RUNS=$HOME/ep/runs EP_DATA=$HOME/ep/data
export PYTHONPATH=$HOME/ep/src
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
source $HOME/venv/bin/activate
EOF
python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GiB")
EOF
echo "bootstrap done"
