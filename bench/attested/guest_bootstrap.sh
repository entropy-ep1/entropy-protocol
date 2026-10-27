#!/bin/bash
# Runs on the Azure confidential H100 guest after launch_azure.sh (driver in place), from run_attested_azure.sh:
#   ssh ubuntu@IP 'bash -s' < guest_bootstrap.sh
# tpm2-tools; a venv with the stack of the evaluation program (torch 2.7.1 CUDA 12.8, transformers 4.53.2) and the
# tokenizers version the authority computed the tokenizer digest with (0.21.2); the model snapshot at the pinned
# revision; the verifier's X25519 key pair, generated here and kept in ~/verifier (mode 0600).
set -euo pipefail
REV=a09a35458c702b33eeacc393d103063234e8bc28
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv tpm2-tools >/dev/null
python3 -m venv "$HOME/venv"
# shellcheck disable=SC1091
source "$HOME/venv/bin/activate"
pip install -q --upgrade pip
pip install -q "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu128
pip install -q "transformers==4.53.2" "tokenizers==0.21.2" "accelerate>=1.2" "cryptography>=43.0.3" "cbor2>=5.6" \
    "markdown-it-py>=3" "nvidia-ml-py==12.550.52" huggingface_hub
pip freeze > "$HOME/pip_freeze.txt"
mkdir -p "$HOME/hf" "$HOME/io" "$HOME/out" "$HOME/verifier" "$HOME/ep/data"
HF_HOME="$HOME/hf" python - "$REV" <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen2.5-7B-Instruct")
if not path.rstrip("/").endswith(sys.argv[1]):
    sys.exit(f"main is no longer {sys.argv[1]}: {path}")
print("model", path)
EOF
python - "$HOME/verifier/verifier.key" "$HOME/io/verifier.pub" <<'EOF'
import os, sys
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
k = X25519PrivateKey.generate()
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
os.write(fd, k.private_bytes_raw().hex().encode()); os.close(fd)
open(sys.argv[2], "w").write(k.public_key().public_bytes_raw().hex())
EOF
python - <<'EOF'
import torch, transformers, tokenizers
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "tokenizers", tokenizers.__version__, "gpu", torch.cuda.get_device_name(0))
EOF
echo "guest bootstrap done"
