#!/bin/bash
# G10 on Qwen2.5-0.5B-Instruct (MPS fp32), both enforcement contexts, one after the other.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src
for align in free prompt; do
  echo "$(date +%H:%M:%S) START g10 $align"
  python3 bench/g10_watermark.py --model Qwen/Qwen2.5-0.5B-Instruct --n 30 --lengths 100,200,400 --nulls 200 \
      --device mps --align $align > bench/logs/g10_qwen_$align.log 2>&1
  echo "$(date +%H:%M:%S) END g10 $align rc=$?"
done
echo "$(date +%H:%M:%S) ALL DONE"
