#!/bin/bash
# Memory probe of the reference code's sealed batch path before the long unattended runs (13.09.2026, 17:40).
# The frozen copy held 13-19 GB on 1.5B GSM8K (its score_batch takes up to 16384 tokens per forward pass: ~5 GB of
# fp16 logits over a 151936-token vocabulary, plus fp32 copies, kept by the MPS allocator). The reference applies the
# head only on the rows it needs, 512 at a time. Here: the peak footprint (top's MEM, sampled every 3 s) of short
# corpora in the configurations the bridge uses. Corpora go to a scratch directory; only this log is kept.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
SCR=${SCR:-/tmp/ep_diag_memory}; mkdir -p "$SCR"
export PYTHONPATH=src EP_RUNS=$SCR/runs EP_DATA=$PWD/data
OUT=$PWD/bench/logs/bridge/diag_memory_reference_code.log
tomb() { awk '{s=$1; n=s; gsub(/[^0-9.]/, "", n); if (s ~ /G/) n*=1024; else if (s ~ /K/) n/=1024;
               else if (s !~ /M/) n/=1048576; print int(n)}'; }
probe() {
  local name=$1; shift
  "$@" > "$SCR/$name.log" 2>&1 & local pid=$! peak=0 m
  while kill -0 $pid 2>/dev/null; do
    m=$(top -l 1 -pid $pid -stats mem 2>/dev/null | tail -1 | tomb)
    [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
    sleep 3
  done
  wait $pid; local rc=$?
  echo "$(date +%T) $name rc=$rc peak footprint ${peak} MB; swap used $(sysctl -n vm.swapusage | awk '{print $6}')" >> "$OUT"
  grep "  batch\|done:\|Error\|error" "$SCR/$name.log" | tail -4 >> "$OUT"
}
echo "$(date '+%F %T') code $(python3 -W ignore -c "import sys; sys.path.insert(0, 'bench'); from _meta import code_digest; print(code_digest())" 2>/dev/null)" >> "$OUT"
Q=(--data gsm8k --field question --chat --max-new 400 --temperature 0 --mode sealed --guard 0.9)
probe q15_gsm_b4 python3 bench/gpu/seal_corpus.py --model Qwen/Qwen2.5-1.5B-Instruct --device mps --dtype float16 "${Q[@]}" \
    --n 8 --batch 4 --name diag_q15_b4
probe q15_gsm_b16 python3 bench/gpu/seal_corpus.py --model Qwen/Qwen2.5-1.5B-Instruct --device mps --dtype float16 "${Q[@]}" \
    --n 16 --batch 16 --name diag_q15_b16
probe phi3_gsm_b4 python3 bench/gpu/seal_corpus.py --model microsoft/Phi-3-mini-4k-instruct --device mps --dtype float16 \
    "${Q[@]}" --n 8 --batch 4 --name diag_phi3_b4
echo "$(date +%T) DIAG DONE" >> "$OUT"
