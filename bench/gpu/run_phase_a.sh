#!/bin/bash
# Phase A (smoke on a real GPU): every experiment of the program on one model with small N; throughput and memory
# measured, so the budget for phases B-F is re-estimated before they start.
# 14.09.2026: the product's default operating point is quality (guard 0.9 at sealing, detector gate 0.9; profiles.py),
# so phase A seals, verifies and detects at both points: "sealed" = strong (the protocol's reference parameters,
# SealParams()), "quality" = --guard 0.9.
# Usage (on the evaluation machine): bash bench/gpu/run_phase_a.sh [model]
set -uo pipefail
source "$HOME/ep.env"; cd "$HOME/ep"
export PYTHONPATH=$HOME/ep/src    # pytest imports the package from src (15.09: every test module failed to collect without it)
M=${1:-Qwen/Qwen2.5-7B-Instruct}; S=${M//\//_}; G=bench/gpu; mkdir -p logs
# a 7B model in fp32 takes ~30 GB of weights: a 24 GB card (A10G, L4) skips the fp32 steps, which move to phase B
GPU_GB=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 2**30))" 2>/dev/null || echo 0)
FP32=1; [ "$GPU_GB" -lt 40 ] && FP32=0
stamp() { date -u +%H:%M:%S; }
# a step that ends with rc=0 leaves logs/A_<step>.ok and is skipped when the phase is started again (15.09.2026: the
# first run lost every sampled step to one CUDA defect; after the fix only those steps ran again)
run() {
  local n=$1; shift
  if [ -f "logs/A_$n.ok" ]; then echo "$(stamp) SKIP $n (done earlier)"; return 0; fi
  echo "$(stamp) START $n"; "$@" > "logs/A_$n.log" 2>&1; local rc=$?; echo "$(stamp) END $n rc=$rc"
  [ "$rc" = 0 ] && touch "logs/A_$n.ok"; return 0
}
# the steps after this one read its output: without it the phase stops, and the machine stays up for a fix (the
# watcher terminates it only at PHASE A DONE; the cap still applies)
need() { [ -f "logs/A_$1.ok" ] || { echo "$(stamp) PHASE A STOPPED: $1 failed (logs/A_$1.log)"; exit 1; }; }

run data       python3 $G/prep_data.py --c4 1200 --wiki 300 --langs en,uk --dolly 100 --only c4,wiki,dolly,gsm8k
run unit_tests python3 -m pytest tests -q -m "not model"
run tiny_model python3 -c "from huggingface_hub import snapshot_download as s; s('distilgpt2', allow_patterns=['*.json', '*.safetensors', '*.txt'])"
run model_tests python3 -m pytest tests -q -m model                  # distilgpt2 on this machine's CPU
# calibration corpus, dumps on this GPU, table
run cal_seal   python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --offset 1000 --n 64 \
                 --batch 32 --max-new 400 --min-new 300 --mode sealed --name A_cal
need cal_seal
CFGS=("float16 1" "float16 32" "bfloat16 32"); [ "$FP32" = 1 ] && CFGS+=("float32 1")
for cfg in "${CFGS[@]}"; do
  set -- $cfg
  run "dump_$1_b$2" python3 $G/calib_dump.py --model "$M" --dtype "$1" --batch "$2" --corpus A_cal
done
need dump_float16_b1
REF=$(ls "$EP_RUNS/$S/calib" | grep '^A_cal__.*_float16_default_b1\.npz$' | sed 's/^A_cal__//; s/\.npz$//')
run table      python3 $G/calib_table.py --model "$M" --corpus A_cal --reference "$REF" --certify $([ "$FP32" = 1 ] && echo fp32,fp16 || echo fp16)
need table
CAL=$EP_RUNS/$S/calibration_A_cal.json
# corpora: sealed (with the table), plain, baselines; bf16 generation + fp16 measurement on a few
run sealed     python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 128 --batch 32 \
                 --max-new 400 --min-new 400 --mode sealed --cal "$CAL" --name A_c4_sealed
run quality    python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 128 --batch 32 \
                 --max-new 400 --min-new 400 --mode sealed --guard 0.9 --cal "$CAL" --name A_c4_quality
run sealed_bf  python3 $G/seal_corpus.py --model "$M" --dtype bfloat16 --measure-dtype float16 --data c4 --field prompt \
                 --offset 200 --n 32 --batch 32 --max-new 400 --min-new 400 --mode sealed --cal "$CAL" --name A_c4_sealed_bf16gen
run plain      python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 128 --batch 32 \
                 --max-new 400 --min-new 400 --mode plain --name A_c4_plain
run kgw        python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 64 --batch 32 \
                 --max-new 400 --min-new 400 --mode kgw --name A_c4_kgw
run synthid    python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 64 --batch 16 \
                 --max-new 400 --min-new 400 --mode synthid --name A_c4_synthid
run chat_seal  python3 $G/seal_corpus.py --model "$M" --dtype float16 --data dolly --field instruction --chat --n 64 \
                 --batch 32 --max-new 512 --mode sealed --cal "$CAL" --name A_dolly_sealed
run chat_plain python3 $G/seal_corpus.py --model "$M" --dtype float16 --data dolly --field instruction --chat --n 64 \
                 --batch 32 --max-new 512 --mode plain --name A_dolly_plain
run gsm_seal   python3 $G/seal_corpus.py --model "$M" --dtype float16 --data gsm8k --field question --chat --n 128 \
                 --batch 32 --max-new 512 --temperature 0 --mode sealed --cal "$CAL" --name A_gsm8k_sealed
run gsm_plain  python3 $G/seal_corpus.py --model "$M" --dtype float16 --data gsm8k --field question --chat --n 128 \
                 --batch 32 --max-new 512 --temperature 0 --mode plain --name A_gsm8k_plain
run gsm_qual   python3 $G/seal_corpus.py --model "$M" --dtype float16 --data gsm8k --field question --chat --n 128 \
                 --batch 32 --max-new 512 --temperature 0 --mode sealed --guard 0.9 --cal "$CAL" --name A_gsm8k_quality
# records on this GPU in three configurations
run verify_fp16 python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus A_c4_sealed
run verify_bf16 python3 $G/verify_corpus.py --model "$M" --dtype bfloat16 --batch 32 --corpus A_c4_sealed --only-same
[ "$FP32" = 1 ] && run verify_fp32 python3 $G/verify_corpus.py --model "$M" --dtype float32 --batch 1 --corpus A_c4_sealed --n 32 --only-same
run verify_bfg  python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus A_c4_sealed_bf16gen
run verify_q    python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus A_c4_quality
# detection, attacks, baselines
run attacks    python3 $G/attacks_text.py --model "$M" --corpus A_c4_sealed --human data:c4:human
run detect     python3 $G/detect_corpus.py --model "$M" --dtype float16 --positives A_c4_sealed --lengths 50,100,200,400 \
                 --nulls data:c4:human,data:wiki_uk:text,runs:A_c4_plain:text --max-null-texts 1000 --null-keys 20 \
                 --attacks runs:attacks_A_c4_sealed --cal "$CAL" --name A
run detect_q   python3 $G/detect_corpus.py --model "$M" --dtype float16 --positives A_c4_quality --lengths 50,100,200,400 \
                 --gate 0.9 --nulls data:c4:human --max-null-texts 500 --null-keys 20 --cal "$CAL" --name A_quality
run base_kgw   python3 $G/baselines.py --model "$M" --dtype float16 --scheme kgw --positives A_c4_kgw --lengths 100,200,400 \
                 --nulls data:c4:human,runs:A_c4_plain:text --max-null-texts 1000 --name A
run base_sid   python3 $G/baselines.py --model "$M" --dtype float16 --scheme synthid --positives A_c4_synthid --lengths 100,200,400 \
                 --nulls data:c4:human,runs:A_c4_plain:text --max-null-texts 1000 --name A
# quality (self-oracle in phase A; the 32B oracle and judges come in phase E), tasks, speed
run q_div      python3 $G/quality.py diversity --model "$M" --a A_c4_sealed --b A_c4_plain
run q_div_q    python3 $G/quality.py diversity --model "$M" --a A_c4_quality --b A_c4_plain
run q_gsm      python3 $G/quality.py gsm8k --model "$M" --a A_gsm8k_sealed --b A_gsm8k_plain
run q_gsm_q    python3 $G/quality.py gsm8k --model "$M" --a A_gsm8k_quality --b A_gsm8k_plain
run speed      python3 $G/speed.py --model "$M" --dtype float16 --batches 1,8,32 --tokens 256
echo "$(stamp) PHASE A DONE (GPU ${GPU_GB} GB, fp32 steps: $([ "$FP32" = 1 ] && echo run || echo skipped))"
