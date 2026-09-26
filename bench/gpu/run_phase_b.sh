#!/bin/bash
# Phase B (calibration across machines): this machine's margins on the SHARED sealed corpus, in every configuration
# it can run, plus the throughput probe that re-prices phase C, plus verification of phase A records on this card.
# The corpus is sealed once (--seal on the first machine) and copied to the others (infra/sync.sh runs-up), so every
# dump covers exactly the same token positions - calib_table refuses dumps whose positions differ.
# The table is built on the Mac across machines (infra/phase_b_table.sh): numpy only, no GPU.
# Usage (on the evaluation machine): bash bench/gpu/run_phase_b.sh [--seal] [model] [texts]
set -uo pipefail
source "$HOME/ep.env"; cd "$HOME/ep"
export PYTHONPATH=$HOME/ep/src
SEAL=0; [ "${1:-}" = "--seal" ] && { SEAL=1; shift; }
M=${1:-Qwen/Qwen2.5-7B-Instruct}; N=${2:-512}; S=${M//\//_}; G=bench/gpu; mkdir -p logs
GPU_GB=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 2**30))" 2>/dev/null || echo 0)
FP32=1; [ "$GPU_GB" -lt 40 ] && FP32=0          # a 7B in fp32 takes ~30 GB of weights
stamp() { date -u +%H:%M:%S; }
run() {
  local n=$1; shift
  if [ -f "logs/B_$n.ok" ]; then echo "$(stamp) SKIP $n (done earlier)"; return 0; fi
  echo "$(stamp) START $n"; "$@" > "logs/B_$n.log" 2>&1; local rc=$?; echo "$(stamp) END $n rc=$rc"
  [ "$rc" = 0 ] && touch "logs/B_$n.ok"; return 0
}
need() { [ -f "logs/B_$1.ok" ] || { echo "$(stamp) PHASE B STOPPED: $1 failed (logs/B_$1.log)"; exit 1; }; }

run versions   bash -c 'python3 -c "import torch,transformers,platform; print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda,\"cudnn\",torch.backends.cudnn.version(),\"transformers\",transformers.__version__,\"python\",platform.python_version()); print(\"gpu\",torch.cuda.get_device_name(0),round(torch.cuda.get_device_properties(0).total_memory/2**30,1),\"GiB cc\",torch.cuda.get_device_capability(0))"; nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader; uname -r'
run data       python3 $G/prep_data.py --c4 $((N + 1200)) --wiki 300 --langs en,uk --only c4,wiki
run unit_tests python3 -m pytest tests -q -m "not model"
run tiny_model python3 -c "from huggingface_hub import snapshot_download as s; s('distilgpt2', allow_patterns=['*.json', '*.safetensors', '*.txt'])"
run model_tests python3 -m pytest tests -q -m model
if [ "$SEAL" = 1 ]; then
  run cal_seal python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --offset 2000 --n "$N" \
                 --batch 32 --max-new 400 --min-new 300 --mode sealed --name B_cal
  need cal_seal
fi
[ -s "$EP_RUNS/$S/B_cal.jsonl" ] || { echo "$(stamp) PHASE B STOPPED: no B_cal.jsonl on this machine"; exit 1; }
# every configuration this card can run: precision x batch x attention kernel
CFGS=("float16 1 default" "float16 32 default" "float16 32 eager" "bfloat16 1 default" "bfloat16 32 default")
[ "$FP32" = 1 ] && CFGS+=("float32 1 default" "float32 32 default")
for cfg in "${CFGS[@]}"; do
  set -- $cfg
  DT=$1; B=$2; A=$3
  if [ "$A" = eager ]; then
    run "dump_${DT}_b${B}_eager" python3 $G/calib_dump.py --model "$M" --dtype "$DT" --batch "$B" --attn eager --corpus B_cal
  else
    run "dump_${DT}_b${B}" python3 $G/calib_dump.py --model "$M" --dtype "$DT" --batch "$B" --corpus B_cal
  fi
done
# throughput: batch 64 is new (phase A stopped at 32 and the card was not full)
run speed python3 $G/speed.py --model "$M" --dtype float16 --batches 1,8,32,64 --tokens 256
# phase A records (fp16-only table) verified here, including fp32: does a verifier outside the table stay correct?
for C in A_c4_sealed_fp16cert A_c4_quality_fp16cert; do
  [ -s "$EP_RUNS/$S/$C.jsonl" ] || continue
  run "verify_${C}_fp16" python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus "$C"
  run "verify_${C}_bf16" python3 $G/verify_corpus.py --model "$M" --dtype bfloat16 --batch 32 --corpus "$C" --only-same
  [ "$FP32" = 1 ] && run "verify_${C}_fp32" python3 $G/verify_corpus.py --model "$M" --dtype float32 --batch 1 \
                       --corpus "$C" --n 64 --only-same
done
# why the KGW baseline weakened with length (phase A: 81 / 75 / 34 % at p <= 1e-2): needs this model's own
# tokenizer and vocabulary, no weights (push the corpus up first: infra/sync.sh runs-up)
[ -s "$EP_RUNS/$S/A_c4_kgw.jsonl" ] && run kgw_diag python3 bench/diag_kgw_length.py "$EP_RUNS" --model "$M" \
  --tokenizer "$M" --corpus A_c4_kgw --out "$EP_RUNS/$S/diag_kgw_length.json"
echo "$(stamp) PHASE B DONE (GPU ${GPU_GB} GB, fp32 steps: $([ "$FP32" = 1 ] && echo run || echo skipped))"
