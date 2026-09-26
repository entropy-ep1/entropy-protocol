#!/bin/bash
# Stage-1 final measurements at the reference operating point (every 5th token, k = 1, HIGH window 3,
# tolerance in logit space, calibration table fp32 + fp16). Every result records the code digest.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src
L=bench/logs/final
mkdir -p "$L" calibrations
stamp() { date "+%H:%M:%S"; }
run() { local name=$1; shift; echo "$(stamp) START $name"; "$@" > "$L/$name.log" 2>&1; echo "$(stamp) END $name rc=$?"; }
tagof() { echo "$1" | tr '/' '_' | tr 'A-Z' 'a-z'; }
MODELS="distilgpt2 gpt2-medium Qwen/Qwen2.5-0.5B-Instruct"

# A. calibration files (k = 1, logit; fp32 + fp16 certified; bf16 measured for information)
for m in $MODELS; do
  t=$(tagof "$m")
  run "cal_$t" python3 -m entropy_protocol.cli calibrate --model "$m" --configs mps:float32,mps:float16,mps:bfloat16 \
      --positions 3000 --certify fp32,fp16 --out "calibrations/${t}_k1_logit.json"
done

# B. gates N = 200 on CPU fp32 with the calibration table   ||   C. hardware gate at the reference point
(
  for m in $MODELS; do
    t=$(tagof "$m")
    run "gates_$t" python3 bench/gates.py --model "$m" --n 200 --cal "calibrations/${t}_k1_logit.json" --tag final
  done
) &
run g4_distilgpt2 python3 bench/g4_backends.py --model distilgpt2 --n 40 --calib 30 --tag final
run g4_qwen python3 bench/g4_backends.py --model Qwen/Qwen2.5-0.5B-Instruct --n 40 --calib 30 --tag final \
    --configs cpu:float32,mps:float32,mps:float16,mps:bfloat16
wait

# D. N = 1000 on distilgpt2
run gates_distilgpt2_N1000 python3 bench/gates.py --model distilgpt2 --n 1000 \
    --cal calibrations/distilgpt2_k1_logit.json --tag final

# E. speed, on a machine with nothing else running
for m in $MODELS; do
  run "g7_$(tagof "$m")" python3 bench/g7_speed.py --model "$m" --configs cpu:float32,mps:float32,mps:float16
done
echo "$(stamp) ALL DONE"
