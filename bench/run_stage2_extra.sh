#!/bin/bash
# Stage-2 additions on the same frozen code as the final run (aa2dddc19298c239), started when the MPS device is free
# and finished before the speed gate G7: G4 on distilgpt2 again (the bench fix skips a text that ends at its first
# token), G5 quality of the reference operating point and of the "record strength" profile.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src
L=bench/logs/stage2
stamp() { date "+%H:%M:%S"; }
run() { local name=$1; shift; echo "$(stamp) START $name"; "$@" > "$L/$name.log" 2>&1; local rc=$?; echo "$(stamp) END $name rc=$rc"; }
CAL=$(python3 -c "import json; t=json.load(open('calibrations/qwen_qwen2.5-0.5b-instruct_stage2.json'))['table']; print(','.join(f'{k}={v}' for k,v in t.items()))")
run g4_distilgpt2_rerun python3 bench/g4_backends.py --model distilgpt2 --n 40 --calib 30 --tag stage2
run g5_qwen_final python3 bench/g5_operating_point.py --model Qwen/Qwen2.5-0.5B-Instruct --oracle gpt2-medium --n 40 \
    --space logit --cal "$CAL" --variants c5_free_hm0,c5_record_strength --device mps --self-ppl --tag stage2final
echo "$(stamp) EXTRA DONE"
