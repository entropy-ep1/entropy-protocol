#!/bin/bash
# Scale bridge on the Mac while the cloud quotas are pending: the stage-2 measurements on larger modern models,
# same prompts, frozen reference code. Qwen2.5-1.5B-Instruct (1.5B, 152k vocabulary) with cross-precision
# calibration and G4; Phi-3-mini-4k-instruct (3.8B, 32k vocabulary) in fp16 only (fp32 does not fit in memory).
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src
L=bench/logs/bridge; mkdir -p "$L" calibrations
stamp() { date "+%H:%M:%S"; }
run() { local name=$1; shift; echo "$(stamp) START $name"; "$@" > "$L/$name.log" 2>&1; local rc=$?; echo "$(stamp) END $name rc=$rc"; }
echo "$(stamp) code $(python3 -W ignore -c "import sys; sys.path.insert(0, 'bench'); from _meta import code_digest; print(code_digest())" 2>/dev/null)"

M=Qwen/Qwen2.5-1.5B-Instruct; T=qwen_qwen2.5-1.5b-instruct
run "cal_$T" python3 -m entropy_protocol.cli calibrate --model $M --configs mps:float32,mps:float16,mps:bfloat16 \
    --positions 3000 --certify fp32,fp16 --out "calibrations/${T}_bridge.json"
CAL=$(python3 -c "import json; t=json.load(open('calibrations/${T}_bridge.json'))['table']; print(','.join(f'{k}={v}' for k, v in t.items()))")
run "gates_$T" python3 bench/gates.py --model $M --n 100 --device mps --dtype float32 --cal "calibrations/${T}_bridge.json" --tag bridge
run "g4_$T" python3 bench/g4_backends.py --model $M --n 40 --calib 30 --tag bridge --configs cpu:float32,mps:float32,mps:float16,mps:bfloat16
run "g10_$T" python3 bench/g10_watermark.py --model $M --n 30 --lengths 100,200,400 --nulls 200 --device mps --tag bridge
run "g5_$T" python3 bench/g5_operating_point.py --model $M --oracle gpt2-medium --n 40 --space logit --cal "$CAL" \
    --variants c5_free_hm0 --device mps --self-ppl --tag bridge

P=microsoft/Phi-3-mini-4k-instruct; U=microsoft_phi-3-mini-4k-instruct
run "gates_$U" python3 bench/gates.py --model $P --n 50 --device mps --dtype float16 --tag bridge
run "g10_$U" python3 bench/g10_watermark.py --model $P --n 20 --lengths 100,200 --nulls 100 --device mps --dtype float16 --tag bridge
run "g5_$U" python3 bench/g5_operating_point.py --model $P --oracle gpt2-medium --n 30 --space logit --variants c5_free_hm0 \
    --device mps --dtype float16 --self-ppl --tag bridge
echo "$(stamp) BRIDGE DONE"
