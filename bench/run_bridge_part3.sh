#!/bin/bash
# Scale bridge, part 3: resumes part 2 after the 13.09 11:37 kernel panic (watchdog timeout under memory pressure;
# part 2 had finished the G10 positives on Qwen2.5-1.5B, not its null or report). One step at a time, nothing else
# heavy alongside. Order: the GSM8K confidence-gate study (staging code ../entropy-core-next, frozen at
# 83efb42d15ed0690), G5 and a full G10 rerun on 1.5B (reference code), G4 on 1.5B, then Phi-3-mini.
set -u
cd "$(dirname "$0")/.."
REF=$PWD; NEXT=$(cd ../entropy-core-next && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
L=$REF/bench/logs/bridge; mkdir -p "$L"
stamp() { date "+%H:%M:%S"; }
swap() { sysctl -n vm.swapusage | awk '{print $6}'; }
run() { local name=$1; shift; echo "$(stamp) START $name (swap used $(swap))"; "$@" > "$L/$name.log" 2>&1; local rc=$?
        echo "$(stamp) END $name rc=$rc"; }
M=Qwen/Qwen2.5-1.5B-Instruct; T=qwen_qwen2.5-1.5b-instruct
CAL=$(python3 -c "import json; t=json.load(open('calibrations/${T}_bridge.json'))['table']; print(','.join(f'{k}={v}' for k, v in t.items()))")
echo "$(stamp) part 3; staging code $(cd "$NEXT" && python3 -W ignore -c "import sys; sys.path.insert(0, 'bench'); from _meta import code_digest; print(code_digest())" 2>/dev/null)"

# GSM8K: plain vs sealed without and with the confidence gate / generation guard (greedy, chat template)
cd "$NEXT"; export PYTHONPATH=src EP_RUNS=$NEXT/runs EP_DATA=$NEXT/data
for MM in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
  S=${MM//\//_}
  G=(python3 bench/gpu/seal_corpus.py --model $MM --device mps --dtype float16 --data gsm8k --field question --chat \
     --n 100 --batch 16 --max-new 400 --temperature 0)
  run "gsm_${S}_plain"   "${G[@]}" --mode plain --name gsm_plain
  run "gsm_${S}_sealed"  "${G[@]}" --mode sealed --name gsm_sealed
  run "gsm_${S}_gate90"  "${G[@]}" --mode sealed --gate 0.9 --name gsm_gate90
  run "gsm_${S}_g90u90"  "${G[@]}" --mode sealed --gate 0.9 --guard 0.9 --name gsm_gate90_guard90
  for V in gsm_sealed gsm_gate90 gsm_gate90_guard90; do
    run "q_${S}_$V" python3 bench/gpu/quality.py gsm8k --model $MM --device mps --a $V --b gsm_plain
  done
  run "det_${S}_sealed" python3 bench/gpu/detect_corpus.py --model $MM --device mps --dtype float16 --positives gsm_sealed \
      --lengths 100,200,400 --name gsm_sealed
  run "det_${S}_gate90" python3 bench/gpu/detect_corpus.py --model $MM --device mps --dtype float16 --positives gsm_gate90 \
      --lengths 100,200,400 --gate 0.9 --name gsm_gate90
  run "det_${S}_g90u90" python3 bench/gpu/detect_corpus.py --model $MM --device mps --dtype float16 --positives gsm_gate90_guard90 \
      --lengths 100,200,400 --gate 0.9 --name gsm_gate90_guard90
done

cd "$REF"; export PYTHONPATH=src; unset EP_RUNS EP_DATA
run "g5_$T" python3 bench/g5_operating_point.py --model $M --oracle gpt2-medium --n 40 --space logit --cal "$CAL" \
    --variants c5_free_hm0 --device mps --self-ppl --tag bridge
run "g10_$T" python3 bench/g10_watermark.py --model $M --n 30 --lengths 100,200,400 --nulls 200 --device mps --tag bridge
run "g4_$T" python3 bench/g4_backends.py --model $M --n 40 --calib 30 --tag bridge --configs cpu:float32,mps:float32,mps:float16,mps:bfloat16
P=microsoft/Phi-3-mini-4k-instruct; U=microsoft_phi-3-mini-4k-instruct
run "gates_$U" python3 bench/gates.py --model $P --n 50 --device mps --dtype float16 --tag bridge
run "g10_$U" python3 bench/g10_watermark.py --model $P --n 20 --lengths 100,200 --nulls 100 --device mps --dtype float16 --tag bridge
run "g5_$U" python3 bench/g5_operating_point.py --model $P --oracle gpt2-medium --n 30 --space logit --variants c5_free_hm0 \
    --device mps --dtype float16 --self-ppl --tag bridge
echo "$(stamp) BRIDGE PART 3 DONE"
