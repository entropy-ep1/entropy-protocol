#!/bin/bash
# Scale bridge, part 5: the operating point. GSM8K on Qwen2.5-0.5B showed that the reference point breaks reasoning
# (accuracy 38% -> 3%; a HIGH checkpoint forces a non-first choice where the model copies a number or a name from the
# question, which the prompt-free gate cannot see), while gate 0.9 + guard 0.9 keeps it (36%) and leaves almost no
# mark on maths. Before choosing a default we need the other side: what the guard costs the mark on prose.
#   1. prose (40 chat prompts, T 0.7), Qwen2.5-0.5B: plain / sealed / gate90 / gate90+guard90 / guard90 / guard95
#   2. GSM8K, Qwen2.5-0.5B: guard90 / guard95 alone (which part protects accuracy)
#   3. prose, Qwen2.5-1.5B: plain / sealed / gate90+guard90 / guard90
# Code: the frozen copy ../entropy-core-gsm (00a100caabf1bdf8, as the GSM8K study); data from ../entropy-core-dev/data.
# Then the rest of part 4: G10 and G4 on 1.5B (reference code), Phi-3-mini.
set -u
cd "$(dirname "$0")/.."
REF=$PWD; GSM=$(cd ../entropy-core-gsm && pwd); DEV=$(cd ../entropy-core-dev && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
L=$REF/bench/logs/bridge; mkdir -p "$L"
stamp() { date "+%H:%M:%S"; }
swap() { sysctl -n vm.swapusage | awk '{print $6}'; }
run() { local name=$1; shift; echo "$(stamp) START $name (swap used $(swap))"; "$@" > "$L/$name.log" 2>&1; local rc=$?
        echo "$(stamp) END $name rc=$rc"; }
echo "$(stamp) part 5; code $(cd "$GSM" && python3 -W ignore -c "import sys; sys.path.insert(0, 'bench'); from _meta import code_digest; print(code_digest())" 2>/dev/null)"

cd "$GSM"; export PYTHONPATH=src EP_RUNS=$GSM/runs EP_DATA=$DEV/data
prose() {   # model tag variant extra-args...
  local MM=$1 S=$2 V=$3; shift 3
  run "prose_${S}_${V}" python3 bench/gpu/seal_corpus.py --model $MM --device mps --dtype float16 --data prose_offline \
      --field prompt --chat --n 40 --batch 20 --max-new 400 --min-new 250 --temperature 0.7 --name prose_$V "$@"
}
detect() {  # model tag variant gate
  run "detp_${2}_${3}" python3 bench/gpu/detect_corpus.py --model $1 --device mps --dtype float16 --positives prose_$3 \
      --lengths 100,200 --gate $4 --name prose_$3
}
qual() {    # model tag variant
  run "qp_${2}_${3}" python3 bench/gpu/quality.py ppl --model $1 --device mps --dtype float32 --oracle gpt2-medium \
      --a prose_$3 --b prose_plain --data prose_offline --field prompt
  run "qd_${2}_${3}" python3 bench/gpu/quality.py diversity --model $1 --a prose_$3 --b prose_plain
}

M5=Qwen/Qwen2.5-0.5B-Instruct; S5=${M5//\//_}
prose $M5 $S5 plain --mode plain
prose $M5 $S5 sealed --mode sealed
prose $M5 $S5 gate90 --mode sealed --gate 0.9
prose $M5 $S5 g90u90 --mode sealed --gate 0.9 --guard 0.9
prose $M5 $S5 guard90 --mode sealed --guard 0.9
prose $M5 $S5 guard95 --mode sealed --guard 0.95
detect $M5 $S5 sealed 0; detect $M5 $S5 gate90 0.9; detect $M5 $S5 g90u90 0.9; detect $M5 $S5 guard90 0; detect $M5 $S5 guard95 0
for V in sealed gate90 g90u90 guard90 guard95; do qual $M5 $S5 $V; done

G=(python3 bench/gpu/seal_corpus.py --model $M5 --device mps --dtype float16 --data gsm8k --field question --chat \
   --n 100 --batch 16 --max-new 400 --temperature 0)
run "gsm_${S5}_guard90" "${G[@]}" --mode sealed --guard 0.9 --name gsm_guard90
run "gsm_${S5}_guard95" "${G[@]}" --mode sealed --guard 0.95 --name gsm_guard95
for V in gsm_guard90 gsm_guard95; do
  run "q_${S5}_$V" python3 bench/gpu/quality.py gsm8k --model $M5 --device mps --a $V --b gsm_plain
  run "det_${S5}_$V" python3 bench/gpu/detect_corpus.py --model $M5 --device mps --dtype float16 --positives $V \
      --lengths 100,200,400 --name $V
done

M15=Qwen/Qwen2.5-1.5B-Instruct; S15=${M15//\//_}
prose $M15 $S15 plain --mode plain
prose $M15 $S15 sealed --mode sealed
prose $M15 $S15 g90u90 --mode sealed --gate 0.9 --guard 0.9
prose $M15 $S15 guard90 --mode sealed --guard 0.9
detect $M15 $S15 sealed 0; detect $M15 $S15 g90u90 0.9; detect $M15 $S15 guard90 0
for V in sealed g90u90 guard90; do qual $M15 $S15 $V; done

cd "$REF"; export PYTHONPATH=src; unset EP_RUNS EP_DATA
T=qwen_qwen2.5-1.5b-instruct
run "g10_$T" python3 bench/g10_watermark.py --model $M15 --n 30 --lengths 100,200,400 --nulls 200 --device mps --tag bridge
run "g4_$T" python3 bench/g4_backends.py --model $M15 --n 40 --calib 30 --tag bridge --configs cpu:float32,mps:float32,mps:float16,mps:bfloat16
P=microsoft/Phi-3-mini-4k-instruct; U=microsoft_phi-3-mini-4k-instruct
run "gates_$U" python3 bench/gates.py --model $P --n 50 --device mps --dtype float16 --tag bridge
run "g10_$U" python3 bench/g10_watermark.py --model $P --n 20 --lengths 100,200 --nulls 100 --device mps --dtype float16 --tag bridge
run "g5_$U" python3 bench/g5_operating_point.py --model $P --oracle gpt2-medium --n 30 --space logit --variants c5_free_hm0 \
    --device mps --dtype float16 --self-ppl --tag bridge
echo "$(stamp) BRIDGE PART 5 DONE"
