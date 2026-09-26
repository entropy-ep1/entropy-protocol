#!/bin/bash
# Phase E on the H100 machine: model-based attacks (classes 4-5), oracle perplexity, LLM judges, then detection of
# the attacked texts with the evaluated model. Corpora come from the phase-C machines (sync.sh down, then up here).
# Usage (on the machine, after bootstrap.sh vllm): bash bench/gpu/run_phase_e.sh <model> [<model> ...]
set -uo pipefail
source "$HOME/ep.env"; cd "$HOME/ep"
G=bench/gpu; mkdir -p logs
VPY=$HOME/vllm/bin/python                     # vLLM environment (judges, paraphrase, translation)
ATTACKER=${ATTACKER:-Qwen/Qwen2.5-32B-Instruct}; JUDGES=${JUDGES:-Qwen/Qwen2.5-32B-Instruct,mistralai/Mistral-Small-24B-Instruct-2501}
NA=${NA:-300}                                 # texts per model-based attack
stamp() { date -u +%H:%M:%S; }
run() { local n=$1; shift; echo "$(stamp) START $n"; "$@" > "logs/E_$n.log" 2>&1; local rc=$?; echo "$(stamp) END $n rc=$rc"; }

for M in "$@"; do
  S=${M//\//_}; CAL=$EP_RUNS/$S/calibration_cal.json
  for C in c4_sealed_L400 c4_kgw_L400 c4_synthid_L400; do
    run "${S}_para_$C"  $VPY $G/attacks_llm.py --model "$M" --corpus "$C" --mode paraphrase --attacker "$ATTACKER" --rounds 3 --n $NA
    run "${S}_trans_$C" $VPY $G/attacks_llm.py --model "$M" --corpus "$C" --mode translate --attacker "$ATTACKER" --langs uk,zh --n $NA
    run "${S}_dipper_$C" python3 $G/attacks_llm.py --model "$M" --corpus "$C" --mode dipper --n $NA
  done
  run "${S}_ppl_c4"    python3 $G/quality.py ppl --model "$M" --oracle "$ATTACKER" --a c4_sealed_L400 --b c4_plain_L400 --data c4 --field prompt
  run "${S}_ppl_dolly" python3 $G/quality.py ppl --model "$M" --oracle "$ATTACKER" --a dolly_sealed --b dolly_plain
  for J in ${JUDGES//,/ }; do
    run "${S}_judge_${J//\//_}" $VPY $G/quality.py judge --model "$M" --judge "$J" --a dolly_sealed --b dolly_plain \
        --data dolly --field instruction --n 1000
  done
  # the attacked texts, detected with the evaluated model (true key) and by the baselines' detectors
  run "${S}_detect_attacks" python3 $G/detect_corpus.py --model "$M" --dtype float16 --attacks runs:attacks_c4_sealed_L400 \
      --cal "$CAL" --name attacks_llm
  for sch in kgw synthid; do
    run "${S}_base_$sch" python3 $G/baselines.py --model "$M" --dtype float16 --scheme $sch --positives "c4_${sch}_L400" \
        --lengths 400 --nulls data:c4:human,runs:c4_plain_L400:text --max-null-texts 10000 --attacks "attacks_c4_${sch}_L400" --name attacks_llm
  done
done
echo "$(stamp) PHASE E DONE"
