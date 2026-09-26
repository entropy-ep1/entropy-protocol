#!/bin/bash
# Phases B-D for one model on one machine, step by step (every step resumable; rerun a step to continue it).
# Usage (on the machine): bash bench/gpu/run_model.sh <model> <step> [<step> ...]
#   steps: data cal_seal cal_dump table corpora verify verify_g4 attacks detect baselines quality speed
#   N (texts per corpus), NULLS (null texts), KEYS (null keys) can be set in the environment.
set -uo pipefail
source "$HOME/ep.env"; cd "$HOME/ep"
M=${1:?model}; shift; S=${M//\//_}; G=bench/gpu; mkdir -p logs
N=${N:-1000}; NULLS=${NULLS:-20000}; KEYS=${KEYS:-50}; DT=${DT:-float16}
CAL=${CAL:-$EP_RUNS/$S/calibration_cal.json}
stamp() { date -u +%H:%M:%S; }
run() { local n=$1; shift; echo "$(stamp) START $S $n"; "$@" > "logs/${S}_$n.log" 2>&1; local rc=$?; echo "$(stamp) END $S $n rc=$rc"; }
seal() { run "$1" python3 $G/seal_corpus.py --model "$M" --dtype "$DT" "${@:2}" --name "$1"; }

for step in "$@"; do case $step in
data)
  run data python3 $G/prep_data.py --c4 $((NULLS + 2 * N + 200)) --wiki 5000 --langs en,uk,es,zh,ar --dolly 2000 ;;
cal_seal)                       # calibration corpus: sealed texts, measured on this machine (reference class fp16)
  seal cal --data c4 --field prompt --offset $((NULLS + 2 * N)) --n 200 --batch 32 --max-new 400 --min-new 300 --mode sealed ;;
cal_dump)                       # this machine's configurations (other GPUs run the same step on their machines)
  for cfg in "float16 1" "float16 32" "bfloat16 1" "bfloat16 32" "float32 1"; do
    set -- $cfg; run "dump_$1_b$2" python3 $G/calib_dump.py --model "$M" --dtype "$1" --batch "$2" --corpus cal
  done
  run dump_eager python3 $G/calib_dump.py --model "$M" --dtype float16 --batch 1 --attn eager --corpus cal ;;
table)                          # after all machines' dumps are gathered in runs/<model>/calib/
  REF=$(ls "$EP_RUNS/$S/calib" | grep "^cal__${REF_GPU:-NVIDIA-L40S}_float16_default_b1\.npz$" | sed 's/^cal__//; s/\.npz$//')
  run table python3 $G/calib_table.py --model "$M" --corpus cal --reference "$REF" --certify fp32,fp16 --out "$CAL" ;;
corpora)
  seal c4_sealed_L1000 --data c4 --field prompt --offset $NULLS --n $N --batch 32 --max-new 1000 --min-new 1000 --mode sealed --cal "$CAL"
  seal c4_sealed_L400  --data c4 --field prompt --offset $((NULLS + N)) --n $N --batch 32 --max-new 400 --min-new 400 --mode sealed --cal "$CAL"
  seal c4_plain_L400   --data c4 --field prompt --offset $((NULLS + N)) --n $N --batch 32 --max-new 400 --min-new 400 --mode plain
  seal c4_kgw_L400     --data c4 --field prompt --offset $((NULLS + N)) --n $N --batch 32 --max-new 400 --min-new 400 --mode kgw
  seal c4_synthid_L400 --data c4 --field prompt --offset $((NULLS + N)) --n $N --batch 16 --max-new 400 --min-new 400 --mode synthid
  # canonical KGW setting (bias before truncation is equivalent with no truncation) and the same for sealing
  seal c4_kgw_L400_p1    --data c4 --field prompt --offset $((NULLS + N)) --n 500 --batch 32 --max-new 400 --min-new 400 --top-p 1.0 --mode kgw
  seal c4_sealed_L400_p1 --data c4 --field prompt --offset $((NULLS + N)) --n 500 --batch 32 --max-new 400 --min-new 400 --top-p 1.0 --mode sealed --cal "$CAL"
  for T in 0 0.7 1.0; do
    seal "c4_sealed_T${T}" --data c4 --field prompt --offset $((NULLS + N)) --n 500 --batch 32 --max-new 400 --min-new 400 \
         --temperature "$T" --mode sealed --cal "$CAL"
  done
  seal dolly_sealed --data dolly --field instruction --chat --n $N --batch 32 --max-new 1024 --mode sealed --cal "$CAL"
  seal dolly_plain  --data dolly --field instruction --chat --n $N --batch 32 --max-new 1024 --mode plain
  seal gsm8k_sealed --data gsm8k --field question --chat --n 1319 --batch 32 --max-new 768 --temperature 0 --mode sealed --cal "$CAL"
  seal gsm8k_plain  --data gsm8k --field question --chat --n 1319 --batch 32 --max-new 768 --temperature 0 --mode plain
  seal he_sealed --data humaneval --field instruction --chat --n 164 --batch 32 --max-new 768 --temperature 0 --mode sealed --cal "$CAL"
  seal he_plain  --data humaneval --field instruction --chat --n 164 --batch 32 --max-new 768 --temperature 0 --mode plain
  for L in uk es zh ar; do
    seal "wiki_${L}_sealed" --data "wiki_$L" --field prompt --n 300 --batch 32 --max-new 400 --min-new 400 --mode sealed --cal "$CAL"
    seal "wiki_${L}_plain"  --data "wiki_$L" --field prompt --n 300 --batch 32 --max-new 400 --min-new 400 --mode plain
  done ;;
verify)
  run verify_L400  python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus c4_sealed_L400
  run verify_L1000 python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus c4_sealed_L1000
  run verify_dolly python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus dolly_sealed ;;
verify_g4)                      # this machine's configurations on the main corpus (run on every GPU type)
  for cfg in "float16 1" "float16 32" "bfloat16 32" "float32 1"; do
    set -- $cfg
    run "g4_$1_b$2" python3 $G/verify_corpus.py --model "$M" --dtype "$1" --batch "$2" --corpus c4_sealed_L400 --only-same \
        --n ${G4N:-1000} --tag "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9' '-')"
  done ;;
attacks)
  run attacks_text python3 $G/attacks_text.py --model "$M" --corpus c4_sealed_L400 --human data:c4:human
  run attacks_kgw  python3 $G/attacks_text.py --model "$M" --corpus c4_kgw_L400 --human data:c4:human
  run attacks_sid  python3 $G/attacks_text.py --model "$M" --corpus c4_synthid_L400 --human data:c4:human ;;
detect)
  NL=data:c4:human,data:wiki_en:text,data:wiki_uk:text,data:wiki_es:text,data:wiki_zh:text,data:wiki_ar:text,runs:c4_plain_L400:text,runs:dolly_plain:text
  run detect_main python3 $G/detect_corpus.py --model "$M" --dtype float16 --positives c4_sealed_L1000 \
      --lengths 50,100,200,400,1000 --nulls "$NL" --max-null-texts $NULLS --null-keys $KEYS \
      --attacks runs:attacks_c4_sealed_L400 --cal "$CAL" --name main
  for T in 0 0.7 1.0; do
    run "detect_T$T" python3 $G/detect_corpus.py --model "$M" --dtype float16 --positives "c4_sealed_T$T" \
        --lengths 100,200,400 --cal "$CAL" --name "T$T"
  done
  for L in uk es zh ar; do
    run "detect_$L" python3 $G/detect_corpus.py --model "$M" --dtype float16 --positives "wiki_${L}_sealed" \
        --lengths 100,200,400 --nulls "data:wiki_$L:text" --max-null-texts 5000 --null-keys 20 --cal "$CAL" --name "lang_$L"
  done
  run detect_chat python3 $G/detect_corpus.py --model "$M" --dtype float16 --positives dolly_sealed --lengths 100,200,400,1000 \
      --cal "$CAL" --name chat ;;
baselines)
  for sch in kgw synthid; do
    C4=c4_${sch}_L400; [ $sch = synthid ] && C4=c4_synthid_L400
    run "base_$sch" python3 $G/baselines.py --model "$M" --dtype float16 --scheme $sch --positives "$C4" --lengths 50,100,200,400 \
        --nulls data:c4:human,runs:c4_plain_L400:text --max-null-texts 10000 --attacks "attacks_$C4" --name main
  done ;;
quality)
  run q_div    python3 $G/quality.py diversity --model "$M" --a c4_sealed_L400 --b c4_plain_L400
  run q_gsm    python3 $G/quality.py gsm8k --model "$M" --a gsm8k_sealed --b gsm8k_plain
  run q_he     python3 $G/quality.py humaneval --model "$M" --a he_sealed --b he_plain ;;
speed)
  run speed python3 $G/speed.py --model "$M" --dtype float16 --batches 1,8,32 --tokens 256 ;;
*) echo "unknown step $step"; exit 2 ;;
esac; done
