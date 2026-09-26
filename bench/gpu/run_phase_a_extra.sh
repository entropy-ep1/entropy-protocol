#!/bin/bash
# Phase A extra (15.09.2026): record density when the table certifies fp16 only. With fp32 in the table the record
# margin is 2 x 3 x 0.68 = 4.09 logits (fp32 sits 0.68 from the fp16 sealer); with fp16 alone it is 2 x 0.42 = 0.84.
# The detector takes its own class's tolerance either way, and generation does not read the table (high_margin 0),
# so the texts must equal those of A_c4_sealed / A_c4_quality byte for byte: a determinism check on this GPU too.
# Starts when phase A has printed PHASE A DONE. Usage (on the evaluation machine): bash bench/gpu/run_phase_a_extra.sh
set -uo pipefail
source "$HOME/ep.env"; cd "$HOME/ep"
export PYTHONPATH=$HOME/ep/src
M=${1:-Qwen/Qwen2.5-7B-Instruct}; S=${M//\//_}; G=bench/gpu
stamp() { date -u +%H:%M:%S; }
run() {
  local n=$1; shift
  if [ -f "logs/A_$n.ok" ]; then echo "$(stamp) SKIP $n (done earlier)"; return 0; fi
  echo "$(stamp) START $n"; "$@" > "logs/A_$n.log" 2>&1; local rc=$?; echo "$(stamp) END $n rc=$rc"
  [ "$rc" = 0 ] && touch "logs/A_$n.ok"; return 0
}
until grep -q "PHASE A DONE\|PHASE A STOPPED" logs/phase_a.log; do sleep 15; done
grep -q "PHASE A DONE" logs/phase_a.log || { echo "$(stamp) phase A did not finish: no extra"; exit 1; }
# the SynthID baseline again, with the repetition mask vectorized (its transformers loop held one CPU core for 13+ min)
run base_sid    python3 $G/baselines.py --model "$M" --dtype float16 --scheme synthid --positives A_c4_synthid \
                  --lengths 100,200,400 --nulls data:c4:human,runs:A_c4_plain:text --max-null-texts 1000 --name A
REF=$(ls "$EP_RUNS/$S/calib" | grep '^A_cal__.*_float16_default_b1\.npz$' | sed 's/^A_cal__//; s/\.npz$//')
CAL16=$EP_RUNS/$S/calibration_A_cal_fp16only.json
run table_fp16  python3 $G/calib_table.py --model "$M" --corpus A_cal --reference "$REF" --certify fp16 --out "$CAL16"
run quality_16  python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 128 --batch 32 \
                  --max-new 400 --min-new 400 --mode sealed --guard 0.9 --cal "$CAL16" --name A_c4_quality_fp16cert
run sealed_16   python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 128 --batch 32 \
                  --max-new 400 --min-new 400 --mode sealed --cal "$CAL16" --name A_c4_sealed_fp16cert
run verify_q16  python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus A_c4_quality_fp16cert
run verify_s16  python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus A_c4_sealed_fp16cert
echo "$(stamp) PHASE A EXTRA DONE"
