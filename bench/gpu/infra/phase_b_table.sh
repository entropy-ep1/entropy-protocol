#!/bin/bash
# Phase B, step 2 (on this Mac): one tolerance table from the dumps of every machine, plus the report of how the
# tolerance grows with the corpus and of what a verifier outside the table can still judge. Numpy only, no GPU.
# Usage: bash phase_b_table.sh [corpus] [model] [certify]
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$DIR/../../.." && pwd)
CORPUS=${1:-B_cal}; M=${2:-Qwen/Qwen2.5-7B-Instruct}; CERT=${3:-fp16}; S=${M//\//_}
WORK=$ROOT/bench/results/phase_b/$CORPUS; mkdir -p "$WORK/$S/calib"
for d in "$ROOT"/runs_gpu/*/"$S"/calib; do
  [ -d "$d" ] || continue
  cp -n "$d/${CORPUS}__"*.npz "$d/${CORPUS}__"*.json "$WORK/$S/calib/" 2>/dev/null || true
done
echo "dumps: $(ls "$WORK/$S/calib" | grep -c '\.npz$') from $(ls -d "$ROOT"/runs_gpu/*/ | wc -l | tr -d ' ') machines"
# the reference: fp16 batch 1 on the card that sealed the corpus (its meta names it), as in phase A; else the first
REF=""
SEALER=$(python3 -c "import json, glob, sys; m = [json.load(open(p))['env']['hardware'] for p in glob.glob(sys.argv[1])]; print(m[0] if m else '')" \
         "$ROOT/runs_gpu/*/$S/$CORPUS.meta.json" 2>/dev/null || true)
[ -n "$SEALER" ] && [ -f "$WORK/$S/calib/${CORPUS}__${SEALER// /-}_float16_default_b1.npz" ] && REF="${SEALER// /-}_float16_default_b1"
[ -n "$REF" ] || REF=$(ls "$WORK/$S/calib" | grep -E "^${CORPUS}__.*_float16_default_b1\.npz$" | head -1 | sed "s/^${CORPUS}__//; s/\.npz$//")
[ -n "$REF" ] || { echo "no fp16 batch-1 dump to use as the reference"; exit 1; }
echo "reference: $REF"
EP_RUNS="$WORK" python3 "$ROOT/bench/gpu/calib_table.py" --model "$M" --corpus "$CORPUS" --reference "$REF" \
  --certify "$CERT" --out "$WORK/calibration_${CORPUS}.json"
python3 "$ROOT/bench/diag_calib_corpus_and_classes.py" "$WORK" --model "$M" --corpus "$CORPUS" \
  --out "$WORK/growth_and_classes.json"
echo "table: $WORK/calibration_${CORPUS}.json"
