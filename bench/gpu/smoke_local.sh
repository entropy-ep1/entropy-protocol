#!/bin/bash
# Local end-to-end smoke test of the GPU harness on a small model (CPU), before any cloud time is spent.
# Usage: bash bench/gpu/smoke_local.sh <scratch dir> [model] [device] [oracle]
set -euo pipefail
cd "$(dirname "$0")"
SP=${1:?scratch dir}; M=${2:-distilgpt2}; DEV=${3:-cpu}; ORACLE=${4:-gpt2-medium}
rm -rf "$SP"; mkdir -p "$SP"
export EP_RUNS=$SP/runs EP_DATA=$SP/data HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore
C=(--model "$M" --device "$DEV")
step() { echo "== $*"; }
step data;      python3 prep_data.py --tiny | tail -1
step sealed;    python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --n 8 --batch 4 --max-new 120 --min-new 100 --mode sealed --name c4_sealed | tail -1
step plain;     python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --n 8 --batch 4 --max-new 120 --min-new 100 --mode plain --name c4_plain | tail -1
step kgw;       python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --n 4 --batch 4 --max-new 120 --min-new 100 --mode kgw --name c4_kgw | tail -1
step synthid;   python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --n 4 --batch 4 --max-new 120 --min-new 100 --mode synthid --name c4_synthid | tail -1
step verify;    python3 verify_corpus.py "${C[@]}" --dtype float32 --corpus c4_sealed --batch 8 | tail -1
step attacks;   python3 attacks_text.py --model "$M" --corpus c4_sealed --human data:c4:human | tail -1
step detect;    python3 detect_corpus.py "${C[@]}" --dtype float32 --positives c4_sealed --lengths 50,100 \
                  --nulls data:wiki_en:text,runs:c4_plain:text --null-keys 5 --attacks runs:attacks_c4_sealed --workers 2 --name smoke | tail -1
step calib;     python3 calib_dump.py "${C[@]}" --dtype float32 --corpus c4_sealed --batch 1 | tail -1
                python3 calib_dump.py "${C[@]}" --dtype float32 --corpus c4_sealed --batch 4 | tail -1
                REF=$(ls "$EP_RUNS/$M/calib" | grep '_b1.npz' | sed 's/.*__//; s/\.npz$//')
                python3 calib_table.py --model "$M" --corpus c4_sealed --reference "$REF" --certify fp32 | tail -1
step baselines; python3 baselines.py "${C[@]}" --dtype float32 --scheme kgw --positives c4_kgw --lengths 50,100 --nulls data:wiki_en:text --name smoke | tail -1
                python3 baselines.py "${C[@]}" --dtype float32 --scheme synthid --positives c4_synthid --lengths 50,100 --nulls data:wiki_en:text --name smoke | tail -1
step quality;   python3 quality.py ppl "${C[@]}" --dtype float32 --oracle "$ORACLE" --a c4_sealed --b c4_plain --data c4 --field prompt | tail -1
                python3 quality.py diversity --model "$M" --a c4_sealed --b c4_plain | tail -1
step speed;     python3 speed.py "${C[@]}" --dtype float32 --batches 1,2 --tokens 32 --verify-tokens 200 --repeats 1 | tail -2
echo "SMOKE OK"
