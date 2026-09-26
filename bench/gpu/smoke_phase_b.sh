#!/bin/bash
# Local dry run of the phase B step list (run_phase_b.sh) on a small model and the CPU: the same shapes of call -
# a shared sealed corpus, dumps in several precisions and both attention kernels, the table, the throughput probe
# with a larger batch, and the phase A records verified in another precision. Cheap insurance against a script
# defect that would otherwise be found on a paid machine.
# Usage: bash bench/gpu/smoke_phase_b.sh <scratch dir> [model]
set -euo pipefail
cd "$(dirname "$0")"
SP=${1:?scratch dir}; M=${2:-distilgpt2}
rm -rf "$SP"; mkdir -p "$SP"
export EP_RUNS=$SP/runs EP_DATA=$SP/data HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore
C=(--model "$M" --device cpu)
step() { echo "== $*"; }
step data;       python3 prep_data.py --tiny | tail -1
step cal_seal;   python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --n 8 --batch 4 \
                   --max-new 120 --min-new 100 --mode sealed --name B_cal | tail -1
step dumps;      for cfg in "float32 1 default" "float32 4 default" "float32 4 eager" "bfloat16 1 default"; do
                   set -- $cfg
                   if [ "$3" = eager ]; then
                     python3 calib_dump.py "${C[@]}" --dtype "$1" --batch "$2" --attn eager --corpus B_cal | tail -1
                   else
                     python3 calib_dump.py "${C[@]}" --dtype "$1" --batch "$2" --corpus B_cal | tail -1
                   fi
                 done
step table;      REF=$(ls "$EP_RUNS/$M/calib" | grep '_float32_default_b1\.npz$' | sed 's/^B_cal__//; s/\.npz$//')
                 python3 calib_table.py --model "$M" --corpus B_cal --reference "$REF" --certify fp32 | tail -1
step speed;      python3 speed.py "${C[@]}" --dtype float32 --batches 1,4 --tokens 32 --verify-tokens 200 --repeats 1 | tail -1
step verify;     python3 verify_corpus.py "${C[@]}" --dtype float32 --batch 4 --corpus B_cal | tail -1
                 python3 verify_corpus.py "${C[@]}" --dtype bfloat16 --batch 4 --corpus B_cal --only-same | tail -1
echo "SMOKE B OK"
