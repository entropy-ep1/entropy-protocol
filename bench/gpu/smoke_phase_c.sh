#!/bin/bash
# Local dry run of the phase C step list (run_phase_c.sh) on a small chat model and the CPU: every distinct shape of
# call - the many-key null with a gate, canonical baselines at --top-p 1.0 with length-matched nulls, and one model
# family end to end (calibration, fp16-only-style table from position 4, sealed / quality / plain, records in two
# precisions, detection with a runs: null, GSM8K through the chat template, the quality check, speed). Numbers here
# mean nothing; the point is that no step dies on a flag, a path or a missing file on a paid machine.
# Usage: bash bench/gpu/smoke_phase_c.sh <scratch dir> [model]
set -euo pipefail
cd "$(dirname "$0")"
SP=${1:?scratch dir}; M=${2:-Qwen/Qwen2.5-0.5B-Instruct}; S=${M//\//_}
rm -rf "$SP"; mkdir -p "$SP"
export EP_RUNS=$SP/runs EP_DATA=$SP/data HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore
C=(--model "$M" --device cpu)
step() { echo "== $*"; }
step data;      python3 prep_data.py --tiny | tail -1
# a calibration table first: the detector and the sealer both read one (--cal)
step cal;       python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --offset 20 --n 6 --batch 3 \
                  --max-new 90 --min-new 80 --mode sealed --name C_s_cal | tail -1
                for cfg in "float32 1" "float32 3" "bfloat16 3"; do set -- $cfg
                  python3 calib_dump.py "${C[@]}" --dtype "$1" --batch "$2" --corpus C_s_cal | tail -1; done
                REF=$(ls "$EP_RUNS/$S/calib" | grep '^C_s_cal__.*_float32_default_b1\.npz$' | sed 's/^C_s_cal__//; s/\.npz$//')
                python3 calib_table.py --model "$M" --corpus C_s_cal --reference "$REF" --certify fp32 --from-position 4 | tail -1
                CAL=$EP_RUNS/$S/calibration_C_s_cal.json; [ -s "$CAL" ]
step sealed;    python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --n 6 --batch 3 --max-new 90 \
                  --min-new 90 --mode sealed --cal "$CAL" --name C_s_sealed | tail -1
                python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --n 6 --batch 3 --max-new 90 \
                  --min-new 90 --mode sealed --guard 0.9 --cal "$CAL" --name C_s_quality | tail -1
                python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --n 6 --batch 3 --max-new 90 \
                  --min-new 90 --mode plain --name C_s_plain | tail -1
step null;      python3 detect_corpus.py "${C[@]}" --dtype float32 --positives C_s_sealed --lengths 40,80 \
                  --nulls data:c4:human --max-null-texts 12 --null-keys 5 --cal "$CAL" --name C_s_null | tail -1
                python3 detect_corpus.py "${C[@]}" --dtype float32 --positives C_s_quality --lengths 40,80 --gate 0.9 \
                  --nulls data:c4:human --max-null-texts 12 --null-keys 5 --cal "$CAL" --name C_s_null_q | tail -1
step baselines; python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --offset 10 --n 4 --batch 2 \
                  --max-new 90 --min-new 90 --top-p 1.0 --mode kgw --name C_s_kgw_p1 | tail -1
                python3 seal_corpus.py "${C[@]}" --dtype float32 --data c4 --field prompt --offset 10 --n 4 --batch 2 \
                  --max-new 90 --min-new 90 --top-p 1.0 --mode synthid --name C_s_synthid_p1 | tail -1
                python3 baselines.py "${C[@]}" --dtype float32 --scheme kgw --positives C_s_kgw_p1 --lengths 40,80 \
                  --nulls data:c4:human --max-null-texts 8 --name C_s_p1 | tail -1
                python3 baselines.py "${C[@]}" --dtype float32 --scheme synthid --positives C_s_synthid_p1 --lengths 40,80 \
                  --nulls data:c4:human --max-null-texts 8 --name C_s_p1 | tail -1
step records;   python3 verify_corpus.py "${C[@]}" --dtype float32 --batch 3 --corpus C_s_sealed | tail -1
                python3 verify_corpus.py "${C[@]}" --dtype float32 --batch 3 --corpus C_s_quality | tail -1
                python3 verify_corpus.py "${C[@]}" --dtype bfloat16 --batch 3 --corpus C_s_sealed --only-same | tail -1
step detect;    python3 detect_corpus.py "${C[@]}" --dtype float32 --positives C_s_sealed --lengths 40,80 \
                  --nulls "data:c4:human,runs:C_s_plain:text" --max-null-texts 6 --null-keys 3 --cal "$CAL" --name C_s | tail -1
step gsm8k;     python3 seal_corpus.py "${C[@]}" --dtype float32 --data gsm8k --field question --chat --n 1 --batch 1 \
                  --max-new 80 --temperature 0 --mode sealed --cal "$CAL" --name C_s_gsm8k_sealed | tail -1
                python3 seal_corpus.py "${C[@]}" --dtype float32 --data gsm8k --field question --chat --n 1 --batch 1 \
                  --max-new 80 --temperature 0 --mode plain --name C_s_gsm8k_plain | tail -1
                python3 quality.py gsm8k "${C[@]}" --a C_s_gsm8k_sealed --b C_s_gsm8k_plain | tail -1
step speed;     python3 speed.py "${C[@]}" --dtype float32 --batches 1,2 --tokens 24 --verify-tokens 120 --repeats 1 | tail -1
step fp32_fit;  python3 - "$M" 48 <<'EOF'
import sys, json, huggingface_hub as h
cfg = json.load(open(h.hf_hub_download(sys.argv[1], "config.json")))
n = cfg.get("num_hidden_layers", 0) * cfg.get("hidden_size", 0) ** 2 * 12 + cfg.get("vocab_size", 0) * cfg.get("hidden_size", 0) * 2
print(f"~{n / 1e9:.2f}B parameters -> fp32 {n * 4 / 2**30:.1f} GiB of weights; fits a 48 GiB card: {n * 4 / 2**30 + 6 < 48}")
EOF
echo "SMOKE C OK"
