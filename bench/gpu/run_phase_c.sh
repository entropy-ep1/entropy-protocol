#!/bin/bash
# Phase C, lean scope (25.09.2026): the numbers the release states, measured rather than extrapolated.
#  1. The detector's null at the scale of its claim. Phase A corroborated 1e-4 on 28 688 pairs; 1e-6 came from the
#     exact Binomial(n, 1/2) null. Here: 20 000 human C4 texts x 50 keys = 1 000 000 trials, at both operating
#     points (strong; quality = detector gate 0.9), on the same model and fp16-only table as phase A's headline.
#  2. Fair baselines. KGW and SynthID sealed canonically: --top-p 1.0, because transformers applies the KGW bias
#     after top-p (found 17.09), and judged against human nulls of the positives' own length (phase B prep).
#  3. Two more families and a bigger model: Qwen2.5-14B, Mistral-7B-v0.3, OLMo-2-7B (all Apache-2.0, none gated).
#     Per model, on this card: calibration corpus, dumps, an fp16-only table (the phase A rule), sealed and quality
#     corpora, records verified in fp16 and bf16, detection at 50-400 tokens against 20 000 null pairs, GSM8K.
# Inputs pushed from the Mac first (infra/sync.sh runs-up <name> <path> FROM=a1): phase A's fp16-only table and its
# two corpora, Qwen_Qwen2.5-7B-Instruct/{calibration_A_cal_fp16only.json, A_c4_sealed_fp16cert.jsonl,
# A_c4_quality_fp16cert.jsonl}. Resumable per step (logs/C_<step>.ok), like phases A and B.
# Usage (on the evaluation machine, after run_phase_b.sh): bash bench/gpu/run_phase_c.sh
set -uo pipefail
source "$HOME/ep.env"; cd "$HOME/ep"
export PYTHONPATH=$HOME/ep/src
G=bench/gpu; mkdir -p logs
GPU_GB=$(python3 -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 2**30))" 2>/dev/null || echo 0)
stamp() { date -u +%H:%M:%S; }
run() {
  local n=$1; shift
  if [ -f "logs/C_$n.ok" ]; then echo "$(stamp) SKIP $n (done earlier)"; return 0; fi
  echo "$(stamp) START $n"; "$@" > "logs/C_$n.log" 2>&1; local rc=$?; echo "$(stamp) END $n rc=$rc"
  [ "$rc" = 0 ] && touch "logs/C_$n.ok"; return 0
}
need() { [ -f "logs/C_$1.ok" ] || { echo "$(stamp) PHASE C STOPPED: $1 failed (logs/C_$1.log)"; exit 1; }; }

# enough C4 for 20 000 null texts beyond every prompt offset used in phases A-C (prompts sit below 2 600)
run data python3 $G/prep_data.py --c4 24000 --wiki 300 --langs en,uk --dolly 100 --only c4,wiki,dolly,gsm8k
need data

# ---- 1. the million-trial null, Qwen2.5-7B, phase A's fp16-only table ----------------------------------------
Q7=Qwen/Qwen2.5-7B-Instruct; S7=${Q7//\//_}; CAL7=$EP_RUNS/$S7/calibration_A_cal_fp16only.json
for f in "$CAL7" "$EP_RUNS/$S7/A_c4_sealed_fp16cert.jsonl" "$EP_RUNS/$S7/A_c4_quality_fp16cert.jsonl"; do
  [ -s "$f" ] || { echo "$(stamp) PHASE C STOPPED: missing $f (push it with infra/sync.sh runs-up)"; exit 1; }
done
run null1m_strong  python3 $G/detect_corpus.py --model "$Q7" --dtype float16 --positives A_c4_sealed_fp16cert \
                     --lengths 200,400 --nulls data:c4:human --max-null-texts 20000 --null-keys 50 --cal "$CAL7" \
                     --name C_null1m_strong
run null1m_quality python3 $G/detect_corpus.py --model "$Q7" --dtype float16 --positives A_c4_quality_fp16cert \
                     --lengths 200,400 --gate 0.9 --nulls data:c4:human --max-null-texts 20000 --null-keys 50 \
                     --cal "$CAL7" --name C_null1m_quality

# ---- 2. fair baselines, Qwen2.5-7B ----------------------------------------------------------------------------
run kgw_p1     python3 $G/seal_corpus.py --model "$Q7" --dtype float16 --data c4 --field prompt --offset 400 --n 256 \
                 --batch 32 --max-new 400 --min-new 400 --top-p 1.0 --mode kgw --name C_c4_kgw_p1
run synthid_p1 python3 $G/seal_corpus.py --model "$Q7" --dtype float16 --data c4 --field prompt --offset 400 --n 256 \
                 --batch 16 --max-new 400 --min-new 400 --top-p 1.0 --mode synthid --name C_c4_synthid_p1
run base_kgw   python3 $G/baselines.py --model "$Q7" --dtype float16 --scheme kgw --positives C_c4_kgw_p1 \
                 --lengths 100,200,400 --nulls data:c4:human --max-null-texts 4000 --name C_p1
run base_sid   python3 $G/baselines.py --model "$Q7" --dtype float16 --scheme synthid --positives C_c4_synthid_p1 \
                 --lengths 100,200,400 --nulls data:c4:human --max-null-texts 4000 --name C_p1

# ---- 3. more models --------------------------------------------------------------------------------------------
family() {                 # family <model> <tag>
  local M=$1 T=$2 S=${1//\//_}
  run "${T}_cal_seal" python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --offset 1000 \
                        --n 64 --batch 32 --max-new 400 --min-new 300 --mode sealed --name "C_${T}_cal"
  [ -f "logs/C_${T}_cal_seal.ok" ] || { echo "$(stamp) $T: no calibration corpus, family skipped"; return 0; }
  local CFGS=("float16 1" "float16 32" "bfloat16 32")
  # fp32 weights: 4 bytes a parameter; the 14B needs ~59 GB and stays in fp16/bf16
  python3 - "$M" "$GPU_GB" <<'EOF' && CFGS+=("float32 1")
import sys, json, huggingface_hub as h
cfg = json.load(open(h.hf_hub_download(sys.argv[1], "config.json")))
n = cfg.get("num_hidden_layers", 0) * cfg.get("hidden_size", 0) ** 2 * 12 + cfg.get("vocab_size", 0) * cfg.get("hidden_size", 0) * 2
sys.exit(0 if n * 4 / 2**30 + 6 < float(sys.argv[2]) else 1)
EOF
  for cfg in "${CFGS[@]}"; do
    set -- $cfg
    run "${T}_dump_$1_b$2" python3 $G/calib_dump.py --model "$M" --dtype "$1" --batch "$2" --corpus "C_${T}_cal"
  done
  [ -f "logs/C_${T}_dump_float16_b1.ok" ] || { echo "$(stamp) $T: no reference dump, family skipped"; return 0; }
  local REF; REF=$(ls "$EP_RUNS/$S/calib" | grep "^C_${T}_cal__.*_float16_default_b1\.npz$" | sed "s/^C_${T}_cal__//; s/\.npz$//")
  run "${T}_table" python3 $G/calib_table.py --model "$M" --corpus "C_${T}_cal" --reference "$REF" --certify fp16 \
                     --from-position 4
  [ -f "logs/C_${T}_table.ok" ] || { echo "$(stamp) $T: no table, family skipped"; return 0; }
  local CAL=$EP_RUNS/$S/calibration_C_${T}_cal.json
  run "${T}_sealed"  python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 128 --batch 32 \
                       --max-new 400 --min-new 400 --mode sealed --cal "$CAL" --name "C_${T}_c4_sealed"
  run "${T}_quality" python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 128 --batch 32 \
                       --max-new 400 --min-new 400 --mode sealed --guard 0.9 --cal "$CAL" --name "C_${T}_c4_quality"
  run "${T}_plain"   python3 $G/seal_corpus.py --model "$M" --dtype float16 --data c4 --field prompt --n 128 --batch 32 \
                       --max-new 400 --min-new 400 --mode plain --name "C_${T}_c4_plain"
  run "${T}_v_fp16"  python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus "C_${T}_c4_sealed"
  run "${T}_v_q"     python3 $G/verify_corpus.py --model "$M" --dtype float16 --batch 32 --corpus "C_${T}_c4_quality"
  run "${T}_v_bf16"  python3 $G/verify_corpus.py --model "$M" --dtype bfloat16 --batch 32 --corpus "C_${T}_c4_sealed" --only-same
  run "${T}_detect"  python3 $G/detect_corpus.py --model "$M" --dtype float16 --positives "C_${T}_c4_sealed" \
                       --lengths 50,100,200,400 --nulls data:c4:human,runs:C_${T}_c4_plain:text --max-null-texts 1000 \
                       --null-keys 20 --cal "$CAL" --name "C_${T}"
  run "${T}_detect_q" python3 $G/detect_corpus.py --model "$M" --dtype float16 --positives "C_${T}_c4_quality" \
                       --lengths 50,100,200,400 --gate 0.9 --nulls data:c4:human --max-null-texts 1000 --null-keys 20 \
                       --cal "$CAL" --name "C_${T}_quality"
  run "${T}_gsm_seal" python3 $G/seal_corpus.py --model "$M" --dtype float16 --data gsm8k --field question --chat --n 128 \
                       --batch 32 --max-new 512 --temperature 0 --mode sealed --cal "$CAL" --name "C_${T}_gsm8k_sealed"
  run "${T}_gsm_qual" python3 $G/seal_corpus.py --model "$M" --dtype float16 --data gsm8k --field question --chat --n 128 \
                       --batch 32 --max-new 512 --temperature 0 --mode sealed --guard 0.9 --cal "$CAL" --name "C_${T}_gsm8k_quality"
  run "${T}_gsm_plain" python3 $G/seal_corpus.py --model "$M" --dtype float16 --data gsm8k --field question --chat --n 128 \
                       --batch 32 --max-new 512 --temperature 0 --mode plain --name "C_${T}_gsm8k_plain"
  run "${T}_q_gsm"   python3 $G/quality.py gsm8k --model "$M" --a "C_${T}_gsm8k_sealed" --b "C_${T}_gsm8k_plain"
  run "${T}_q_gsm_q" python3 $G/quality.py gsm8k --model "$M" --a "C_${T}_gsm8k_quality" --b "C_${T}_gsm8k_plain"
  run "${T}_speed"   python3 $G/speed.py --model "$M" --dtype float16 --batches 1,32 --tokens 256
}
family mistralai/Mistral-7B-Instruct-v0.3 mistral7
family allenai/OLMo-2-1124-7B-Instruct   olmo7
family Qwen/Qwen2.5-14B-Instruct         qwen14
echo "$(stamp) PHASE C DONE (GPU ${GPU_GB} GB)"
