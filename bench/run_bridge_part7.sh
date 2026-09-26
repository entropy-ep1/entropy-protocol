#!/bin/bash
# Scale bridge, part 7 (13.09 evening; Serhii: "Перестроить"): Phi-3-mini-4k-instruct (3.8B, SentencePiece tokenizer,
# fp16 on MPS) on the reference code (entropy-core-dev merged into entropy-core, then the MPS cache fix), at both
# operating points and unsealed. In order of what the operating-point decision needs first:
#   1. alignment audit (canonical generation on a SentencePiece tokenizer)                    20 chat prompts x 2 modes
#   2. GSM8K (50 problems, greedy): plain / quality / strong; accuracy
#   3. prose: plain / quality / strong; detection (quality: gate 0.9; strong: none); perplexity, diversity
#   4. detection on GSM8K
#   5. gates (verification G1-G3/G6 + detection) at the quality and the strong operating points   30 texts each
# Phi-3 keeps a key and a value per head (no grouped attention): 393 KB of cache per token; at batch 4 it holds
# 9.1-10.2 GB on MPS with the cache fix and a 12-14 GB footprint, so batch 4. Runs after part 8, whose first step ran
# the test suite on this code (tests_ok_<code>); without that marker the suite runs here first.
# Runs under bench/mem_watchdog.sh; a step that reads a corpus runs only if that corpus is complete.
# Rerun after a stop (20:31): what is complete is kept (the audit, finished corpora and gates; sealing resumes by id).
# Sealing runs in processes of CH texts: at 20:06 and 20:28 both sealed GSM8K corpora (one process each) were
# stopped at their tenth batch at 16 GB, while four batches in one process stay at 12-14 GB after every batch
# (bench/logs/bridge/diag_phi3_growth_seal_vs_genonly.log); something outside the allocator grows slowly across
# batches (to find). Same code and batch size: the texts do not depend on the process they were made in.
set -u
cd "$(dirname "$0")/.."
REF=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
L=$REF/bench/logs/bridge; mkdir -p "$L" "$REF/bench/results"
stamp() { date "+%H:%M:%S"; }
swap() { sysctl -n vm.swapusage | awk '{print $6}'; }
run() { local name=$1; shift; echo "$(stamp) START $name (swap used $(swap))"; "$@" > "$L/$name.log" 2>&1; local rc=$?
        echo "$(stamp) END $name rc=$rc"; return $rc; }
export PYTHONPATH=src EP_RUNS=$REF/bench/results/runs EP_DATA=$REF/data
CODE=$(python3 -W ignore -c "import sys; sys.path.insert(0, 'bench'); from _meta import code_digest; print(code_digest())" 2>/dev/null)
echo "$(stamp) part 7; code $CODE"
P=microsoft/Phi-3-mini-4k-instruct; U=microsoft_phi-3-mini-4k-instruct; S=${P//\//_}
texts() { local f="$EP_RUNS/$S/$1.jsonl"; if [ -f "$f" ]; then grep -c . "$f"; else echo 0; fi; }
need() {  # need corpus:count ... -> false (and a SKIP line) unless every corpus has its count of texts
  local c have
  for c in "$@"; do have=$(texts "${c%:*}")
    if [ "$have" -lt "${c#*:}" ]; then echo "$(stamp) SKIP: corpus ${c%:*} has $have of ${c#*:} texts"; return 1; fi
  done
}
todo_in() {  # todo_in CORPUS DATA OFFSET N -> items of DATA[OFFSET:OFFSET+N] not yet in CORPUS
  python3 - "$EP_RUNS/$S/$1.jsonl" "$EP_DATA/$2.jsonl" "$3" "$4" <<'PY'
import json, os, sys
done = {json.loads(l)["id"] for l in open(sys.argv[1]) if l.strip()} if os.path.exists(sys.argv[1]) else set()
ids = [json.loads(l)["id"] for l in open(sys.argv[2]) if l.strip()][int(sys.argv[3]):int(sys.argv[3]) + int(sys.argv[4])]
print(sum(i not in done for i in ids))
PY
}
CH=16
seal_to() {  # seal_to CORPUS DATA FIELD TOTAL args... : seal_corpus in processes of CH texts until CORPUS is complete
  local name=$1 data=$2 field=$3 total=$4 off n; shift 4
  for ((off = 0; off < total; off += CH)); do
    n=$(( total - off < CH ? total - off : CH ))
    [ "$(todo_in "$name" "$data" $off $n)" -eq 0 ] && continue
    run "${name}_${S}_o$off" python3 bench/gpu/seal_corpus.py --model $P --device mps --dtype float16 --data "$data" \
        --field "$field" --chat --batch 4 --offset $off --n $n --name "$name" "$@"
  done
}
rows_in() { python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['rows']))" "$1" 2>/dev/null || echo 0; }

if [ -f "$L/tests_ok_$CODE" ]; then
  echo "$(stamp) tests: passed on this code in part 8 ($L/tests_$CODE.log)"
else
  run "tests_$CODE" python3 -W ignore -m pytest tests -q -p no:cacheprovider
  run "js_vectors_$CODE" node impl-js/check-vectors.mjs
  if ! tail -3 "$L/tests_$CODE.log" | grep -q " passed" || tail -3 "$L/tests_$CODE.log" | grep -q "failed\|error" \
     || ! grep -q "checks passed" "$L/js_vectors_$CODE.log"; then
    echo "$(stamp) ABORT: the reference does not pass its tests (see $L/tests_$CODE.log, js_vectors_$CODE.log)"; exit 1
  fi
  touch "$L/tests_ok_$CODE"
fi

if [ "$(rows_in bench/results/align_${U}.json)" -ge 40 ]; then
  echo "$(stamp) align: complete ($(python3 -c "import json; print(json.load(open('bench/results/align_${U}.json'))['code'])"))"
else
  run "align_$U" python3 bench/align_audit.py --model $P --chat --device mps --dtype float16 \
      --out bench/results/align_${U}.json
fi

GS=(--max-new 400 --temperature 0)
seal_to gsm_plain gsm8k question 50 "${GS[@]}" --mode plain
seal_to gsm_quality gsm8k question 50 "${GS[@]}" --mode sealed --guard 0.9
seal_to gsm_strong gsm8k question 50 "${GS[@]}" --mode sealed
for V in quality strong; do
  need gsm_$V:50 gsm_plain:50 && run "q_${S}_gsm_$V" python3 bench/gpu/quality.py gsm8k --model $P --device mps \
      --a gsm_$V --b gsm_plain
done

PS=(--max-new 400 --min-new 250 --temperature 0.7)
seal_to prose_plain prose_offline prompt 40 "${PS[@]}" --mode plain
seal_to prose_quality prose_offline prompt 40 "${PS[@]}" --mode sealed --guard 0.9
seal_to prose_strong prose_offline prompt 40 "${PS[@]}" --mode sealed
D=(python3 bench/gpu/detect_corpus.py --model $P --device mps --dtype float16 --token-budget 8192)
need prose_quality:40 && run "detp_${S}_quality" "${D[@]}" --positives prose_quality --lengths 100,200 --gate 0.9 \
    --name prose_quality
need prose_strong:40 && run "detp_${S}_strong" "${D[@]}" --positives prose_strong --lengths 100,200 --gate 0 \
    --name prose_strong
for V in quality strong; do
  need prose_$V:40 prose_plain:40 && run "qp_${S}_$V" python3 bench/gpu/quality.py ppl --model $P --device mps \
      --dtype float32 --oracle gpt2-medium --a prose_$V --b prose_plain --data prose_offline --field prompt
  need prose_$V:40 prose_plain:40 && run "qd_${S}_$V" python3 bench/gpu/quality.py diversity --model $P --a prose_$V \
      --b prose_plain
done

need gsm_quality:50 && run "det_${S}_gsm_quality" "${D[@]}" --positives gsm_quality --lengths 100,200,400 --gate 0.9 \
    --name gsm_quality
need gsm_strong:50 && run "det_${S}_gsm_strong" "${D[@]}" --positives gsm_strong --lengths 100,200,400 --gate 0 \
    --name gsm_strong

for V in quality strong; do
  if ls bench/results/gates_${S}_mps_float16_N30_p7$V.json > /dev/null 2>&1; then
    echo "$(stamp) gates $V: complete"
  else
    run "gates_${U}_$V" python3 bench/gates.py --model $P --n 30 --device mps --dtype float16 --profile $V --tag p7$V
  fi
done
echo "$(stamp) BRIDGE PART 7 DONE"
