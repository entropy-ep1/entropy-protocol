#!/bin/bash
# Scale bridge, part 8 (13.09.2026, 18:15): Qwen2.5-1.5B-Instruct on the reference code at both operating points and
# unsealed, replacing part 6. Part 6 ran on the frozen copy ../entropy-core-gsm, whose batch path (like the
# reference's before the MPS cache fix) filled the 16 GB Mac: 13-19 GB footprints, stopped at 17:28 and 17:34. With
# HFBackend.tidy the same sealing holds 4.3 GB at batch 16 (bench/logs/bridge/diag_mps_memory_*.log), so batch 16.
#   0. the reference's whole test suite and the JS vectors (the code changed at 18:10); abort if they fail
#   1. GSM8K, 100 problems, greedy: plain / quality (guard 0.9) / strong; accuracy; batch 4 vs 16 on 16 ids
#   2. prose, 40 prompts, T 0.7: plain / quality / strong; detection (quality: gate 0.9; strong: none);
#      perplexity (gpt2-medium oracle) and diversity against plain
#   3. detection on GSM8K (quality: gate 0.9; strong: none)
# Runs under bench/mem_watchdog.sh; a step that reads a corpus runs only if that corpus is complete.
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
echo "$(stamp) part 8; code $CODE"
M=Qwen/Qwen2.5-1.5B-Instruct; S=${M//\//_}
texts() { local f="$EP_RUNS/$S/$1.jsonl"; if [ -f "$f" ]; then grep -c . "$f"; else echo 0; fi; }
need() {  # need corpus:count ... -> false (and a SKIP line) unless every corpus has its count of texts
  local c have
  for c in "$@"; do have=$(texts "${c%:*}")
    if [ "$have" -lt "${c#*:}" ]; then echo "$(stamp) SKIP: corpus ${c%:*} has $have of ${c#*:} texts"; return 1; fi
  done
}

run "tests_$CODE" python3 -W ignore -m pytest tests -q -p no:cacheprovider
run "js_vectors_$CODE" node impl-js/check-vectors.mjs
if ! tail -3 "$L/tests_$CODE.log" | grep -q " passed" || tail -3 "$L/tests_$CODE.log" | grep -q "failed\|error" \
   || ! grep -q "checks passed" "$L/js_vectors_$CODE.log"; then
  echo "$(stamp) ABORT: the reference does not pass its tests (see $L/tests_$CODE.log, js_vectors_$CODE.log)"; exit 1
fi
touch "$L/tests_ok_$CODE"

G=(python3 bench/gpu/seal_corpus.py --model $M --device mps --dtype float16 --data gsm8k --field question --chat \
   --max-new 400 --temperature 0)
run "gsm_${S}_plain" "${G[@]}" --n 100 --batch 16 --mode plain --name gsm_plain
run "gsm_${S}_quality" "${G[@]}" --n 100 --batch 16 --mode sealed --guard 0.9 --name gsm_quality
run "gsm_${S}_strong" "${G[@]}" --n 100 --batch 16 --mode sealed --name gsm_strong
for V in quality strong; do
  need gsm_$V:100 gsm_plain:100 && run "q_${S}_gsm_$V" python3 bench/gpu/quality.py gsm8k --model $M --device mps \
      --a gsm_$V --b gsm_plain
done
# the same corpus name (the document ids, hence the checkpoints, derive from it) in a separate directory, at batch 4
B4=$REF/bench/results/runs_batch4
run "gsm_${S}_quality_b4" env EP_RUNS=$B4 "${G[@]}" --n 16 --batch 4 --mode sealed --guard 0.9 --name gsm_quality
[ -f "$B4/$S/gsm_quality.jsonl" ] && need gsm_quality:100 && run "batchinv_${S}_gsm_quality" python3 \
    bench/compare_batch_invariance.py "$B4/$S/gsm_quality.jsonl" "$EP_RUNS/$S/gsm_quality.jsonl"

prose() { run "prose_${S}_$1" python3 bench/gpu/seal_corpus.py --model $M --device mps --dtype float16 --data prose_offline \
          --field prompt --chat --n 40 --batch 16 --max-new 400 --min-new 250 --temperature 0.7 --name prose_$1 "${@:2}"; }
prose plain --mode plain
prose quality --mode sealed --guard 0.9
prose strong --mode sealed
need prose_quality:40 && run "detp_${S}_quality" python3 bench/gpu/detect_corpus.py --model $M --device mps \
    --dtype float16 --positives prose_quality --lengths 100,200 --gate 0.9 --name prose_quality
need prose_strong:40 && run "detp_${S}_strong" python3 bench/gpu/detect_corpus.py --model $M --device mps --dtype float16 \
    --positives prose_strong --lengths 100,200 --gate 0 --name prose_strong
for V in quality strong; do
  need prose_$V:40 prose_plain:40 && run "qp_${S}_$V" python3 bench/gpu/quality.py ppl --model $M --device mps \
      --dtype float32 --oracle gpt2-medium --a prose_$V --b prose_plain --data prose_offline --field prompt
  need prose_$V:40 prose_plain:40 && run "qd_${S}_$V" python3 bench/gpu/quality.py diversity --model $M --a prose_$V \
      --b prose_plain
done

need gsm_quality:100 && run "det_${S}_gsm_quality" python3 bench/gpu/detect_corpus.py --model $M --device mps \
    --dtype float16 --positives gsm_quality --lengths 100,200,400 --gate 0.9 --name gsm_quality
need gsm_strong:100 && run "det_${S}_gsm_strong" python3 bench/gpu/detect_corpus.py --model $M --device mps \
    --dtype float16 --positives gsm_strong --lengths 100,200,400 --gate 0 --name gsm_strong
echo "$(stamp) BRIDGE PART 8 DONE"
