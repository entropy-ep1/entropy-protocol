#!/bin/bash
# Scale bridge, part 6: confirm the operating-point candidate on Qwen2.5-1.5B. On 0.5B, guard 0.9 at sealing with the
# detector's gate at 0.9 kept GSM8K accuracy (38% = unsealed) and prose quality (x1.01) with 90% detection at 1e-6
# on 200 prose tokens. Here: GSM8K with guard 0.9 alone on 1.5B (accuracy, detection), and the detector's gate on
# the 1.5B prose corpora sealed in part 5. Code: the frozen copy ../entropy-core-gsm (00a100caabf1bdf8).
# Batch 4, not 16: that code scores full [batch x length x vocabulary] logits; at batch 16 the first start (17:20)
# held a 19 GB footprint on this 16 GB Mac with the swap full, and was stopped at 17:28 after 2 of 7 batches (kept as
# runs/.../gsm_guard90_b16_killed.jsonl). With the row-by-row prefill, greedy decoding should not depend on the batch
# size (the plain and strong baselines ran at 16); those 32 texts, compared with the same ids here, check that.
set -u
cd "$(dirname "$0")/.."
REF=$PWD; GSM=$(cd ../entropy-core-gsm && pwd); DEV=$(cd ../entropy-core-dev && pwd)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 TOKENIZERS_PARALLELISM=false
L=$REF/bench/logs/bridge; mkdir -p "$L"
stamp() { date "+%H:%M:%S"; }
swap() { sysctl -n vm.swapusage | awk '{print $6}'; }
run() { local name=$1; shift; echo "$(stamp) START $name (swap used $(swap))"; "$@" > "$L/$name.log" 2>&1; local rc=$?
        echo "$(stamp) END $name rc=$rc"; }
cd "$GSM"; export PYTHONPATH=src EP_RUNS=$GSM/runs EP_DATA=$DEV/data
M=Qwen/Qwen2.5-1.5B-Instruct; S=${M//\//_}
run "gsm_${S}_guard90" python3 bench/gpu/seal_corpus.py --model $M --device mps --dtype float16 --data gsm8k --field question \
    --chat --n 100 --batch 4 --max-new 400 --temperature 0 --mode sealed --guard 0.9 --name gsm_guard90
run "q_${S}_gsm_guard90" python3 bench/gpu/quality.py gsm8k --model $M --device mps --a gsm_guard90 --b gsm_plain
for G in 0.9 0.8; do
  run "det_${S}_gsm_guard90_d$G" python3 bench/gpu/detect_corpus.py --model $M --device mps --dtype float16 \
      --positives gsm_guard90 --lengths 100,200,400 --gate $G --name gsm_guard90_d$G
  run "detp_${S}_guard90_d$G" python3 bench/gpu/detect_corpus.py --model $M --device mps --dtype float16 \
      --positives prose_guard90 --lengths 100,200 --gate $G --name prose_guard90_d$G
  run "detp_${S}_g90u90_d$G" python3 bench/gpu/detect_corpus.py --model $M --device mps --dtype float16 \
      --positives prose_g90u90 --lengths 100,200 --gate $G --name prose_g90u90_d$G
done
echo "$(stamp) BRIDGE PART 6 DONE"
