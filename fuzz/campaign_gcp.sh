#!/bin/bash
# A coverage-guided fuzzing campaign (atheris / libFuzzer) for every target in fuzz/targets.py, all at once, on one
# GCP machine: the committed tree is shipped, each target runs for MINUTES from its corpus with a 10 s per-input
# timeout and a 2 GB memory limit, the corpora grown by the run are merged (minimized) and brought back with every
# crash, timeout or out-of-memory input and the libFuzzer statistics. The platform deletes the machine at the cap
# (--max-run-duration), and the script deletes it at the end and confirms it is gone.
# Evidence: bench/results/fuzz/<stamp>/; grown corpora: fuzz/corpus/<target>/ (merged).
# Usage: GCP_PROJECT=<project> bash fuzz/campaign_gcp.sh [minutes] [machine type]
set -euo pipefail
MINUTES=${1:-60}
TYPE=${2:-e2-highcpu-32}
PROJECT=${GCP_PROJECT:?set GCP_PROJECT to the project the machine runs in}
ZONE=us-central1-a
REPO=$(cd "$(dirname "$0")/.." && pwd)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
VM="ep-fuzz-$(date -u +%m%d%H%M)"
OUT="$REPO/bench/results/fuzz/$STAMP"
mkdir -p "$OUT"
G=(gcloud compute --project "$PROJECT")
SSH=(gcloud compute ssh "$VM" --project "$PROJECT" --zone "$ZONE" --quiet --command)
CREATED=""

cleanup() {
  set +e
  if [ -n "$CREATED" ]; then
    echo "cleanup: deleting $VM"
    "${G[@]}" instances delete "$VM" --zone "$ZONE" --quiet </dev/null
    "${G[@]}" instances describe "$VM" --zone "$ZONE" >/dev/null 2>&1 </dev/null && echo "cleanup: $VM STILL EXISTS" \
      || echo "cleanup: $VM is gone"
  fi
}
trap cleanup EXIT

git -C "$REPO" diff --quiet && git -C "$REPO" diff --cached --quiet || { echo "uncommitted changes"; exit 1; }
git -C "$REPO" rev-parse HEAD > "$OUT/commit.txt"
TARGETS=$(cd "$REPO" && PYTHONPATH=src:fuzz "$REPO/.venv/bin/python" -W ignore -c \
  "import targets; print(' '.join(sorted(targets.TARGETS)))")
echo "targets: $TARGETS"

echo "machine $VM ($TYPE) $(date -u +%H:%M:%SZ)"
"${G[@]}" instances create "$VM" --zone "$ZONE" --machine-type "$TYPE" --image-family ubuntu-2404-lts-amd64 \
  --image-project ubuntu-os-cloud --boot-disk-size 50GB --max-run-duration 3h --instance-termination-action DELETE \
  --labels project=entropy-fuzz </dev/null
CREATED=1
for _ in $(seq 1 30); do "${SSH[@]}" true </dev/null >/dev/null 2>&1 && break; sleep 10; done

echo "tools"
"${SSH[@]}" 'sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev clang libclang-rt-18-dev nodejs >/dev/null && python3 -m venv ~/v && ~/v/bin/pip -q install --upgrade pip && ~/v/bin/pip -q install "cryptography>=43.0.3" "cbor2>=5.6" "signxml>=4" "lxml>=4.9" "markdown-it-py>=3" && (~/v/bin/pip -q install atheris || CLANG_BIN=$(which clang) ~/v/bin/pip -q install --no-binary atheris atheris) && ~/v/bin/python -c "import atheris, sys; print(\"atheris ok\", sys.version.split()[0])" && node --version' </dev/null
git -C "$REPO" archive --format=tar HEAD | "${SSH[@]}" 'mkdir -p ep && tar -C ep -xf -'

echo "campaign: every target for $MINUTES minutes, in parallel ($(date -u +%H:%M:%SZ))"
"${SSH[@]}" "cd ~ && for t in $TARGETS; do mkdir -p work/\$t/new && (cd work/\$t && PYTHONPATH=\$HOME/ep/src:\$HOME/ep/fuzz \
  PATH=\$HOME/v/bin:\$PATH setsid nohup timeout $((MINUTES * 60 + 300)) python3 -W ignore \$HOME/ep/fuzz/run.py \$t \
  -max_total_time=$((MINUTES * 60)) -timeout=10 -rss_limit_mb=2048 -print_final_stats=1 new \$HOME/ep/fuzz/corpus/\$t \
  > log 2>&1 < /dev/null &); done; sleep 5; \
  ls work | wc -l" </dev/null
sleep $((MINUTES * 60 + 60))
for _ in $(seq 1 30); do
  "${SSH[@]}" "pgrep -f 'fuzz/run[.]py' >/dev/null" </dev/null || break
  sleep 20
done

echo "merge corpora, collect findings ($(date -u +%H:%M:%SZ))"
"${SSH[@]}" "cd ~ && for t in $TARGETS; do mkdir -p merged/\$t && (cd work/\$t && PYTHONPATH=\$HOME/ep/src:\$HOME/ep/fuzz \
  PATH=\$HOME/v/bin:\$PATH timeout 900 python3 -W ignore \$HOME/ep/fuzz/run.py \$t -merge=1 -timeout=10 \
  -rss_limit_mb=2048 \$HOME/merged/\$t \$HOME/ep/fuzz/corpus/\$t new > merge.log 2>&1 || true); \
  [ -n \"\$(ls work/\$t/new)\" ] && [ -z \"\$(ls merged/\$t)\" ] && echo \"MERGE FAILED: \$t\"; done; \
  find work -path '*/new/*' -size -65k > raw.lst; \
  tar -czf results.tgz work/*/log work/*/merge.log work/*/crash-* work/*/timeout-* work/*/oom-* work/*/slow-unit-* merged \
  -T raw.lst 2>/dev/null; ls -la results.tgz" </dev/null
"${G[@]}" scp --zone "$ZONE" --quiet "$VM:~/results.tgz" "$OUT/results.tgz" </dev/null
tar -C "$OUT" -xzf "$OUT/results.tgz"
"$REPO/.venv/bin/python" -I - "$OUT" <<'EOF'
import json, pathlib, re, sys
out = pathlib.Path(sys.argv[1]); rows = {}
for log in sorted((out / "work").glob("*/log")):
    t = log.parent.name; text = log.read_text(errors="replace")
    stats = dict(re.findall(r"^stat::(\w+):\s+(\d+)", text, re.M))
    cov = re.findall(r"cov: (\d+) ft: (\d+)", text)
    rows[t] = {"execs": int(stats.get("number_of_executed_units", 0)), "execs_per_sec": int(stats.get("average_exec_per_sec", 0)),
               "peak_rss_mb": int(stats.get("peak_rss_mb", 0)), "cov": int(cov[-1][0]) if cov else 0,
               "ft": int(cov[-1][1]) if cov else 0,
               "findings": sorted(p.name for p in log.parent.iterdir() if re.match(r"(crash|timeout|oom|slow-unit)-", p.name))}
(out / "summary.json").write_text(json.dumps(rows, indent=1))
print(json.dumps({t: (r["execs"], r["cov"], len(r["findings"])) for t, r in rows.items()}))
EOF
echo "done ($(date -u +%H:%M:%SZ)): $OUT"
