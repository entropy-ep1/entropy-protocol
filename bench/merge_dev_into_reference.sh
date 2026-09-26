#!/bin/bash
# Merge entropy-core-dev into the reference entropy-core (13.09.2026 evening). Run only when no bridge step imports
# from entropy-core/src. The reference's own results, logs, calibrations and bridge scripts stay; the code before the
# merge is kept as a tarball next to it.
set -euo pipefail
cd "$(dirname "$0")/../.."
REF=entropy-core; DEV=entropy-core-dev
BEFORE=$(cd $REF && python3 -W ignore -c "import sys; sys.path.insert(0, 'bench'); from _meta import code_digest; print(code_digest())")
tar czf "entropy-core_${BEFORE}_before_merge.tar.gz" -C $REF src tests spec impl-js bench/gpu bench/gates.py ARCHITECTURE.md
for d in src tests spec impl-js bench/gpu data; do
  mkdir -p "$REF/$d"
  rsync -a --exclude __pycache__ --exclude .pytest_cache "$DEV/$d/" "$REF/$d/"
done
cp "$DEV/bench/align_audit.py" "$DEV/bench/gates.py" "$REF/bench/"
cp "$DEV/ARCHITECTURE.md" "$REF/ARCHITECTURE.md"
AFTER=$(cd $REF && python3 -W ignore -c "import sys; sys.path.insert(0, 'bench'); from _meta import code_digest; print(code_digest())")
DEVD=$(cd $DEV && python3 -W ignore -c "import sys; sys.path.insert(0, 'bench'); from _meta import code_digest; print(code_digest())")
echo "reference before: $BEFORE  after: $AFTER  dev: $DEVD"
diff -rq --exclude __pycache__ --exclude .pytest_cache $DEV/src $REF/src && echo "src identical to dev"
diff -rq --exclude __pycache__ --exclude .pytest_cache $DEV/tests $REF/tests && echo "tests identical to dev"
