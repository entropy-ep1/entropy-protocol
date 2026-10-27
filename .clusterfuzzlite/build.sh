#!/bin/bash -eu
# ClusterFuzzLite build: one fuzzer per target of fuzz/targets.py (the differential targets need Node and run in the
# test suite and in fuzz/campaign_gcp.sh instead), each with its seed corpus from fuzz/corpus/<target>/.
pip3 install "cryptography>=43.0.3" "cbor2>=5.6" "signxml>=4" "lxml>=4.9" "markdown-it-py>=3"
pip3 install --no-deps .
TARGETS=$(cd fuzz && python3 -c "import targets; print(' '.join(n for n in sorted(targets.TARGETS) if not n.endswith('_diff')))")
for t in $TARGETS; do
  cat > "fuzz/fuzz_${t}.py" <<PY
import sys
import atheris
with atheris.instrument_imports():
    import targets
atheris.Setup(sys.argv, lambda data: targets.run("${t}", data))
atheris.Fuzz()
PY
  # signxml loads its XML schemas as package data (signxml.schemas) through importlib.resources, which PyInstaller
  # does not follow: without --collect-all the first RIM that reaches schema validation fails to import it
  compile_python_fuzzer "fuzz/fuzz_${t}.py" --add-data "fuzz/targets.py:." --paths fuzz --collect-all signxml
  if [ -d "fuzz/corpus/${t}" ]; then
    (cd "fuzz/corpus/${t}" && zip -q -r "$OUT/fuzz_${t}_seed_corpus.zip" .)
  fi
done
