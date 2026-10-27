"""Fuzz targets (fuzz/targets.py) in every test run: each committed corpus input replayed - the seeds from real data
and every input a campaign ever found (fuzz/corpus/<target>/) - and each target driven by Hypothesis, from mutated
seeds and from arbitrary bytes. A target may raise only its declared errors; a property it checks must hold. The
long, coverage-guided campaigns run with atheris (fuzz/run.py, .clusterfuzzlite/)."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

hypothesis = pytest.importorskip("hypothesis")
pytest.importorskip("signxml")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "fuzz"))
import targets  # noqa: E402

NEEDS_NODE = {"canon_diff", "cbor_diff"}
NAMES = sorted(n for n in targets.TARGETS if n not in NEEDS_NODE or shutil.which("node"))


def corpus(name: str) -> list[bytes]:
    d = ROOT / "fuzz/corpus" / name
    return [f.read_bytes() for f in sorted(d.iterdir())] if d.is_dir() else []


@pytest.mark.parametrize("name", NAMES)
def test_corpus_replays(name):
    for data in corpus(name):
        targets.run(name, data)


@st.composite
def mutated(draw, seeds: list[bytes]):
    data = bytearray(draw(st.sampled_from(seeds)) if seeds else draw(st.binary(max_size=64)))
    for _ in range(draw(st.integers(1, 8))):
        op = draw(st.integers(0, 3))
        k = draw(st.integers(0, max(len(data), 1) - 1)) if data else 0
        if op == 0 and data:
            data[k] = draw(st.integers(0, 255))
        elif op == 1:
            data[k:k] = draw(st.binary(min_size=1, max_size=8))
        elif op == 2 and data:
            del data[k:k + draw(st.integers(1, 16))]
        elif data:
            data = data[:k]                                  # truncation
    return bytes(data)


@pytest.mark.parametrize("name", NAMES)
def test_target_under_hypothesis(name):
    seeds = corpus(name)

    @settings(max_examples=60 if name in NEEDS_NODE | {"pack", "appraise", "rim"} else 200, deadline=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large], database=None)
    @given(st.one_of(mutated(seeds), st.binary(max_size=512)))
    def run(data):
        targets.run(name, data)

    run()
