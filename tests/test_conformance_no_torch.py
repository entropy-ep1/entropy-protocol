"""The specification's conformance path needs no model stack.

Formats, statements, the transparency log, record delivery, the visible-text layer and the exact detection tail are
what a second implementation is checked against (spec/make_vectors.py). Importing them must not import torch, so a
conformance run - and anyone checking statements or evidence - does not have to install one.
"""
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_vector_path_imports_without_torch():
    code = (
        f"import sys; sys.path.insert(0, {SRC!r});"
        "from entropy_protocol import cose, crypto, delivery, hpke, statements, tlog;"
        "from entropy_protocol.canon import canonicalize;"
        "from entropy_protocol.stats import binomial_tail;"
        "from entropy_protocol.plain import normalize;"
        "from entropy_protocol.record import dumps;"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False", "the conformance path imported torch"


def test_detect_reexports_the_same_tail():
    from entropy_protocol import detect, stats
    assert detect.binomial_tail is stats.binomial_tail
