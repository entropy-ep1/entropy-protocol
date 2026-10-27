#!/usr/bin/env python3
"""Run one fuzz target (fuzz/targets.py).

    python fuzz/run.py <target> [libFuzzer options...] [corpus dir]     coverage-guided, with atheris (Linux)
    python fuzz/run.py <target> --replay <dir|file>...                  every input once, anywhere

A finding is any exception the target does not declare, a PropertyFailure, a hang or an out-of-memory; libFuzzer
writes the input as crash-*/timeout-*/oom-* in the working directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# libFuzzer's merge and fork modes re-execute argv[0] with the options alone; the target travels in the environment
TARGET_ENV = "EP1_FUZZ_TARGET"


def replay(name: str, paths: list[str]) -> int:
    import targets
    n = 0
    for p in paths:
        for f in sorted(Path(p).rglob("*")) if Path(p).is_dir() else [Path(p)]:
            if f.is_file():
                targets.run(name, f.read_bytes())
                n += 1
    print(f"{name}: {n} inputs replayed, no finding")
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1].startswith("-") and os.environ.get(TARGET_ENV):
        name, args = os.environ[TARGET_ENV], sys.argv[1:]    # re-executed by libFuzzer (merge, fork): options only
    else:
        if len(sys.argv) < 2:
            print(__doc__)
            return 2
        name, args = sys.argv[1], sys.argv[2:]
        if args[:1] == ["--replay"]:
            return replay(name, args[1:])
        os.environ[TARGET_ENV] = name
    import atheris
    with atheris.instrument_imports():
        import targets
    atheris.Setup([sys.argv[0], *args], lambda data: targets.run(name, data))
    atheris.Fuzz()
    return 0


if __name__ == "__main__":
    sys.exit(main())
