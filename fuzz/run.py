#!/usr/bin/env python3
"""Run one fuzz target (fuzz/targets.py).

    python fuzz/run.py <target> [libFuzzer options...] [corpus dir]     coverage-guided, with atheris (Linux)
    python fuzz/run.py <target> --replay <dir|file>...                  every input once, anywhere

A finding is any exception the target does not declare, a PropertyFailure, a hang or an out-of-memory; libFuzzer
writes the input as crash-*/timeout-*/oom-* in the working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    name = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "--replay":
        return replay(name, sys.argv[3:])
    import atheris
    with atheris.instrument_imports():
        import targets
    atheris.Setup([sys.argv[0], *sys.argv[2:]], lambda data: targets.run(name, data))
    atheris.Fuzz()
    return 0


if __name__ == "__main__":
    sys.exit(main())
