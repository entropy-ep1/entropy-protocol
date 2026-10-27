"""The table of code points assigned in Unicode 14.0, for canonical form v5's version pin (canon.py, impl-js).

    python spec/make_unicode14.py <DerivedAge.txt of Unicode 17.0.0>

Source: https://www.unicode.org/Public/17.0.0/ucd/DerivedAge.txt (Unicode License v3), SHA-256 below; every line
gives the version that assigned a range. Canonical form v5 normalizes as Unicode 14.0 does - the version of the
oldest Python the package supports - by cutting the text at every code point not assigned in 14.0 (such a code point
is, to Unicode 14.0, a starter that nothing composes or reorders across) and normalizing the pieces: by Unicode's
normalization stability policy every later version normalizes those pieces the same way. Writes
src/entropy_protocol/_unicode14.py and impl-js/unicode14.mjs.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHA256 = "f8ecdf768bdc210f"            # first 16 hex of the file's SHA-256 (full value in the generated headers)


def ranges(text: str) -> list[tuple[int, int]]:
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        cps, age = (x.strip() for x in line.split(";"))
        major, minor = (int(x) for x in age.split("."))
        if (major, minor) > (14, 0):
            continue
        a, _, b = cps.partition("..")
        out.append((int(a, 16), int(b or a, 16)))
    out.sort()
    merged: list[list[int]] = []
    for a, b in out:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def main() -> int:
    src = Path(sys.argv[1]).read_bytes()
    digest = hashlib.sha256(src).hexdigest()
    if not digest.startswith(SHA256):
        sys.exit(f"unexpected DerivedAge.txt (sha256 {digest})")
    text = src.decode("utf-8")
    if not re.search(r"^# DerivedAge-17\.0\.0\.txt", text, re.M):
        sys.exit("not DerivedAge-17.0.0.txt")
    rs = ranges(text)
    head = (f"Code points assigned in Unicode 14.0, from DerivedAge-17.0.0.txt (sha256 {digest}), by "
            f"spec/make_unicode14.py: {len(rs)} ranges. Generated - do not edit.")
    py = ['"""' + head + '"""', "from __future__ import annotations", "", "from bisect import bisect_right", "",
          "STARTS = (" + ", ".join(f"0x{a:X}" for a, _ in rs) + ",)",
          "ENDS = (" + ", ".join(f"0x{b:X}" for _, b in rs) + ",)", "", "",
          "def assigned(cp: int) -> bool:",
          "    i = bisect_right(STARTS, cp) - 1",
          "    return i >= 0 and cp <= ENDS[i]", ""]
    (ROOT / "src/entropy_protocol/_unicode14.py").write_text(_wrap("\n".join(py)))
    js = [f"// {head}",
          "const STARTS = [" + ", ".join(f"0x{a:X}" for a, _ in rs) + "];",
          "const ENDS = [" + ", ".join(f"0x{b:X}" for _, b in rs) + "];",
          "export function assigned14(cp) {",
          "  let lo = 0, hi = STARTS.length - 1, i = -1;",
          "  while (lo <= hi) { const m = (lo + hi) >> 1; if (STARTS[m] <= cp) { i = m; lo = m + 1; } else hi = m - 1; }",
          "  return i >= 0 && cp <= ENDS[i];",
          "}", ""]
    (ROOT / "impl-js/unicode14.mjs").write_text(_wrap("\n".join(js)))
    print(f"{len(rs)} ranges; sha256 {digest}")
    return 0


def _wrap(text: str, width: int = 120) -> str:
    """Break the long tuple and array lines at commas so the generated files stay readable."""
    out = []
    for line in text.split("\n"):
        lead = "// " if line.startswith("//") else "    "
        while len(line) > width:
            cut = max(line.rfind(", ", 0, width), line.rfind(" ", 0, width)) + 1
            out.append(line[:cut].rstrip())
            line = lead + line[cut:].lstrip()
        out.append(line)
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
