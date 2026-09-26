"""Does a sealed corpus depend on the generation batch size? Compare two corpora of the same model, mode and ids.

Greedy decoding (temperature 0) with the row-by-row prompt pass should give the same text for an id at any batch
size; this checks it on the ids both corpora have. For sealed corpora both must carry the same corpus name (document
ids, hence checkpoints, derive from it): e.g. bench/run_bridge_part8.sh seals 16 GSM8K questions at batch 4 into a
separate directory under the name of the batch-16 corpus.

Usage: python3 bench/compare_batch_invariance.py A.jsonl B.jsonl
"""
import json
import sys


def rows(path: str) -> dict:
    return {r["id"]: r for r in (json.loads(line) for line in open(path) if line.strip())}


def main():
    a, b = rows(sys.argv[1]), rows(sys.argv[2])
    common = sorted(set(a) & set(b))
    same_text = [i for i in common if a[i]["text"] == b[i]["text"]]
    fields = ("tokens", "enforced", "entries", "dropped")
    same_rec = [i for i in same_text if all(a[i].get(f) == b[i].get(f) for f in fields)]
    print(f"A {sys.argv[1]}: {len(a)} rows, batch {sorted({r.get('batch') for r in a.values()})}")
    print(f"B {sys.argv[2]}: {len(b)} rows, batch {sorted({r.get('batch') for r in b.values()})}")
    print(f"common ids {len(common)}; identical text {len(same_text)}; identical text and record fields "
          f"({', '.join(fields)}) {len(same_rec)}")
    for i in common:
        if a[i]["text"] != b[i]["text"]:
            ta, tb = a[i]["text"], b[i]["text"]
            k = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]), min(len(ta), len(tb)))
            print(f"  differs: {i}: first difference at character {k} of {len(ta)}/{len(tb)}: "
                  f"{ta[max(0, k - 30):k + 30]!r} vs {tb[max(0, k - 30):k + 30]!r}")


if __name__ == "__main__":
    main()
