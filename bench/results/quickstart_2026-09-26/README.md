# The quick start, run once (26.09.2026)

`quickstart.sh` is the README's quick start; `run.out` is its output (library warnings removed) on a laptop CPU,
Qwen2.5-0.5B-Instruct in float32, 100 s wall time. `master.key` is a throwaway key published so the run can be
checked: with it, `sealed.json`'s token opens and `candidate.txt` verifies.

| File | What |
|---|---|
| `sealed.json` | the sealed answer and its custody token |
| `candidate.txt` | the answer as delivered: `VERIFIED`, 51 of 51 checkpoints; detection 42 of 44, p = 5.6e-11 |
| `edited.txt` | one word changed ("historian" to "sailor"): `MODIFIED`, the first sentence reported, 7 checkpoints diverge (`verify_edited.json`) |
