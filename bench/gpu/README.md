# GPU evaluation harness (PRIVATE until the covering provisional application is filed)

Plan and budget: `../../../GPU_EVAL_PLAN_RU.md`. Every script is resumable (one JSON line per finished item) and
writes `runs/<model>/...` with the code digest, model identity and environment of the run.

| Script | Protocol item |
|---|---|
| `prep_data.py` | data with commercial-use licences: C4, Wikipedia (5 languages), Dolly-15k, GSM8K, HumanEval |
| `seal_corpus.py` | corpora: `sealed` (reference operating point), `plain`, `kgw`, `synthid`; batched |
| `verify_corpus.py` | G1-G3, G6 at scale; G4 = the same corpus on another GPU / precision / kernel / batch |
| `calib_dump.py`, `calib_table.py` | calibration across machines; class 11 (noise flips at kept checkpoints) |
| `attacks_text.py` | attack classes 1-3 (+ truncation, sentence deletion) |
| `attacks_llm.py` | classes 4-6: LLM paraphrase (recursive), DIPPER, translation round trip, emoji attack |
| `detect_corpus.py` | G10: power by length, >= 10^6 (text, key) null pairs, other key, attacks, localization |
| `baselines.py` | KGW and SynthID-Text on the same models, prompts, lengths, attacks |
| `quality.py` | oracle perplexity, LLM judges, GSM8K, HumanEval, diversity |
| `speed.py` | G7 on GPUs: sealing overhead, verification throughput, memory |
| `smoke_local.sh` | the whole chain on distilgpt2 / CPU (run before any cloud time) |
| `run_phase_a.sh` | phase A on one GPU machine |
| `infra/launch.sh`, `bootstrap.sh`, `sync.sh`, `status.sh`, `terminate.sh` | AWS machines with a hard time cap |

Machines: `bash infra/launch.sh <name> <type> <hours> spot` → `ssh ... 'bash -s' < infra/bootstrap.sh` →
`bash infra/sync.sh up <name>` → run → `bash infra/sync.sh down <name>` (results to `runs_gpu/<name>/`).
Each machine powers itself off after its cap (`shutdown -h +N`, shutdown behaviour "terminate").
