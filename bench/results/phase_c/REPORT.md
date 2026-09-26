# Phase C — a million-trial null, canonical baselines, three more models (26.09.2026)

One NVIDIA L40S (AWS g6e.xlarge), fp16, `bench/gpu/run_phase_c.sh`, 03:09–07:12 UTC. Every number is a file under
`runs_gpu/c1/`; the summaries named below are JSON produced by the scripts in `bench/gpu/`.

## 1. The exact null, measured on two million pairs

Record-free detection reports a p-value from Binomial(n, ½). The claim is that it is exact: for any text not sealed
under the key, P(p ≤ α) ≤ α, with no calibration. Phase A checked it on 28,688 pairs, to 10⁻⁴. Here: 20,000 human C4
texts × 50 keys = **1,000,000 pairs** at each operating point of Qwen2.5-7B, plus 20,000 pairs per operating point on
each of three more models and 2,560 pairs of each model's own unsealed text (`detect_C_*.summary.json`).

| α | Qwen2.5-7B strong, 10⁶ pairs | Qwen2.5-7B quality, 10⁶ pairs | All four models, 2,127,680 pairs | Bound (pairs × α) |
|---|---|---|---|---|
| 10⁻² | 7,028 | 7,098 | 14,995 | 21,277 |
| 10⁻³ | 650 | 615 | 1,350 | 2,128 |
| 10⁻⁴ | 72 | 71 | 154 | 213 |
| 10⁻⁵ | 5 | 9 | 16 | 21.3 |
| 10⁻⁶ | 2 | 2 | 4 | 2.1 |

At every level down to 10⁻⁵ the false-positive count is 0.63–0.75 of the bound (a discrete exact test is
conservative). At 10⁻⁶, 4 against a bound of 2.1: the probability of 4 or more under a Poisson count with mean 2.1 is
0.17 — consistent with the bound, and the first measurement at this level. Pairs that share a text are correlated
(the same text under 50 keys), which widens the spread of these counts, not their expectation.

## 2. Baselines: KGW and SynthID-Text on the same model

KGW (γ = 0.25, δ = 2, left hash) and SynthID-Text (tournament depth 30, 5-gram context) as implemented in
transformers 4.53.2, 256 texts each of 400 tokens at temperature 0.8 and top-p 1.0, scored against 4,000 human C4
texts cut to the same length (`baseline_{kgw,synthid}_C_p1.summary.json`).

| | 100 tokens | 200 | 400 |
|---|---|---|---|
| KGW, detected at an empirical FPR of 1 % | 92.2 % | 98.1 % | 100 % |
| SynthID (mean score), same | 97.7 % | 99.2 % | 99.6 % |
| KGW's own p ≤ 10⁻², on human text (nominal 1 %) | 1.03 % | 1.48 % | **2.2 %** |
| KGW's own p ≤ 10⁻⁴, on human text (nominal 0.01 %) | **0.1 %** | 0.025 % | **0.1 %** |
| SynthID's own p ≤ 10⁻³, on human text (nominal 0.1 %) | 0.2 % | 0.125 % | 0.2 % |

Both are strong detectors of their own mark. The difference is the null: KGW's z-test p-value overstates its
confidence on human text — at a nominal 10⁻⁴, 4 of 4,000 human texts were flagged where 0.4 are expected
(P ≈ 0.001 by chance); at 400 tokens and a nominal 1 %, 88 where 40 are expected. SynthID's binomial p-value on the
mean g-value is close, about twice nominal at 10⁻³ (8 against 4). ep1's p-value stays below its bound on two
million pairs (section 1). The ep1 corpora were sealed at temperature 0.8 and top-p 0.9, the baselines at top-p 1.0; the comparison is of null calibration and of what each method certifies, not of
power at identical decoding.

A correction to phase A: its KGW figures (81 / 75 / 34 % at p ≤ 0.01 for 100 / 200 / 400 tokens) counted p = 0.0 as
a miss; see `bench/results/phase_b/REPORT.md`. This run used the corrected count.

## 3. Three more model families

Mistral-7B-Instruct-v0.3, OLMo-2-1124-7B-Instruct and Qwen2.5-14B-Instruct (all Apache-2.0, not gated), each with its
own calibration (64 texts), 128 texts per corpus, 400 tokens, temperature 0.8, top-p 0.9.

| | Qwen2.5-7B (phase A) | Mistral-7B-v0.3 | OLMo-2-7B | Qwen2.5-14B |
|---|---|---|---|---|
| fp16 tolerance (logits) | 0.42 | 0.73 | **2.43** | 0.89 |
| Records `VERIFIED`: sealed fp16 · quality fp16 · sealed bf16 | 128 · 128 · 128 | 128 · 128 · 128 | 128 · 128 · 128 | 128 · 128 · 128 |
| False `MODIFIED` | 0 | 0 | 0 | 0 |
| Edits reported `MODIFIED` (with location) | 99.2–100 % | 99.2–100 % | 98.4–100 % | 99.2–100 % |
| Detection at 10⁻⁶, strong, 200 · 400 tokens | 96.1 · 100 % | 95.3 · 99.2 % | 20.3 · 98.4 % | 78.9 · 100 % |
| Detection at 10⁻⁶, quality, 200 · 400 tokens | 61.7 · 95.3 % | 21.9 · 83.6 % | 0 · **0.8 %** | 17.2 · 82.8 % |
| GSM8K: unsealed · quality · strong (128 problems) | 87.5 · 85.2 · 39.1 | 52.3 · 50.0 · 14.1 | 71.9 · 69.5 · 36.7 | 88.3 · 90.6 · 51.6 |
| Verification, tokens/s (batch 1) | 9,089 | 10,507 | 9,696 | 4,727 |
| Sealing overhead, batch 1 · 32 | 23 · 77 % | 24 · 72 % | 20 · 72 % | 16 · 65 % |

- **Verification is model-independent**: every record of every model verified in fp16 and in bf16 (a class outside
  its table); no identical text was ever reported modified.
- **The quality operating point costs no measurable accuracy** on any model (exact McNemar test on the paired
  problems: p = 0.51–0.71). **The strong one breaks arithmetic on all of them** (p < 0.001).
- **Detection strength depends on the model.** OLMo-2 in fp16 is numerically noisier (tolerance 2.43, 2.7 to 5.8
  times the others), so more positions are excluded as near ties; at the quality operating point, which already frees
  the confident positions, too few checkpoints remain: its mark is not reliably detectable (52 % at p ≤ 0.01 on 400
  tokens). Its records verify regardless, and the strong operating point detects 98.4 % at 10⁻⁶ on 400 tokens. A
  sealer should check detection power on its own model before relying on record-free detection.
- **Short text**: at 50 tokens none of the three new models reaches 10⁻⁴ even at the strong operating point (the best,
  Mistral: 54 % at 10⁻²) — the limit stated in the README.

## Cost and custody

The L40S ran 5 h 05 min in all (phases B and C), about $9.50; phase C alone about $7.50. Results were pulled to the
operator's machine every 30 minutes; the machine was deleted by its watcher at 07:12 UTC only after every remote file
(20 MB) matched its local copy by size.
