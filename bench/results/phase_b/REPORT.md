# Phase B — one record, four GPU architectures (26.09.2026)

Records sealed on an NVIDIA L40S on 15.09.2026 (phase A) were verified again on four GPU architectures, in three
precisions, on three clouds, with two PyTorch builds and three driver versions. A shared calibration corpus (512
sealed texts, 201,565 token positions) was scored by every configuration each card can run, so that disagreements
between machines could be counted position by position. Model: Qwen2.5-7B-Instruct. Every number below is a file
under `runs_gpu/{c1,b2,b3,b4}/` or `bench/results/phase_b/B_cal/`, produced by `bench/gpu/run_phase_b.sh`.

## Result

**2,304 verifications of unchanged text on four architectures: 100 % `VERIFIED`, 0 false `MODIFIED`, 0
`INDETERMINATE`.**

| Card | Architecture | Cloud | fp16 | bf16 (class outside the record's table) | fp32 |
|---|---|---|---|---|---|
| L40S | Ada (8.9) | AWS | 256 / 256 | 256 / 256 | 128 / 128 |
| L4 | Ada (8.9) | Google Cloud | 256 / 256 | 256 / 256 | — (24 GB) |
| A10G | Ampere (8.6) | AWS | 256 / 256 | 256 / 256 | — (24 GB) |
| H100 NVL, confidential-computing mode | Hopper (9.0) | Azure | 256 / 256 | 256 / 256 | 128 / 128 |

Each cell: records `VERIFIED` / records checked; two corpora of 128 texts (strong and quality operating points), 64
each in fp32. In fp16 on every card the edited copies were reported `MODIFIED` in 99.2–100 % of cases with location,
with 0 false `VERIFIED`; transported copies (CRLF, e-mail, no-break spaces, BOM, an embedded C2PA manifest, invisible
characters) failed 0 times. Table: `bench/gpu/infra/phase_b_verify_table.py`.

| | L40S | L4 | A10G | H100 NVL |
|---|---|---|---|---|
| torch · CUDA · cuDNN | 2.7.0 · 12.8 · 9.10 | 2.7.1 · 12.8 · 9.7 | 2.7.0 · 12.8 · 9.10 | 2.7.1 · 12.8 · 9.7 |
| driver · Python | 580.126 · 3.12 | 580.178 · 3.10 | 580.126 · 3.12 | 595.91 · 3.12 |

## Where machines disagree, and how rarely it matters

A record entry can be misjudged only where the verifier's signed margin has the opposite sign to the sealer's. Against
the reference (L40S, fp16, batch 1), over all usable positions (`bench/diag_flips_by_margin.py` →
`B_cal/flips_by_margin.json`):

| Reference margin at least | 0.84 (phase A records) | 2.0 | 5.0 |
|---|---|---|---|
| positions (share of all) | 144,871 (72 %) | 87,324 (43 %) | 29,439 (15 %) |
| fp16, any card, batch 1 or 32 (7 configurations) | 0–1 flips each | 0 | 0 |
| fp32, L40S and H100 (4 configurations) | 1 flip each | 0 | 0 |
| bf16, any card (8 configurations) | 28–39 flips each | 6–10 | 2–3 |

- **fp16 and fp32:** two positions in the whole corpus, at tokens 11 and 16 of their texts, where the classes genuinely
  disagree by up to 2.5 logits (at one, fp32 and the Ada cards' fp16 batch-1 pass agree against every other fp16 pass
  and bf16; at the other, fp32 and bf16 agree against fp16). After token 32, fp16 on every card stays within 1.6
  logits of the reference (0.8–0.9 at batch 32), and no position flips at margin ≥ 0.84.
- **bf16** disagrees everywhere in a text, occasionally by 10+ logits. It verified all 1,024 phase A records here, but
  at a rate of about 2 × 10⁻⁴ per entry an occasional record will come out `INDETERMINATE` under a bf16 verifier.
- **In every case the consequence is `INDETERMINATE`, never `MODIFIED`**: modification is decided by the content layer
  (digests of the canonical text), and an identical text cannot be reported as modified.

### The "3 × max" tolerance does not converge

The calibration rule used so far — tolerance = 3 × the largest deviation seen on a corpus — measured 0.42 logits for
fp16 on phase A's 64 texts. On 512 texts it measures 7.8: the maximum is set by a handful of positions at the start of
texts and keeps growing with the corpus (the growth by subsample size is in `B_cal/growth_and_classes.json`). A
table built this way (`B_cal/calibration_B_cal.json`) would keep 0.2 % of positions in a record, so **it is not
adopted**; records keep phase A's calibration, which the flip counts above support directly.

For the next format version: place the first checkpoint at token 32 instead of 4 (removes both fp16/fp32 flips and
cuts the fp16 maximum from 2.6 to 1.5 logits; bf16's disagreements are spread through the text and remain), and
state the tolerance as a quantile with an explicit `INDETERMINATE` budget, with the verifier excluding its own
low-margin entries (the two-tier scheme).

## Other findings

- **fp16 with eager attention overflows** on Qwen2.5-7B (non-finite logits) on every card; the pass refuses rather than
  judging. SDPA in fp16 is unaffected. Use SDPA, or bf16/fp32.
- **Throughput.** Verification: 14,891 tokens/s on the H100 at batch 1 (19,101 at 32), 9,089 on the L40S, 2,300 on the
  A10G, 2,077 on the L4. Sealing overhead at batch 1–8: 14–27 %; at batch 64: 83–109 %. Batch-1 generation on the
  H100 (33 tokens/s) was slower than on the L40S (40) despite four times the memory bandwidth — consistent with the
  cost of per-token host round trips through encrypted bounce buffers in confidential-computing mode (not measured
  separately). Peak memory with sealing: 14.7–16.7 GB.
- **KGW baseline, corrected.** Phase A reported the KGW watermark detected in 81 / 75 / 34 % of texts at p ≤ 0.01 for
  100 / 200 / 400 tokens. That decline was ours: the transformers detector returns p = 0.0 above z ≈ 7.6, and the
  count `(p or 1) <= alpha` turned the strongest detections into misses. Correctly counted: **81 / 97 / 100 %**
  (`runs_gpu/c1/…/diag_kgw_length_cuda.json`). Fixed in `baselines.exact_rate` before phase C's baselines. Also: a
  KGW detector on the CPU reads a corpus generated on CUDA as noise (the greenlist permutations differ by device).
- **A 3.12-only f-string** in a diagnostic script failed on the Python 3.10 machine; CI now runs 3.10 and 3.12.

## Cost

A10G 54 min, L4 55 min, H100 about 45 min, L40S 62 min for phase B: about **$10.50**. Every machine was deleted by its
watcher only after each result file was compared by size with the local copy.
