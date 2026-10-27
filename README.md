# Entropy Protocol (ep1)

**A receipt for machine-written text.** Seal text at the moment a language model writes it. Later, anyone you
give the receipt to can check — deterministically, without the prompt, and without the receipt revealing a word of
the text — whether a document is that text, and if it is not, exactly where it was changed.

> **What a missing mark means.** ep1 speaks only about text that was sealed. A text with no ep1 mark was not
> sealed; that says nothing about whether a person or a machine wrote it. ep1 is not a classifier of AI authorship
> and never judges unsealed text.

This repository is the reference implementation, the specification of its formats, test vectors, a second
independent implementation, and the evaluation program behind every number below.

---

## Why a receipt, not a guess

Tools that infer authorship from a single text cannot be both useful and fair: they must accuse some honest writers,
and their error rates are estimated, not known. Statistical watermarks report a score whose null distribution depends
on the model and the text. Metadata attached to a file does not survive a copy out of a chat window.

ep1 takes the other road. At generation it writes a **record** with two layers:

- a **content layer** — digests of the canonical text and of its sentence blocks: exact integrity and the location of
  every edit;
- a **model layer** — checkpoints: a character offset and an ordinal property of the model's choice there (its first
  choice, or not). No token, no word, no text.

A verifier re-reads the candidate once with the same model and returns a determination:

| Verdict | Meaning |
|---|---|
| `VERIFIED` | the candidate is the sealed text (canonical form: line endings, invisible characters, typographic quotes do not matter) |
| `VERIFIED_TEXT` | a rendered copy (Markdown shown as a reader sees it) whose visible text is the sealed one |
| `MODIFIED` | changed — with the changed blocks, deletion points and diverging ranges |
| `INDETERMINATE` | the verifier's numerics cannot decide; **an identical text is never reported as MODIFIED** |
| `NOT_VERIFIABLE` | no valid record |

Without any record, a holder of the **detection key** can test whether a text was sealed under that key. The
p-value is **exact**: for any text not sealed under the key, the count of matching checkpoints follows
Binomial(n, ½) by construction — no model- or text-dependent calibration of the null.

## What ep1 proves — and what it does not

**Proves**, given a record: whether a candidate is the sealed text and, if not, where it differs.
**Proves**, given the detection key: whether a text was sealed under the key, with an exact false-positive
probability.

**Does not prove**, and we say so first:

| Limit | Detail |
|---|---|
| Unsealed text | no mark means *not sealed*, not *not AI*; ep1 does not judge unsealed text |
| Rewriting | paraphrase and translation are modifications and are reported as such; after a rewrite of about a quarter of the words the record-free signal is gone |
| Short text | below about 50 tokens record-free detection has too few checkpoints |
| The model | record-free detection is as strong as the model is numerically steady: OLMo-2-7B in fp16 leaves too few checkpoints at the quality operating point (52 % at p ≤ 0.01 on 400 tokens), while its records verify as every model's do — measure detection on your own model before relying on it |
| The issuer | whoever holds the master key can seal a text its model did not write — unless sealing runs attested (below): the signing key is then certified only for a confidential VM running the declared model and code; excluding the operator of that VM also needs a locked, measured image |
| Same model | the verifier needs the same weights and tokenizer, in a certified numerical class; outside it the answer is `INDETERMINATE`, never a false verdict |
| The key | a holder of the detection key can imitate the mark — hand it only to trusted detectors, or run detection as a service |
| Measured range | the exact null is measured down to 1e-6, on 2.1 million (text, key) pairs across four models; smaller p-values rest on the construction, not on a measurement |
| Cost | sealing adds 14–27 % latency at batch 1–8, 48–77 % at batch 32 and 83–109 % at batch 64 (four cards, four models) |
| Time stamps | we check a time-stamp token's imprint; validating the authority's signature and chain is the reader's trust decision |

## Measured

One NVIDIA L40S, Qwen2.5-7B-Instruct, fp16, 128 texts of 400 tokens per corpus (phase A, 15.09.2026, total cost
$4.60). Every number is a file under [`runs_gpu/a1/`](runs_gpu/a1) and a command in [`bench/gpu/`](bench/gpu).

| | Result |
|---|---|
| Records of unchanged text in five verifier configurations — fp16; bf16 outside the calibration table; fp32; bf16 generation with fp16 measurement; the quality operating point | 100% `VERIFIED` (128 texts each; 32 for fp32 and for bf16 generation); **0 false `MODIFIED`** |
| Records after transport (CRLF, e-mail, no-break spaces, BOM, an embedded C2PA text manifest, invisible characters) | 0 failures |
| Edited texts (word replaced, one character, sentence deleted or inserted, tail truncated) | `MODIFIED` in 99.2–100%, with location |
| Exact null, 20,000 human C4 pairs | 155 at 1e-2 (200 allowed), 16 at 1e-3 (20), 1 at 1e-4 (2), 0 at 1e-5 and 1e-6 |
| Record-free detection at 1e-6, 200 / 400 tokens | 96.1% / 100% (strong) · 61.7% / 95.3% (quality) |
| GSM8K accuracy: unsealed / quality / strong | 87.5% / 85.2% / 39.1% |
| Verification throughput · peak memory | 9,275 tokens/s · 15.41 GB |

*Quality* (the default) leaves the model free where it is confident; *strong* enforces every checkpoint and breaks
arithmetic — use it only where the mark matters more than the answer.

**On other hardware** (phase B, 26.09.2026, [report](bench/results/phase_b/REPORT.md)). The same records, sealed on
the L40S, verified again on an L4 (Ada), an A10G (Ampere) and an H100 NVL in confidential-computing mode (Hopper), in
fp16, bf16 and fp32, on three clouds: **2,304 verifications, 100 % `VERIFIED`, 0 false `MODIFIED`**. Where machines
disagree at all — at most 1 position in 144,871 for fp16 and fp32, about 2 in 10,000 for bf16 — the answer is
`INDETERMINATE`, never `MODIFIED`.

| Card | Verification, tokens/s (batch 1) | Sealed generation, tokens/s (batch 1 · 64) |
|---|---|---|
| H100 NVL | 14,891 | 27 · 803 |
| L40S | 9,089 | 33 · 606 |
| A10G | 2,300 | 24 · 394 |
| L4 | 2,077 | 14 · 242 |

**The null at 10⁻⁶, two other watermarks, three more models** (phase C, 26.09.2026,
[report](bench/results/phase_c/REPORT.md)).

| | Result |
|---|---|
| Exact null: 2,127,680 (text, key) pairs of human and unsealed text, four models | 0.63–0.75 of the bound at every level down to 1e-5; at 1e-6, 4 against a bound of 2.1 (P = 0.17 under the bound) |
| The same check for KGW and SynthID-Text — their own p-values on 4,000 human texts | KGW flags 0.1 % at a nominal 1e-4 and 2.2 % at a nominal 1 % (400 tokens); SynthID about twice nominal at 1e-3 |
| Mistral-7B-v0.3, OLMo-2-7B, Qwen2.5-14B: records of unchanged text, fp16 and bf16 | 100 % `VERIFIED` (768 records, 1,152 verifications), 0 false `MODIFIED` |
| Record-free detection at 1e-6, 400 tokens, strong · quality | Mistral 99.2 · 83.6 %; OLMo-2 98.4 · 0.8 %; Qwen2.5-14B 100 · 82.8 % |
| GSM8K accuracy, quality operating point against unsealed, four models | no measurable cost (McNemar p = 0.51–0.71); *strong* breaks arithmetic on all four |

## Attested sealing on a confidential H100

The keys, the model and the sealing code run in a confidential VM (AMD SEV-SNP) with an H100 in confidential-computing
mode. An authority appraises the hardware evidence — the processor's report, the vTPM quote and boot log, the GPU's
SPDM report — and certifies a key only if every check holds; each statement then carries the certificate and the digest
of the attestation result (EAR). Design: [ARCHITECTURE §7d](ARCHITECTURE.md). One run
([files](bench/results/attested/run-20261027T172258Z): Azure `Standard_NCC40ads_H100_v5`, Qwen2.5-7B-Instruct, fp16):

| | Result |
|---|---|
| Enrollment of a statement key and a detection key, each generated inside the guest | certified after 47 checks each, none failed or skipped: AMD chain to the pinned ARK and AMD's CRL; the TCB floors of AMD's bulletins; the BadRAM and RMPocalypse fixes in the report; the paravisor's launch endorsed by its ID key; the binding (key, model, code, parameters) in the runtime claims and in the vTPM quote; the boot chain and the kernel image in the event log; the H100's report over the authority's nonce; NVIDIA's signed manifests; OCSP for nine certificates |
| 20 prose texts sealed in the VM, read by a separate verifier | as sealed 20/20 `VERIFIED`; one word changed 20/20 `MODIFIED`, with the range; after transport 20/20 `VERIFIED` |
| Record-free detection in the VM, 200 tokens, quality, 1e-6 | 14 of 20 sealed texts; 0 of 10 unsealed texts of the same model |
| Evidence packages | 20 of 20 check offline, 40 checks each: the appraisal repeated from AMD's and NVIDIA's roots at the result's issue time, the certificates, the statements, log receipts, RFC 3161 time stamps |

Refusals are tested on the run's own data ([`tests/test_attested_run.py`](tests/test_attested_run.py)): a request
bound to another key, evidence moved from another enrollment, a changed model digest, a spent handle, a changed text,
a result signed by another authority. A live refusal is kept as well: a host whose paravisor build had a launch
measurement the policy did not list ([files](bench/results/attested/run-20261027T170502Z-refused-launch-measurement));
the policy pins the key that signs the paravisor's launch instead. Not shown by this run: the operator had SSH access
to the guest. Excluding the operator needs a locked image — no interactive access, a measured root file system — whose
measurements become the reference values.

```bash
pip install -e ".[attested]"
entropy evidence --check bench/results/attested/run-20261027T172258Z/packs/5aWyrqFNxAxJwkXE.cbor
```

## Quick start

```bash
git clone https://github.com/entropy-ep1/entropy-protocol.git && cd entropy-protocol
pip install -e ".[markdown]"      # Python >= 3.10; brings torch and transformers
M=Qwen/Qwen2.5-0.5B-Instruct
entropy keygen --out master.key
entropy seal --model $M --key master.key --max-new-tokens 400 \
  --prompt "Write a short story about a lighthouse keeper who finds a letter in a bottle." --out sealed.json
python3 -c 'import json; d = json.load(open("sealed.json")); open("candidate.txt", "w").write(d["text"])'
TOKEN=$(python3 -c 'import json; print(json.load(open("sealed.json"))["token"])')
entropy verify --model $M --text-file candidate.txt --token "$TOKEN" --key master.key   # "verdict": "VERIFIED"
entropy detect --model $M --text-file candidate.txt --key master.key                    # "p_value", exact
```

On a laptop CPU this takes under two minutes. One run ([files](bench/results/quickstart_2026-09-26)): `VERIFIED`
with all 51 checkpoints of the record matched; record-free detection found 42 of 44 checkpoints and reported
p = 5.6 × 10⁻¹¹. Change one word in `candidate.txt` and
verify again: `MODIFIED`, with the character range of the sentence that changed. Short or formulaic answers carry few checkpoints — detection needs
about 50 tokens or more, and is strongest on open-ended text.

The whole product in one command, on a CPU, nothing leaving the machine: [`bench/demo_review.sh`](bench/demo_review.sh)
(keys → an OpenAI-compatible sealing gateway on `127.0.0.1` → a sealed answer → three candidates checked: as
delivered, as a rendered copy, with one word changed → record-free detection with its exact p-value → the same
verdict through an MCP tool → signed statements, a transparency log and an evidence package checked offline).

`entropy --help` lists the rest: `serve` (sealing gateway), `mcp` (verifier as assistant tools), `commit` /
`register` / `serve-log` / `submit` (SCITT statements and receipts), `timestamp` (RFC 3161), `evidence` (a
single-file package with an offline check), `deliver` (a record encrypted to one verifier's public key).

## Reproduce the numbers

The evaluation keys are **public by design**: key *i* is `SHA-256("ep-gpu-eval/2026-09/master/<i>")`
([`bench/gpu/common.py`](bench/gpu/common.py)), so anyone can re-derive them and recompute every detection. The
phases are scripts with a hard cost cap per machine; each step leaves a log and resumes where it stopped.

| Phase | What | Script |
|---|---|---|
| A | every experiment on one 7B model | [`run_phase_a.sh`](bench/gpu/run_phase_a.sh) |
| B | the same records verified on other GPU architectures; calibration across machines | [`run_phase_b.sh`](bench/gpu/run_phase_b.sh) |
| C | a million-trial null; KGW and SynthID-Text against length-matched human nulls; three more model families | [`run_phase_c.sh`](bench/gpu/run_phase_c.sh) |

Each has a CPU dry run (`smoke_phase_*.sh`) that exercises every call on a small model before a paid machine starts.

## Specification and a second implementation

- Formats: [`spec/ep1.cddl`](spec/ep1.cddl) — the record, the statements, receipts, the evidence package. No floating
  point on the wire: tolerances in integer micro-units, p-values from counts.
- Test vectors: [`spec/test-vectors/ep1-vectors.json`](spec/test-vectors/ep1-vectors.json), regenerated by
  [`spec/make_vectors.py`](spec/make_vectors.py).
- An independent JavaScript implementation of every model-free part: [`impl-js/`](impl-js) — 83/83 vectors. CI
  regenerates the vectors with the Python reference, requires them to be byte-identical to the committed file, and
  runs the JavaScript implementation against them.
- Design and rationale: [`ARCHITECTURE.md`](ARCHITECTURE.md).

Built on published standards: SCITT architecture (RFC 9943), COSE Receipts (RFC 9942), Certificate Transparency
v2 Merkle trees (RFC 9162), COSE Hash Envelope (RFC 9995), CWT claims in COSE (RFC 9597), RFC 3161 time stamps
(carried per RFC 9921), SCRAPI (draft-ietf-scitt-scrapi), COSE-HPKE (draft-ietf-cose-hpke).

## Status

0.1.0 — a reference implementation ahead of standardization. Formats may change before a specification is adopted
by a standards body; records name their canonical form and version, and older records keep verifying.

## License and patents

Code: [Apache License 2.0](LICENSE). Patent applications cover this work; see [PATENTS.md](PATENTS.md) for the
commitment made to implementers of the specification.

## Authors

Roman Oliinyk and Serhii Nikolaichuk. Cite as in [CITATION.cff](CITATION.cff). Security reports:
[SECURITY.md](SECURITY.md).
