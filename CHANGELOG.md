# Changelog

## 0.1.1 — 2026-11-04

Two fixes from our own review of 0.1.0. No use of either is known; the attested run's packages are re-issued.

- **The evidence check accepted a foreign text in a genuine package.** `entropy evidence --check` tied the disclosed
  text to the generation statement only through the verifier's verdict, and checked that verdict under the verifier
  key the package itself carries. A genuine attested package whose text was replaced, with a verdict about the new
  text signed by any key and that key plus a log of one's own, checked as complete (exit 0, verdict `VERIFIED`).
  Packages are now `ep1-pack/v3`: they open the generation statement's commitment (the record's digest and salt,
  never the record), and the check recomputes the commitment from the text, so a foreign text fails whoever signed
  a verdict about it. The check also answers in three levels: `FAILED`; `INTACT` — the package agrees with itself
  under keys it brings, which no one vouches for, so treat it as unsigned; `TRUSTED` — only with `--trust`, the
  issuer's published trust list (its keys; for an attested key, the authority's root and the policy it applied).
  A weakened appraisal policy and a package made from nothing are never `TRUSTED`. For an attested key, a statement
  must be signed inside its certificate's validity and before the attestation result expires.
- **Attested appraisal: a vTPM quote could leave out what the logs replay.** The quote must now be over the SHA-256
  bank alone, every replayed PCR must be quoted and equal, and the kernel image's sections are read only when PCR 11
  replays to its quoted value. Not reachable with the policy the bench writes (PCRs 4 and 7 pinned); reachable with
  one that pins only the sections.

Changed: the report's `ok` is true only for `TRUSTED`; `entropy evidence --check` exits 0 `TRUSTED`, 3 `INTACT`,
1 `FAILED`; `entropy evidence --key` adds the opening when building. The 20 packages of the attested run are
re-issued as v3 with the run's trust list (`bench/results/attested/run-20261027T172258Z/packs-v3`): 20/20 `TRUSTED`;
`bench/attested/forgeries.py` runs five forgeries on each and none is accepted. The 0.1.0 packages stay as issued and
check `INTACT`.

## 0.1.0 — 2026-10-27

First public release of the reference implementation.

- Sealing at generation: two-layer records (content layer: canonical-text and block digests; model layer: character
  offsets and ordinal predicates, no text), content-keyed checkpoint selection, enforcement in the verifier's
  prompt-free context, canonical generation (the sealer emits exactly the tokens the verifier re-reads).
- Verification: `VERIFIED`, `VERIFIED_TEXT` (a rendered copy), `MODIFIED` with located changes, `INDETERMINATE`;
  an identical text is never reported as modified.
- Record-free detection with an exact Binomial(n, 1/2) null, paragraph-start readings and localized windows for
  mixed documents; many keys in one model pass.
- Calibration measured per numerical class, over the positions that can hold a checkpoint.
- Signed generation, verification and detection statements (COSE), an RFC 9162 transparency log with RFC 9942
  receipts, a SCRAPI client and service, RFC 3161 time stamps, a single-file evidence package with an offline check.
- An OpenAI-compatible sealing gateway (`entropy serve`) and the verifier as MCP tools for an assistant (`entropy mcp`).
- A second implementation of every model-free part in JavaScript (`impl-js/`), checked against the test vectors in
  CI; the vector path of the Python reference imports no machine-learning stack.
- Attested sealing and attested detection (`attested/`): statement and detection keys generated inside a confidential
  VM with a confidential GPU and certified by an authority after it appraises the hardware evidence (AMD SEV-SNP,
  the paravisor's vTPM, NVIDIA H100); EAR attestation results; evidence packages `ep1-pack/v2` whose offline check
  repeats the appraisal from the manufacturers' roots. Run end to end on an Azure confidential H100.
- Canonical form v5: the deletions before NFC (idempotent), NFC pinned to Unicode 14.0; visible-text form v2 on it;
  detection statements name their canonical form.
- One CBOR profile for untrusted input in both implementations; fuzzing (`fuzz/`): 26 targets with seed corpora from
  real data, differential targets across the two implementations, Hypothesis in every test run, ClusterFuzzLite on
  every pull request, coverage-guided atheris campaigns.
- Packaging: the model stack (torch, transformers) is the `model` extra; checking statements, receipts and evidence
  packages needs `cryptography` and `cbor2` only (`attested` adds NVIDIA's signed manifests).
- Releases signed with Sigstore, one bundle per file, with GitHub build-provenance attestations; README: how to
  verify them.
- The GPU evaluation program (`bench/gpu/`) and its results: phase A (Qwen2.5-7B-Instruct, NVIDIA L40S), phase B
  (the same records verified on four GPU architectures), phase C (a two-million-pair null, KGW and SynthID-Text,
  Mistral-7B-v0.3, OLMo-2-7B and Qwen2.5-14B).
