# Changelog

## 0.1.0 — unreleased

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
  real data, differential targets across the two implementations, Hypothesis in every test run, atheris campaigns.
- The GPU evaluation program (`bench/gpu/`) and its results: phase A (Qwen2.5-7B-Instruct, NVIDIA L40S), phase B
  (the same records verified on four GPU architectures), phase C (a two-million-pair null, KGW and SynthID-Text,
  Mistral-7B-v0.3, OLMo-2-7B and Qwen2.5-14B).
