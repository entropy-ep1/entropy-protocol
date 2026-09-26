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
- The GPU evaluation program (`bench/gpu/`) and its results: phase A (Qwen2.5-7B-Instruct, NVIDIA L40S), phase B
  (the same records verified on four GPU architectures), phase C (a two-million-pair null, KGW and SynthID-Text,
  Mistral-7B-v0.3, OLMo-2-7B and Qwen2.5-14B).
