# Entropy Protocol ep1 — architecture of the reference implementation (v0.1.0.dev0, stage 2 + ideal-product work)

PRIVATE — do not publish before the covering provisional application is filed. Seed of
`draft-oliinyk-scitt-generation-records`. Formats: `spec/ep1.cddl`; test vectors: `spec/test-vectors/`.

## 1. Goal
Seal AI-generated text at the moment of generation so that
(a) anyone holding a verification key can later determine — deterministically, without the prompt and without
the record revealing any text — whether a candidate text is the sealed text, and if not, where it differs; and
(b) anyone holding the detection key can tell, without the record, whether a text was sealed under the key,
with an exact false-positive probability.
Survive ordinary transport (copy/paste, line endings, BOM, trailing whitespace, no-break spaces, invisible format
characters, an embedded C2PA text manifest, blank lines around a paste).

Non-goals: detecting AI authorship of unsealed text; surviving paraphrase or translation (those are
modifications and are reported as such).

## 2. Roles
| Role | Holds | Can |
|---|---|---|
| Generator | master key `K` | generate sealed text; custody the record under `K_log`; issue the generation statement |
| Custodian | `K` or `K_log` | re-issue a document's record to a named verifier |
| Verifier | `K_ver` for one document, or its own X25519 key pair (records delivered by public key, §7) | verify; issue a signed verification statement; cannot mark, cannot derive `K`, `K_doc`, `K_bits`, `K_wm` |
| Detector | `K_wm` | record-free detection; issue a signed detection statement. Holding `K_wm` also lets one imitate the mark (as with every keyed watermark) — hand it only to trusted detectors or run detection as a service |
| Relying party | Ed25519 public keys | check statements and receipts without any secret |
| Transparency log | log key | append-only log of statements; RFC 9942 receipts (`tlog.py`; SCITT in production) |

## 3. The record (CBOR, then AEAD) — `spec/ep1.cddl`
```
Record = { h: Header, e: [[char_offset, prop] ...], b: [[start, end, digest64] ...] }
Header = { model, weights_digest, tokenizer_digest, canon, context="bos+text", k, interval, tol_u, min_entries,
           indet_pm, text_digest, engine, extra, tol_space, cal_u, components, env, selection, wm_context, salt }
```
On the wire every number is an integer: `tol_u`/`cal_u` in micro-units of the tolerance space, `indet_pm` per-mille
(in memory: `tol`, `cal`, `indet_frac`).
- **Content layer** — `text_digest` (SHA-256 of the canonical text) and content-defined sentence blocks with 64-bit
  digests: exact integrity and block localization.
- **Model layer** — checkpoints: canonical character offset + expected predicate (`LOW` = the model's first choice;
  `HIGH` = outside its top-k; reference k = 1). No token identifier, no text.
- `components` = measured model components (weights, tokenizer, chat template, generation defaults; RFC 10013
  style); `env` = sealing environment (device, hardware, dtype, class, torch/transformers, attention kernel).
- `salt` = 16 random bytes for the public commitment.
- Token: `"ep1." + base64url(CBOR{v, doc, n, ct})`, ChaCha20-Poly1305, AAD = CBOR{v, doc}.

## 4. Keys (HKDF-SHA256, domain-separated) — spec [0121]-[0124]
`K_bits = KDF(K,"ep1/bits")` · `K_doc = KDF(K,"ep1/doc|"+doc)` · `K_log = KDF(K_doc,"ep1/log")` ·
`K_ver = KDF(K_doc,"ep1/verifier|"+verifier_id)` · **`K_wm = KDF(K,"ep1/wm")`**.

## 5. Sealing (`seal.py`) — "measure the record the way the verifier will measure it"
1. **Checkpoint selection.** `selection="content"` (reference): at generated position j ≥ `context` (4), one
   HMAC under `K_wm` over the preceding 4 token ids decides whether j is a checkpoint (probability 1/`interval`)
   and which predicate it carries (LOW/HIGH, 1/2 each). A detector recomputes it from the text alone; an edit
   desynchronizes only the positions whose context it touches. `selection="interval"`: every `interval`-th
   position, predicate from the per-document bitstream `K_bits` (spec [0104]; claim 1 embodiment).
2. **Enforcement.** `align="prompt"`: the predicate is enforced on the generation distribution (prompt + text).
   `align="free"`: a second cache holds the **verifier's context** (start token + text, no prompt); the predicate
   is enforced there — LOW = the first choice in that context; HIGH = the most probable token under the prompt
   distribution among tokens outside the verifier-context top-k and at least the calibrated margin below its k-th
   logit. The property then holds exactly where it will be measured (match rate 97 % vs 91 %; see G10).
   `free_after` = W enforces positions before W in the generation context (early in a text the prompt-free view
   knows too little of the topic) and later ones in the verifier's context.
   Other positions: nucleus sampling. Special tokens never enter the text; every end-of-sequence id ends it; where
   the model's first choice at a checkpoint is to end, the text ends (no entry).
3. Decode; canonicalize (`ep1-canon/v5`: delete CR, BOM, zero-width space, word joiner, soft hyphen; delete a
   U+FEFF-opened run of ≥ 8 variation selectors as a unit (C2PA 2.4 A.8 wrapper), then any run of ≥ 8 selectors;
   keep single selectors and ZWJ/ZWNJ; **then NFC as Unicode 14.0 defines it** — the text is cut at every code point
   not assigned in 14.0 (to 14.0 a starter nothing composes or reorders across) and the pieces normalized, which
   Unicode's normalization stability policy makes exact under any later library version; no-break spaces → space;
   **typographic folding** — curly quotes and guillemets → `"`, curly apostrophes and U+02BC → `'`, hyphen/en
   dash/minus → `-`, em dash → `--`, ellipsis → `...` (the inverse of a word processor's autocorrect); strip trailing
   whitespace per line and around the text; blocks by sentence ends including scripts' own full stops, `seg2`),
   keeping a map from indices of the decoded text to canonical indices. The form is idempotent: after the deletions
   only starters are replaced or removed. Records name their form; v3 and v4 records still verify (v4 normalized
   before deleting, which is not idempotent where a deleted character separated a base from its mark or marks of
   different classes, and neither v3 nor v4 is pinned to a Unicode version). The 14.0 table (706 ranges) is generated
   for both implementations from Unicode's DerivedAge (`spec/make_unicode14.py`).
4. Anchor each enforced checkpoint at the canonical offset of the first non-space character of its token.
5. Re-read the canonical text alone (one forward pass); keep a checkpoint only if its predicate holds there with
   margin ≥ 2 × the largest tolerance of the calibration table and it maps to a distinct canonical token.
6. Header + entries + blocks → CBOR → encrypt under `K_log`.

## 5a. Canonical generation (`canongen.py`, default on) — "generate the text the verifier will re-read"
The verifier tokenizes the canonical text; wherever that differs from the tokens the seal enforced, checkpoints
near the difference are read in another context. Canonical generation makes the two identical token for token:
- **Nothing the canonical form changes is written**: the folded typographic characters, invisible characters, the
  replacement character. Tokens are byte strings and a character may be split across tokens, so a state machine
  tracks the unfinished UTF-8 character: inside it only continuation bytes follow and the text cannot end; a token
  never stops inside a forbidden character past its lead byte; where the verifier's first choice is banned, no
  token may stop inside any character at that step (else a byte-level model completes a half-written quote into a
  stray neighbour).
- **No space or tab before a line break** (inside a token, or a line break after a token ending in one).
- **The first token as the verifier reads it**: the canonical form strips leading whitespace, so the verifier's
  view (its cache and the token contexts that select checkpoints) takes " that" as "that" (a first token with no
  single-token reading, and an end-of-text first token, are not emitted).
- **No join the verifier would re-tokenize differently**: each chosen token is re-tokenized with its left context
  and a few probe continuations (end, letter, space + letter, line break, full stop, digit — some tokenizers split a
  whitespace run by what follows it); a rejected token is banned and the choice made again, then the most probable
  joinable token is taken, else the model's own choice (a local break; a break already in the window is not held
  against later tokens). A HIGH checkpoint never takes a byte prefix of the first choice (" question" for
  " questionnaire") or a token that only adds whitespace to it ("\n\n" for "\n", ".\n\n" for ".").
- **LOW left free where it cannot be satisfied** (the first choice is a stop or a banned token); the detector
  recognizes these positions from the text and the model alone (§6b).
Measured (alignment audit, 20 texts × 200 tokens each; `bench/align_audit.py`): distilgpt2 — detection match
96.7 % → **100 %** (678/678), record entries 554/820 → **814/814**, token differences 37 → **0**; Qwen2.5-0.5B chat —
99.9 % → **100 %** (683/683). Word/Docs autocorrect + CRLF: 40/40 VERIFIED (v3 would have failed 55–80 % of them).
**Operating points** (`profiles.py`, measured 13.09 on Qwen2.5-0.5B/1.5B): *strong* (the reference) enforces every
selected checkpoint — GSM8K accuracy 38 % → 3 % (0.5B), 64 % → 11 % (1.5B): a HIGH checkpoint overrules the model where
it copies a number or a name from the question, which the prompt-free view cannot see. *quality* = guard 0.9 at
sealing + gate 0.9 at detection — GSM8K and prose quality as unsealed; prose, 200 tokens: 90 % at 1e-6 and 100 % at 1e-4
(0.5B), 60 % / 97.5 % (1.5B, with the sealing gate as well); maths keeps almost no mark (low entropy). The record is exact
under either. **Default: quality** (decided 13.09.2026) on the product's surfaces — `entropy seal / detect / serve` and
`gateway.Gateway`; `SealParams()` stays the protocol's reference parameters (strong), which the specification and its
vectors are written against. A record carries no guard, so records of both points have one form; the detector's gate
for any sealing parameters is max(gate, guard) (`profiles.detect_gate_for`), and an explicit `--gate 0` is honoured.
**Confidence gate** (`gate`): a checkpoint only where the verifier's view is below the gate for its first choice;
key-independent (the exact null holds on the gated positions); the record keeps only entries below the gate in its
own measurement pass, so every entry is a position the detector counts. **Guard** (`guard`): no checkpoint where the
prompt-conditioned view is sure (invisible to the detector: dilution only).

## 6. Verification (`verify.py`) and verdicts
Open the token (else `NOT_VERIFIABLE`) → check model/tokenizer digests (else `INDETERMINATE`) → canonicalize the
candidate → one forward pass → evaluate each checkpoint (unalignable = divergence; margin < the verifier class's
tolerance = indeterminate) →
- canonical digest equal: `VERIFIED` if no divergence, else `INDETERMINATE` (environment differs).
  **An identical text is never reported as MODIFIED.**
- canonical digest different: `MODIFIED`, with changed blocks, deletion points and diverging checkpoint ranges.
  Blocks are aligned in order (longest matching runs first, like diff), so a repeated sentence is attributed to
  the copy that actually changed.
The result reports the verifier's environment next to the sealing environment.

## 6d. Visible-text layer (`plain.py`) — a rendered copy verifies as its text
The commonest path of a model's answer is a copy out of a chat window, where its Markdown is already rendered: no
`**`, `#` or list markers, links as their text, lines joined differently. Against the content layer that copy is
`MODIFIED` although every visible word is the sealed one. New records therefore carry a second, visible-text layer:
`plain` (form id) and `plain_digest` in the header, blocks `p` in the body — on the wire only when present, so records
without it keep their bytes (every earlier test vector is unchanged).
- **Sealer**: renders the canonical text as Markdown (markdown-it-py: CommonMark + GFM tables and strikethrough; raw
  HTML shown as text, as chat windows do), keeps what a reader sees — one line per paragraph, heading, list item,
  table row and line of code; image alt texts are not visible — then normalizes. Without markdown-it-py the layer is
  not written.
- **Verifier**: never renders. The candidate is taken literally and normalized: the form's canonical form (`plain/v2`:
  canonical form v5; `plain/v1`: v4); per line one
  marker at its start removed (bullets, an ordinal `12.`/`12)`, a task box `[ ]`/`[x]` or the ballot box a copy
  shows); whitespace runs one space; empty lines dropped; lines joined by one space (the digest does not depend on
  how a copy breaks its lines). Blocks: seg2 segmentation inside each line (a change is located to its list item or
  table row even where items end without a full stop).
- **Verdict `VERIFIED_TEXT`**: canonical digest differs, visible digest equal. The visible text is the sealed one;
  formatting (markup, line breaks, list numbering) is not covered and the model layer is not evaluated on it. A
  mismatch stays `MODIFIED`, with a `text_view` (the candidate's visible text, changed ranges and deletion points in
  it) next to the canonical ranges; the check page shows whichever flags less.
- **Why literal on the verifier's side**: markup in a candidate stays text, so a Markdown source whose link target
  was changed is not the visible text of the sealed one (`MODIFIED`), and a rendered copy carries no link targets to
  change. The comparison is SHA-256 over the whole visible text: no similarity, no threshold.
- Checked: `tests/test_plain.py` (rendered constructs vs what a copy gives back, literal candidates, a rendered copy
  → `VERIFIED_TEXT`, one edited list item located exactly, a changed link target caught, records without the layer);
  the JS implementation reproduces normalization, blocks and the layer (`spec/test-vectors`: `visible_text`,
  `visible_text_v2`, `record_v4_visible`).

## 6a. Tolerance and calibration (`calibrate.py`) — spec [0115]-[0117]
- **Measured, not chosen**, on a corpus of *sealed* generations (forced tokens put the model where numerical
  noise is larger: Qwen2.5 fp16 0.36 on sealed vs 0.19 on plain text). δ = max |margin_cfg − margin_ref| over the
  positions that can hold a checkpoint (index ≥ `seal.first_checkpoint(params)`: `context` = 4 for content
  selection) and both predicates; class tolerance = c·δ, c = 3. The table records `from_position`; sealing and
  detection refuse a table that did not measure every position their parameters can use (`seal.table_covers`).
- **Why not all positions (15.09, first GPU run, Qwen2.5-7B, L40S).** fp16 batch 32 against batch 1: 1.99 logits at
  position 3, at most 0.141 from position 4 on (p99.99 0.11). Over all positions the fp16 tolerance was 5.96 instead
  of 0.42, the record margin 11.9 instead of 4.1, and records kept 1.6% of positions instead of 18.9%. The first
  tokens after BOS never carry a record entry or a detector count, so their noise does not bound anything.
- **CUDA classes measured on the L40S (7B, positions ≥ 4, against fp16 b1):** fp16 b32 0.141, fp32 b1 0.68,
  bf16 b32 8.34 (p99.99 2.25): bf16 stays outside the table (one flip at a position a record would keep), fp16 and
  fp32 flip at none of 4712.
- **Across machines (phase B, 26.09, 512 sealed texts, 201,565 positions, four architectures).** Against the
  sealer's fp16 batch-1 pass, at the record margin of the phase A records (0.84): fp16 and fp32 on every card flip
  at most 1 position in 144,871 (two positions in all, positions 11 and 16 counting from 0), bf16 about 2 in 10,000. The rule
  c·max does not converge on a larger corpus — 0.42 on 64 texts, 7.8 on 512, set by a few positions near the start
  of texts — so its phase B table is not adopted; next format version: first checkpoint at token 32 and a quantile
  tolerance with an explicit `INDETERMINATE` budget, the verifier dropping its own low-margin entries.
- **Logit space** (log-probability ratio): one tolerance fits the top token and the tail alike.
- **Table in the header**: seal margin ≥ 2 × largest listed tolerance; the verifier applies its own class;
  outside the table → evaluated with the largest tolerance and flagged `calibrated = False`.
- **Reference certified set: fp32 + fp16.** Generate in any precision; measure the record in a certified one.

## 6b. Record-free detection (`detect.py`)
Canonicalize, tokenize, one forward pass; at every position with a distinct 4-token context, recompute the
content selection under `K_wm`; count how often the predicate holds among decisive checkpoints (near ties below
the detector's tolerance excluded — for k = 1 the LOW and HIGH margins coincide, so the exclusion cannot depend on
the key). **Exact null:** for text not generated under `K_wm`, exactly one of LOW/HIGH holds at every position
(k = 1) and the keyed predicate is independent of the text, so each checkpoint matches with probability exactly
1/2, independently across distinct contexts: matches ~ Binomial(n, 1/2) and the p-value is exact — no model-
or text-dependent null calibration, unlike z-score (KGW) or trained-detector (SynthID) watermarks.
**Localized search** (`localize=True`, mixed documents): windows of 8, 16, 32, … checkpoints in half-window steps;
the smallest window p-value is Bonferroni-corrected by the number of windows (about n/8) and reported with the
window's canonical character range, as a decision separate from the whole-text test. **Paragraph-start readings**:
a passage pasted into a longer document is also read from each paragraph start as a text of its own (start token
+ at most 1024 tokens) — the context it was sealed in; the best window over all readings is corrected by the
windows examined in all of them (Bonferroni holds under any dependence). An AI paragraph between human paragraphs:
corrected p 1.4e-4 → 7.9e-9 in the test document; human text stays clean.
**Exclusions**, all read from the text and the model, never the key (so the null stays exact): near ties below the
tolerance; positions above the confidence gate; **unsatisfiable positions** — where canonical generation could not
have emitted the verifier's first choice (a stop, a banned token, a join it would re-read differently), so a sealer
left LOW free there.

## 6c. Batched mode (`batch.py`) — accelerators and the evaluation program
- **Generation**: one forward pass per step over 2B rows — B generation contexts (prompt + text, left-padded) and,
  with verifier-context enforcement, B verification contexts (start token + text) stacked in the same batch, so
  the weights are read once per step for both views. Each row has its own document, key and decision function;
  rows that end leave the batch (cache compaction). Sampling on the device.
- **Prompt pass row by row**: each prompt is run alone (no padding, as the single path) and the rows' caches are
  left-padded with zeros. A left-padded prompt pass turned the padding into NaN on MPS in half precision (sdpa and
  eager alike) and 0 × NaN reached the real positions; batch now equals single there too.
- **Scoring**: the verifier's pass over right-padded batches without an attention mask, as the single pass;
  only the statistics a verdict needs leave the device (logit, probability and rank of the token present, the
  k+1 largest logits and probabilities, the first choice), in bounded chunks for long documents. Probabilities
  come from logsumexp (one formula for the gate in generator, record and detector; no full softmax), and the
  non-finite check reads per-row logsumexp and minimum instead of a vocabulary-sized mask.
- **Output head only where needed** (`HFBackend.split`, `logits_at`): where a model's logits are exactly its head
  over the last hidden state (checked once per backend on a short input; else the full forward), every scoring pass
  runs the base model and applies the head to the rows it needs — the record's anchors, the detector's checkpoints,
  512-row chunks of a batch — never the [length × vocabulary] tensor (5 GB for 16k positions of a 152k vocabulary;
  on the 16 GB Mac that tensor pushed a verification run into 26 GB of swap). Logits identical to the full forward
  (max difference 0.0 on distilgpt2 and Qwen2.5-0.5B).
- **Allocator cache on MPS** (`HFBackend.tidy`): MPS keeps freed blocks cached up to 1.4 × the recommended working
  set (16.6 GB on a 16 GB Mac), and a generation allocates new sizes at every step (the cache grows by a position),
  so the cache, not the tensors, filled the machine: 4 GSM8K questions sealed in fp16, Qwen2.5-1.5B 2.9 GB live and
  11.7 GB held (14 GB footprint); Phi-3-mini, whose cache has no grouped attention (about 1 GB reallocated per step
  at batch 4), past 16 GB. After every forward pass the cache goes back to the system once it exceeds the live
  tensors by 1 GB, and after every scoring group: 1.5B holds at most 3.9 GB at batch 4 and 4.3 GB at batch 16, Phi-3
  9.1 GB (7.8 GB live); the same tokens (output digests equal) in the same time. No effect on other devices.
- **Numerics**: batching changes the order of floating-point reductions. That is a configuration effect like a
  change of hardware: the environment of a batched seal or verification reports its batch size and calibration
  measures batch sizes as configuration members. Single-text outputs are byte-identical to the unbatched path.
- **Generate in one precision, measure in another** (`meter`): e.g. generation in bf16, the record's measurement
  pass in fp16 — the record reports the measuring environment.
- **Many keys, one pass**: distinct-context positions, token ranks and the near-tie exclusion do not depend on
  the detection key, so a text is tested against any number of keys with one model pass (one HMAC per position
  per key) — the empirical null over ≥10^6 (text, key) pairs, and a detector serving many issuers of one model.

## 7. Statements and transparency (`statements.py`, `cose.py`, `tlog.py`) — IETF building blocks
- **Generation statement**: COSE_Sign1 (Ed25519, alg −19), CWT claims {iss, sub = doc id, iat}, `typ`, COSE Hash
  Envelope (RFC 9995: 258 = SHA-256, 259 = commitment content type). Payload = SHA-256 of
  `{v, doc, rec: SHA-256(record plaintext), txt: SHA-256(salt ‖ text_digest), canon, wts, tok}` — neither text nor
  record in the log; the salt defeats confirmation of short texts; any copy of the record reproduces it.
- **Verification statement** (issued by the verifier, its own `iss`, same `sub`): verdict, counts, canonical
  ranges, content hash of the candidate, model digests, decoding configuration, nonce, engine, and a reference to
  the generation statement (SHA-256 of its logged form). Optional Hash Envelope form for public registration.
- **Detection statement**: detected, exact p-value, counts, technique id, content hash, model digests — the signed
  detection result the EU Code of Practice asks detectors to provide.
- **Transparency log**: entry = statement with emptied unprotected header; RFC 9162 SHA-256 Merkle tree;
  receipts per RFC 9942 (vds 1, inclusion proof in vdp −1, detached root payload).
- **SCRAPI** (`scrapi.py`, draft-ietf-scitt-scrapi-11): `POST /entries` (application/cose → 201 + Location +
  Receipt), `GET /entries/{id}`, `GET /.well-known/scitt-keys` (COSE Key Set), problem details in
  application/concise-problem-details+cbor; registration policy = the statement verifies under a registered issuer
  key. The same client registers with any SCITT Transparency Service. "Register before release": the generation
  statement is registered before the text is returned.
- **No floating point on the wire**: tolerances in integer micro-units, fractions per-mille, p-values derived from
  counts (found by the second implementation, `impl-js/`).
- **Record delivered by public key** (`hpke.py`, `delivery.py`): the issuer opens the record with `K_log` and
  encrypts it to one verifier's X25519 key — COSE_Encrypt0 with COSE-HPKE integrated encryption
  (draft-ietf-cose-hpke-27: alg HPKE-4 = 42, "ek" = -4 unprotected, AAD = Enc_structure, info empty; code points
  are the draft's suggestions until IANA assigns them); the document id is a CWT `sub` in the protected header.
  Unlike `K_ver`, no symmetric secret is derived for or shared with the verifier. CLI: `hpkekey`, `deliver`,
  `verify --delivered`.

## 7a. Sealing gateway and presentation typography (`gateway.py`, `typography.py`) — internal tool
`entropy serve`: an OpenAI-compatible HTTP surface (standard library): `/v1/chat/completions` and `/v1/completions`
return the sealed text with an `entropy` object (doc id, custody token, entries, canonical form); `/v1/verify`
(text + token → verdict, the gateway holds the master key), `/v1/detect` (record-free, with paragraph readings),
`/v1/models`, a local check page at `/`. One model, one request at a time; localhost unless configured; optional
bearer API key. **Presentation typography**: sealed text is canonical (straight quotes, `--`); a request may ask
for `"typography": "smart", "lang": "ru"` and gets «ёлочки» / „…“ / “…”, apostrophes, dashes and the ellipsis by
substitution with forms v4 folds back — so the delivered text verifies as the sealed one (code and Markdown
structure untouched; the result is checked and returned unchanged if its canonical form differed).
**Memory budget (macOS)**: MPS keeps a compiled graph for every tensor shape a process has run, outside the
allocator, so a long-running gateway grows with the prompt and text lengths it meets (Phi-3, sealing: 10 batches of
new lengths +1.8 GB, the same batch 8 times +0.13 GB; precise `phys_footprint` from `task_info`). After each model
pass the gateway reads its footprint; over `--max-footprint-gb` (default 80 % of the machine's memory, 0 = off) the
answer that crossed it goes out in full, model requests get 503 with `Retry-After`, and `entropy serve` re-executes
itself (same PID and port, all memory back; a model reload, 3–30 s). Checked live: answer, restart, answer.

## 7b. Verifier as a tool for an assistant (`mcp.py`) — internal tool
`entropy mcp`: an MCP server over stdio (JSON-RPC 2.0, newline-delimited; revisions 2025-06-18 / 2025-03-26 /
2024-11-05). Tools: `verify` (text + custody token → verdict, counts, changed ranges), `detect` (text → exact
p-value, with paragraph-start readings so a sealed passage inside a document is found and located),
`operating_point` (model, weights digest, canonical form, profile, the detector's gate). The keys stay in the
process of whoever runs it: the assistant sends text and receives a verdict, never a key or a record. The text in a
call is data, never instructions — the server measures it and nothing else; it writes nothing, and one model is
loaded lazily on the first call. A failed check is a tool error, not a dead server.

## 7c. Time stamp and evidence package (`ts3161.py`, `pack.py`)
- **RFC 3161 time stamp** over a statement's *logged form* (so a receipt can still be added afterwards):
  `entropy timestamp --statement s.cose [--tsa URL]`. Without `--tsa` only the DER request is written, to be sent
  by hand. The token travels in the statement's unprotected header (RFC 9921 `3161-ttc`, label 269) or beside it in
  a package. We build the request, read the token's claims (imprint, generation time, serial, policy) and check the
  imprint against the digest we asked to stamp; **we do not validate the authority's CMS signature or its
  certificate chain** — that needs a trust list, which belongs to the reader. The package says so and prints the
  `openssl ts -verify` command.
- **Evidence package** (`entropy evidence`): one CBOR file (`ep1-pack/v1`) holding the disclosed text, the signed
  statements about it, their receipts, the public keys those signatures verify under, any time-stamp token and a
  one-line note on provenance. The record never goes in: it stays the issuer's secret, and the statements speak
  about it in digests. `--check` re-runs offline every check that needs no model — each signature, the digest that
  ties the text to the verdict, the reference that ties the verdict to the generation statement, the same-document
  claim, inclusion in the log, the time-stamp imprint — and reports what it cannot do (re-read the text with the
  model; validate the authority's chain). `--out-dir` explodes the package into files with a `README.md` that
  names each part and gives the exact commands. Trust stays with the reader: the package carries its own keys, so
  the reader compares them with the issuer's published key.

## 7d. Attested sealing and attested detection (`attested/`) — spec [0100]–[0109]
The record binds a text to a model's choices, but whoever holds `K` can seal any text (§8, last row). Here the keys,
the model and the sealing code run inside a confidential VM with a confidential GPU, and every statement carries
evidence of that, appraised by a credential authority before it certifies the statement key.

**First platform: Azure `Standard_NCC40ads_H100_v5`** — an AMD SEV-SNP guest (Genoa) under Microsoft's paravisor,
with an H100 NVL in confidential-computing mode. The paravisor owns REPORT_DATA; a guest binds its own 64 octets
through the vTPM: it writes them to NV index 0x01400002 (guest attestation input, 64 octets), and the paravisor
writes a fresh report to NV 0x01400001 whose runtime claims (JSON) carry them as `user-data` (hex), together with
the vTPM attestation key `HCLAkPub` and the VM configuration; REPORT_DATA holds the hash of those runtime claims
(type in the report's request data). Layout: header 32 octets (`HCLA`, version 2, report size, request type,
status), the SNP report (1184 octets, slot 0x4a0), request data (data size, version, report type, hash type,
variable size), runtime claims. Source: microsoft/openvmm `vm/devices/tpm/tpm_guest_tests/src/report.rs`,
`openhcl/openhcl_attestation_protocol/src/igvm_attest/get.rs`, `vm/devices/tpm/tpm_protocol/src/lib.rs`.
Other platforms put the binding straight into the processor's guest field (SEV-SNP REPORT_DATA at 0x50; TDX
REPORTDATA).

**Binding value** (64 octets): SHA-512 over length-prefixed fields (4-octet big-endian length, then the octets):
`"ep1/attested-binding/v1"`, the freshness handle (the authority's nonce), `rrp_id` (the authority's identity, held
in the sealer's configuration, never taken from the wire), `target` (the sealing service's identity), SHA-256 of
the certificate signing request (DER), the statement public key (SubjectPublicKeyInfo DER), the weights digest, the
tokenizer digest, the code identity (SHA-256 of the sorted list of [path, SHA-256(file)] of the installed package),
the sealing class, and SHA-256 of the sealing parameters (canonical CBOR). The GPU report is taken over the same
handle (32 octets); the vTPM quote's qualifying data is SHA-256 of the binding value.

**Enrollment** (attestation-based, after the message shapes of draft-novak-rats-tacra): the authority returns
`{freshness_kind: "present-nonce", handle, rrp_id, mode: "enroll", expires_in}`; the sealer answers with
`{freshness_kind, target, credential_type: "ep1-statement-key", handle, evidence, profile: "ep1/azure-cgpu/v1",
binding: {method: "binding-input", hash: "sha512"}, csr}`. The conduit between them is untrusted.

**Appraisal** — each check is named in the result:
- `cpu`: VCEK → ASK → ARK to the pinned ARK-Genoa (AMD's RSASSA-PSS certificates encode the PSS trailer field at its
  default, which DER forbids: cryptography 42.0.8 refuses them, 43.0.3 and later read and verify them — hence the
  dependency floor), the ASK not on AMD's CRL; the report's ECDSA P-384 signature under the VCEK; the VCEK's TCB
  extensions equal the reported TCB; TCB at or above the floors of AMD's bulletins (SNP 0x1B, microcode 0x56 on Genoa);
  ALIAS_CHECK_COMPLETE (BadRAM) and mitigation vector bit 1 (RMPocalypse); DEBUG off; the launch endorsed — its
  measurement in the reference set, or an ID block signed by a pinned ID key (the firmware launches only a measurement
  the ID key signed, AMD 56860 rev. 1.59 §8.19.2; Azure runs several paravisor builds at once under one key);
  REPORT_DATA = hash of the runtime claims, the rest zero.
- `vtpm`: `user-data` = the binding value; `HCLAkPub` present; secure boot and the vTPM on; the quote (TPMS_ATTEST)
  signed by that AK, its qualifying data = SHA-256(binding), its PCR digest = the quoted PCRs; the TCG event log (each
  event carrying exactly one digest of every bank the Spec ID event announces, at that bank's size), with systemd's
  after-boot measurements (PCR 11), replays to them; the boot chain (PCR 4, 7) and the kernel image's
  sections (`.linux`, `.initrd`, `.cmdline`, measured by systemd-stub) in the reference set. The paravisor renews its
  report on read, at most every 2 s: the attester reads until the report carries what it wrote.
- `gpu0`: the certificate chain to NVIDIA's pinned device identity root; the SPDM measurement response signed by the
  device's attestation key over the handle; the firmware ID in the device certificate = the one in the report; the
  measurements against NVIDIA's reference integrity manifests (driver and VBIOS; RIM index i is SPDM block i + 1; the
  manifests' chain checked here to NVIDIA's pinned RIM root, because signxml 5 refuses it under RFC 5280); revocation
  of the GPU chain and both manifest chains by OCSP, the responses kept.
- binding: recomputed from the request's fields equals `user-data`; the handle was issued by this authority and has
  not expired or been used; the CSR's signature verifies under its own key, which is the key in the binding.

**Attestation result** — EAR (draft-ietf-rats-ear-04) as a CWT, COSE_Sign1 under the authority's root key itself
(never an end-entity key: sealers hold end-entity certificates of the same root, and a result that verified under
any of them could be forged by a sealer for itself): profile `tag:ietf.org,2026:rats/ear#04`, `iat`/`exp`, verifier identity, `eat_nonce` = handle, `ear_status`,
submods `cpu`, `vtpm`, `gpu0` (each with status, trustworthiness vector, policy identifiers, attester claims), and
`ear_device_topology` `{"cpu": ["vtpm", "gpu0"]}` — which GPU is bound to which guest. The appraised identities
(statement key, model digests, code identity, class, parameters, binding value) travel as an ep1 extension.

**Certificate**: X.509 for the Ed25519 statement key, issued by the authority's root; subject = `target`; extended
key usage id-kp-documentSigning; valid from five minutes before issue (the sealer checks it against its own clock)
for the result's lifetime. The sealer pins the authority's root and `rrp_id` in its own configuration. Statements signed in the environment carry `x5chain` (RFC 9360, label 33) and
SHA-256 of the attestation result in the protected header, so both reach the log with the statement.

**Attested detection**: the same enrollment with `credential_type: "ep1-detection-key"`; `K_wm` is generated inside
the environment and never leaves it; detection is a service that returns detection statements signed under the
certified key. Whoever runs the service can no longer imitate the mark with `K_wm` (§8), and the exact null is
unchanged.

**Evidence package `ep1-pack/v2`** adds `att`: the result, the enrollment request (evidence + CSR), the collateral
used (AMD chain and CRL, GPU chain, RIMs, OCSP responses) and the policy (reference values). The offline check
appraises the evidence again with the same code at the result's issue time, verifies the result's signature to the
authority's root (pinned by the reader), the certificate, the statement key = the key in the binding, and then the
existing chain (text → statements → log → time stamp). It names every root, reference value or revocation status it
was not given.

**What it proves, what it does not.** Hardware-signed: a genuine AMD SEV-SNP processor at the reported TCB, the
paravisor's measurement, debug off, the vTPM key bound to that report, the boot chain in the quoted PCRs, a genuine
H100 whose firmware and driver match NVIDIA's manifests; the statement key was generated by software in this guest,
the only party that can write the guest input. Reported by that software: the model digests, the code identity, the
class and the parameters inside the binding — trustworthy as far as the guest's software is (boot chain appraised;
user space measured when the image measures it). Excluding the operator needs a locked image — no interactive
access, measured root file system — whose measurements become the reference values. Time comes from outside the
environment (RFC 3161 over the logged form, the log's registration), never from the guest's clock.

## 7e. The ep1 CBOR profile, and fuzzing (`cose.check_profile`, `impl-js`, `fuzz/`)
Both implementations decode untrusted CBOR only within one profile, checked before any value is built: definite
lengths, each within the bytes that are left; integers and lengths in their shortest form; tags 16 (COSE_Encrypt0)
and 18 (COSE_Sign1) only; `false`, `true` and `null` as the only simple values; no floating point; map keys integers,
text or byte strings, never twice in one map; at most 64 levels of nesting; text valid UTF-8. Two decoders that
read the same bytes differently are how signed formats get bypassed elsewhere; within the profile both accept
exactly the same inputs and decode them to the same values.

`fuzz/targets.py` holds one target per parser of untrusted input — records, tokens, COSE, delivered records,
statements, receipts, time stamps, evidence packages, the paravisor's report, SNP reports, TPM quotes, signatures
and event logs, systemd's log, GPU reports, NVIDIA's manifests, enrollment messages, attestation results, the
authority's appraisal — with the errors each may raise (the offline package check and the appraisal never raise:
they report). Property targets check that canonical form v5 is idempotent and that its index map stays inside the
text; differential targets give the same input to both implementations (canonical forms, blocks, visible text,
CBOR). Seeds are real data (`fuzz/make_corpus.py`); every test run replays each corpus and drives each target with
Hypothesis (`tests/test_fuzz.py`); coverage-guided campaigns run with atheris (`fuzz/run.py`,
`fuzz/campaign_gcp.sh`), and whatever they find joins the corpus.

## 8. What is proven, against whom (threat model)
| Adversary | Outcome |
|---|---|
| Forger without the key alters a sealed text | `MODIFIED` with location (content layer is exact) |
| Forger without the key presents a new text as sealed | no valid record → `NOT_VERIFIABLE`; detection p-value ~ uniform |
| Ordinary transport of an unaltered text | `VERIFIED`; detection unchanged |
| Verifier holding `K_ver` tries to mark or to re-key | cannot: `K_bits`, `K_doc`, `K`, `K_wm` not derivable |
| Verifier holding `K_ver` forges a record for another text under the same document id | possible with a symmetric key — defeated by the registered generation statement (commitment mismatch) |
| Detector holding `K_wm` | can detect and imitate the mark; cannot open records |
| Different hardware / precision at verification | tolerance → `INDETERMINATE`, never a false `MODIFIED` on identical text |
| Paraphrase, translation, heavy rewriting | `MODIFIED`; detection signal lost (as for every watermark) |
| **Issuer holding `K` seals a text its model did not write** | **not prevented by the record alone.** Proof of model origin against the issuer requires attested generation (TEE + RATS evidence) — stage 3. |

## 9. Honest status
| Component | Status |
|---|---|
| Canonical form v5 (deletions before NFC, NFC pinned to Unicode 14.0, typographic folding, script sentence ends), content layer with ordered alignment, model layer, verdicts | implemented, tested; v3 and v4 records verify |
| ep1 CBOR profile in both implementations; 26 fuzz targets (properties, differential) with seed corpora from real data | implemented; replayed and Hypothesis-driven in every test run |
| Canonical generation (token-identical re-read), confidence gate and guard, unsatisfiable-position exclusion | implemented, tested; alignment audit 100 % on distilgpt2 and Qwen2.5-0.5B |
| Paragraph-start readings for localized detection (single and batched) | implemented, tested |
| Sealing gateway (`entropy serve`, OpenAI-compatible) and presentation typography | implemented, tested (local, internal) |
| HKDF keys, AEAD token, verifier re-issue, detection key | implemented, tested |
| Calibrated tolerance (logit space, per-class table, sealed-text corpus) | implemented, tested |
| Content-keyed selection, verifier-context enforcement, record-free detection with exact p-value | implemented, tested; gate G10 |
| COSE statements (generation / verification / detection), local RFC 9162 log with RFC 9942 receipts | implemented, tested; test vectors |
| SCRAPI Transparency Service and client | implemented, tested (local); interop with third-party services pending |
| Second implementation (JavaScript, model-free parts) | 72/72 test-vector checks; CI regenerates the vectors with Python and runs both |
| Record delivered by public key: HPKE (RFC 9180, vectors A.1.1/A.2.1) + COSE-HPKE integrated encryption (draft-27) | implemented, tested; second implementation opens it |
| RFC 3161 time stamps over statements (`entropy timestamp`; imprint checked, the authority's chain is the reader's trust decision) | implemented, tested |
| TEE-attested sealing and detection (§7d) | implemented; run end to end on an Azure confidential H100 (`bench/results/attested/`): two keys certified after 48 checks each, 20 texts, 20 packages checking offline |
| Backends: HF transformers, single and batched | implemented, tested · vLLM / llama.cpp — pending |
| Model identity: weights digest over exactly the files the loader reads (index of shards) | implemented, tested |
| GPU evaluation program (`bench/gpu/`): phase A (15.09, L40S, Qwen2.5-7B), phases B and C (26.09: L40S, L4, A10G, H100; Qwen2.5-7B and 14B, Mistral-7B-v0.3, OLMo-2-7B; a two-million-pair null; KGW and SynthID-Text) | run; results in `runs_gpu/`, reports in `bench/results/phase_{b,c}/REPORT.md` |

Known behaviour to measure at scale: on a small base model that falls into blank-line loops (distilgpt2: 57 % of
unsealed texts), whether content-keyed enforcement breaks the loop depends on the key (0–56 % of sealed texts
degenerate across 8 keys; interval selection breaks loops for every key). Evaluation corpora therefore rotate
over ≥ 5 keys; a key-independent loop guard is designed but not adopted before 7–14B measurements.

## 10. New matter for the provisional (do not publish before filing)
Canonical sealing (canonical text, context-free, char anchors, keep holding checkpoints with margin); the
two-layer record with "identical canonical text is never MODIFIED"; ordered block alignment and deletion-point
localization; logit-space tolerance with a per-class calibration table, seal-time margin and verifier-class lookup;
calibration on sealed text; generation in any precision with measurement in a certified precision; canonical
form v3 (invisible characters, embedded variation-selector payloads); **content-keyed checkpoint selection with
record-free detection whose null is exactly Binomial(n, ½) (k = 1 complementary predicates, context
de-duplication, key-independent near-tie exclusion)**; **enforcement in the verifier's context via a second,
prompt-free cache (LOW = verifier-context first choice; HIGH = best prompt-distribution token outside the
verifier-context top-k with the calibrated margin)**; salted commitment over the record plaintext registered as a
COSE Hash Envelope before release; verification and detection statements referencing the generation statement;
both enforcement contexts stacked in one batch (weights read once per step); generation and record measurement in
different precisions; detection against many keys from one model pass (key-independent positions, ranks and
exclusion); model identity over exactly the loader's files.
Added in the ideal-product work (stage 2+): **key-independent confidence gate** read in the verifier's view, with
record entries restricted to positions below the gate in the record's own measurement pass; **canonical form v4**
(typographic folding as the inverse of word-processor autocorrect; sentence blocks by scripts' full stops);
**canonical generation** — generating exactly the token sequence the verifier re-reads from the canonical text:
bans of every character the canonical form changes, a UTF-8 state machine with forbidden-prefix and continuation
rules, the partial-character ban when the verifier's first choice is banned, the line-end whitespace rule, the
first token taken in the verifier's view as the verifier reads it after stripping, join checks by re-tokenization
with probe continuations and their fallbacks, HIGH exclusion of the first choice's byte prefixes and
whitespace-only extensions; **detector exclusion of positions where the sealer could not satisfy LOW** (read from
text and model, preserving the exact Binomial null); **paragraph-start readings** for localized detection with a
Bonferroni correction across readings; **presentation typography by fold-inverse substitution** so a localized,
typeset text verifies as the canonical sealed one; a **sealing gateway** returning custody tokens with each
completion. Added 14.09: a **visible-text layer** — the record also binds the text as a reader sees it once its
Markdown is rendered (renderer on the sealing side only; the verifier normalizes the candidate literally: line-start
list markers, whitespace, line joins), with its own blocks located per visible line, giving the verdict
`VERIFIED_TEXT` for a rendered copy while a changed link target in a source stays `MODIFIED`; and a **memory budget
with self-re-execution** for a long-running sealing service on unified-memory accelerators; an **evidence package** whose offline check re-establishes the chain text → verdict → generation statement → log → time stamp without any secret, and a **verifier exposed as assistant tools** that never hands out a key or a record. Added 15.09: the
**tolerance measured only over positions that can hold a checkpoint** (the table carries its first measured
position; sealing and detection refuse a table that does not cover their first checkpoint).
