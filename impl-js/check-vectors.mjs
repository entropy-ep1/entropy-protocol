// Checks the second implementation against spec/test-vectors/ep1-vectors.json (written by the Python reference).
// Usage: node impl-js/check-vectors.mjs
import fs from "node:fs";
import * as ep1 from "./ep1.mjs";

const v = JSON.parse(fs.readFileSync(new URL("../spec/test-vectors/ep1-vectors.json", import.meta.url)));
let failures = 0, checks = 0;
const eq = (name, got, want) => {
  checks++;
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) { failures++; console.log(`FAIL ${name}\n  got  ${g.slice(0, 300)}\n  want ${w.slice(0, 300)}`); }
};
const hex = (b) => Buffer.from(b).toString("hex");

eq("canon id (v3 vectors)", ep1.CANON_V3, v.canon_id);
const master = Buffer.from(v.keys.master, "hex");
const k = ep1.keys(master, v.keys.doc, v.keys.verifier);
eq("k_bits", hex(k.kBits), v.keys.k_bits); eq("k_doc", hex(k.kDoc), v.keys.k_doc);
eq("k_log", hex(k.kLog), v.keys.k_log); eq("k_ver", hex(k.kVer), v.keys.k_ver); eq("k_wm", hex(k.kWm), v.keys.k_wm);
eq("bitstream", v.bitstream.j.map((j) => ep1.bit(k.kBits, v.keys.doc, j)), v.bitstream.bits);
eq("bitstream 0.75", v.bitstream.j.map((j) => ep1.bit(k.kBits, v.keys.doc, j, 0.75)), v.bitstream["bits_low_share_0.75"]);
eq("content selection", v.content_selection.contexts.map((c) => ep1.wmDecide(k.kWm, c, v.content_selection.interval)),
   v.content_selection.decisions);
for (const [i, c] of v.canonicalization.entries()) eq(`canonicalization[${i}]`, ep1.canonicalize(c.input, ep1.CANON_V3), c.output);
eq("segmentation", ep1.segment(v.segmentation.text, ep1.CANON_V3).map((s) => [...s, ep1.blockDigest(v.segmentation.text, s)]),
   v.segmentation.blocks);
// canonical form v5 (the reference): v4's steps with the deletions before NFC, idempotent
const c5 = v.canonicalization_v5;
eq("canon id v5", ep1.CANON_V5, c5.canon_id);
for (const [i, c] of c5.cases.entries()) {
  eq(`canonicalization v5[${i}]`, ep1.canonicalize(c.input, ep1.CANON_V5), c.output);
  eq(`canonicalization v5[${i}] idempotent`, ep1.canonicalize(c.output, ep1.CANON_V5), c.output);
  if (c.v4 !== null) eq(`canonicalization v5[${i}] as v4`, ep1.canonicalize(c.input, ep1.CANON_V4), c.v4);
}
eq("canonical v5 segmentation text", ep1.canonicalize(c5.segmentation.text, ep1.CANON_V5), c5.segmentation.canonical);
eq("segmentation v5", ep1.segment(c5.segmentation.canonical, ep1.CANON_V5).map((s) => [...s, ep1.blockDigest(c5.segmentation.canonical, s)]),
   c5.segmentation.blocks);
const vt2 = v.visible_text_v2;
eq("visible form id v2", ep1.PLAIN_V2, vt2.form);
for (const [i, c] of vt2.cases.entries()) {
  const [vis, spans] = ep1.segmentVisible(c.input, ep1.PLAIN_V2);
  eq(`visible text v2[${i}]`, [vis, spans.map((s) => [...s, ep1.blockDigest(vis, s)])], [c.visible, c.blocks]);
}
// canonical form v4: typographic folding and blocks for scripts written without spaces
const c4 = v.canonicalization_v4;
eq("canon id v4", ep1.CANON_V4, c4.canon_id);
for (const [i, c] of c4.cases.entries()) eq(`canonicalization v4[${i}]`, ep1.canonicalize(c.input, ep1.CANON_V4), c.output);
eq("canonical v4 segmentation text", ep1.canonicalize(c4.segmentation.text, ep1.CANON_V4), c4.segmentation.canonical);
eq("segmentation v4", ep1.segment(c4.segmentation.canonical, ep1.CANON_V4).map((s) => [...s, ep1.blockDigest(c4.segmentation.canonical, s)]),
   c4.segmentation.blocks);

// record: open both tokens, re-encode the record deterministically
const opened = ep1.openToken(k.kLog, v.record.token_log);
eq("record plaintext (custody token)", hex(opened.plaintext), v.record.cbor);
eq("record plaintext (verifier token)", hex(ep1.openToken(k.kVer, v.record.token_verifier).plaintext), v.record.cbor);
eq("record re-encoding", hex(ep1.encode(ep1.decode(opened.plaintext))), v.record.cbor);
const rec = ep1.decode(opened.plaintext);
eq("record blocks recompute", ep1.segment(v.record.text, ep1.CANON_V3).map((s) => [...s, ep1.blockDigest(v.record.text, s)]),
   rec.get("b"));

// a record in canonical form v4 with the confidence gate (wm_gate_pm)
const r4 = v.record_v4_gated;
const o4 = ep1.openToken(k.kLog, r4.token_log);
eq("v4 gated record plaintext", hex(o4.plaintext), r4.cbor);
eq("v4 gated record re-encoding", hex(ep1.encode(ep1.decode(o4.plaintext))), r4.cbor);
const h4 = ep1.decode(o4.plaintext).get("h");
eq("v4 gated record header", [h4.get("canon"), h4.get("wm_gate_pm")], [ep1.CANON_V4, r4.wm_gate_pm]);
eq("v4 gated record text", ep1.canonicalize(r4.raw, ep1.CANON_V4), r4.text);
eq("v4 gated record blocks", ep1.segment(r4.text, ep1.CANON_V4).map((s) => [...s, ep1.blockDigest(r4.text, s)]),
   ep1.decode(o4.plaintext).get("b"));

// visible-text layer (plain.py): the verifier's normalization, and a v4 record of a Markdown answer carrying it
const vt = v.visible_text;
eq("visible form id", ep1.PLAIN_V1, vt.form);
for (const [i, c] of vt.cases.entries()) {
  const [vis, spans] = ep1.segmentVisible(c.input, ep1.PLAIN_V1);
  eq(`visible text[${i}]`, vis, c.visible);
  eq(`visible blocks[${i}]`, spans.map((s) => [...s, ep1.blockDigest(vis, s)]), c.blocks);
}
const r5 = v.record_v4_visible;
const o5 = ep1.openToken(k.kLog, r5.token_log);
eq("visible record plaintext", hex(o5.plaintext), r5.cbor);
eq("visible record re-encoding", hex(ep1.encode(ep1.decode(o5.plaintext))), r5.cbor);
const rec5 = ep1.decode(o5.plaintext), h5 = rec5.get("h");
eq("visible record header", [h5.get("canon"), h5.get("plain")], [ep1.CANON_V4, ep1.PLAIN_V1]);
const hexOf = (str) => ep1.sha256(Buffer.from(str, "utf8")).toString("hex");
eq("Markdown source: canonical digest = the record's (VERIFIED)", hexOf(ep1.canonicalize(r5.markdown, ep1.CANON_V4)), h5.get("text_digest"));
const [cv, cs] = ep1.segmentVisible(r5.copy, ep1.PLAIN_V1);
eq("rendered copy: canonical digest differs", hexOf(ep1.canonicalize(r5.copy, ep1.CANON_V4)) !== h5.get("text_digest"), true);
eq("rendered copy: visible digest = the record's (VERIFIED_TEXT)", hexOf(cv), h5.get("plain_digest"));
eq("rendered copy: visible blocks = the record's", cs.map((s) => [...s, ep1.blockDigest(cv, s)]), rec5.get("p"));
eq("edited copy: visible text", ep1.plainNormalize(r5.edited, ep1.PLAIN_V1), r5.edited_visible);
eq("edited copy: visible digest differs (MODIFIED)", hexOf(ep1.plainNormalize(r5.edited, ep1.PLAIN_V1)) !== h5.get("plain_digest"), true);

// generation statement: commitment recomputed from the record, signature, envelope
const g = v.generation_statement;
const signer = ep1.publicKeyFromSeed(Buffer.from(g.signer_seed, "hex"));
const c = ep1.commitment(opened.plaintext, opened.doc);
eq("commitment cbor", hex(ep1.encode(c)), g.commitment_cbor);
const st = ep1.verify1(Buffer.from(g.cose_sign1, "hex"), signer);
eq("statement payload = SHA-256(commitment)", hex(st.payload), g.commitment_sha256);
eq("statement hash envelope headers", [st.prot.get(258), st.prot.get(259), st.prot.get(15).get(2)],
   [-16, "application/ep1-commitment+cbor", v.keys.doc]);
eq("statement kid", hex(st.prot.get(4)), hex(ep1.sha256(ep1.rawPublicKey(signer)).subarray(0, 16)));
eq("statement reference", hex(ep1.sha256(ep1.entryOf(Buffer.from(g.cose_sign1, "hex")))), g.ref);

// verification statement
const vs = v.verification_statement;
const vst = ep1.verify1(Buffer.from(vs.cose_sign1, "hex"), ep1.publicKeyFromSeed(Buffer.from(vs.signer_seed, "hex")));
eq("verification payload", hex(vst.payload), vs.payload_cbor);
eq("verification payload re-encoding", hex(ep1.encode(ep1.decode(vst.payload))), vs.payload_cbor);

// Merkle log and receipt
const m = v.merkle;
const leaves = m.entries.map((e) => ep1.leafHash(ep1.entryOf(Buffer.from(e, "hex"))));
eq("leaf hashes", leaves.map(hex), m.leaf_hashes);
eq("inclusion size 7 index 3", hex(ep1.rootFromInclusion(leaves[3], 3, 7, m.inclusion_size7_index3.map((p) => Buffer.from(p, "hex")))), m.roots[6]);
const r = ep1.verifyReceipt(Buffer.from(m.receipt_index6, "hex"), Buffer.from(m.entries[6], "hex"),
                            ep1.publicKeyFromSeed(Buffer.from(m.log_seed, "hex")));
eq("receipt root", r.root, m.roots[6]);
{ // the unprotected header is not signed: an integer where the inclusion proof belongs must be refused, as in Python
  const rc = ep1.decode(Buffer.from(m.receipt_index6, "hex")), p = rc instanceof ep1.Tag ? rc.value : rc;
  const bad = ep1.encode(new ep1.Tag(18, [p[0], new Map([[396, new Map([[-1, [ep1.encode(5)]]])]]), p[2], p[3]]));
  let refused = "";
  try { ep1.verifyReceipt(bad, Buffer.from(m.entries[6], "hex"), ep1.publicKeyFromSeed(Buffer.from(m.log_seed, "hex"))); }
  catch (e) { refused = e.message; }
  eq("receipt with a malformed inclusion proof refused", refused, "malformed inclusion proof");
}

for (const b of v.binomial_tail) eq(`binomial tail ${b.n},${b.m}`, ep1.binomialTail(b.n, b.m), b.p);

// HPKE (RFC 9180 A.2.1) and a record delivered by public key (COSE-HPKE)
const h = v.hpke_rfc9180_a21, hb = (x) => Buffer.from(x, "hex");
eq("HPKE RFC 9180 A.2.1 open", hex(ep1.hpkeOpen(hb(h.enc), hb(h.skRm), hb(h.info), hb(h.aad), hb(h.ct), 3)), h.pt);
const dr = v.delivered_record;
const del = ep1.openDelivered(hb(dr.cose_encrypt0), hb(dr.recipient_sk));
eq("delivered record plaintext", hex(del.plaintext), v.record.cbor);
eq("delivered record headers", [del.prot.get(1), del.contentType, del.doc], [dr.alg, dr.content_type, v.keys.doc]);

console.log(`${checks - failures}/${checks} checks passed`);
process.exit(failures ? 1 : 0);
