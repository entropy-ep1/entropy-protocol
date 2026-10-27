// Entropy Protocol ep1 — second implementation of the model-free parts, written from spec/ep1.cddl and
// ARCHITECTURE.md in JavaScript (Node.js >= 20, no dependencies): deterministic CBOR, canonical forms v3 to v5,
// segmentation, key derivation, keyed selection, record token, commitment, COSE_Sign1 (Ed25519),
// RFC 9162 inclusion proofs and RFC 9942 receipts, exact binomial tail, HPKE (RFC 9180, base mode) and a record
// delivered by public key (COSE-HPKE integrated encryption, draft-ietf-cose-hpke-27).
import crypto from "node:crypto";
import { assigned14 } from "./unicode14.mjs";

// ------------------------------------------------------------------ CBOR (RFC 8949), deterministic encoding
export class Tag { constructor(tag, value) { this.tag = tag; this.value = value; } }

function head(major, n) {
  const m = major << 5;
  if (typeof n === "bigint") {
    const b = Buffer.alloc(9); b[0] = m | 27; b.writeBigUInt64BE(n, 1); return b;
  }
  if (n < 24) return Buffer.from([m | n]);
  if (n < 0x100) return Buffer.from([m | 24, n]);
  if (n < 0x10000) { const b = Buffer.alloc(3); b[0] = m | 25; b.writeUInt16BE(n, 1); return b; }
  if (n < 0x100000000) { const b = Buffer.alloc(5); b[0] = m | 26; b.writeUInt32BE(n, 1); return b; }
  const b = Buffer.alloc(9); b[0] = m | 27; b.writeBigUInt64BE(BigInt(n), 1); return b;
}

function float16Bits(v) {           // the float16 encoding of v if it is exact, else null
  if (Number.isNaN(v)) return 0x7e00;
  if (v === Infinity) return 0x7c00;
  if (v === -Infinity) return 0xfc00;
  if (v === 0) return Object.is(v, -0) ? 0x8000 : 0;
  const sign = v < 0 ? 0x8000 : 0, a = Math.abs(v);
  const sub = a / 2 ** -24;                                   // subnormal: a = m * 2^-24, m < 1024
  if (Number.isInteger(sub) && sub < 1024) return sign | sub;
  const e0 = Math.floor(Math.log2(a));
  for (const e of [e0 - 1, e0, e0 + 1]) {                     // normal: a = (1 + m/1024) * 2^e, -14 <= e <= 15
    if (e < -14 || e > 15) continue;
    const m = (a / 2 ** e - 1) * 1024;
    if (Number.isInteger(m) && m >= 0 && m < 1024) return sign | ((e + 15) << 10) | m;
  }
  return null;
}

function encodeFloat(v) {
  const h = float16Bits(v);
  if (h !== null) { const b = Buffer.alloc(3); b[0] = 0xf9; b.writeUInt16BE(h, 1); return b; }
  const f32 = new Float32Array([v]);
  if (f32[0] === v) { const b = Buffer.alloc(5); b[0] = 0xfa; b.writeFloatBE(v, 1); return b; }
  const b = Buffer.alloc(9); b[0] = 0xfb; b.writeDoubleBE(v, 1); return b;
}

export function encode(v) {
  if (v === null || v === undefined) return Buffer.from([0xf6]);
  if (v === false) return Buffer.from([0xf4]);
  if (v === true) return Buffer.from([0xf5]);
  if (v instanceof Tag) return Buffer.concat([head(6, v.tag), encode(v.value)]);
  if (Buffer.isBuffer(v) || v instanceof Uint8Array) return Buffer.concat([head(2, v.length), Buffer.from(v)]);
  if (typeof v === "string") { const b = Buffer.from(v, "utf8"); return Buffer.concat([head(3, b.length), b]); }
  if (typeof v === "bigint") return v >= 0n ? head(0, v) : head(1, -1n - v);
  if (typeof v === "number") {
    if (Number.isInteger(v) && !Object.is(v, -0)) return v >= 0 ? head(0, v) : head(1, -1 - v);
    return encodeFloat(v);
  }
  if (Array.isArray(v)) return Buffer.concat([head(4, v.length), ...v.map(encode)]);
  if (v instanceof Map || typeof v === "object") {
    const entries = v instanceof Map ? [...v.entries()] : Object.entries(v);
    const enc = entries.map(([k, x]) => [encode(k), encode(x)]);
    enc.sort((a, b) => Buffer.compare(a[0], b[0]));           // RFC 8949 section 4.2.1: bytewise key order
    return Buffer.concat([head(5, enc.length), ...enc.flat()]);
  }
  throw new Error(`cannot encode ${typeof v}`);
}

// Strict where the reference (cbor2) is strict and bounded where untrusted input could make it loop: a byte past the
// end is an error, not a zero; an array or map may not declare more items than there are bytes left (each item takes
// at least one); text must be valid UTF-8. (Differential fuzzing against the reference, 27.10.2026: a few bytes
// declaring a 2^60-item array kept the old decoder reading zeros past the end; ill-formed UTF-8 was replaced, not
// refused.)
const UTF8 = new TextDecoder("utf-8", { fatal: true });
export function decode(buf) {
  let i = 0;
  const need = (k) => { if (i + k > buf.length) throw new Error("truncated CBOR"); };
  const short = () => { throw new Error("integer or length not in its shortest form"); };
  const readN = (ai) => {
    if (ai < 24) return ai;
    if (ai === 24) { need(1); const n = buf[i++]; if (n < 24) short(); return n; }
    if (ai === 25) { need(2); const n = buf.readUInt16BE(i); i += 2; if (n < 0x100) short(); return n; }
    if (ai === 26) { need(4); const n = buf.readUInt32BE(i); i += 4; if (n < 0x10000) short(); return n; }
    if (ai === 27) {
      need(8); const n = buf.readBigUInt64BE(i); i += 8; if (n < 0x100000000n) short();
      return n <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(n) : n;
    }
    throw new Error("indefinite lengths are not deterministic CBOR");
  };
  const count = (n) => {                                        // a declared length checked against what is left
    if (typeof n === "bigint" || n > buf.length - i) throw new Error("CBOR length beyond the input");
    return n;
  };
  // the ep1 CBOR profile (cose.py check_profile): tags 16 and 18 only, false/true/null only, no floating point,
  // at most 64 levels of nesting
  const item = (depth) => {
    if (depth > 64) throw new Error("CBOR nested too deeply");
    need(1);
    const ib = buf[i++], major = ib >> 5, ai = ib & 31;
    if (major === 7) {
      if (ai === 20) return false; if (ai === 21) return true; if (ai === 22) return null;
      throw new Error("simple value or floating point outside the ep1 profile");
    }
    const n = readN(ai);
    switch (major) {
      case 0: return n;
      case 1: return typeof n === "bigint" ? -1n - n : -1 - n;
      case 2: { count(n); const b = buf.subarray(i, i + n); i += n; return Buffer.from(b); }
      case 3: { count(n); const s = UTF8.decode(buf.subarray(i, i + n)); i += n; return s; }
      case 4: { count(n); const a = []; for (let k = 0; k < n; k++) a.push(item(depth + 1)); return a; }
      case 5: {
        count(n); const m = new Map(), seen = new Set();
        for (let k = 0; k < n; k++) {
          if (i >= buf.length || buf[i] >> 5 > 3) throw new Error("map key that is not an integer, text or byte string");
          const a = i, key = item(depth + 1), raw = buf.toString("hex", a, i);
          if (seen.has(raw)) throw new Error("duplicate map key");
          seen.add(raw); m.set(key, item(depth + 1));
        }
        return m;
      }
      case 6: if (n !== 16 && n !== 18) throw new Error(`tag ${n} outside the ep1 profile`); return new Tag(n, item(depth + 1));
    }
    throw new Error("bad CBOR");
  };
  const out = item(0);
  if (i !== buf.length) throw new Error("trailing bytes");
  return out;
}


// ------------------------------------------------------------------ canonical form v3
export const CANON_V3 = "ep1-canon/v3:nfc+del-cr-bom-zw-shy+del-vs-runs+nbsp+rstrip-lines+strip";
export const CANON_V4 = "ep1-canon/v4:nfc+del-cr-bom-zw-shy+del-vs-runs+nbsp+typo+rstrip-lines+strip;seg2";
// v5: the deletions before NFC, so the form is idempotent (see canon.py)
export const CANON_V5 = "ep1-canon/v5:del-cr-bom-zw-shy+del-vs-runs+nfc-u14+nbsp+typo+rstrip-lines+strip;seg2";
export const CANON_ID = CANON_V5;                              // new records; records name their version
const FOLD = new Map([[0x201c, 0x22], [0x201d, 0x22], [0x201e, 0x22], [0x201f, 0x22], [0xab, 0x22], [0xbb, 0x22],
  [0x2018, 0x27], [0x2019, 0x27], [0x201a, 0x27], [0x201b, 0x27], [0x2bc, 0x27],
  [0x2010, 0x2d], [0x2011, 0x2d], [0x2012, 0x2d], [0x2013, 0x2d], [0x2212, 0x2d]]);
const FOLD_MULTI = new Map([[0x2014, [0x2d, 0x2d]], [0x2015, [0x2d, 0x2d]], [0x2026, [0x2e, 0x2e, 0x2e]]]);   // em dash, ellipsis
const SCRIPT_STOPS = new Set([0x3002, 0xff01, 0xff1f, 0xff0e, 0x061f, 0x06d4, 0x0964, 0x0965, 0x1362, 0x0589]);
const CLOSERS = new Set([0x29, 0x5d, 0x7d, 0x22, 0x27, 0x300d, 0x300f, 0x3011, 0x300b, 0x3009, 0xff09, 0xff3d]);
const DELETE = new Set([0x0d, 0xfeff, 0x200b, 0x2060, 0x00ad]);
const SPACE = new Set([0x00a0, 0x2007, 0x202f]);
const isVS = (c) => (c >= 0xfe00 && c <= 0xfe0f) || (c >= 0xe0100 && c <= 0xe01ef);
const WS = new Set([0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029,
  0x202f, 0x205f, 0x3000]);         // Python str.isspace() for the code points that can reach step 6

// a string from code points without spreading them into one call (that overflows the stack on long texts)
const fromCodePoints = (cps) => {
  let out = "";
  for (let k = 0; k < cps.length; k += 8192) out += String.fromCodePoint(...cps.slice(k, k + 8192));
  return out;
};

function deletions(cps) {                                     // v5 steps 1-2, on the source as given
  const kept = [];
  for (let k = 0; k < cps.length;) {
    const c = cps[k];
    if (c === 0xfeff) {
      let e = k + 1; while (e < cps.length && isVS(cps[e])) e++;
      k = e - (k + 1) >= 8 ? e : k + 1; continue;
    }
    if (!DELETE.has(c)) kept.push(c);
    k++;
  }
  const out = [];
  for (let k = 0; k < kept.length;) {
    if (isVS(kept[k])) {
      let e = k; while (e < kept.length && isVS(kept[e])) e++;
      if (e - k >= 8) { k = e; continue; }
    }
    out.push(kept[k]); k++;
  }
  return out;
}

function finish(cps) {                                        // trailing spaces of lines, then of the text
  const res = [];
  for (let k = 0; k < cps.length;) {
    if (cps[k] === 0x20 || cps[k] === 0x09) {
      let e = k; while (e < cps.length && (cps[e] === 0x20 || cps[e] === 0x09)) e++;
      if (e === cps.length || cps[e] === 0x0a) { k = e; continue; }
    }
    res.push(cps[k]); k++;
  }
  let a = 0, b = res.length;
  while (a < b && WS.has(res[a])) a++;
  while (b > a && WS.has(res[b - 1])) b--;
  return fromCodePoints(res.slice(a, b));
}

// NFC as Unicode 14.0 defines it: cut at every code point not assigned in 14.0, normalize the pieces (see canon.py)
function nfc14(cps) {
  const out = [];
  let piece = [];
  const flush = () => { if (piece.length) { for (const ch of fromCodePoints(piece).normalize("NFC")) out.push(ch.codePointAt(0)); piece = []; } };
  for (const c of cps) {
    if (assigned14(c)) piece.push(c);
    else { flush(); out.push(c); }
  }
  flush();
  return out;
}

function canonicalizeV5(text) {
  const raw = Array.from(text, (ch) => ch.codePointAt(0));
  const nfc = nfc14(deletions(raw));
  const out = [];
  for (const c of nfc) {
    if (FOLD_MULTI.has(c)) out.push(...FOLD_MULTI.get(c));
    else if (FOLD.has(c)) out.push(FOLD.get(c));
    else out.push(SPACE.has(c) ? 0x20 : c);
  }
  return finish(out);
}

export function canonicalize(text, version = CANON_ID) {
  if (version === CANON_V5) return canonicalizeV5(text);
  if (version !== CANON_V3 && version !== CANON_V4) throw new Error("unsupported canonical form");
  const v4 = version === CANON_V4;
  const src = Array.from(text.normalize("NFC"), (ch) => ch.codePointAt(0));
  const kept = [];
  for (let k = 0; k < src.length;) {                          // steps 2-3, 5, 6 (C2PA wrapper as a unit)
    const c = src[k];
    if (c === 0xfeff) {
      let e = k + 1; while (e < src.length && isVS(src[e])) e++;
      k = e - (k + 1) >= 8 ? e : k + 1; continue;
    }
    if (!DELETE.has(c)) {
      if (v4 && FOLD_MULTI.has(c)) kept.push(...FOLD_MULTI.get(c));   // v4: em dash -> "--", ellipsis -> "..."
      else if (v4 && FOLD.has(c)) kept.push(FOLD.get(c));            // v4: quotes, apostrophes, hyphens, en dash
      else kept.push(SPACE.has(c) ? 0x20 : c);
    }
    k++;
  }
  const tmp = [];
  for (let k = 0; k < kept.length;) {                         // step 4: remaining selector runs of 8 or more
    if (isVS(kept[k])) {
      let e = k; while (e < kept.length && isVS(kept[e])) e++;
      if (e - k >= 8) { k = e; continue; }
    }
    tmp.push(kept[k]); k++;
  }
  const res = [];
  for (let k = 0; k < tmp.length;) {                          // step 6: spaces/tabs ending a line or the text
    if (tmp[k] === 0x20 || tmp[k] === 0x09) {
      let e = k; while (e < tmp.length && (tmp[e] === 0x20 || tmp[e] === 0x09)) e++;
      if (e === tmp.length || tmp[e] === 0x0a) { k = e; continue; }
    }
    res.push(tmp[k]); k++;
  }
  let a = 0, b = res.length;
  while (a < b && WS.has(res[a])) a++;
  while (b > a && WS.has(res[b - 1])) b--;
  return fromCodePoints(res.slice(a, b));
}

export function segment(ctext, version = CANON_ID) {         // offsets in code points, as in the record
  const cps = Array.from(ctext, (ch) => ch.codePointAt(0));
  const spans = [];
  const v4 = version === CANON_V4 || version === CANON_V5;    // seg2
  let start = 0;
  const emit = (a, b) => {
    let s = a, e = b;
    while (s < e && WS.has(cps[s])) s++;
    while (e > s && WS.has(cps[e - 1])) e--;
    if (s < e) spans.push([s, e]);
  };
  const stop = (c) => c === 0x2e || c === 0x21 || c === 0x3f;
  for (let i = 0; i < cps.length; i++) {
    const c = cps[i];
    if (c === 0x0a) { emit(start, i); start = i + 1; }
    else if (!v4) {
      if (stop(c) && (i + 1 === cps.length || WS.has(cps[i + 1]))) { emit(start, i + 1); start = i + 1; }
    } else if (stop(c) || SCRIPT_STOPS.has(c)) {              // v4: closers stay with their sentence
      let j = i + 1; while (j < cps.length && CLOSERS.has(cps[j])) j++;
      if (SCRIPT_STOPS.has(c) || j === cps.length || WS.has(cps[j])) { emit(start, j); start = j; i = j - 1; }
    }
  }
  emit(start, cps.length);
  return spans;
}

export const sha256 = (b) => crypto.createHash("sha256").update(b).digest();
export function blockDigest(ctext, [a, b]) {
  return sha256(Buffer.from(String.fromCodePoint(...Array.from(ctext, (c) => c.codePointAt(0)).slice(a, b)), "utf8")).toString("hex").slice(0, 16);
}

// ------------------------------------------------------------------ visible-text layer (plain.py), verifier's side
// A candidate is taken literally (never rendered): canonical form v4; per line one list marker at its start (bullets,
// an ordinal "12." / "12)", a task box "[ ]" / "[x]" or a ballot box) removed; whitespace runs one space; empty lines
// dropped; lines joined by one space. Blocks: canonical v4 segmentation inside each line, as code-point spans.
export const PLAIN_V1 = "ep1-plain/v1:canon-v4+commonmark-gfm-visible+list-marker+ws";
export const PLAIN_V2 = "ep1-plain/v2:canon-v5+commonmark-gfm-visible+list-marker+ws";
export const PLAIN_ID = PLAIN_V2;
const canonOf = (form) => {
  if (form === PLAIN_V2) return CANON_V5;
  if (form === PLAIN_V1) return CANON_V4;
  throw new Error("unsupported visible-text form");
};
const MARKER = /^[ \t]*(?:(?:[-*+\u2022\u25e6\u25aa\u2023\u25cf\u00b7]|[0-9]{1,9}[.)])[ \t]+)?(?:(?:\[[ xX]\]|[\u2610\u2611\u2612])[ \t]+)?/u;
const VWS = /[ \t\n\x0b\x0c\u1680\u2000-\u200a\u2028\u2029\u205f\u3000]+/gu;
function visibleLines(text, form = PLAIN_ID) {
  const out = [];
  for (const raw of canonicalize(text, canonOf(form)).split("\n")) {
    const line = raw.replace(MARKER, "").replace(VWS, " ").replace(/^ +| +$/g, "");
    if (line) out.push(line);
  }
  return out;
}
export const plainNormalize = (text, form = PLAIN_ID) => visibleLines(text, form).join(" ");
export function segmentVisible(text, form = PLAIN_ID) {      // [visible text, blocks as code-point spans]
  const lines = visibleLines(text, form), spans = [];
  let pos = 0;
  for (const line of lines) {
    for (const [a, b] of segment(line, canonOf(form))) spans.push([pos + a, pos + b]);
    pos += Array.from(line).length + 1;
  }
  return [lines.join(" "), spans];
}

// ------------------------------------------------------------------ keys and keyed selection
export const hkdf = (key, info) => Buffer.from(crypto.hkdfSync("sha256", key, Buffer.alloc(0), Buffer.from(info, "utf8"), 32));
export const keys = (master, doc, verifier) => {
  const kDoc = hkdf(master, `ep1/doc|${doc}`);
  return { kBits: hkdf(master, "ep1/bits"), kDoc, kLog: hkdf(kDoc, "ep1/log"),
           kVer: hkdf(kDoc, `ep1/verifier|${verifier}`), kWm: hkdf(master, "ep1/wm") };
};
const hmac = (k, m) => crypto.createHmac("sha256", k).update(m).digest();
export function bit(kBits, doc, j, lowShare = 0.5) {
  const d = hmac(kBits, Buffer.from(`${doc}:${j}`, "utf8"));
  if (lowShare === 0.5) return d[0] & 1;
  return d.readUInt16BE(1) < Math.round(lowShare * 65536) ? 0 : 1;
}
export function wmDecide(kWm, context, interval) {
  const msg = Buffer.concat([Buffer.from("ep1/wm|"), ...context.map((t) => { const b = Buffer.alloc(4); b.writeUInt32BE(t); return b; })]);
  const d = hmac(kWm, msg);
  return d.readUInt32BE(0) % interval ? null : d[4] & 1;
}

// ------------------------------------------------------------------ record token
export function openToken(key, token) {
  if (!token.startsWith("ep1.")) throw new Error("not an ep1 token");
  const env = decode(Buffer.from(token.slice(4), "base64url"));
  const doc = env.get("doc"), nonce = env.get("n"), ct = env.get("ct");
  const aad = encode(new Map([["v", 1], ["doc", doc]]));
  const d = crypto.createDecipheriv("chacha20-poly1305", key, nonce, { authTagLength: 16 });
  d.setAAD(aad, { plaintextLength: ct.length - 16 });
  d.setAuthTag(ct.subarray(ct.length - 16));
  return { doc, plaintext: Buffer.concat([d.update(ct.subarray(0, ct.length - 16)), d.final()]) };
}

export function commitment(plaintext, doc) {
  const h = decode(plaintext).get("h");
  return new Map([["v", 1], ["doc", doc], ["rec", sha256(plaintext)],
    ["txt", sha256(Buffer.concat([Buffer.from(h.get("salt"), "hex"), Buffer.from(h.get("text_digest"), "hex")]))],
    ["canon", h.get("canon")], ["wts", Buffer.from(h.get("weights_digest"), "hex")],
    ["tok", Buffer.from(h.get("tokenizer_digest"), "hex")]]);
}

// ------------------------------------------------------------------ COSE_Sign1 with Ed25519
const PKCS8_ED25519 = Buffer.from("302e020100300506032b657004220420", "hex");
export const publicKeyFromSeed = (seed) => crypto.createPublicKey(crypto.createPrivateKey({ key: Buffer.concat([PKCS8_ED25519, seed]), format: "der", type: "pkcs8" }));
export const rawPublicKey = (pub) => Buffer.from(pub.export({ format: "jwk" }).x, "base64url");

export function verify1(bytes, pub, detachedPayload = null) {
  let obj = decode(bytes);
  if (obj instanceof Tag) { if (obj.tag !== 18) throw new Error("not COSE_Sign1"); obj = obj.value; }
  const [protBytes, unprot, carried, sig] = obj;
  const prot = protBytes.length ? decode(protBytes) : new Map();
  if (![-19, -8].includes(prot.get(1))) throw new Error("unsupported algorithm");
  const payload = carried ?? detachedPayload;
  const tbs = encode(["Signature1", protBytes, Buffer.alloc(0), payload]);
  if (!crypto.verify(null, tbs, pub, sig)) throw new Error("signature does not verify");
  return { prot, unprot, payload };
}

export const entryOf = (statement) => {
  let obj = decode(statement); if (obj instanceof Tag) obj = obj.value;
  return encode(new Tag(18, [obj[0], new Map(), obj[2], obj[3]]));
};

// ------------------------------------------------------------------ RFC 9162 Merkle tree, RFC 9942 receipts
export const leafHash = (entry) => sha256(Buffer.concat([Buffer.from([0]), entry]));
const node = (l, r) => sha256(Buffer.concat([Buffer.from([1]), l, r]));
export function rootFromInclusion(leaf, index, size, path) {
  if (index >= size) throw new Error("leaf index outside the tree");
  let fn = index, sn = size - 1, r = leaf;
  for (const p of path) {
    if (sn === 0) throw new Error("path too long");
    if ((fn & 1) || fn === sn) { r = node(p, r); while (!(fn & 1) && fn !== 0) { fn >>= 1; sn >>= 1; } }
    else r = node(r, p);
    fn >>= 1; sn >>= 1;
  }
  if (sn !== 0) throw new Error("path too short");
  return r;
}
export function verifyReceipt(receipt, statement, logPub) {
  let obj = decode(receipt); if (obj instanceof Tag) obj = obj.value;
  if (!(Array.isArray(obj) && obj.length === 4 && Buffer.isBuffer(obj[0]) && obj[1] instanceof Map)) throw new Error("not a COSE_Sign1");
  const prot = decode(obj[0]);
  if (!(prot instanceof Map) || prot.get(395) !== 1) throw new Error("unsupported VDS");
  const vdp = obj[1].get(396), proofs = vdp instanceof Map ? vdp.get(-1) : undefined;
  if (!(Array.isArray(proofs) && proofs.length && Buffer.isBuffer(proofs[0]))) throw new Error("no inclusion proof");
  const proof = decode(proofs[0]);                  // [tree size, leaf index, [sibling hash, ...]], as tlog.py
  if (!(Array.isArray(proof) && proof.length === 3 && proof.slice(0, 2).every((v) => Number.isSafeInteger(v) && v >= 0)
        && Array.isArray(proof[2]) && proof[2].every((h) => Buffer.isBuffer(h) && h.length === 32))) {
    throw new Error("malformed inclusion proof");
  }
  const [size, index, path] = proof;
  const root = rootFromInclusion(leafHash(entryOf(statement)), index, size, path);
  verify1(receipt, logPub, root);
  return { size, index, root: root.toString("hex") };
}

// ------------------------------------------------------------------ HPKE (RFC 9180): base mode, single shot
// DHKEM(X25519, HKDF-SHA256) (kem 0x0020), HKDF-SHA256 (kdf 1), AES-128-GCM (aead 1) or ChaCha20-Poly1305 (aead 3)
const i2osp = (n, len) => { const b = Buffer.alloc(len); b.writeUIntBE(n, 0, len); return b; };
const extract = (salt, ikm) => hmac(salt.length ? salt : Buffer.alloc(32), ikm);
function expand(prk, info, len) {
  let out = Buffer.alloc(0), block = Buffer.alloc(0);
  for (let i = 1; out.length < len; i++) { block = hmac(prk, Buffer.concat([block, info, Buffer.from([i])])); out = Buffer.concat([out, block]); }
  return out.subarray(0, len);
}
const V1 = Buffer.from("HPKE-v1");
const labeledExtract = (suite, salt, label, ikm) => extract(salt, Buffer.concat([V1, suite, Buffer.from(label), ikm]));
const labeledExpand = (suite, prk, label, info, len) => expand(prk, Buffer.concat([i2osp(len, 2), V1, suite, Buffer.from(label), info]), len);
const KEM_SUITE = Buffer.concat([Buffer.from("KEM"), i2osp(0x20, 2)]);
const X25519_PKCS8 = Buffer.from("302e020100300506032b656e04220420", "hex");
const X25519_SPKI = Buffer.from("302a300506032b656e032100", "hex");
const AEADS = { 1: ["aes-128-gcm", 16], 3: ["chacha20-poly1305", 32] };

export function hpkeOpen(enc, skR, info, aad, ct, aeadId = 3) {
  const priv = crypto.createPrivateKey({ key: Buffer.concat([X25519_PKCS8, skR]), format: "der", type: "pkcs8" });
  const pkR = Buffer.from(crypto.createPublicKey(priv).export({ format: "jwk" }).x, "base64url");
  const dh = crypto.diffieHellman({ privateKey: priv, publicKey: crypto.createPublicKey({ key: Buffer.concat([X25519_SPKI, enc]), format: "der", type: "spki" }) });
  const shared = labeledExpand(KEM_SUITE, labeledExtract(KEM_SUITE, Buffer.alloc(0), "eae_prk", dh), "shared_secret", Buffer.concat([enc, pkR]), 32);
  const [cipher, nk] = AEADS[aeadId];
  const suite = Buffer.concat([Buffer.from("HPKE"), i2osp(0x20, 2), i2osp(1, 2), i2osp(aeadId, 2)]);
  const ctx = Buffer.concat([Buffer.from([0]), labeledExtract(suite, Buffer.alloc(0), "psk_id_hash", Buffer.alloc(0)),
                             labeledExtract(suite, Buffer.alloc(0), "info_hash", info)]);
  const secret = labeledExtract(suite, shared, "secret", Buffer.alloc(0));
  const key = labeledExpand(suite, secret, "key", ctx, nk), nonce = labeledExpand(suite, secret, "base_nonce", ctx, 12);
  const d = crypto.createDecipheriv(cipher, key, nonce, { authTagLength: 16 });
  d.setAAD(aad, { plaintextLength: ct.length - 16 });
  d.setAuthTag(ct.subarray(ct.length - 16));
  return Buffer.concat([d.update(ct.subarray(0, ct.length - 16)), d.final()]);
}

// a record delivered to a verifier: COSE_Encrypt0 (tag 16), alg HPKE-3 (41) / HPKE-4 (42), "ek" (-4) unprotected,
// HPKE info empty, AAD = Enc_structure ["Encrypt0", protected, h''] (draft-ietf-cose-hpke-27 code points)
export function openDelivered(bytes, skR) {
  let obj = decode(bytes);
  if (obj instanceof Tag) { if (obj.tag !== 16) throw new Error("not COSE_Encrypt0"); obj = obj.value; }
  const [protBytes, unprot, ct] = obj;
  const prot = decode(protBytes);
  const aead = { 41: 1, 42: 3 }[prot.get(1)];
  if (!aead) throw new Error("not COSE-HPKE integrated encryption");
  const plaintext = hpkeOpen(unprot.get(-4), skR, Buffer.alloc(0), encode(["Encrypt0", protBytes, Buffer.alloc(0)]), ct, aead);
  return { prot, doc: prot.get(15).get(2), contentType: prot.get(3), plaintext };
}

// ------------------------------------------------------------------ exact binomial tail P(X >= m), X ~ Bin(n, 1/2)
export function binomialTail(n, m) {
  if (m <= 0) return 1; if (m > n) return 0;
  let c = 1n, sum = 0n;                                         // c = C(n, i), iterating i = 0..n
  for (let i = 0; i <= n; i++) { if (i >= m) sum += c; c = c * BigInt(n - i) / BigInt(i + 1); }
  return ratioPow2(sum, n);
}

// sum / 2^n rounded to the nearest double, ties to even: what Python's int / int gives (correctly rounded). The
// plain Number(sum) / 2 ** n overflows from n = 1024 (both sides Infinity: 0 at 1024, NaN beyond) - a document with
// more than 1023 checkpoints - and rounds twice in the subnormal range (found by differential fuzzing, 27.10.2026).
function ratioPow2(sum, n) {
  if (sum === 0n) return 0;
  const bl = sum.toString(2).length;
  const e = bl - 1 - n;                                         // exponent of the leading bit
  if (e < -1075) return 0;
  if (e === -1075) return sum > (1n << BigInt(bl - 1)) ? 5e-324 : 0;  // above half the least subnormal, or a tie to 0
  const bits = e >= -1022 ? 53 : e + 1075;                      // significant bits a double holds at that exponent
  let shift = bl - bits, q;
  if (shift > 0) {
    const s = BigInt(shift);
    q = sum >> s;
    const rem = sum - (q << s), half = 1n << (s - 1n);
    if (rem > half || (rem === half && (q & 1n))) q += 1n;
  } else {
    q = sum << BigInt(-shift);
  }
  let x = Number(q), k = shift - n;                             // the value is q * 2^k, exactly representable
  while (k < -1000) { x *= 2 ** -1000; k += 1000; }             // each step exact: x stays normal until the last
  return x * 2 ** k;
}
