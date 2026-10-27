"""TPM 2.0 quotes and the TCG event log, read from the specifications.

TPMS_ATTEST (TPM 2.0 Library Part 2, big-endian): magic 0xff544347, type (0x8018 quote), qualifiedSigner TPM2B_NAME,
extraData TPM2B_DATA (the qualifying data), clockInfo {clock u64, resetCount u32, restartCount u32, safe u8},
firmwareVersion u64, TPMS_QUOTE_INFO {pcrSelect TPML_PCR_SELECTION, pcrDigest TPM2B_DIGEST}.
TPMT_SIGNATURE: sigAlg u16 (RSASSA 0x0014, RSAPSS 0x0016, ECDSA 0x0018), hash u16, then the signature.
Event log (TCG PC Client Platform Firmware Profile, little-endian): a SHA-1-format Spec ID event, then
TCG_PCR_EVENT2 {pcrIndex, eventType, TPML_DIGEST_VALUES, eventSize, event}. Replay: PCR = H(PCR || digest) for
every event but EV_NO_ACTION; a StartupLocality EV_NO_ACTION event sets PCR0's starting value to its locality.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

TPM_GENERATED = 0xFF544347
ST_ATTEST_QUOTE = 0x8018
ALG_SHA1, ALG_SHA256, ALG_SHA384, ALG_SHA512 = 0x0004, 0x000B, 0x000C, 0x000D
ALG_RSASSA, ALG_RSAPSS, ALG_ECDSA = 0x0014, 0x0016, 0x0018
DIGEST_SIZE = {ALG_SHA1: 20, ALG_SHA256: 32, ALG_SHA384: 48, ALG_SHA512: 64}
HASH = {ALG_SHA1: hashlib.sha1, ALG_SHA256: hashlib.sha256, ALG_SHA384: hashlib.sha384, ALG_SHA512: hashlib.sha512}
CRYPTO_HASH = {ALG_SHA256: hashes.SHA256, ALG_SHA384: hashes.SHA384, ALG_SHA512: hashes.SHA512}
EV_NO_ACTION = 0x00000003


class TPMError(ValueError):
    """Malformed TPM structure or a check that fails."""


class _R:
    def __init__(self, b: bytes, little: bool = False):
        self.b, self.o, self.order = b, 0, "little" if little else "big"

    def take(self, n: int) -> bytes:
        if self.o + n > len(self.b):
            raise TPMError("structure truncated")
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def u(self, n: int) -> int:
        return int.from_bytes(self.take(n), self.order)

    def tpm2b(self) -> bytes:
        return self.take(self.u(2))


@dataclass
class Quote:
    signer: bytes
    extra: bytes
    clock: int
    reset_count: int
    restart_count: int
    safe: int
    firmware: int
    selection: list          # [(hash alg, [pcr indices])]
    pcr_digest: bytes


def parse_attest(msg: bytes) -> Quote:
    r = _R(msg)
    if r.u(4) != TPM_GENERATED:
        raise TPMError("not TPM-generated (magic)")
    if r.u(2) != ST_ATTEST_QUOTE:
        raise TPMError("not a quote")
    signer, extra = r.tpm2b(), r.tpm2b()
    clock, reset, restart, safe = r.u(8), r.u(4), r.u(4), r.u(1)
    fw = r.u(8)
    sel = []
    for _ in range(r.u(4)):
        alg, size = r.u(2), r.u(1)
        bits = r.take(size)
        sel.append((alg, [i for i in range(size * 8) if bits[i // 8] >> (i % 8) & 1]))
    digest = r.tpm2b()
    if r.o != len(msg):
        raise TPMError("trailing octets after the quote")
    return Quote(signer, extra, clock, reset, restart, safe, fw, sel, digest)


def parse_signature(sig: bytes) -> tuple[int, int, bytes]:
    """(sigAlg, hash, signature octets) of an RSA TPMT_SIGNATURE."""
    r = _R(sig)
    alg, h = r.u(2), r.u(2)
    if alg not in (ALG_RSASSA, ALG_RSAPSS):
        raise TPMError(f"signature algorithm 0x{alg:04x} is not RSA")
    s = r.tpm2b()
    if r.o != len(sig):
        raise TPMError("trailing octets after the signature")
    return alg, h, s


def verify_quote(msg: bytes, sig: bytes, rsa_public_key) -> None:
    alg, h, s = parse_signature(sig)
    if h not in CRYPTO_HASH:
        raise TPMError(f"hash 0x{h:04x} not accepted")
    pad = padding.PKCS1v15() if alg == ALG_RSASSA else padding.PSS(mgf=padding.MGF1(CRYPTO_HASH[h]()),
                                                                     salt_length=padding.PSS.AUTO)
    try:
        rsa_public_key.verify(s, msg, pad, CRYPTO_HASH[h]())
    except InvalidSignature as exc:
        raise TPMError("quote signature does not verify") from exc


def pcr_digest(q: Quote, values: dict) -> bytes:
    """The digest a quote commits to: H(PCR values in selection order); ``values`` maps (alg, index) -> digest."""
    if len(q.selection) != 1:
        raise TPMError("one bank per quote expected")
    alg, idx = q.selection[0]
    return HASH[alg](b"".join(values[(alg, i)] for i in idx)).digest()


def split_values(blob: bytes, q: Quote) -> dict:
    """tpm2_quote's "values" output (concatenated digests in selection order) as (alg, index) -> digest."""
    out, off = {}, 0
    for alg, idx in q.selection:
        n = DIGEST_SIZE[alg]
        for i in idx:
            out[(alg, i)] = blob[off:off + n]
            off += n
    if off != len(blob):
        raise TPMError("PCR values do not match the selection")
    return out


@dataclass
class Event:
    pcr: int
    type: int
    digests: dict
    data: bytes


@dataclass
class EventLog:
    algs: dict                       # alg -> digest size (from the Spec ID event)
    events: list = field(default_factory=list)
    locality: int = 0


def parse_event_log(blob: bytes) -> EventLog:
    r = _R(blob, little=True)
    r.u(4)
    if r.u(4) != EV_NO_ACTION:
        raise TPMError("first event is not the Spec ID event")
    r.take(20)
    spec = _R(r.take(r.u(4)), little=True)
    if spec.take(16) != b"Spec ID Event03\x00":
        raise TPMError("not a crypto-agile log")
    spec.u(4), spec.take(3), spec.u(1)
    algs = {}
    for _ in range(spec.u(4)):
        a, n = spec.u(2), spec.u(2)
        algs[a] = n
    log = EventLog(algs=algs)
    while r.o < len(blob):
        pcr, etype = r.u(4), r.u(4)
        digests = {}
        for _ in range(r.u(4)):
            a = r.u(2)
            if a not in algs:
                raise TPMError(f"digest of algorithm 0x{a:04x} not announced")
            digests[a] = r.take(algs[a])
        data = r.take(r.u(4))
        if etype == EV_NO_ACTION and data.startswith(b"StartupLocality\x00"):
            log.locality = data[16]
        log.events.append(Event(pcr, etype, digests, data))
    return log


def replay(log: EventLog, alg: int = ALG_SHA256) -> dict:
    """PCR index -> value after replaying every measured event of the bank."""
    n = DIGEST_SIZE[alg]
    pcrs: dict = {}
    for e in log.events:
        if e.type == EV_NO_ACTION:
            continue
        start = (bytes(n - 1) + bytes([log.locality])) if e.pcr == 0 else bytes(n)
        pcrs[e.pcr] = HASH[alg](pcrs.get(e.pcr, start) + e.digests[alg]).digest()
    return pcrs


def parse_systemd_log(blob: bytes) -> list[tuple[int, bytes]]:
    """systemd's TPM2 measurement log (/run/log/systemd/tpm2-measure.log, JSON-SEQ, RFC 7464): (pcr, SHA-256 digest)
    of every measurement systemd made after the firmware handed over (systemd-pcrphase into PCR 11, and others)."""
    import json
    out = []
    for rec in blob.split(b"\x1e"):
        rec = rec.strip()
        if not rec:
            continue
        r = json.loads(rec)
        d = next((x["digest"] for x in r.get("digests", []) if x.get("hashAlg") == "sha256"), None)
        if d is None or "pcr" not in r:
            raise TPMError("systemd log record without a PCR or a SHA-256 digest")
        out.append((int(r["pcr"]), bytes.fromhex(d)))
    return out


def replay_with(log: EventLog, later: list[tuple[int, bytes]], alg: int = ALG_SHA256) -> dict:
    """The firmware log's replay, then the measurements made after boot, in order."""
    pcrs = replay(log, alg)
    n = DIGEST_SIZE[alg]
    for pcr, digest in later:
        pcrs[pcr] = HASH[alg](pcrs.get(pcr, bytes(n)) + digest).digest()
    return pcrs


def uki_sections(log: EventLog) -> dict:
    """systemd-stub's PCR 11 measurements of a unified kernel image: section name -> SHA-256 of its contents (the
    stub measures each section's name, then its contents, as EV_IPL events)."""
    out, name = {}, None
    for e in log.events:
        if e.pcr != 11 or e.type != 0x0D:
            continue
        try:
            text = e.data.decode("utf-16-le").rstrip("\x00")
        except UnicodeDecodeError:
            text = ""
        if name is None:
            name = text
        else:
            out[name] = e.digests[ALG_SHA256]
            name = None
    return out
