"""EAT Attestation Result (draft-ietf-rats-ear-04) as a CWT: COSE_Sign1 over the CBOR claims-set.

Labels (draft-ietf-rats-ear-04 §3, RFC 9711): eat_profile 265, iat 6, exp 4, eat_nonce 10, submods 266,
ear_status 1000, ear_trustworthiness_vector 1001, ear_raw_evidence 1002, ear_appraisal_policy_ids 1003,
ear_verifier_id 1004 {developer 0, build 1}, ear_attester_claims 1005, ear_verifier_claims 1006,
ear_device_topology 1007. Tiers (AR4SI): none 0, affirming 2, warning 32, contraindicated 96. Trustworthiness vector:
instance-identity 0, configuration 1, executables 2, file-system 3, hardware 4, runtime-opaque 5, storage-opaque 6,
sourced-data 7. ep1's identities travel in a private-space claim (§4.1: integers below -65536): EP1 = -71001.
Signed with Ed25519 by the authority's root key itself (pki: never by an end-entity key, which a sealer also holds);
the protected header carries the root certificate (x5chain, 33) for convenience, and the reader compares it with
the root it pins.
"""
from __future__ import annotations

import time

import cbor2

from .. import cose

PROFILE = "tag:ietf.org,2026:rats/ear#04"
EAT_PROFILE, IAT, EXP, NONCE, SUBMODS = 265, 6, 4, 10, 266
STATUS, TVECTOR, RAW_EVIDENCE, POLICY_IDS, VERIFIER_ID, ATTESTER_CLAIMS, VERIFIER_CLAIMS, TOPOLOGY = range(1000, 1008)
NONE, AFFIRMING, WARNING, CONTRAINDICATED = 0, 2, 32, 96
INSTANCE_IDENTITY, CONFIGURATION, EXECUTABLES, FILE_SYSTEM, HARDWARE, RUNTIME_OPAQUE, STORAGE_OPAQUE, SOURCED_DATA = \
    range(8)
EP1 = -71001
X5CHAIN = 33
TYP = "application/eat+cwt"


def tier_of(value: int) -> int:
    """The tier of a trustworthiness claim value (AR4SI §2.3: -1..1 none; 2..31 and -2..-32 affirming; 32..95 and
    -33..-96 warning; 96..127 and -97..-128 contraindicated)."""
    if value in (-1, 0, 1):
        return NONE
    v = value if value > 0 else -value - 1       # -2..-32 -> 1..31, -33..-96 -> 32..95, -97..-128 -> 96..127
    if v >= 96:
        return CONTRAINDICATED
    if v >= 32:
        return WARNING
    return AFFIRMING


def worst(tiers) -> int:
    order = {AFFIRMING: 0, NONE: 1, WARNING: 2, CONTRAINDICATED: 3}
    return max(tiers, key=lambda t: order[t], default=NONE)


def appraisal(vector: dict, policy_ids: list[str], attester: dict | None = None, verifier: dict | None = None,
              nonce: bytes | None = None) -> dict:
    """One submod: status = the worst tier across the vector."""
    a = {STATUS: worst(tier_of(v) for v in vector.values()), TVECTOR: vector, POLICY_IDS: policy_ids}
    if attester:
        a[ATTESTER_CLAIMS] = attester
    if verifier:
        a[VERIFIER_CLAIMS] = verifier
    if nonce is not None:
        a[NONCE] = nonce
    return a


def claims(submods: dict, verifier_id: tuple[str, str], *, nonce: bytes, lifetime: int, topology: dict | None = None,
           ep1: dict | None = None, iat: int | None = None) -> dict:
    now = int(time.time()) if iat is None else int(iat)
    c = {EAT_PROFILE: PROFILE, IAT: now, EXP: now + int(lifetime), VERIFIER_ID: {0: verifier_id[0], 1: verifier_id[1]},
         NONCE: nonce, SUBMODS: submods, STATUS: worst(s[STATUS] for s in submods.values())}
    if topology:
        c[TOPOLOGY] = topology
    if ep1:
        c[EP1] = ep1
    return c


def sign(c: dict, key, chain_der: list[bytes]) -> bytes:
    protected = {cose.ALG: cose.ED25519, cose.TYP: TYP, X5CHAIN: chain_der[0] if len(chain_der) == 1 else chain_der}
    return cose.sign1(protected, cose.dumps(c), key)


def read(ear: bytes, public_key) -> dict:
    """Verify the signature under ``public_key`` (the result-signer's, already chained to a root) and return the claims."""
    prot, _, payload = cose.verify1(ear, public_key)
    if prot.get(cose.TYP) != TYP:
        raise cose.COSEError("not an attestation result")
    c = cose._thaw(cbor2.loads(payload))
    if c.get(EAT_PROFILE) != PROFILE:
        raise cose.COSEError("not an EAR of this profile")
    return c


def chain_of(ear: bytes) -> list[bytes]:
    """The x5chain of a signed result, leaf first (not verified here)."""
    prot = cose.decode1(ear)[0]
    v = prot.get(X5CHAIN)
    return [v] if isinstance(v, bytes) else list(v or [])
