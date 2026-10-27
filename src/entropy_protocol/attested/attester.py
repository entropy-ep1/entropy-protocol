"""The sealer's side of an enrollment, inside the confidential VM (ARCHITECTURE §7d).

A fresh Ed25519 statement key is generated in this process and never written anywhere; its CSR, the model and code
identities and the authority's handle go into the binding value; the binding goes into the paravisor's report (vTPM
guest input), SHA-256 of it into the quote's qualifying data, and the handle into every GPU's report. The authority
named in the binding is the one this sealer is configured with: an initiation naming another is refused.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .. import cose
from . import enroll, pki
from .binding import BindingInput, code_identity, params_digest


@dataclass
class Enrollment:
    key: Ed25519PrivateKey
    request: bytes
    binding: BindingInput


def evidence_azure_cgpu(binding_value: bytes, handle: bytes, claims: dict) -> bytes:
    from . import azure_cgpu as az
    ev = {"hcl": az.guest_report(binding_value), "quote": az.quote(hashlib.sha256(binding_value).digest()),
          "thim": az.thim(), "eventlog": az.event_log(), "gpus": az.gpus(handle), "claims": claims}
    ima = az.ima_log()
    if ima:
        ev["ima"] = ima
    sd = az.systemd_tpm_log()
    if sd:
        ev["systemd_tpm_log"] = sd
    akc = az.ak_cert()
    if akc:
        ev["akcert"] = akc
    return cose.dumps(ev)


def prepare(*, initiation: bytes, rrp_id: str, target: str, weights: bytes, tokenizer: bytes, sealing_class: str,
            params: dict, credential_type: str = enroll.CRED_STATEMENT, collect=evidence_azure_cgpu) -> Enrollment:
    ini = enroll.read_initiation(initiation)
    if ini["rrp_id"] != rrp_id:
        raise enroll.EnrollError(f"initiation names authority {ini['rrp_id']!r}, this sealer is configured for "
                                 f"{rrp_id!r}")
    key = Ed25519PrivateKey.generate()
    csr = pki.make_csr(key, target)
    bi = BindingInput(handle=ini["handle"], rrp_id=rrp_id, target=target, csr_digest=hashlib.sha256(csr).digest(),
                      statement_key=pki.spki(key.public_key()), weights=weights, tokenizer=tokenizer,
                      code=code_identity(), sealing_class=sealing_class, params=params_digest(params))
    claims = {"weights": weights, "tokenizer": tokenizer, "code": bi.code, "class": sealing_class,
              "params": bi.params, "params_map": params}
    ev = collect(bi.value(), ini["handle"], claims)
    req = enroll.request(target=target, credential_type=credential_type, handle=ini["handle"], evidence=ev, csr=csr)
    return Enrollment(key=key, request=req, binding=bi)
