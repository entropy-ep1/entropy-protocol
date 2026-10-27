"""NVIDIA reference integrity manifests (RIMs): fetch, signature, golden measurements, comparison.

Semantics follow NVIDIA's own local GPU verifier (nvtrust tag 2026.06.04.001, `rim/`, `verifier.py`,
`cc_admin_utils.py`), re-implemented so the appraisal does not depend on NVIDIA's code or service at check time:
- ids: driver `NV_GPU_DRIVER_GH100_<driver version>`, VBIOS `NV_GPU_VBIOS_<project>_<project sku>_<chip sku>_<vbios
  version>`; served by https://rim.attestation.nvidia.com/v1/rim/<id> as JSON {"rim": base64 of the SWID XML};
- a RIM is an ISO/IEC 19770-2 SWID tag with an XML signature (verified with signxml against NVIDIA's pinned RIM
  root); its Payload holds one Resource per measurement index {index, active, alternatives, size, name,
  Hash0..Hash<n-1> in the namespace http://www.w3.org/2001/04/xmlenc#sha384};
- golden = the active measurements of both RIMs (an index active in both is an error); every golden index i must have
  a runtime measurement equal to one of its alternatives, of the stated size - RIM indices count from 0, SPDM
  measurement blocks from 1, so golden i is block i + 1 (nvtrust `get_measurements`: list[index - 1]); index 35 is
  skipped when the report's opaque data says NVDEC0 is disabled (0x55).
"""
from __future__ import annotations

import base64
import json
import urllib.request

RIM_URL = "https://rim.attestation.nvidia.com/v1/rim/"
HASH_NS = "{http://www.w3.org/2001/04/xmlenc#sha384}"
NVDEC_DISABLED = 0x55


class RIMError(ValueError):
    """A RIM that cannot be read, is not signed by NVIDIA, or does not match."""


def driver_id(driver_version: str) -> str:
    return "NV_GPU_DRIVER_GH100_" + driver_version


def vbios_id(project: str, project_sku: str, chip_sku: str, vbios_version: str) -> str:
    return f"NV_GPU_VBIOS_{project}_{project_sku}_{chip_sku}_{vbios_version}"


def fetch(rim_id: str, timeout: int = 20) -> bytes:
    """The RIM's XML as NVIDIA's service returns it (decoded from its JSON envelope)."""
    with urllib.request.urlopen(RIM_URL + rim_id, timeout=timeout) as r:  # noqa: S310 - fixed https URL
        return base64.b64decode(json.loads(r.read())["rim"])


def _chain(tree) -> list:
    """The certificates the signature carries (ds:X509Certificate), ordered leaf first by issuer name."""
    from cryptography import x509
    certs = [x509.load_der_x509_certificate(base64.b64decode("".join(e.text.split())))
             for e in tree.iter("{http://www.w3.org/2000/09/xmldsig#}X509Certificate")]
    if not certs:
        raise RIMError("RIM signature carries no certificate")
    issuers = {c.issuer for c in certs if c.issuer != c.subject}
    leaf = [c for c in certs if c.subject not in issuers]
    if len(leaf) != 1:
        raise RIMError("cannot tell the RIM signer among its certificates")
    chain = leaf
    while chain[-1].issuer != chain[-1].subject:
        up = next((c for c in certs if c.subject == chain[-1].issuer), None)
        if up is None:
            raise RIMError("RIM certificate chain is incomplete")
        chain.append(up)
    return chain


def verify(xml: bytes, at) -> object:
    """The signed XML subtree, if the signature verifies under the signer's certificate and that certificate chains
    to NVIDIA's pinned RIM root; RIMError otherwise.

    The chain is checked here, signature by signature, not by signxml: signxml 5.2 validates paths under RFC 5280
    and refuses NVIDIA's RIM chain (a validity date before 2050 encoded as GeneralizedTime; checked 27.10.2026 on
    NV_GPU_DRIVER_GH100_595.91.07), while NVIDIA's own verifier pins signxml 3.2.0, which does not. The signature
    itself (ecdsa-sha384, C14N 1.1) is signxml's, under the signer certificate we validated."""
    from cryptography.hazmat.primitives import serialization
    from lxml import etree
    from signxml import XMLVerifier
    from cryptography.exceptions import InvalidSignature
    from signxml.exceptions import SignXMLException

    from .nvidia import GPUError, RIM_ROOT_SHA256, verify_chain
    parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False)
    try:
        tree = etree.fromstring(xml, parser)
        chain = _chain(tree)
        verify_chain(chain, RIM_ROOT_SHA256, at)
        leaf_pem = chain[0].public_bytes(serialization.Encoding.PEM).decode()
        return XMLVerifier().verify(xml, x509_cert=leaf_pem, parser=parser).signed_xml
    except (SignXMLException, InvalidSignature, etree.LxmlError, GPUError, ValueError) as exc:
        raise RIMError(f"RIM signature does not verify: {exc}") from exc     # every signxml and lxml error included


def golden(tree) -> dict:
    """index -> {"active", "size", "values": [hex], "name"} from the Payload."""
    payload = next((el for el in tree.iter() if isinstance(el.tag, str) and el.tag.endswith("Payload")), None)
    if payload is None:
        raise RIMError("RIM has no Payload")
    out: dict = {}
    for res in payload:
        a = res.attrib
        try:
            idx, size = int(a["index"]), int(a["size"])
            values = [a[f"{HASH_NS}Hash{i}"].lower() for i in range(int(a["alternatives"]))]
        except (KeyError, ValueError) as exc:
            raise RIMError(f"malformed measurement {a.get('name', '?')!r} ({type(exc).__name__}: {exc})") from exc
        if idx in out:
            raise RIMError(f"index {idx} appears twice")
        out[idx] = {"active": a.get("active") != "False", "size": size, "name": a.get("name", ""), "values": values}
    if not out:
        raise RIMError("RIM has no measurements")
    return out


def compare(runtime: dict, driver: dict, vbios: dict, nvdec_status: int | None) -> list[tuple[int, str]]:
    """Mismatched (RIM index, source) pairs; empty means the GPU is in a state NVIDIA's manifests describe.
    ``runtime`` maps SPDM block index (from 1) to the measurement."""
    gold = {i: (m, "driver") for i, m in driver.items() if m["active"]}
    for i, m in vbios.items():
        if m["active"]:
            if i in gold:
                raise RIMError(f"driver and VBIOS RIMs both define index {i}")
            gold[i] = (m, "vbios")
    if len(gold) > len(runtime):
        return [(-1, "more golden measurements than runtime ones")]
    bad = []
    for i, (m, src) in sorted(gold.items()):
        if i == 35 and nvdec_status == NVDEC_DISABLED:
            continue
        v = runtime.get(i + 1, b"")
        if not (len(v) == m["size"] and v.hex() in m["values"]):
            bad.append((i, src))
    return bad
