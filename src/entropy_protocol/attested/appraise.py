"""Appraisal of an `ep1/azure-cgpu/v1` enrollment (ARCHITECTURE §7d): every check named, the EAR built from them.

The same function serves the authority at enrollment and the offline check later; what differs is where the
collateral comes from (fetched live, or read back from the evidence package) and the time of the check (now, or the
result's issue time). Nothing here reaches the network: `collateral` brings RIMs and OCSP responses in.

Policy (JSON; reference values with their provenance):
  {"id", "lifetime", "amd": {"product", "min_tcb": {...}, "measurements": [hex]},
   "vtpm": {"pcrs": {"<index>": [hex]}},
   "gpu": {"count", "rim": "required" | "off", "ocsp": "required" | "off"},
   "identities": {"weights": [hex], "tokenizer": [hex], "code": [hex], "class": [...], "params": [hex]}}
An empty allow-list accepts any value and says so in the result (the check is reported as not performed).
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass, field

from cryptography import x509

from .. import cose
from . import amd, ear, hcl, nvidia, ocsp, pki, rim, tpm
from .binding import BindingInput


@dataclass
class Check:
    name: str
    ok: bool | None          # None = not performed (and why, in detail)
    detail: str = ""


@dataclass
class Appraisal:
    checks: list = field(default_factory=list)
    submods: dict = field(default_factory=dict)
    identities: dict = field(default_factory=dict)
    statement_key: object = None
    target: str = ""

    @property
    def ok(self) -> bool:
        return all(c.ok is not False for c in self.checks) and bool(self.checks)

    def failed(self) -> list:
        return [c for c in self.checks if c.ok is False]


def _thim_chain(thim: dict) -> tuple[bytes, bytes, bytes]:
    from cryptography.hazmat.primitives import serialization
    vcek = x509.load_pem_x509_certificate(thim["vcekCert"].encode())
    chain = x509.load_pem_x509_certificates(thim["certificateChain"].encode())
    enc = serialization.Encoding.DER
    ask = next(c for c in chain if c.issuer != c.subject)
    ark = next(c for c in chain if c.issuer == c.subject)
    return vcek.public_bytes(enc), ask.public_bytes(enc), ark.public_bytes(enc)


def appraise(req: dict, *, rrp_id: str, policy: dict, at: dt.datetime, collateral: dict) -> Appraisal:
    """Every check, named. The evidence is untrusted: a shape no check anticipated fails closed, as "evidence.format",
    and never raises (an authority must answer every request)."""
    try:
        return _appraise(req, rrp_id=rrp_id, policy=policy, at=at, collateral=collateral)
    except Exception as exc:  # noqa: BLE001
        a = Appraisal(target=str(req.get("target", "")) if isinstance(req, dict) else "")
        a.checks.append(Check("evidence.format", False, f"{type(exc).__name__}: {exc}"))
        return a


def _appraise(req: dict, *, rrp_id: str, policy: dict, at: dt.datetime, collateral: dict) -> Appraisal:
    """``req`` is a read enrollment request (enroll.read_request); ``collateral`` = {"rims": {id: xml},
    "ocsp": [{"serial": hex, "nonce": bytes, "response": bytes, ...}], "amd_crl": der}."""
    a = Appraisal(target=req["target"])
    ev = cose._thaw(cose.loads(req["evidence"]))
    add = lambda name, ok, detail="": a.checks.append(Check(name, ok, detail))  # noqa: E731
    pol_amd, pol_vtpm, pol_gpu = policy.get("amd", {}), policy.get("vtpm", {}), policy.get("gpu", {})
    claims = ev.get("claims", {})

    # ---- binding: the request's own fields recomputed, the key that signed the CSR
    try:
        key, cn = pki.read_csr(req["csr"])
        add("binding.csr_proof_of_possession", True)
        a.statement_key = key
    except pki.PKIError as exc:
        add("binding.csr_proof_of_possession", False, str(exc))
        key, cn = None, ""
    add("binding.csr_subject_is_target", cn == req["target"], f"CSR names {cn!r}")
    bi = None
    try:
        bi = BindingInput(handle=req["handle"], rrp_id=rrp_id, target=req["target"],
                          csr_digest=hashlib.sha256(req["csr"]).digest(),
                          statement_key=pki.spki(key) if key else b"", weights=claims["weights"],
                          tokenizer=claims["tokenizer"], code=claims["code"], sealing_class=claims["class"],
                          params=claims["params"])
        expected = bi.value()
    except (KeyError, ValueError) as exc:
        add("binding.fields", False, str(exc))
        expected = b""
    a.identities = {"weights": claims.get("weights", b""), "tokenizer": claims.get("tokenizer", b""),
                    "code": claims.get("code", b""), "class": claims.get("class", ""),
                    "params": claims.get("params", b""), "binding": expected,
                    "key": hashlib.sha256(pki.spki(key)).digest() if key else b""}
    if "params_map" in claims:
        from .binding import params_digest
        add("binding.params_map_digest", params_digest(claims["params_map"]) == claims["params"])
    for name, allowed in policy.get("identities", {}).items():
        value = claims.get(name)
        shown = value.hex() if isinstance(value, bytes) else str(value)
        if not allowed:
            add(f"identity.{name}", None, f"no reference values; observed {shown}")
        else:
            add(f"identity.{name}", shown in allowed, f"observed {shown}")

    # ---- cpu: the paravisor's report under AMD's chain
    cpu_ok = {"hardware": False, "config": False, "identity": False}
    h = None
    try:
        h = hcl.parse(ev["hcl"])
        add("cpu.hcl.parse", True, f"report type {h.report_type}, hash type {h.hash_type}")
    except (hcl.HCLError, amd.SNPError, KeyError) as exc:
        add("cpu.hcl.parse", False, str(exc))
    if h is not None:
        product = pol_amd.get("product", "Genoa")
        try:
            vcek_der, ask_der, ark_der = _thim_chain(ev["thim"])
            ext = amd.verify_chain(vcek_der, ask_der, ark_der, product, at)
            add("cpu.chain.to_pinned_ark", True, f"ARK-{product}; {ext.get('productName', '?')}")
            if pol_amd.get("crl", "required") != "off":
                crl = collateral.get("amd_crl")
                try:
                    if crl is None:
                        raise amd.SNPError("no CRL")
                    amd.check_crl(crl, ask_der, ark_der, at)
                    add("cpu.chain.ask_not_revoked", True, "AMD CRL")
                except (amd.SNPError, ValueError) as exc:
                    add("cpu.chain.ask_not_revoked", False, str(exc))
            amd.verify_report(h.snp, vcek_der)
            add("cpu.report.signature", True)
            parts = amd.tcb_parts(h.snp.reported_tcb, product)
            same = all(ext.get(k) == v for k, v in parts.items())
            add("cpu.report.tcb_matches_vcek", same, f"reported {parts}")
            minimum = pol_amd.get("min_tcb", {})
            add("cpu.report.tcb_at_least_minimum", amd.tcb_at_least(parts, minimum) if minimum else None,
                f"minimum {minimum}" if minimum else "no minimum in the policy")
            cpu_ok["hardware"] = same and (not minimum or amd.tcb_at_least(parts, minimum))
        except (amd.SNPError, KeyError, StopIteration, ValueError) as exc:
            add("cpu.chain.to_pinned_ark", False, str(exc))
        add("cpu.report.debug_off", not h.snp.debug, f"policy 0x{h.snp.policy:x}")
        if pol_amd.get("alias_check_complete", True):
            add("cpu.report.alias_check_complete", h.snp.alias_check_complete,
                f"platform info 0x{h.snp.platform_info:x}")
        bits = pol_amd.get("mitigation_vector_bits", [])
        if bits:
            add("cpu.report.mitigation_vector", all(h.snp.current_mit_vector >> b & 1 for b in bits),
                f"current 0x{h.snp.current_mit_vector:x}, required bits {bits}")
        refs, id_keys = pol_amd.get("measurements", []), pol_amd.get("id_key_digests", [])
        m, idk = h.snp.measurement.hex(), h.snp.id_key_digest.hex()
        by_measurement, by_key = m in refs, h.snp.id_block and idk in id_keys
        endorsed = (by_measurement or by_key) if (refs or id_keys) else None
        add("cpu.report.launch_endorsed", endorsed,
            ("measurement in the reference set" if by_measurement else
             f"ID block signed by pinned ID key {idk[:16]}… (family {h.snp.family_id.hex()[:8]}, image "
             f"{h.snp.image_id.hex()[:8]}, guest SVN {h.snp.guest_svn})" if by_key else
             f"measurement {m}, ID key {idk if h.snp.id_block else 'none'}: neither in the policy")
            + ("" if (refs or id_keys) else "; no reference values"))
        cpu_ok["config"] = not h.snp.debug and (endorsed is not False) and \
            (h.snp.alias_check_complete or not pol_amd.get("alias_check_complete", True)) and \
            all(h.snp.current_mit_vector >> b & 1 for b in bits)
        rd = hcl.report_data_binds_claims(h)
        add("cpu.report.report_data_is_hash_of_runtime_claims", rd)
        cpu_ok["identity"] = rd
    a.submods["cpu"] = ear.appraisal(
        {ear.HARDWARE: 2 if cpu_ok["hardware"] else 96, ear.CONFIGURATION: 2 if cpu_ok["config"] else 96,
         ear.RUNTIME_OPAQUE: 2 if cpu_ok["hardware"] else 96, ear.INSTANCE_IDENTITY: 2 if cpu_ok["identity"] else 96},
        [policy.get("id", "")],
        attester={"measurement": h.snp.measurement, "reported_tcb": h.snp.reported_tcb, "policy": h.snp.policy,
                  "chip_id": h.snp.chip_id, "vmpl": h.snp.vmpl, "id_key_digest": h.snp.id_key_digest,
                  "family_id": h.snp.family_id, "image_id": h.snp.image_id, "guest_svn": h.snp.guest_svn}
        if h else None)

    # ---- vtpm: user data, the attestation key, the quote and the boot chain
    vt: dict = {"identity": False, "exec": False, "config": False}
    if h is not None:
        ud_ok = bool(expected) and h.user_data == expected
        add("vtpm.runtime_claims.user_data_is_binding", ud_ok)
        vmc = h.claims.get("vm-configuration", {})
        add("vtpm.runtime_claims.secure_boot", vmc.get("secure-boot") is True)
        add("vtpm.runtime_claims.tpm_enabled", vmc.get("tpm-enabled") is True)
        vt["config"] = vmc.get("secure-boot") is True and vmc.get("tpm-enabled") is True
        jwk = h.jwk("HCLAkPub")
        try:
            akpub = hcl.rsa_from_jwk(jwk or {})
            q = ev["quote"]
            tpm.verify_quote(q["msg"], q["sig"], akpub)
            add("vtpm.quote.signature_by_bound_ak", True)
            quote = tpm.parse_attest(q["msg"])
            qd = bool(expected) and quote.extra == hashlib.sha256(expected).digest()
            add("vtpm.quote.qualifying_data_is_binding_digest", qd)
            values = tpm.split_values(q["pcrs"], quote)
            add("vtpm.quote.pcr_digest", tpm.pcr_digest(quote, values) == quote.pcr_digest)
            log = tpm.parse_event_log(ev["eventlog"])
            later = tpm.parse_systemd_log(ev["systemd_tpm_log"]) if ev.get("systemd_tpm_log") else []
            replayed = tpm.replay_with(log, later)
            # PCR 11 is extended again after boot (systemd-pcrphase): without systemd's log it cannot be replayed
            uncovered = {11} if not later and any(e.pcr == 11 for e in log.events) else set()
            judged = {i: v for i, v in replayed.items() if i not in uncovered}
            mism = [i for i, v in judged.items() if values.get((tpm.ALG_SHA256, i)) not in (None, v)]
            add("vtpm.eventlog.replays_to_quoted_pcrs", not mism, f"PCRs replayed {sorted(judged)}; mismatched "
                f"{mism}" if mism else f"PCRs replayed {sorted(judged)}"
                + (f"; not covered without systemd's log: {sorted(uncovered)}" if uncovered else ""))
            uki = tpm.uki_sections(log)
            ref_uki = pol_vtpm.get("uki", {})
            bad_uki = [k for k, allowed in ref_uki.items() if uki.get(k, b"").hex() not in allowed]
            if ref_uki:
                add("vtpm.uki_sections.in_reference_set", not bad_uki,
                    f"outside the reference set: {bad_uki}" if bad_uki else ", ".join(sorted(ref_uki)))
            vt["claims"] = {"pcr4": values.get((tpm.ALG_SHA256, 4), b""), "pcr7": values.get((tpm.ALG_SHA256, 7), b""),
                            "uki": uki}
            refs = pol_vtpm.get("pcrs", {})
            bad = [i for i, allowed in refs.items() if values.get((tpm.ALG_SHA256, int(i)), b"").hex() not in allowed]
            add("vtpm.pcrs.in_reference_set", (not bad) if refs else None,
                f"outside the reference set: {bad}" if bad else ("" if refs else "no reference values"))
            vt["identity"] = ud_ok and qd
            vt["exec"] = (not mism) and (not bad if refs else True) and not bad_uki
        except (hcl.HCLError, tpm.TPMError, KeyError) as exc:
            add("vtpm.quote.signature_by_bound_ak", False, str(exc))
    a.submods["vtpm"] = ear.appraisal(
        {ear.INSTANCE_IDENTITY: 2 if vt["identity"] else 96, ear.EXECUTABLES: 3 if vt["exec"] else 33,
         ear.CONFIGURATION: 2 if vt["config"] else 32}, [policy.get("id", "")], attester=vt.get("claims"))

    # ---- gpus: chain, signature over the handle, manifests, revocation
    gpus = ev.get("gpus", [])
    want = pol_gpu.get("count")
    add("gpu.count", (len(gpus) == want) if want else bool(gpus), f"{len(gpus)} GPU(s)")
    for g in gpus:
        label = f"gpu{g.get('index', 0)}"
        gk = {"hardware": False, "exec": False, "config": False}
        try:
            chain = nvidia.load_chain(g["chain"])
            nvidia.verify_chain(chain, nvidia.DEVICE_ROOT_SHA256, at)
            add(f"{label}.chain.to_pinned_root", True, chain[0].subject.rfc4514_string())
            rep = nvidia.parse(g["report"])
            nvidia.verify_report(rep, chain[0])
            add(f"{label}.report.signature", True)
            add(f"{label}.report.nonce_is_handle", rep.nonce == req["handle"])
            add(f"{label}.report.fwid_matches_certificate",
                rep.opaque.get("fwid", b"") == nvidia.certificate_fwid(chain[0]))
            gk["hardware"] = rep.nonce == req["handle"]
            drv = nvidia.text(rep.opaque.get("driver_version", b""))
            vb = nvidia.vbios_version(rep.opaque.get("vbios_version", b""))
            add(f"{label}.report.versions_match_nvml", drv == g.get("driver") and vb.lower() == g.get("vbios", "").lower(),
                f"report {drv} / {vb}; NVML {g.get('driver')} / {g.get('vbios')}")
            cc = g.get("cc", {})
            gk["config"] = cc.get("feature") == 1 and cc.get("devtools") == 0
            add(f"{label}.cc_mode_on_devtools_off", gk["config"], f"NVML {cc}")
            revocation = pol_gpu.get("ocsp", "required") != "off"
            if pol_gpu.get("rim", "required") != "off":
                gk["exec"] = _rims(add, label, rep, drv, vb, collateral, at, revocation)
            else:
                add(f"{label}.rim", None, "manifests not checked by policy")
            if revocation:
                _ocsp_chain(add, label, chain, chain[1:-1], collateral, at, nvidia.DEVICE_ROOT_SHA256)
            else:
                add(f"{label}.ocsp", None, "revocation not checked by policy")
        except (nvidia.GPUError, KeyError, ValueError) as exc:
            add(f"{label}.chain.to_pinned_root", False, str(exc))
        a.submods[label] = ear.appraisal(
            {ear.HARDWARE: 2 if gk["hardware"] else 96, ear.EXECUTABLES: 2 if gk["exec"] else 33,
             ear.CONFIGURATION: 2 if gk["config"] else 96, ear.RUNTIME_OPAQUE: 2 if gk["config"] else 96},
            [policy.get("id", "")], nonce=req["handle"])
    return a


def rim_ids(rep: nvidia.GPUReport) -> dict:
    """The driver and VBIOS manifest ids for a report, as NVIDIA's verifier composes them from its opaque data."""
    up = lambda k: nvidia.text(rep.opaque.get(k, b"")).upper()  # noqa: E731
    vb = nvidia.vbios_version(rep.opaque.get("vbios_version", b""))
    return {"driver": rim.driver_id(nvidia.text(rep.opaque.get("driver_version", b""))),
            "vbios": rim.vbios_id(up("project"), up("project_sku"), up("chip_sku"), vb.replace(".", "").upper())}


def _rims(add, label, rep, drv, vb, collateral, at, revocation: bool) -> bool:
    ids = rim_ids(rep)
    gold = {}
    for kind, rid in ids.items():
        xml = collateral.get("rims", {}).get(rid)
        if xml is None:
            add(f"{label}.rim.{kind}", False, f"{rid} not available")
            return False
        try:
            gold[kind] = rim.golden(rim.verify(xml, at))
            add(f"{label}.rim.{kind}.signature", True, rid)
        except (rim.RIMError, KeyError) as exc:
            add(f"{label}.rim.{kind}.signature", False, f"{rid}: {exc}")
            return False
        if revocation:
            from lxml import etree
            rc = rim._chain(etree.fromstring(xml, etree.XMLParser(resolve_entities=False, no_network=True)))
            _ocsp_chain(add, f"{label}.rim.{kind}", rc, rc[:-1], collateral, at, nvidia.RIM_ROOT_SHA256)
    nvdec = rep.opaque.get("nvdec0_status", b"")
    bad = rim.compare(rep.measurements, gold["driver"], gold["vbios"], nvdec[0] if nvdec else None)
    add(f"{label}.rim.measurements_match", not bad, f"mismatched {bad}" if bad else
        f"{len(rep.measurements)} measurements")
    return not bad


def _ocsp_chain(add, label, chain, asked, collateral, at, root_sha256) -> None:
    """Revocation of ``asked`` (members of ``chain``, leaf first) from the kept responses. For a GPU chain NVIDIA's
    verifier asks from the second certificate to the one below the root (the leaf is the per-boot attestation key);
    for a RIM chain, every certificate below the root."""
    responses = collateral.get("ocsp", [])
    for cert in asked:
        issuer = chain[chain.index(cert) + 1]
        sn = format(cert.serial_number, "x")
        name = f"{label}.ocsp.{cert.subject.rfc4514_string().split(',')[0].split('=')[-1][:32]}"
        last = "no response"
        for r in (x for x in responses if x["serial"] == sn):
            try:
                add(name, ocsp.check(r["response"], cert, issuer, r["nonce"], root_sha256, at, chain) == "good", "good")
                break
            except nvidia.GPUError as exc:
                last = str(exc)
        else:
            add(name, False, last)
