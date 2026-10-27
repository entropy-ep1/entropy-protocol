"""Build the authority's appraisal policy for a run on Azure confidential H100s (ARCHITECTURE §7d), each reference
value with where it comes from.

    python bench/attested/make_policy.py --capture bench/results/attested/capture-20261027T162455Z \
        --cal runs_gpu/a1/Qwen_Qwen2.5-7B-Instruct/calibration_A_cal_fp16only.json \
        --tokenizer f884026e05f6dfffe68b72d580b588006d98b3bc0a128bb1ac425c76f7fc438c --out policy.json

- AMD floors for Genoa: the live AMD bulletins as read on 22.10.2026 (library attacks.md §9 correction): TCB[SNP]
  >= 0x1B and SEV mitigation vector bit 1 (AMD-SB-3020, RMPocalypse, rev. 2026-02-23), TCB[MICROCODE] >= 0x56
  (AMD-SB-3027, StackWarp), PLATFORM_INFO ALIAS_CHECK_COMPLETE (AMD-SB-3015, BadRAM).
- The paravisor's launch measurement, PCRs 4 and 7 and the kernel image's sections: first observed in the capture
  (same image, same onboarding package) - a reference taken from one observation, and named so.
- Model: the weights digest the calibration was measured for (phase A), the tokenizer digest computed on the
  authority's own machine from the pinned revision; class fp16 (the calibrated class).
- Code: the identity of this checkout's package (binding.code_identity); parameters: SHA-256 of the sealing
  parameters the service derives from the same calibration and profile (service.params_map).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from entropy_protocol.attested import binding, hcl, service, tpm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--profile", default="quality")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--id", default="ep1/azure-cgpu/qwen2.5-7b-instruct")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cap = Path(a.capture)
    h = hcl.parse((cap / "r0_hcl_report.bin").read_bytes())
    q = tpm.parse_attest((cap / "r0_quote_msg.bin").read_bytes())
    values = tpm.split_values((cap / "r0_quote_pcrs.bin").read_bytes(), q)
    uki = tpm.uki_sections(tpm.parse_event_log((cap / "event_log.bin").read_bytes()))
    cal = json.loads(Path(a.cal).read_text())
    params = service.sealing_params(a.cal, a.max_new_tokens, a.profile, cal["weights_digest"])
    observed = f"first observed in {cap.name} (Azure Standard_NCC40ads_H100_v5, Ubuntu 24.04 CVM image " \
               f"24.04.202607310, Microsoft onboarding V4.4.1)"
    policy = {
        "id": a.id,
        "lifetime": 86400,
        "amd": {"product": "Genoa", "crl": "required", "min_tcb": {"snpSPL": 0x1B, "ucodeSPL": 0x56},
                "alias_check_complete": True, "mitigation_vector_bits": [1],
                "measurements": [h.snp.measurement.hex()]},
        "vtpm": {"pcrs": {"4": [values[(tpm.ALG_SHA256, 4)].hex()], "7": [values[(tpm.ALG_SHA256, 7)].hex()]},
                 "uki": {k: [uki[k].hex()] for k in (".linux", ".initrd", ".cmdline")}},
        "gpu": {"count": 1, "rim": "required", "ocsp": "required"},
        "identities": {"weights": [cal["weights_digest"]], "tokenizer": [a.tokenizer],
                       "code": [binding.code_identity().hex()], "class": ["fp16"],
                       "params": [binding.params_digest(service.params_map(params)).hex()]},
        "provenance": {
            "amd": "AMD-SB-3020 (rev. 2026-02-23) TCB[SNP] >= 0x1B + mitigation vector bit 1; AMD-SB-3027 microcode "
                   ">= 0x56; AMD-SB-3015 ALIAS_CHECK_COMPLETE; read live 22.10.2026",
            "amd.measurements": observed, "vtpm.pcrs": observed, "vtpm.uki": observed,
            "identities.weights": f"{cal['model']} (phase A calibration, {Path(a.cal).name})",
            "identities.tokenizer": "computed by the authority from the pinned revision a09a35458c702b33eeacc393d103063234e8bc28",
            "identities.code": "binding.code_identity() of the checkout that ships to the sealer",
            "identities.params": f"service.params_map(profile {a.profile}, {Path(a.cal).name}, "
                                 f"max_new_tokens {a.max_new_tokens})"}}
    Path(a.out).write_text(json.dumps(policy, indent=1))
    print(json.dumps(policy, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
