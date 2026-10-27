# A refusal kept as evidence: another paravisor build (27.10.2026, 17:05–17:21Z)

The authority's policy pinned the paravisor's launch measurement first observed in
`../capture-20261027T162455Z` (6587d41e…). This machine (another host, West Europe) ran another build of Azure's
paravisor: measurement 63e15b3f…, family id 01232000…, guest SVN 65549 (the capture: family 00252000…, SVN 3). Every
other check held - AMD chain and CRL, TCB floors, BadRAM and RMPocalypse fixes, the binding in the runtime claims
and in the quote, the boot chain and the kernel image's sections, the H100's report, NVIDIA's manifests and OCSP,
the model, code and parameter identities - and the authority refused the key on that one check
(`authority/issued/0001/appraisal.json`, status 96, no certificate). The service never received a certificate and
sealed nothing.

Both builds carry the same ID_KEY_DIGEST (942fd93e…): the firmware launches a guest with an ID block only if the
block's expected launch digest equals the measurement and the block's signature verifies under that ID key (AMD
56860 rev. 1.59 §8.19.2). The policy now pins the ID key, not a single measurement (`cpu.report.launch_endorsed`);
the same request, appraised again with that policy and the same collateral, affirms (tests/test_attested_run.py).
