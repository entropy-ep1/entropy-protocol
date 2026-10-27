"""Attested sealing and attested detection (ARCHITECTURE §7d; spec [0100]-[0109]).

The keys, the model and the sealing code run in a confidential VM with a confidential GPU; a credential authority
appraises the hardware evidence, which carries a binding value over the statement key, the model and the code, and
only then certifies the statement key. Attester side: `azure_cgpu` (evidence from inside the guest), `binding`.
"""
