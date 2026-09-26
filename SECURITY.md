# Security

Please report a vulnerability privately through GitHub's **Report a vulnerability** button on this repository's
Security tab. Do not open a public issue for it.

In scope, in order of what we most want to hear about: a way to make a changed text verify as `VERIFIED`; a way to
make an unchanged text verify as `MODIFIED`; a way to learn anything about a sealed text from its record or its
generation statement; a way to open a record, or derive a key, without the key it is encrypted under; a false
positive of record-free detection at a rate above the stated exact null; a flaw in the formats of `spec/ep1.cddl` or
in the offline check of an evidence package.

Out of scope, because the documentation already says so: that a holder of the detection key can imitate the mark;
that paraphrase or translation removes the record-free signal; that an issuer holding the master key can seal a text
its model did not write (attested generation is not part of this release); that we do not validate an RFC 3161
authority's signature or certificate chain.
