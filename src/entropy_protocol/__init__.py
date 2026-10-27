"""Entropy Protocol — reference implementation (ep1).

Seal AI-generated text at the moment of generation; verify it anywhere, without revealing it.
Verdicts are determinations, not probabilities: VERIFIED, MODIFIED (with character ranges),
INDETERMINATE, or NOT_VERIFIABLE.

Apache-2.0; see LICENSE, NOTICE and PATENTS.md.
"""

__version__ = "0.1.0rc1"
ENGINE = f"entropy-protocol/{__version__}"
