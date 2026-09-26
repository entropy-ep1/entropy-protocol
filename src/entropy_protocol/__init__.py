"""Entropy Protocol — reference implementation (ep1).

Seal AI-generated text at the moment of generation; verify it anywhere, without revealing it.
Verdicts are determinations, not probabilities: VERIFIED, MODIFIED (with character ranges),
INDETERMINATE, or NOT_VERIFIABLE.

PRIVATE until the covering provisional application is filed. Apache-2.0 at publication.
"""

__version__ = "0.1.0.dev0"
ENGINE = f"entropy-protocol/{__version__}"
