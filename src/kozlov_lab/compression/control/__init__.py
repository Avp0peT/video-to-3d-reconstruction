"""Control compressors — the F5.0 sanity family.

A control is not a method: it exists to prove that the *harness* is sound.  If the identity
control does not preserve recognition at 100%, then nothing measured about real compression can
be believed, and the correct response is to stop rather than to report a degradation.
"""

from __future__ import annotations

from .identity import IdentityCompressor

__all__ = ["IdentityCompressor"]
