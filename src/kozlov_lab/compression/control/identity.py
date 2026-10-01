"""``C_0(X) = X`` — the identity control.

Its purpose is to make the *pipeline* falsifiable.  Recognition preservation must be 100% for
this compressor on every exact-compatible pair, so a single failure means the experiment, the
taxonomy or the definition of "preserved" is wrong — not that compression degrades recognition.

This control earned its place immediately: running it exposed that the Phase 5 corpus used
non-integer float coordinates, which the exact recogniser refuses, and that the harness was
mislabelling that refusal as "code undefined after compression".  The whole "0 / 768 preserved"
headline was an artefact of that, and would have been published had the control not existed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model import CompressionResult, Compressor, CompressorContract

__all__ = ["IdentityCompressor"]


class IdentityCompressor(Compressor):
    """Return the image unchanged.

    ``cover_guarantee="guaranteed"`` with radius 0: the input covers itself exactly.  The
    declared radius is 0 and the achieved radius is verified to be 0 in the tests, which is
    Control D of the audit — the metadata must not be the only evidence that a no-op was a no-op.
    """

    contract = CompressorContract(
        name="IdentityCompressor",
        family="control",
        surjective_on=True,
        cover_guarantee="guaranteed",
        cardinality_guarantee="fixed",
        deterministic=True,
        complexity="O(1)",
        source="sanity control (F5.0); NOT from Kozlov 2002 and not a compression method at all",
    )

    def fit(self, points: Any, **_: Any) -> CompressionResult:
        matrix = self._matrix(points)
        indices = list(range(matrix.shape[0]))
        return self._result(
            matrix,
            indices,
            self.contract,
            requested_radius=0.0,
            achieved_radius=0.0,
            build_ms=0.0,
            peak_bytes=0,
            diagnostics={"no_op": True, "control": "identity"},
        )
