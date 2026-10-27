"""Segmentation sources for the visual lab.

Only *explicit manual* masks exist today (``photo-io-v1`` §4).  Automatic
segmentation is out of scope for T01A and deliberately absent rather than
stubbed, so nothing can accidentally treat a placeholder as a working segmenter.

The package is named in the contract's ``implementation_sha256`` definition, so
its source files participate in the cache fingerprint even though a run that
uses only manual masks imports nothing from here: a change to mask decoding must
invalidate cached results.
"""

from __future__ import annotations

__all__: list[str] = []
