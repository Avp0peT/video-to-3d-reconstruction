"""Photo and manual-mask IO implementing contract ``photo-io-v1``.

Scope, deliberately narrow: single-frame JPEG/PNG photos with explicitly paired
manual binary masks, decoded to canonically oriented ``uint8 RGB`` arrays plus a
canonical ``L`` mask, with per-input records and run evidence.

**Not implemented here, on purpose** — video, frame extraction, automatic
segmentation, camera or distortion estimation, pose, SfM/MVS, point clouds,
skeletons, GUI.  ``camera`` / ``distortion`` / ``pose`` / ``geometry_scale`` are
recorded as JSON ``null`` because they are *undefined at this stage*; substituting
a zero, an identity matrix, or a "no distortion" claim would silently turn an
unknown into an assertion.

This package does not implement Kozlov's mathematical definitions.  It only
produces validated inputs for them.
"""

from __future__ import annotations

from .hashing import CONTRACT_VERSION

__all__ = ["CONTRACT_VERSION"]
