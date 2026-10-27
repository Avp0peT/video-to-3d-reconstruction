"""Native sparse reconstruction from T01 canonical images (``sfm-native-v1``).

Scope, deliberately narrow: take a finished ``photo-io-v1`` run, hand its
canonical PNGs to the pinned pycolmap CPU backend for SIFT extraction, exhaustive
matching and incremental SfM, and preserve the resulting native models together
with a per-input identity map.

**Not here, on purpose** — object-point selection from masks (that is T02B),
undistortion, coordinate conversion into the older camera type (T03), dense
reconstruction, meshing, texturing, Open3D, neural methods, video.  Masks are
carried through the identity map only, so this round produces a scene model, not
an object point cloud.

Two properties this package will not trade away:

* the native backend runs in a **subprocess**, so a native abort costs one stage's
  log instead of the whole run's evidence, and every stage has a real exit code
  rather than an invented one;
* the model is written in **both** binary and text form and each is read back
  separately, because a reader that prefers the binary would hide an error that
  only the text form has.
"""

from __future__ import annotations

from .colmap_adapter import (
    CONTRACT_VERSION,
    COORDINATE_CONVENTION,
    BackendError,
    BackendUnavailable,
    InputError,
)

__all__ = [
    "CONTRACT_VERSION",
    "COORDINATE_CONVENTION",
    "BackendError",
    "BackendUnavailable",
    "InputError",
]
