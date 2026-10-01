"""Our own exploratory compressors (Phase 5G) — no author attribution.

Everything here is this project's idea.  The naming rule (G5.9) forbids ``KozlovImproved*`` and
similar: an experimental construction may not borrow the author's name to sound supported.
"""

from __future__ import annotations

from .affine_aware import AffineAwareSampler

__all__ = ["AffineAwareSampler"]
