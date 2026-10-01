"""Image-code layer — **not implemented yet** (Phase 3, README section 7).

Status: BLOCKED on open question **Q1** (the exact definition of Kozlov's formal
image code) and gated by the Phase 2 acceptance gate.  The permission gate for
implementing anything here is ``docs/specs/official_image_code.md``: while it is
absent or incomplete, Phase 3 stays LOCKED and this package stays empty.  The
pedagogical construction that *is* allowed lives in :mod:`kozlov_lab.teaching`.

Status
------
Blocked by the Phase 2 gate:

    "Phase 2 没有通过性质测试，不允许进入 Phase 3。"

The gate is verified by ``tests/test_gate_phase2.py``.  Once it passes, this
package implements::

    frame_code.py    AffineFrameCode  = (frame, internal coordinates of all points)
    image_code.py    ImageCode        = canonical ordering of all frame codes
    canonical.py     quantisation + canonical representation (+ SHA-256 as an *index only*)

Non-negotiable requirement carried by README section 7: a hash is **not** the
mathematical image code, only a computer index for it.  That sentence must
appear in the user documentation as well, not just in a docstring.

Also blocked: the exact encoding definition of the textbook.  README section 16
forbids inventing it — the specification must come from
``docs/specs/image_code.md`` after a Theory-Agent pass over the primary
sources (Kozlov 2009; Kozlov 2026).  Until that spec is marked
``status: verified``, only the *teaching* encoder may be implemented, and it
must say so in the UI.
"""

from __future__ import annotations
