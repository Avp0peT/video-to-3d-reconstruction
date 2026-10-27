"""CPU-only mask review metadata; agreement is not segmentation accuracy.

Arrays accept bool or integer binary labels {0,1} / {0,255}; floats and
mixed {0,1,255} encodings are rejected. No mask is written or combined.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PropagationPolicy:
    min_iou: float = 0.8
    max_area_ratio: float = 1.5

    def __post_init__(self) -> None:
        if not isfinite(self.min_iou) or not 0 <= self.min_iou <= 1:
            raise ValueError("min_iou must be finite in [0, 1]")
        if not isfinite(self.max_area_ratio) or self.max_area_ratio < 1:
            raise ValueError("max_area_ratio must be finite and >= 1")


def _binary(mask: Any) -> Any:
    import numpy as np

    array = np.asarray(mask)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError("mask must be a nonzero-size 2D array")
    if array.dtype.kind not in "bui":
        raise ValueError("mask must have bool or binary integer labels")
    labels = set(np.unique(array).tolist())
    if not (labels <= {0, 1} or labels <= {0, 255}):
        raise ValueError("mask labels must be {0,1} or {0,255}")
    return array != 0


def _boundary(mask: Any) -> Any:
    import numpy as np

    padded = np.pad(mask, 1, constant_values=False)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def _compare(reference: Any, candidate: Any) -> dict[str, Any]:
    import numpy as np

    if reference.shape != candidate.shape:
        raise ValueError("mask dimensions differ; resizing is forbidden")
    a, b = int(reference.sum()), int(candidate.sum())
    intersection = int(np.count_nonzero(reference & candidate))
    union = a + b - intersection
    ba, bb = _boundary(reference), _boundary(candidate)
    boundary_intersection = int(np.count_nonzero(ba & bb))
    boundary_union = int(np.count_nonzero(ba | bb))
    return {
        "shape": list(reference.shape), "intersection_pixels": intersection,
        "union_pixels": union, "iou": intersection / union if union else None,
        "reference_foreground_pixels": a, "candidate_foreground_pixels": b,
        "area_ratio": max(a, b) / min(a, b) if min(a, b) else None,
        "reference_boundary_pixels": int(ba.sum()),
        "candidate_boundary_pixels": int(bb.sum()),
        "boundary_intersection_pixels": boundary_intersection,
        "boundary_union_pixels": boundary_union,
        "boundary_iou": boundary_intersection / boundary_union if boundary_union else None,
        "boundary_definition": "foreground pixels lacking at least one 4-neighbor; outside is background",
        "interpretation": "agreement_only_not_accuracy",
    }


def compare_masks(reference_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    """Read binary single-channel images without thresholding or conversion."""
    import numpy as np
    from PIL import Image

    with Image.open(reference_path) as image:
        reference = _binary(np.array(image))
    with Image.open(candidate_path) as image:
        candidate = _binary(np.array(image))
    return _compare(reference, candidate)


def choose_propagation(
    forward: Any | None, reverse: Any | None, original: Any | None = None, *,
    policy: PropagationPolicy | None = None,
    forward_confidence: float | None = None,
    reverse_confidence: float | None = None,
) -> dict[str, Any]:
    """Recommend an existing direction for review, never promote to SAM direct OK.

    ``original`` means an existing direct OK mask and is always retained.
    Confidence arguments are caller-supplied mean scores, not probabilities.
    """
    policy = policy or PropagationPolicy()
    f = _binary(forward) if forward is not None else None
    r = _binary(reverse) if reverse is not None else None
    o = _binary(original) if original is not None else None
    arrays = [a for a in (f, r, o) if a is not None]
    if arrays and any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("mask dimensions differ; resizing is forbidden")
    for score in (forward_confidence, reverse_confidence):
        if score is not None and not isfinite(score):
            raise ValueError("confidence must be finite or None")
    metrics = _compare(f, r) if f is not None and r is not None else None
    accepted = bool(metrics and metrics["iou"] is not None
                    and metrics["reference_foreground_pixels"] > 0
                    and metrics["candidate_foreground_pixels"] > 0
                    and metrics["iou"] >= policy.min_iou
                    and metrics["area_ratio"] <= policy.max_area_ratio)
    source = None
    if accepted:
        source = "reverse" if (forward_confidence is not None
                  and reverse_confidence is not None
                  and reverse_confidence > forward_confidence) else "forward"
    empty = f is not None and r is not None and not f.any() and not r.any()
    return {
        "status": "ORIGINAL_RETAINED" if o is not None else (
            "PROPAGATED_CANDIDATE" if accepted else "EMPTY" if empty else "UNRESOLVED"),
        "retained_source": "original" if o is not None else None,
        "recommended_source": source, "review_candidate_only": True,
        "direct_sam_ok": False, "metrics": metrics,
        "thresholds": {"min_iou": policy.min_iou, "max_area_ratio": policy.max_area_ratio},
        "reason": "bidirectional_agreement" if accepted else (
            "missing_direction" if f is None or r is None else "empty_or_disagreement"),
    }
