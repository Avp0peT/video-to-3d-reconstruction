"""Candidate boundary observations; never selection or semantic acceptance.

Image edges can also be texture, an occluder, or a hand corner. Agreement and
point inclusion therefore cannot establish physical object identity.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


def _points(items, id_key, height, width):
    if not isinstance(items, list) or len(items) > 768:
        raise ValueError("Each evidence class must be a list of at most 768 points")
    result, seen = [], set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Evidence must contain dictionaries")
        identity = item.get(id_key)
        valid = (isinstance(identity, int) and not isinstance(identity, bool)) if id_key == "track_id" else (isinstance(identity, str) and bool(identity))
        if not valid or identity in seen:
            raise ValueError("Evidence IDs must be valid and unique within their class")
        seen.add(identity)
        xy = item.get("query_point_xy")
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            raise ValueError("query_point_xy must contain two real coordinates")
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) or not math.isfinite(float(v)) for v in xy):
            raise ValueError("Coordinates must be finite real numbers, excluding booleans")
        x, y = map(float, xy)
        if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
            raise ValueError("Coordinates must lie inside the image")
        sx, sy = math.floor(x + .5), math.floor(y + .5)
        result.append({id_key: identity, "query_point_xy": [x, y], "sample_xy": [sx, sy]})
    return result


def _membership(points, mask):
    observations = [dict(point, inside=bool(mask[point["sample_xy"][1], point["sample_xy"][0]])) for point in points]
    count = sum(p["inside"] for p in observations)
    return {"count": len(points), "inside_count": count, "inside_fraction": count / len(points) if points else None,
            "status": "OBSERVED" if points else "UNAVAILABLE", "points": observations}


def assess_candidate_boundary(image_rgb_uint8, candidates, positive_evidence, exterior_evidence, cv):
    """Return JSON-safe fixed-parameter CPU evidence without changing inputs.

    ``exterior_evidence`` is reference-relative, not physical background truth.
    Clipped boundary pixels are unknown and retained in the all-boundary denominator.
    """
    image = image_rgb_uint8
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("Image must be a nonempty uint8 HWC RGB image")
    height, width = image.shape[:2]
    if not isinstance(candidates, list) or len(candidates) > 8:
        raise ValueError("Provide a list of at most 8 candidates")
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Candidates must contain dictionaries")
        identity, mask = candidate.get("candidate_id"), candidate.get("mask")
        if not isinstance(identity, (str, int)) or isinstance(identity, bool) or identity == "" or identity in seen:
            raise ValueError("Candidate IDs must be nonempty strings or integers and unique; bool is invalid")
        seen.add(identity)
        if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != (height, width) or not np.all((mask == 0) | (mask == 255)) or not np.any(mask):
            raise ValueError("Candidate masks must be nonempty uint8 binary 0/255 arrays matching the image")
    positives = _points(positive_evidence, "track_id", height, width)
    exteriors = _points(exterior_evidence, "witness_id", height, width)
    gray = cv.cvtColor(image.copy(), cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 1)
    edges = cv.Canny(blurred, 40, 100, apertureSize=3, L2gradient=True)
    has_edges = bool(np.any(edges))
    distances = cv.distanceTransform((edges == 0).astype(np.uint8), cv.DIST_L2, cv.DIST_MASK_PRECISE) if has_edges else None
    yy, xx = np.indices((height, width))
    measurable_region = (xx >= 3) & (yy >= 3) & (xx < width - 3) & (yy < height - 3)
    output = []
    for candidate in candidates:
        mask = candidate["mask"]
        eroded = cv.erode(mask.copy(), np.ones((3, 3), np.uint8), borderType=cv.BORDER_CONSTANT, borderValue=0)
        boundary = (mask != 0) & (eroded == 0)
        measurable = boundary & measurable_region
        total, measured = int(boundary.sum()), int(measurable.sum())
        aligned = int(np.count_nonzero(distances[measurable] <= 3)) if has_edges else 0
        values = distances[measurable] if has_edges and measured else None
        reasons = []
        if not has_edges:
            reasons.append("NO_IMAGE_EDGES")
        if measured != total:
            reasons.append("CLIPPED_BOUNDARY_UNKNOWN")
        if not measured:
            reasons.append("NO_MEASURABLE_BOUNDARY")
        pos, ext = _membership(positives, mask), _membership(exteriors, mask)
        if pos["inside_count"] < pos["count"]:
            reasons.append("POSITIVE_POINT_CONFLICT")
        if ext["inside_count"]:
            reasons.append("REFERENCE_EXTERIOR_CONFLICT")
        output.append({"candidate_id": candidate["candidate_id"], "area_px": int(np.count_nonzero(mask)),
                       "boundary": {"total_px": total, "measurable_px": measured, "clipped_px": total - measured,
                                    "measurable_fraction": measured / total,
                                    "clipped_boundary_status": "UNKNOWN" if measured != total else "NONE",
                                    "aligned_px": aligned,
                                    "image_edge_alignment": aligned / measured if has_edges and measured else None,
                                    "image_edge_alignment_all_boundary": aligned / total if has_edges and measured else None,
                                    "distance_p50_px": float(np.percentile(values, 50)) if values is not None else None,
                                    "distance_p90_px": float(np.percentile(values, 90)) if values is not None else None,
                                    "status": ("PARTIALLY_OBSERVED" if measured != total else "OBSERVED") if has_edges and measured else "UNKNOWN"},
                       "positive_evidence": pos, "exterior_evidence": ext, "reasons": reasons})
    return {"status": "EVIDENCE_ONLY", "selected_candidate_id": None, "physical_identity": "UNVERIFIED",
            "publishable": False, "publishable_for_geometry": False,
            "parameters": {"blur_kernel": [3, 3], "blur_sigma": 1, "canny_thresholds": [40, 100],
                           "canny_aperture": 3, "canny_l2gradient": True, "alignment_distance_px": 3,
                           "clipped_unknown_band_px": 3, "point_sampling": "floor(x+0.5), coordinates within closed pixel-center bounds"},
            "canny_edge_px": int(np.count_nonzero(edges)), "candidates": output,
            "reasons": [] if candidates else ["NO_CANDIDATES"],
            "limitations": ["Image edges may represent interior texture or occluders.",
                            "No point conflict does not establish a correct boundary.",
                            "Reference exterior is not physical background truth."]}
