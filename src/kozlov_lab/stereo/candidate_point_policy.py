"""Pure derived filtering of existing measured points, never mask acceptance.

Caller must bind this JSON row to immutable actual source evidence and masks.
This module cannot establish source authenticity or physical object identity.
"""
from __future__ import annotations

from copy import deepcopy
import math


def _integer(value):
    return type(value) is int and value >= 0


def _xy(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in value))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _source_points(points, identity):
    _require(isinstance(points, list), "Evidence points must be lists")
    result = {}
    for point in points:
        _require(isinstance(point, dict), "Invalid evidence point")
        key = point.get(identity)
        valid = _integer(key) if identity == "track_id" else isinstance(key, str) and bool(key)
        _require(valid and key not in result, "Invalid or duplicate point identity")
        _require(_xy(point.get("query_point_xy")), "Invalid point coordinates")
        result[key] = point
    return result


def _membership(summary, source, identity):
    _require(isinstance(summary, dict), "Missing membership summary")
    points = summary.get("points")
    observed = _source_points(points, identity)
    _require(observed.keys() == source.keys(), "Assessment point identities differ from source")
    inside = 0
    for key, point in observed.items():
        xy = point["query_point_xy"]
        _require(xy == source[key]["query_point_xy"], "Assessment coordinates differ from source")
        sample = point.get("sample_xy")
        _require(isinstance(sample, list) and len(sample) == 2
                 and all(_integer(v) for v in sample)
                 and sample == [math.floor(v + .5) for v in xy], "Invalid point sampling")
        _require(type(point.get("inside")) is bool, "Inside membership must be boolean")
        inside += point["inside"]
    count = len(source)
    _require(_integer(summary.get("count")) and summary["count"] == count,
             "Membership count contradicts points")
    _require(_integer(summary.get("inside_count")) and summary["inside_count"] == inside,
             "Inside count contradicts points")
    fraction = summary.get("inside_fraction")
    _require((fraction is None if not count else
              type(fraction) in (int, float) and math.isfinite(fraction) and fraction == inside / count),
             "Inside fraction contradicts points")
    _require(summary.get("status") == ("OBSERVED" if count else "UNAVAILABLE"),
             "Membership status contradicts points")
    return count, inside


def decide_candidate_points(row: dict) -> dict:
    """Derive point-scoped support, preserving source statuses and input data.

    Sparse positives are disclosed as a count, with no tuned minimum. Even a
    unique survivor grants no new point seeding, full-mask, or geometry rights.
    """
    original = row.get("original_status") if isinstance(row, dict) else None
    result = {"status": "INVALID_EVIDENCE", "original_status": original,
              "scope": "EXISTING_MEASURED_POINTS_ONLY", "full_mask_accepted": False,
              "new_point_seeding_allowed": False, "publishable_for_geometry": False,
              "physical_identity": "UNVERIFIED", "provisional_candidate_id": None,
              "supported_observations": [], "candidate_results": [],
              "clipped_boundary_status": "UNKNOWN",
              "limitations": ["Support is conditional on source measured point evidence.",
                              "Sparse positives do not establish full mask correctness.",
                              "Reference exterior is not physical background truth."]}
    try:
        _require(isinstance(row, dict) and original in ("OK", "NO_MATCH", "NEEDS_SELECTION"), "Unsupported original status")
        _require(_integer(row.get("input_index")), "Invalid input_index")
        _require(isinstance(row.get("frame_id"), str) and bool(row["frame_id"]), "Invalid frame_id")
        for field in ("before_index", "after_index"):
            _require(row.get(field) is None or _integer(row[field]), "Invalid reference frame index")
        if original in ("NO_MATCH", "OK"):
            result["status"] = "NO_CANDIDATE" if original == "NO_MATCH" else "ORIGINAL_OK_UNCHANGED"
            return result
        candidates = row.get("candidates")
        _require(isinstance(candidates, list), "Missing candidate list")
        ids = [c.get("candidate_id") if isinstance(c, dict) else None for c in candidates]
        _require(all(isinstance(c, str) and bool(c) for c in ids) and len(set(ids)) == len(ids),
                 "Invalid or duplicate candidate ID")
        assessment = row.get("assessment")
        _require(isinstance(assessment, dict) and assessment.get("status") == "EVIDENCE_ONLY"
                 and assessment.get("selected_candidate_id") is None
                 and assessment.get("publishable_for_geometry") is False
                 and assessment.get("publishable") is False
                 and row.get("status") == "EVIDENCE_ONLY"
                 and row.get("selected_candidate_id") is None
                 and row.get("publishable_for_geometry") is False, "Source status contradicts evidence-only scope")
        assessed = assessment.get("candidates")
        _require(isinstance(assessed, list) and all(isinstance(c, dict) for c in assessed), "Invalid assessments")
        assessed_ids = [c.get("candidate_id") for c in assessed]
        _require(len(assessed_ids) == len(ids) and all(isinstance(c, str) for c in assessed_ids)
                 and len(set(assessed_ids)) == len(ids) and set(assessed_ids) == set(ids), "Assessment candidate IDs differ")
        positive = _source_points(row.get("positive_evidence"), "track_id")
        exterior = _source_points(row.get("exterior_evidence"), "witness_id")
        for point in exterior.values():
            _require(point.get("reference_exterior_witness") is True and point.get("verified_background") is False,
                     "Exterior identity is not reference-relative")
        result["positive_evidence_count"] = len(positive)
        result["reference_exterior_evidence_count"] = len(exterior)
        viable, exterior_held = [], []
        for candidate in assessed:
            _require("status" not in candidate or candidate["status"] == "EVIDENCE_ONLY",
                     "Candidate status contradicts evidence-only scope")
            count, inside = _membership(candidate.get("positive_evidence"), positive, "track_id")
            _, outside_inside = _membership(candidate.get("exterior_evidence"), exterior, "witness_id")
            conflicts = []
            if inside < count:
                conflicts.append("POSITIVE_POINT_CONFLICT")
            if outside_inside:
                conflicts.append("REFERENCE_EXTERIOR_CONFLICT")
            reasons = candidate.get("reasons")
            _require(isinstance(reasons, list) and all(isinstance(r, str) for r in reasons), "Invalid candidate reasons")
            for reason in ("POSITIVE_POINT_CONFLICT", "REFERENCE_EXTERIOR_CONFLICT"):
                _require((reason in reasons) == (reason in conflicts), "Candidate reason contradicts measured points")
            result["candidate_results"].append({"candidate_id": candidate["candidate_id"],
                                                "conflicts": conflicts, "positive_inside_count": inside,
                                                "reference_exterior_inside_count": outside_inside,
                                                "point_status": ("POSITIVE_REJECTED" if inside < count else
                                                                 "REFERENCE_EXTERIOR_HELD" if outside_inside else
                                                                 "POSITIVE_COMPATIBLE")})
            if inside == count and outside_inside:
                exterior_held.append(candidate["candidate_id"])
            if not conflicts:
                viable.append(candidate["candidate_id"])
        if not positive or not exterior:
            result["status"] = "INSUFFICIENT_EVIDENCE"
        elif exterior_held:
            # Conditional exterior evidence cannot establish that this positive-
            # compatible alternative is false and thereby select another mask.
            result["status"] = "REFERENCE_CONFLICT_HOLD"
        elif len(viable) > 1:
            result["status"] = "AMBIGUOUS"
        elif not viable:
            result["status"] = "CONFLICT"
        else:
            result["status"] = "PROVISIONAL_POINT_SUPPORT_ONLY"
            result["provisional_candidate_id"] = viable[0]
            result["supported_observations"] = deepcopy(row["positive_evidence"])
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        result.update(status="INVALID_EVIDENCE", provisional_candidate_id=None, supported_observations=[],
                      reason=str(exc))
    return result
