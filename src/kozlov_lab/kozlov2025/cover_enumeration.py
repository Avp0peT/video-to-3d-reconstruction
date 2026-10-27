"""Bounded complete-cover enumeration, S-2025 PDF4/p149 and PDF10/p155.

Project-defined RGS search, not m-ostov, exact ellipse centres or strict fits.
"""
import hashlib
import json
import math
import time

from .independent_cover import normalize, verify_independent_cover_2d, verify_cover_certificate

__all__ = ["enumerate_independent_covers_2d"]


def _bell(n):
    row = [1]
    for _ in range(n):
        next_row = [row[-1]]
        for value in row:
            next_row.append(next_row[-1] + value)
        row = next_row
    return row[0]


def _partitions(labels):
    assignment = [0]
    def visit(maximum):
        if len(assignment) == len(labels):
            groups = [[] for _ in range(maximum + 1)]
            for label, index in zip(labels, assignment):
                groups[index].append(label)
            yield groups
            return
        for index in range(maximum + 2):
            assignment.append(index)
            yield from visit(max(maximum, index))
            assignment.pop()
    yield from visit(0)


def enumerate_independent_covers_2d(points, *, max_candidates=877,
                                  total_max_work=2000000, seconds=30):
    """Enumerate all covers of a complete supplied set of 1..7 distinct points.

    Work counts solver operations, not verifier or Python instructions. Wall
    time includes validation and verification, with cooperative boundary checks.
    Returns exact validated covers even on partial search; complete stays false.
    """
    start = time.monotonic()
    result = {"schema": "cover-enumeration-v1", "contract": "complete-small-set-covers-v1",
              "status": "UNSUPPORTED_INPUT", "complete": False,
              "visited_candidates": 0, "judged_candidates": 0, "rejected_candidates": 0,
              "independent_covers": [], "uncertain_candidates": [], "work_used": 0,
              "source_pages": {"pdf": [4, 10], "journal": [149, 155]},
              "scope": "complete supplied 2D set only; not m-ostov/fits/semantic discovery"}
    if (type(max_candidates) is not int or not 0 <= max_candidates <= 877
            or type(total_max_work) is not int or not 0 <= total_max_work <= 2000000
            or type(seconds) not in (int, float) or not 0 <= seconds <= 30
            or not math.isfinite(seconds)):
        result["reason"] = "invalid bounded budget"
        return result
    deadline = start + seconds
    result["budgets"] = {"max_candidates": max_candidates, "total_max_work": total_max_work,
                         "seconds": seconds}
    def expired():
        return time.monotonic() >= deadline
    def finish(status, reason):
        result.update(status=status, reason=reason, elapsed_seconds=time.monotonic()-start)
        if 'expected_candidates' in result:
            result['unvisited_candidates'] = result['expected_candidates']-result['visited_candidates']
            result['unresolved_candidates'] = result['expected_candidates']-result['judged_candidates']
        return result
    if expired():
        return finish("PARTIAL_UNKNOWN", "time budget exhausted before validation")
    if isinstance(points, dict) and len(points) > 7:
        return finish("UNSUPPORTED_SIZE", "complete enumeration limited to 1..7 points; no sampling")
    try:
        labels = sorted(points) if isinstance(points, dict) and all(type(i) is int for i in points) else []
        normalized, _, missing = normalize(points, [labels])
    except (ValueError, TypeError, ArithmeticError, AttributeError, KeyError, IndexError) as exc:
        return finish("UNSUPPORTED_INPUT", str(exc))
    n = len(labels)
    encoding = [[hex(label), [[str(v.numerator), str(v.denominator)] for v in normalized[label]]]
                for label in labels]
    result.update(declared_labels=labels, points=n, nonempty_parts_count=2**n-1,
                  expected_candidates=_bell(n),
                  source_coordinate_sha256=hashlib.sha256(json.dumps(encoding, separators=(',', ':')).encode()).hexdigest())
    for parts in _partitions(labels):
        if expired():
            return finish("PARTIAL_UNKNOWN", "shared time budget exhausted")
        if result["visited_candidates"] >= max_candidates:
            return finish("PARTIAL_UNKNOWN", "candidate budget exhausted")
        index = result["visited_candidates"]
        result["visited_candidates"] += 1
        certificate = verify_independent_cover_2d(normalized, parts,
            max_work=total_max_work-result["work_used"], seconds=max(0, deadline-time.monotonic()))
        result["work_used"] += certificate.get("work_used", 0)
        status = certificate["status"]
        if expired() or status not in ("VALID_INDEPENDENT_COVER", "NOT_INDEPENDENT"):
            result["uncertain_candidates"].append({"index": index, "parts": parts, "solver_status": status})
            return finish("PARTIAL_UNKNOWN", "candidate undecided or time exhausted")
        verified = verify_cover_certificate(normalized, parts, certificate)
        if not verified or expired():
            result["uncertain_candidates"].append({"index": index, "parts": parts,
                                                   "solver_status": status, "certificate_verified": verified})
            return finish("PARTIAL_UNKNOWN", "independent verification failed or time exhausted")
        result["judged_candidates"] += 1
        if status == "VALID_INDEPENDENT_COVER":
            result["independent_covers"].append({"candidate_index": index, "parts": parts,
                "certificate": certificate, "certificate_verified": True,
                "source_coordinate_sha256": result["source_coordinate_sha256"]})
        else:
            result["rejected_candidates"] += 1
    if expired():
        return finish("PARTIAL_UNKNOWN", "time exhausted before final completeness check")
    if result["judged_candidates"] != result["expected_candidates"]:
        return finish("PARTIAL_UNKNOWN", "candidate count inconsistent")
    result["complete"] = True
    return finish("COMPLETE", "all set partitions decided and independently verified")
