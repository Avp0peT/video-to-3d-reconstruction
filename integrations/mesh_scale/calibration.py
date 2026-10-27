"""Pure, standard-library mesh scale arithmetic; no physical-accuracy claim.

Coordinates and returned matrices are binary64 values.  Vertex order and repeated
mesh vertices are preserved.  Only the two endpoints of a measurement must have
different coordinates.  The matrices describe the affine map; evaluation uses
``scale * (coordinate - origin)`` to avoid unnecessary cancellation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}
ROUNDTRIP_ULPS = 8
ROUNDTRIP_REFERENCE_CAP = 1e-12
ROUNDTRIP_POLICY = "per_component_origin_relative_ulp_v2"


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number, not bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} is not representable as a finite float") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{label} must be a sequence")
    return value


def _point(value: Any, label: str) -> tuple[float, float, float]:
    values = _sequence(value, label)
    if len(values) != 3:
        raise ValueError(f"{label} must have exactly three coordinates")
    return tuple(_number(component, f"{label}[{axis}]") for axis, component in enumerate(values))


def _points(values: Any) -> list[tuple[float, float, float]]:
    values = _sequence(values, "points")
    if len(values) < 2:
        raise ValueError("points must contain at least two vertices")
    return [_point(point, f"points[{index}]") for index, point in enumerate(values)]


def _index(value: Any, count: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        raise ValueError(f"{label} must be an integer vertex index in [0, {count})")
    return value


def _pair(value: Any, count: int, label: str) -> tuple[int, int]:
    pair = _sequence(value, label)
    if len(pair) != 2:
        raise ValueError(f"{label} must contain exactly two vertex indices")
    first = _index(pair[0], count, f"{label}[0]")
    second = _index(pair[1], count, f"{label}[1]")
    if first == second:
        raise ValueError(f"{label} endpoints must be distinct")
    return first, second


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{label} overflowed or became non-finite")
    return value


def _distance(points: list[tuple[float, float, float]], pair: tuple[int, int]) -> float:
    difference = [_finite(b - a, "endpoint difference")
                  for a, b in zip(points[pair[0]], points[pair[1]])]
    distance = _finite(math.hypot(*difference), "endpoint distance")
    if distance <= 0.0:
        raise ValueError("measurement endpoints must have different coordinates")
    return distance


def _reference(value: Any, count: int, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    pair = _pair(value.get("vertex_indices"), count, f"{label}.vertex_indices")
    length = _number(value.get("length"), f"{label}.length", positive=True)
    unit = value.get("unit")
    if not isinstance(unit, str) or unit not in UNIT_TO_MM:
        raise ValueError(f"{label}.unit must be mm, cm, m, or in")
    length_mm = _finite(length * UNIT_TO_MM[unit], f"{label}.length_mm")
    if length_mm <= 0.0:
        raise ValueError(f"{label}.length_mm underflowed to zero")
    return {"vertex_indices": list(pair), "length": length, "unit": unit,
            "length_mm": length_mm}


def _matrices(origin: tuple[float, float, float], scale: float) -> tuple[list[list[float]], list[list[float]]]:
    reciprocal = _finite(1.0 / scale, "inverse scale")
    if reciprocal <= 0.0:
        raise ValueError("inverse scale underflowed to zero")
    translation = [_finite(-scale * value, "matrix translation") for value in origin]
    if any(value != 0.0 and offset == 0.0 for value, offset in zip(origin, translation)):
        raise ValueError("matrix translation underflowed to zero")
    matrix = [[scale, 0.0, 0.0, translation[0]],
              [0.0, scale, 0.0, translation[1]],
              [0.0, 0.0, scale, translation[2]],
              [0.0, 0.0, 0.0, 1.0]]
    inverse = [[reciprocal, 0.0, 0.0, origin[0]],
               [0.0, reciprocal, 0.0, origin[1]],
               [0.0, 0.0, reciprocal, origin[2]],
               [0.0, 0.0, 0.0, 1.0]]
    return matrix, inverse


def fit_transform(points: Sequence[Sequence[Real]], reference: Mapping[str, Any],
                  origin_vertex: int | None = None) -> dict[str, Any]:
    """Fit a positive uniform scale to one known segment, in millimetres.

    ``origin_vertex`` changes translation only.  A successful fit does not assert
    that every mesh coordinate can be transformed: ``apply_transform`` checks
    that separately before returning any output.
    """
    validated = _points(points)
    normalized = _reference(reference, len(validated), "reference")
    pair = tuple(normalized["vertex_indices"])
    distance = _distance(validated, pair)
    scale = _finite(normalized["length_mm"] / distance, "factor_to_mm")
    if scale <= 0.0:
        raise ValueError("factor_to_mm underflowed to zero")
    if origin_vertex is None:
        origin = (0.0, 0.0, 0.0)
    else:
        origin_vertex = _index(origin_vertex, len(validated), "origin_vertex")
        origin = validated[origin_vertex]
    matrix, inverse = _matrices(origin, scale)
    normalized["distance_model"] = distance
    return {"origin": list(origin), "origin_vertex": origin_vertex,
            "factor_to_mm": scale, "matrix4x4": matrix, "inverse4x4": inverse,
            "reference": normalized}


def _transform(value: Any) -> tuple[tuple[float, float, float], float]:
    if not isinstance(value, Mapping):
        raise ValueError("transform must be a mapping")
    origin = _point(value.get("origin"), "transform.origin")
    scale = _number(value.get("factor_to_mm"), "transform.factor_to_mm", positive=True)
    expected = _matrices(origin, scale)
    for key, wanted in zip(("matrix4x4", "inverse4x4"), expected):
        rows = _sequence(value.get(key), f"transform.{key}")
        if len(rows) != 4:
            raise ValueError(f"transform.{key} must have four rows")
        matrix = []
        for index, row in enumerate(rows):
            row = _sequence(row, f"transform.{key}[{index}]")
            if len(row) != 4:
                raise ValueError(f"transform.{key} rows must have four columns")
            matrix.append([_number(entry, f"transform.{key} entry") for entry in row])
        if matrix != wanted:
            raise ValueError(f"transform.{key} disagrees with origin and factor_to_mm")
    return origin, scale


def _transform_reference_distance(points: list[tuple[float, float, float]],
                                  transform: Mapping[str, Any], scale: float) -> float:
    """Recompute the local error-budget scale instead of trusting stored metadata."""
    stored = transform.get("reference")
    reference = _reference(stored, len(points), "transform.reference")
    distance = _distance(points, tuple(reference["vertex_indices"]))
    stored_distance = _number(stored.get("distance_model"),
                              "transform.reference.distance_model", positive=True)
    if stored_distance != distance:
        raise ValueError("transform reference distance_model disagrees with input endpoints")
    stored_length_mm = _number(stored.get("length_mm"),
                               "transform.reference.length_mm", positive=True)
    if stored_length_mm != reference["length_mm"]:
        raise ValueError("transform reference length_mm disagrees with length and unit")
    expected_scale = _finite(reference["length_mm"] / distance, "reference scale")
    if expected_scale != scale:
        raise ValueError("transform factor_to_mm disagrees with reference endpoints and length")
    return distance


def apply_transform(points: Sequence[Sequence[Real]], transform: Mapping[str, Any]
                    ) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    """Apply the map without reordering, with explicit binary64 roundtrip gates.

    Every component gets its own budget of eight ULPs of its origin-relative
    coordinate.  Absolute coordinate recovery is additionally capped at 1e-12
    times the calibration segment length, recomputed from the input vertices.
    No world-coordinate magnitude or global ULP maximum is used as a gate.
    Nonzero-to-zero underflow and complete-XYZ vertex collapse are rejected;
    a collision in just one coordinate component is not a collapsed vertex.
    These are arithmetic checks, not a claim about reconstruction accuracy.
    """
    validated = _points(points)
    origin, scale = _transform(transform)
    reference_distance = _transform_reference_distance(validated, transform, scale)
    reference_cap = ROUNDTRIP_REFERENCE_CAP * reference_distance
    result = []
    seen_points: dict[tuple[float, float, float], tuple[float, float, float]] = {}
    maximum_error = maximum_relative_error = maximum_recentered_error = 0.0
    maximum_budget = maximum_relative_budget = 0.0
    for index, point in enumerate(validated):
        transformed = []
        for axis, (source, center) in enumerate(zip(point, origin)):
            relative = _finite(source - center, "origin-relative coordinate")
            target = _finite(scale * relative, "transformed coordinate")
            if relative != 0.0 and target == 0.0:
                raise ValueError(f"vertex {index} axis {axis}: transformed coordinate underflow")
            recovered_relative = _finite(target / scale, "inverse relative coordinate")
            recovered = _finite(recovered_relative + center, "inverse coordinate")
            recentered = _finite(recovered - center, "recentered inverse coordinate")
            error = _finite(abs(recovered - source), "inverse absolute error")
            relative_error = _finite(abs(recovered_relative - relative), "relative inverse error")
            recentered_error = _finite(abs(recentered - relative), "recentered inverse error")
            relative_budget = ROUNDTRIP_ULPS * math.ulp(relative)
            budget = min(relative_budget, reference_cap)
            if error > budget or relative_error > relative_budget or recentered_error > relative_budget:
                raise ValueError(f"vertex {index} axis {axis}: inverse roundtrip exceeds numeric budget")
            transformed.append(target)
            maximum_error = max(maximum_error, error)
            maximum_relative_error = max(maximum_relative_error, relative_error)
            maximum_recentered_error = max(maximum_recentered_error, recentered_error)
            maximum_budget = max(maximum_budget, budget)
            maximum_relative_budget = max(maximum_relative_budget, relative_budget)
        transformed_point = tuple(transformed)
        if transformed_point in seen_points and seen_points[transformed_point] != point:
            raise ValueError(f"vertex {index}: distinct source XYZ vertices collapsed")
        seen_points[transformed_point] = point
        result.append(transformed_point)
    return result, {
        "max_inverse_abs_error": maximum_error,
        "max_origin_relative_inverse_abs_error": maximum_relative_error,
        "max_recentered_inverse_abs_error": maximum_recentered_error,
        "roundtrip_policy": ROUNDTRIP_POLICY,
        "numeric_tolerance": {
            "ulps_per_component": ROUNDTRIP_ULPS,
            "inverse_absolute_budget": "min(8 * ulp(source - origin), 1e-12 * reference_distance_model)",
            "origin_relative_budget": "8 * ulp(source - origin)",
            "reference_distance_model": reference_distance,
            "reference_distance_source": "recomputed from input points at transform.reference.vertex_indices",
            "reference_relative_cap": ROUNDTRIP_REFERENCE_CAP,
            "reference_absolute_cap": reference_cap,
            "max_inverse_absolute_budget": maximum_budget,
            "max_origin_relative_budget": maximum_relative_budget,
            "error_and_budget_unit": "input_model_coordinate_unit",
            "global_maximum_used_as_gate": False,
        },
        "forward_evaluation": "factor_to_mm * (source - origin)",
        "inverse_evaluation": "transformed / factor_to_mm + origin",
        "distinct_vertex_collapse_rejected": True,
        "collapse_check_scope": "complete_xyz",
        "roundtrip_pass": True,
        "physical_accuracy_assessed": False,
    }


def measure_checks(points_mm: Sequence[Sequence[Real]], checks: Sequence[Mapping[str, Any]],
                   reference_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Measure supplied independent segments; missing thresholds remain unknown.

    Reversing endpoint order does not create a new check or make a calibration
    segment independent.  Threshold results concern supplied lengths only.
    """
    validated = _points(points_mm)
    reference_pair = _pair(reference_ids, len(validated), "reference_ids")
    _distance(validated, reference_pair)
    calibration_pair = tuple(sorted(reference_pair))
    checks = _sequence(checks, "checks")
    used = set()
    results = []
    for index, check in enumerate(checks):
        normalized = _reference(check, len(validated), f"checks[{index}]")
        pair = tuple(normalized["vertex_indices"])
        canonical = tuple(sorted(pair))
        if canonical == calibration_pair:
            raise ValueError("independent check must not reuse the calibration endpoint pair")
        if canonical in used:
            raise ValueError("duplicate independent check endpoint pair")
        used.add(canonical)
        allowed = None
        if "allowed_absolute_error_mm" in check:
            allowed = _number(check["allowed_absolute_error_mm"], "allowed_absolute_error_mm")
            if allowed < 0.0:
                raise ValueError("allowed_absolute_error_mm must be nonnegative")
        measured = _distance(validated, pair)
        signed_error = _finite(measured - normalized["length_mm"], "signed error")
        absolute_error = abs(signed_error)
        relative_error = _finite(absolute_error / normalized["length_mm"], "relative error")
        results.append({
            "vertex_indices": normalized["vertex_indices"],
            "length": normalized["length"], "unit": normalized["unit"],
            "model_distance_mm": measured,
            "reference_length_mm": normalized["length_mm"],
            "signed_error_mm": signed_error,
            "absolute_error_mm": absolute_error,
            "relative_error": relative_error,
            "allowed_absolute_error_mm": allowed,
            "within_threshold": None if allowed is None else absolute_error <= allowed,
        })
    return results
