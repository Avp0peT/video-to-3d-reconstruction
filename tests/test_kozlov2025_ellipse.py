"""Minimum enclosing ellipse: independent geometry and binary64 proof checks.

These tests never use the implementation's containment/certificate helpers as
their oracle.  Fractions below describe the *stored float values*, not the
decimal numbers from which those values may have originated.
"""

from __future__ import annotations

from fractions import Fraction
from dataclasses import FrozenInstanceError
import json
import math

import numpy as np
import pytest

from kozlov_lab.kozlov2025.ellipse import EllipseInputError, minimum_enclosing_ellipse


TRIANGLE = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
CLOUD = np.array(
    [[-3.0, -0.5], [-1.0, 2.0], [1.5, 1.0], [4.0, -1.0], [0.5, -2.0], [0.0, 0.25]]
)
SUCCESS = {"APPROXIMATE_CERTIFIED", "DEGENERATE_POINT", "DEGENERATE_SEGMENT"}


def _f(value):
    return Fraction.from_float(float(value))


def _read_fraction(record):
    assert isinstance(record, dict)
    assert isinstance(record["numerator"], str)
    assert isinstance(record["denominator"], str)
    assert int(record["denominator"]) > 0
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def _exact_contains(points, center, factor):
    """Evaluate every operation in ||B(x-c)||^2 <= 1 over the rationals."""
    c = [_f(v) for v in center]
    b = [[_f(v) for v in row] for row in factor]
    answers = []
    for point in points:
        d = [_f(point[j]) - c[j] for j in range(2)]
        transformed = [sum(b[i][j] * d[j] for j in range(2)) for i in range(2)]
        answers.append(sum(v * v for v in transformed) <= 1)
    return np.asarray(answers, dtype=bool)


def _exact_area_ratio_squared(result):
    """Reconstruct a dual lower bound from the exported float weights."""
    p = [[_f(v) for v in row] for row in result.unique_points]
    raw_weights = [_f(v) for v in result.weights]
    assert all(w >= 0 for w in raw_weights)
    total = sum(raw_weights)
    assert total > 0
    weights = [w / total for w in raw_weights]
    mean = [sum(w * point[j] for w, point in zip(weights, p)) for j in range(2)]
    covariance = [
        [
            sum(w * (point[i] - mean[i]) * (point[j] - mean[j]) for w, point in zip(weights, p))
            for j in range(2)
        ]
        for i in range(2)
    ]
    det_c = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0]
    b = [[_f(v) for v in row] for row in result.factor]
    det_b = b[0][0] * b[1][1] - b[0][1] * b[1][0]
    assert det_c > 0
    assert det_b != 0
    return 1 / (4 * det_b * det_b * det_c)


def _assert_serializable(result):
    payload = result.to_dict()
    assert isinstance(payload, dict)
    restored = json.loads(json.dumps(payload, allow_nan=False))
    assert restored["status"] == result.status
    assert restored["converged"] is result.converged
    assert isinstance(result.converged, bool)
    assert result.converged is (result.status in SUCCESS)


def _assert_certified(result, original_points, tolerance):
    assert result.status == "APPROXIMATE_CERTIFIED"
    assert result.converged is True
    assert isinstance(result.center, np.ndarray) and result.center.shape == (2,)
    assert isinstance(result.factor, np.ndarray) and result.factor.shape == (2, 2)
    assert np.isfinite(result.center).all()
    assert np.isfinite(result.factor).all()
    assert isinstance(result.unique_points, np.ndarray)
    np.testing.assert_array_equal(result.unique_points, np.unique(original_points, axis=0))
    assert isinstance(result.weights, np.ndarray)
    assert result.weights.shape == (len(result.unique_points),)
    assert np.isfinite(result.weights).all()
    assert isinstance(result.iterations, int) and result.iterations >= 0

    # This is the enclosure assertion; the result's own boolean is insufficient.
    assert _exact_contains(original_points, result.center, result.factor).all()
    reported = result.contains(original_points)
    assert isinstance(reported, np.ndarray) and reported.dtype == np.dtype(bool)
    np.testing.assert_array_equal(reported, np.ones(len(original_points), dtype=bool))

    certificate = result.certificate
    assert certificate["containment_exact"] is True
    assert certificate["proof_scope"] == "stored_binary64_inputs"
    assert certificate["requested_area_tolerance"] == tolerance
    ratio_squared = _exact_area_ratio_squared(result)
    assert _read_fraction(certificate["area_ratio_squared"]) == ratio_squared
    assert 1 <= ratio_squared <= (1 + _f(tolerance)) ** 2
    upper_bound = certificate["area_ratio_upper_bound"]
    assert math.isfinite(upper_bound)
    assert _f(upper_bound) ** 2 >= ratio_squared, "the displayed bound must round upward"
    assert upper_bound <= 1 + tolerance

    singular_values = np.linalg.svd(result.factor, compute_uv=False)
    expected_semiaxes = np.sort(1 / singular_values)[::-1]
    np.testing.assert_allclose(result.semiaxes, expected_semiaxes, rtol=2e-12, atol=0)
    assert result.semiaxes[0] >= result.semiaxes[1] > 0
    assert result.area == pytest.approx(math.pi * np.prod(expected_semiaxes), rel=2e-12)
    assert result.major_axis_length == pytest.approx(2 * expected_semiaxes[0], rel=2e-12)
    _assert_serializable(result)


@pytest.mark.parametrize(
    "points",
    [TRIANGLE, [[-2.0, 1.0], [1.0, 5.0], [4.0, -3.0]], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
)
def test_triangle_matches_analytic_minimum(points):
    points = np.asarray(points, dtype=float)
    result = minimum_enclosing_ellipse(points)
    _assert_certified(result, points, 1e-3)
    edges = points[1:] - points[0]
    expected_area = 2 * math.pi * abs(np.linalg.det(edges)) / (3 * math.sqrt(3))
    np.testing.assert_allclose(result.center, points.mean(axis=0), rtol=0, atol=1e-12)
    assert result.area == pytest.approx(expected_area, rel=2e-11)


@pytest.mark.parametrize("a,b", [(3.0, 2.0), (0.25, 4.0)])
def test_rectangle_has_known_area_and_axes(a, b):
    points = np.array([[-a, -b], [-a, b], [a, -b], [a, b]])
    result = minimum_enclosing_ellipse(points)
    _assert_certified(result, points, 1e-3)
    np.testing.assert_allclose(result.center, [0, 0], rtol=0, atol=1e-13)
    np.testing.assert_allclose(result.semiaxes, np.sort([a, b])[::-1] * math.sqrt(2), rtol=2e-11)
    assert result.area == pytest.approx(2 * math.pi * a * b, rel=2e-11)


def test_regular_octagon_has_unit_circle_as_minimum():
    angles = np.arange(8) * (math.pi / 4)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    result = minimum_enclosing_ellipse(points)
    _assert_certified(result, points, 1e-3)
    np.testing.assert_allclose(result.center, [0, 0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.semiaxes, [1, 1], rtol=2e-11)
    assert result.area == pytest.approx(math.pi, rel=2e-11)


@pytest.mark.parametrize("linear", [np.array([[0.6, -0.8], [0.8, 0.6]]), np.array([[2.0, 1.5], [0.0, 0.5]])])
def test_rotation_and_shear_transform_the_whole_ellipse(linear):
    points = np.array([[-3.0, -2.0], [-3.0, 2.0], [3.0, -2.0], [3.0, 2.0]])
    shift = np.array([7.0, -11.0])
    transformed_points = points @ linear.T + shift
    base = minimum_enclosing_ellipse(points)
    transformed = minimum_enclosing_ellipse(transformed_points)
    _assert_certified(base, points, 1e-3)
    _assert_certified(transformed, transformed_points, 1e-3)
    np.testing.assert_allclose(transformed.center, linear @ base.center + shift, rtol=0, atol=2e-11)
    inverse = np.linalg.inv(linear)
    expected_shape = inverse.T @ (base.factor.T @ base.factor) @ inverse
    np.testing.assert_allclose(transformed.factor.T @ transformed.factor, expected_shape, rtol=2e-10, atol=2e-12)
    assert transformed.area == pytest.approx(abs(np.linalg.det(linear)) * base.area, rel=2e-10)


def test_nonsymmetric_cloud_has_independently_verified_gap():
    result = minimum_enclosing_ellipse(CLOUD, area_tolerance=1e-2)
    _assert_certified(result, CLOUD, 1e-2)


def test_nonsymmetric_affine_areas_agree_within_the_certified_gap():
    tolerance = 1e-2
    linear = np.array([[1.25, -0.5], [0.25, 2.0]])
    shift = np.array([-4.0, 6.0])
    transformed_points = CLOUD @ linear.T + shift
    base = minimum_enclosing_ellipse(CLOUD, area_tolerance=tolerance)
    transformed = minimum_enclosing_ellipse(transformed_points, area_tolerance=tolerance)
    _assert_certified(base, CLOUD, tolerance)
    _assert_certified(transformed, transformed_points, tolerance)
    # Both feasible areas are between the optimum and (1+tau) times the optimum.
    area_ratio = transformed.area / (abs(np.linalg.det(linear)) * base.area)
    assert 1 / (1 + tolerance) - 1e-12 <= area_ratio <= 1 + tolerance + 1e-12


def test_duplicates_and_permutation_do_not_change_geometry_or_mutate_input():
    original = CLOUD.copy()
    repeated = CLOUD[[5, 2, 0, 1, 4, 3, 0, 2, 5, 5]].copy()
    repeated_before = repeated.copy()
    base = minimum_enclosing_ellipse(CLOUD, area_tolerance=1e-2)
    result = minimum_enclosing_ellipse(repeated, area_tolerance=1e-2)
    _assert_certified(result, repeated, 1e-2)
    np.testing.assert_array_equal(CLOUD, original)
    np.testing.assert_array_equal(repeated, repeated_before)
    np.testing.assert_allclose(result.center, base.center, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.factor.T @ result.factor, base.factor.T @ base.factor, rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(result.weights, base.weights, rtol=0, atol=1e-13)
    assert result.area == pytest.approx(base.area, rel=2e-12)


def test_independent_containment_oracle_rejects_a_shrunken_returned_ellipse():
    result = minimum_enclosing_ellipse(TRIANGLE)
    _assert_certified(result, TRIANGLE, 1e-3)
    shrunken_factor = result.factor * 1.05
    assert not _exact_contains(TRIANGLE, result.center, shrunken_factor).all()


def test_contains_agrees_with_exact_oracle_for_inside_and_outside_queries():
    points = np.array([[-3.0, -2.0], [-3.0, 2.0], [3.0, -2.0], [3.0, 2.0]])
    result = minimum_enclosing_ellipse(points)
    queries = np.vstack([points, [0.0, 0.0], [1.0, 1.0], [20.0, 0.0], [0.0, -20.0]])
    expected = _exact_contains(queries, result.center, result.factor)
    assert expected.tolist() == [True, True, True, True, True, True, False, False]
    np.testing.assert_array_equal(result.contains(queries), expected)


@pytest.mark.parametrize("points", [[[2.5, -3.0]], [[2.5, -3.0], [2.5, -3.0], [2.5, -3.0]]])
def test_single_unique_point_is_an_exact_degenerate_result(points):
    result = minimum_enclosing_ellipse(points)
    assert result.status == "DEGENERATE_POINT"
    assert result.area == 0
    assert result.factor is None
    assert result.major_axis_length == 0
    np.testing.assert_array_equal(result.center, [2.5, -3.0])
    np.testing.assert_array_equal(result.semiaxes, [0, 0])
    np.testing.assert_array_equal(result.unique_points, [[2.5, -3.0]])
    assert [_read_fraction(v) for v in result.certificate["center_exact"]] == [Fraction(5, 2), Fraction(-3)]
    np.testing.assert_array_equal(result.contains([[2.5, -3.0], [2.5, np.nextafter(-3.0, 0.0)]]), [True, False])
    _assert_serializable(result)


@pytest.mark.parametrize(
    "points",
    [[[0.0, 1.0], [8.0, 17.0]], [[2.0, 5.0], [0.0, 1.0], [1.0, 3.0], [8.0, 17.0], [1.0, 3.0]]],
)
def test_collinear_segment_uses_extreme_endpoints_not_the_centroid(points):
    result = minimum_enclosing_ellipse(points)
    assert result.status == "DEGENERATE_SEGMENT"
    assert result.factor is None
    assert result.area == 0
    np.testing.assert_array_equal(result.center, [4.0, 9.0])
    assert [_read_fraction(v) for v in result.certificate["center_exact"]] == [Fraction(4), Fraction(9)]
    assert _read_fraction(result.certificate["squared_length_exact"]) == 320
    assert result.major_axis_length == pytest.approx(math.sqrt(320), rel=1e-14)
    np.testing.assert_allclose(result.semiaxes, [math.sqrt(320) / 2, 0], rtol=1e-14)
    np.testing.assert_array_equal(result.contains(points), np.ones(len(points), dtype=bool))
    np.testing.assert_array_equal(result.contains([[4.0, 9.0], [9.0, 19.0], [4.0, 9.000000000000002]]), [True, False, False])
    _assert_serializable(result)


def test_segment_midpoint_certificate_survives_unrepresentable_float_midpoint():
    x = float(2**53)
    points = [[x, 0.0], [x + 2.0, 2.0]]
    result = minimum_enclosing_ellipse(points)
    assert result.status == "DEGENERATE_SEGMENT"
    assert [_read_fraction(v) for v in result.certificate["center_exact"]] == [Fraction(2**53 + 1), Fraction(1)]
    assert _read_fraction(result.certificate["squared_length_exact"]) == 8
    np.testing.assert_array_equal(result.contains(points), [True, True])
    # The rounded displayed midpoint is not on this exact segment.
    np.testing.assert_array_equal(result.contains([[x, 1.0], [x + 2.0, 1.0]]), [False, False])
    _assert_serializable(result)


def test_exactly_rank_two_near_collinear_input_never_silently_collapses():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, np.nextafter(2.0, math.inf)]])
    exact_cross = _f(points[1, 0]) * _f(points[2, 1]) - _f(points[1, 1]) * _f(points[2, 0])
    assert exact_cross != 0
    result = minimum_enclosing_ellipse(points, area_tolerance=1e-2)
    assert result.status in {"APPROXIMATE_CERTIFIED", "MAX_ITERATIONS", "NUMERICAL_FAILURE"}
    if result.status == "APPROXIMATE_CERTIFIED":
        _assert_certified(result, points, 1e-2)
    else:
        assert result.converged is False
        _assert_serializable(result)


@pytest.mark.parametrize("budget", [{"max_iterations": 0}, {"max_work": 1}])
def test_insufficient_budget_does_not_claim_certified_convergence(budget):
    result = minimum_enclosing_ellipse(CLOUD, area_tolerance=1e-8, **budget)
    assert result.status == "MAX_ITERATIONS"
    assert result.converged is False
    if "max_iterations" in budget:
        assert result.iterations == 0
    _assert_serializable(result)


def test_zero_updates_can_certify_an_already_optimal_symmetric_start():
    result = minimum_enclosing_ellipse(TRIANGLE, max_iterations=0)
    _assert_certified(result, TRIANGLE, 1e-3)
    assert result.iterations == 0


@pytest.mark.parametrize(
    "points",
    [[], [0, 1], [[0, 0], [1]], [[False, 0], [1, 1]], [[0, 0], [math.nan, 1]], [[0, 0], [1, math.inf]], [[0j, 0], [1, 1]], [["0", "1"], ["2", "3"]]],
)
def test_invalid_coordinates_are_rejected(points):
    with pytest.raises(EllipseInputError):
        minimum_enclosing_ellipse(points)


@pytest.mark.parametrize(
    "options",
    [
        {"area_tolerance": math.nan},
        {"area_tolerance": 1e-9},
        {"area_tolerance": 0.251},
        {"max_iterations": -1},
        {"max_iterations": True},
        {"max_iterations": 100001},
        {"max_work": True},
        {"max_work": 100000001},
    ],
)
def test_invalid_accuracy_or_resource_limits_are_rejected(options):
    with pytest.raises(EllipseInputError):
        minimum_enclosing_ellipse(TRIANGLE, **options)


def test_input_size_limit_applies_before_duplicate_elimination():
    with pytest.raises(EllipseInputError):
        minimum_enclosing_ellipse(np.zeros((4097, 2)))


def test_certified_geometry_cannot_be_modified_while_retaining_a_stale_certificate():
    result = minimum_enclosing_ellipse(TRIANGLE)
    before = result.factor.copy()
    with pytest.raises(ValueError):
        result.factor[0, 0] *= 2
    with pytest.raises(ValueError):
        result.factor.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.center = np.zeros(2)
    np.testing.assert_array_equal(result.factor, before)
    _assert_certified(result, TRIANGLE, 1e-3)


def test_certificate_and_serialized_output_do_not_alias_internal_proof():
    result = minimum_enclosing_ellipse(TRIANGLE)
    expected = _read_fraction(result.certificate["area_ratio_squared"])
    external = result.certificate
    external["containment_exact"] = False
    external["area_ratio_squared"]["numerator"] = "0"
    payload = result.to_dict()
    payload["certificate"]["area_ratio_squared"]["numerator"] = "0"
    payload["factor"][0][0] = 1e99
    assert result.certificate["containment_exact"] is True
    assert _read_fraction(result.certificate["area_ratio_squared"]) == expected
    _assert_certified(result, TRIANGLE, 1e-3)


def test_mixed_python_coordinates_reject_integer_that_binary64_would_round():
    # np.asarray(..., dtype=float) would erase the distinction between the first
    # two x coordinates, so validation must inspect the original Python values.
    points = [[2**53 + 1, 0.0], [2**53, 1.0], [2**53 + 2, 0.0]]
    with pytest.raises(EllipseInputError):
        minimum_enclosing_ellipse(points)


def test_exactly_representable_large_integer_is_accepted():
    coordinate = 2**54
    result = minimum_enclosing_ellipse([[coordinate, 0.0]])
    assert result.status == "DEGENERATE_POINT"
    np.testing.assert_array_equal(result.center, [float(coordinate), 0.0])
    assert [_read_fraction(v) for v in result.certificate["center_exact"]] == [Fraction(coordinate), Fraction(0)]
    _assert_serializable(result)


def test_overflowing_tolerance_raises_domain_error_not_overflow_error():
    with pytest.raises(EllipseInputError):
        minimum_enclosing_ellipse(TRIANGLE, area_tolerance=10**400)


def test_subnormal_segment_keeps_exact_length_and_marks_unrepresentable_half_axis():
    tiny = float.fromhex("0x0.0000000000001p-1022")
    points = [[0.0, 0.0], [tiny, 0.0]]
    result = minimum_enclosing_ellipse(points)
    assert result.status == "DEGENERATE_SEGMENT"
    assert result.major_axis_length == tiny
    assert result.major_axis_length > 0
    assert result.semiaxes is None, "an underflowed half-axis must not be reported as zero"
    squared_length = _read_fraction(result.certificate["squared_length_exact"])
    assert squared_length == _f(tiny) ** 2
    assert squared_length > 0
    assert [_read_fraction(v) for v in result.certificate["center_exact"]] == [_f(tiny) / 2, Fraction(0)]
    np.testing.assert_array_equal(result.contains(points), [True, True])
    np.testing.assert_array_equal(result.contains([[2 * tiny, 0.0], [0.0, tiny]]), [False, False])
    _assert_serializable(result)
    assert result.to_dict()["semiaxes"] is None
