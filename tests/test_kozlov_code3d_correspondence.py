"""Numerical geometric witnesses for unordered 3D point sets, not indexed codes."""

from __future__ import annotations

import numpy as np
import pytest

from kozlov_lab.stereo.official.kozlov_code3d import _point_bijection, is_a_equivalent


TETRAHEDRON = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
IRREGULAR = np.vstack([TETRAHEDRON, [2, 3, 5], [-2, 1, 4]])


def _assert_witness(first, second, result):
    assert result["equivalent"] is True
    assert not result.get("undecided", False)
    permutation = result["permutation"]
    assert sorted(permutation) == list(range(len(first)))
    matrix = np.asarray(result["matrix"])
    translation = np.asarray(result["translation"])
    assert abs(np.linalg.det(matrix)) > 1e-10
    residual = np.abs(first @ matrix.T + translation - second[permutation]).max()
    assert residual <= result["tolerance"]
    assert result["max_residual"] == pytest.approx(residual)


@pytest.mark.parametrize("source_order", [list(range(6)), [4, 2, 0, 5, 1, 3]])
@pytest.mark.parametrize("reflection", [1.0, -1.0])
def test_shuffled_affine_body_has_a_complete_bijection_witness(source_order, reflection):
    matrix = np.array([[2, 0.5, 0], [0, 1, -0.25], [0, 0, reflection]])
    first = IRREGULAR[source_order]
    second = (IRREGULAR @ matrix.T + [3, -2, 1])[[5, 2, 4, 0, 3, 1]]
    _assert_witness(first, second, is_a_equivalent(first, second))


def test_symmetric_tetrahedron_relabeling_returns_a_witness():
    second = TETRAHEDRON[[2, 0, 3, 1]]
    _assert_witness(TETRAHEDRON, second, is_a_equivalent(TETRAHEDRON, second))


def test_same_cardinality_with_different_convex_hulls_is_not_equivalent():
    # Invertible affine maps preserve extreme points: A has four, B has five.
    first = np.vstack([TETRAHEDRON, [0.2, 0.2, 0.2]])
    second = np.vstack([TETRAHEDRON, [2, 2, 2]])[[4, 2, 1, 0, 3]]
    assert is_a_equivalent(first, second)["equivalent"] is False
    assert is_a_equivalent(second, first)["equivalent"] is False


def test_point_matching_can_reassign_a_previous_match():
    first = np.array([[0.1, 0, 0], [0, 0, 0]])
    second = np.array([[0, 0, 0], [0.2, 0, 0]])
    # A greedy first match consumes B[0], but the valid bijection is [1, 0].
    assert _point_bijection(first, second, 0.11) == [1, 0]


def test_nearby_points_on_both_sides_do_not_establish_a_bijection():
    first = np.zeros((6, 3))
    second = np.zeros((6, 3))
    first[:, 0] = [0, 0.05, 0.1, 10, 10.05, 10.1]
    second[:, 0] = [0, 0.05, 10, 10.025, 10.075, 10.1]
    # Every point has a nearby neighbour, but the first three share only two targets.
    assert _point_bijection(first, second, 0.11) is None


@pytest.mark.parametrize("body", [np.empty((0, 3)), TETRAHEDRON[:3],
                                  np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
                                  np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])])
def test_no_full_rank_control_is_an_abstention_not_a_positive(body):
    result = is_a_equivalent(body, body[::-1])
    assert result["undecided"] is True
    assert result["equivalent"] is None
    assert "matrix" not in result and "permutation" not in result


def test_rank_collapse_cannot_be_an_invertible_affine_equivalence():
    flattened = IRREGULAR.copy()
    flattened[:, 2] = 0
    assert is_a_equivalent(IRREGULAR, flattened)["equivalent"] is False
    assert is_a_equivalent(flattened, IRREGULAR)["equivalent"] is False


def test_caller_tolerance_controls_the_numeric_residual_gate():
    second = IRREGULAR.copy()
    second[5, 0] += 1e-7
    assert is_a_equivalent(IRREGULAR, second, tol=1e-10)["equivalent"] is False
    _assert_witness(IRREGULAR, second, is_a_equivalent(IRREGULAR, second, tol=1e-6))


def test_different_cardinality_cannot_have_a_bijection():
    assert is_a_equivalent(IRREGULAR, TETRAHEDRON)["equivalent"] is False


@pytest.mark.parametrize("tol", [-1, float("nan"), float("inf")])
def test_invalid_tolerance_is_rejected(tol):
    with pytest.raises(ValueError, match="tol"):
        is_a_equivalent(IRREGULAR, IRREGULAR, tol=tol)


def test_nonfinite_points_are_rejected():
    invalid = IRREGULAR.copy()
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        is_a_equivalent(invalid, IRREGULAR)
