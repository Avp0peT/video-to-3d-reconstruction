"""Phase 1 acceptance tests for the geometric transformation layer (README section 5).

The README fixes, for each family, what must be *verified* rather than merely
asserted:

* isometry   -> ``||x_i - x_j|| == ||x'_i - x'_j||``
* similarity -> length **ratios** preserved
* affine     -> ``det A != 0``; collinearity and parallelism preserved

Each of those preservation claims is paired here with a **negative control**
(a transform under which the claim fails), so that a green suite really means
"the laboratory measures a difference" and not "everything returned True".
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kozlov_lab.config import FLOAT_TOL
from kozlov_lab.datasets import shapes
from kozlov_lab.geometry import (
    AffineTransform,
    PointSet2D,
    apply_affine,
    collinearity_relation_change,
    parallelism_relation_change,
)


# --------------------------------------------------------------------------- #
# core API
# --------------------------------------------------------------------------- #
def test_apply_affine_matches_manual_matrix_product():
    pts = np.array([[1.0, 2.0], [3.0, -1.0]])
    mat = np.array([[2.0, 0.5], [-1.0, 3.0]])
    tvec = np.array([10.0, -4.0])

    got = apply_affine(pts, mat, tvec)
    expected = pts @ mat.T + tvec
    np.testing.assert_allclose(got, expected, atol=0, rtol=0)

    # PointSet2D input is accepted and yields a plain array (no silent aliasing)
    got2 = apply_affine(PointSet2D(pts), mat, tvec)
    np.testing.assert_allclose(got2, expected, atol=0, rtol=0)
    assert got2 is not pts


def test_apply_affine_rejects_bad_shapes():
    with pytest.raises(ValueError):
        apply_affine([[0, 0]], [[1, 2, 3], [4, 5, 6]], [0, 0])
    with pytest.raises(ValueError):
        apply_affine([[0, 0]], [[1, 0], [0, 1]], [1, 2, 3])


def test_empty_point_list_is_allowed_and_returns_empty():
    out = apply_affine([], np.eye(2), [0, 0])
    assert out.shape == (0, 2)


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("identity", "identity"),
        ("translation", "translation"),
        ("rotation", "rotation"),
        ("rotation_about_center", "rotation"),
        ("reflection", "reflection"),
        ("similarity", "similarity"),
        ("scale_nonuniform", "affine"),
        ("shear", "affine"),
        ("shear_both", "affine"),
        ("affine_general", "affine"),
        ("singular", "singular"),
    ],
)
def test_classification(transform_bank, name, expected):
    assert transform_bank[name].classify() == expected


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("identity", "isometry"),
        ("translation", "isometry"),
        ("rotation", "isometry"),
        ("reflection", "isometry"),
        ("similarity", "similarity"),
        ("shear", "affine"),
        ("affine_general", "affine"),
        ("singular", "singular"),
    ],
)
def test_invariant_group(transform_bank, name, group):
    assert transform_bank[name].invariant_group() == group


def test_nonuniform_scaling_is_not_a_similarity():
    """A negative control for ``classify``: 2x/0.5 scaling is affine, not similar."""
    t = AffineTransform.scaling(2.0, 0.5)
    assert t.classify() == "affine"
    assert t.is_invertible  # det = 1


# --------------------------------------------------------------------------- #
# invariances, each with its negative control
# --------------------------------------------------------------------------- #
def test_rotation_preserves_all_distances(triangle_set):
    t = AffineTransform.rotation(37.0, center=[0.4, -0.2])
    x = triangle_set.as_array()
    y = t.apply(x)
    d0 = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    d1 = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1)
    np.testing.assert_allclose(d1, d0, rtol=0, atol=1e-12)


def test_nonuniform_scaling_breaks_distances_but_keeps_ratios_along_one_axis():
    """Negative control for the isometry claim, positive control for 'affine'."""
    t = AffineTransform.scaling(3.0, 1.0)
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    y = t.apply(x)
    d0 = np.linalg.norm(x[0] - x[1])
    d1 = np.linalg.norm(y[0] - y[1])
    assert d1 == pytest.approx(3.0 * d0)
    assert collinearity_relation_change(x, y)[0] == frozenset()
    assert collinearity_relation_change(x, y)[1] == frozenset()


def test_similarity_preserves_angles(triangle_set):
    t = AffineTransform.similarity(3.5, 71.0, center=[1.0, 0.0])
    x = triangle_set.as_array()
    y = t.apply(x)
    for (i, j, k) in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
        np.testing.assert_allclose(_angle(x, i, j, k), _angle(y, i, j, k), atol=1e-12)


def test_shear_breaks_angles_but_keeps_collinearity_and_parallelism(square_set):
    """The canonical demonstration of README section 5.

    ``正方形 -> 仿射变换 -> 平行四边形``: right angles are destroyed, while
    collinearity and the parallel relation survive.
    """
    t = AffineTransform.shear(0.8, 0.0)
    x = square_set.as_array()
    y = t.apply(x)

    # collinearity: the relation sets must be identical AND non-empty
    only_x, only_y, n_rel = collinearity_relation_change(x, y)
    assert n_rel > 0, "fixture must contain collinear triples, otherwise this test is vacuous"
    assert only_x == frozenset() and only_y == frozenset()

    # parallelism: same, and non-empty thanks to the gridded square
    only_x, only_y, n_rel = parallelism_relation_change(x, y)
    assert n_rel > 0, "fixture must contain parallel segment pairs"
    assert only_x == frozenset() and only_y == frozenset()

    # negative control: the *angle* really did change.
    # indices (0, 3, 11) = (0,0), (2,0), (0,2/3): a clean right angle that the
    # shear visibly destroys, unlike a vertex triple that lies on one side.
    assert abs(_angle(x, 0, 3, 11) - _angle(y, 0, 3, 11)) > 0.1


def test_affine_map_does_not_preserve_collinearity_when_it_is_not_affine():
    """Falsifiability check: compare a shape against an *unrelated* shape.

    The collinearity/parallelism relation must be reported as broken.  Without
    this test, a buggy implementation that always returns "no witnesses" would
    pass every preservation test above.
    """
    x = shapes.square(2.0, n_per_side=3).as_array()
    y = shapes.grid(4, 4).as_array()[: x.shape[0]]
    only_x, only_y, n_rel = collinearity_relation_change(x, y)
    assert n_rel > 0
    assert len(only_x) + len(only_y) > 0


# --------------------------------------------------------------------------- #
# algebra: composition and inverse
# --------------------------------------------------------------------------- #
def test_composition_is_matrix_like(transform_bank, triangle_set):
    f = transform_bank["affine_general"]
    g = transform_bank["rotation"]
    x = triangle_set.as_array()
    np.testing.assert_allclose(
        (f @ g).apply(x), f.apply(g.apply(x)), rtol=0, atol=1e-12
    )
    # and it is NOT commutative in general
    assert not np.allclose((g @ f).matrix, (f @ g).matrix)


@pytest.mark.parametrize(
    "name",
    ["translation", "rotation", "rotation_about_center", "reflection", "similarity",
     "scale_nonuniform", "shear", "shear_both", "affine_general"],
)
def test_inverse_round_trip_is_exact(transform_bank, name, grid_set):
    t = transform_bank[name]
    x = grid_set.as_array()
    back = t.inverse().apply(t.apply(x))
    np.testing.assert_allclose(back, x, rtol=0, atol=1e-11)


def test_singular_transform_cannot_be_inverted(transform_bank):
    assert not transform_bank["singular"].is_invertible
    with pytest.raises(ValueError):
        transform_bank["singular"].inverse()


def test_shear_is_singular_exactly_when_kx_ky_equals_one():
    assert AffineTransform.shear(2.0, 0.5).det == pytest.approx(0.0, abs=1e-15)
    assert not AffineTransform.shear(2.0, 0.5).is_invertible
    assert AffineTransform.shear(2.0, 0.4).is_invertible


def test_identity_is_neutral_element(transform_bank, triangle_set):
    x = triangle_set.as_array()
    ident = AffineTransform.identity()
    np.testing.assert_allclose((ident @ ident).matrix, np.eye(2))
    np.testing.assert_allclose(ident.apply(x), x, atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# control points (used later by the recognition layer)
# --------------------------------------------------------------------------- #
def test_from_control_points_reproduces_the_generating_transform(grid_set):
    src = grid_set.as_array()
    # (0, 1, 4) on a 4x4 grid is (0,0), (1,0), (0,1): non-collinear, and *not*
    # the diagonal triple (0, 5, 10) which is collinear and would be rejected.
    control = src[[0, 1, 4]]
    t = AffineTransform.random_invertible(12345, max_condition=20.0)
    dst = t.apply(src)
    recovered = AffineTransform.from_control_points(control, t.apply(control))
    np.testing.assert_allclose(recovered.matrix, t.matrix, rtol=0, atol=1e-12)
    np.testing.assert_allclose(recovered.translation, t.translation, rtol=0, atol=1e-12)
    np.testing.assert_allclose(recovered.apply(src), dst, rtol=0, atol=1e-11)


def test_from_control_points_rejects_collinear_source():
    with pytest.raises(ValueError):
        AffineTransform.from_control_points(
            [[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]]
        )


def test_from_control_points_requires_exactly_three_points():
    with pytest.raises(ValueError):
        AffineTransform.from_control_points([[0, 0], [1, 0]], [[0, 0], [1, 0]])


# --------------------------------------------------------------------------- #
# random transform generator — the substrate of Experiment A
# --------------------------------------------------------------------------- #
def test_random_invertible_respects_condition_budget():
    """The budget is a hard bound, and it is not achieved by scaling.

    A tempting-but-wrong implementation divides the whole matrix by a scalar;
    that leaves the condition number untouched (``cond = s_max / s_min``, and a
    scalar cancels).  The bound is therefore checked directly on
    ``condition_number``, and separately the *ratio* of the singular values.
    """
    g = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        t = AffineTransform.random_invertible(g, max_condition=50.0)
        assert t.is_invertible
        cond = t.condition_number
        worst = max(worst, cond)
        assert cond <= 50.0 * (1 + 1e-6), f"condition number {cond} exceeded the budget"
    # the generator actually explores the budget instead of emitting near-identity maps
    assert worst > 5.0, f"worst condition number was only {worst}; generator is too tame"


@pytest.mark.parametrize("budget", [1.0, 1.5, 3.0, 10.0, 1000.0])
def test_condition_budget_holds_exactly_for_every_budget(budget):
    g = np.random.default_rng(int(budget * 1000))
    for _ in range(100):
        t = AffineTransform.random_invertible(
            g, max_condition=budget, scale_range=(1.0, 1.0 + budget), shear_range=(-0.9, 0.9)
        )
        assert t.condition_number <= budget * (1 + 1e-6), (budget, t.condition_number)
        # the singular values are the object being controlled, not the entries
        singular = t.singular_values
        assert singular[0] / singular[-1] <= budget * (1 + 1e-6)


def test_random_invertible_is_deterministic_given_a_seed():
    a = AffineTransform.random_invertible(42).matrix
    b = AffineTransform.random_invertible(42).matrix
    np.testing.assert_array_equal(a, b)
    c = AffineTransform.random_invertible(43).matrix
    assert not np.array_equal(a, c)


def test_parameters_and_homogeneous_form():
    t = AffineTransform(np.array([[2.0, 1.0], [0.0, 3.0]]), np.array([5.0, -1.0]))
    params = t.parameters()
    assert params["a12"] == 1.0 and params["tx"] == 5.0
    assert params["det"] == pytest.approx(6.0)
    np.testing.assert_allclose(t.as_matrix3(), [[2, 1, 5], [0, 3, -1], [0, 0, 1]])


def test_transform_json_round_trip(transform_bank):
    for name, t in transform_bank.items():
        back = AffineTransform.from_json(t.to_json())
        assert back == t, name


def test_apply_set_preserves_identity_and_records_the_transform(grid_set):
    t = AffineTransform.rotation(30.0)
    moved = t.apply_set(grid_set)
    assert moved.name == f"T({grid_set.name})"
    assert moved.meta["apply_affine_matrix"] == t.matrix.tolist()
    assert grid_set.name != moved.name  # original untouched


def test_equality_uses_tolerance_not_identity():
    a = AffineTransform(np.eye(2), np.zeros(2))
    b = AffineTransform(np.eye(2) + 1e-15, np.zeros(2))
    assert a == b
    c = AffineTransform(np.eye(2) + 1e-6, np.zeros(2))
    assert a != c
    assert (a == "not a transform") is False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _angle(points: np.ndarray, vertex: int, p: int, q: int) -> float:
    """Stable angle via ``atan2(|u x v|, u.v)`` — see ``invariants._angle_at``."""
    u = points[p] - points[vertex]
    v = points[q] - points[vertex]
    cross = abs(float(u[0] * v[1] - u[1] * v[0]))
    dot = float(u[0] * v[0] + u[1] * v[1])
    return math.atan2(cross, dot)


def test_helper_sanity():
    """Guard the test helper itself: a wrong helper would silently pass everything."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert _angle(square, 0, 1, 3) == pytest.approx(math.pi / 2)
    assert _angle(square, 0, 1, 2) == pytest.approx(math.pi / 4)
    assert FLOAT_TOL < 1e-6
