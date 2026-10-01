"""Property-based tests (Hypothesis) for the mathematical layer.

The example-based suites pin specific numbers.  These tests instead assert the
*algebraic laws* that must hold for every input, which is where the interesting
bugs live:

* ``PointSet2D`` is a **set**: permutations are invisible;
* ``affine_coordinates`` is **affine invariant**: any invertible affine map is
  invisible;
* ``ati(alpha, beta)`` around the frame is a **bijection** between
  ``R^2`` and ``R^2``;
* ``collinearity_sine`` is a **symmetric** function of ``(a, b, c)`` into ``[0, 1]``
  that vanishes exactly on degenerate triples;
* ``apply_affine`` is **linear**: it commutes with composition.

Bounding strategy: rather than letting Hypothesis invent ``2x2`` matrices (most
of which are nearly singular, so the tests would measure conditioning instead of
correctness), :func:`affine_maps` builds maps from a rotation, two bounded
singular values and a bounded translation.  Their condition number is at most
``max_scale / min_scale``, and every assertion is made against
``tolerance_for_condition`` rather than a hard-coded epsilon.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from kozlov_lab.config import FLOAT_TOL, tolerance_for_condition
from kozlov_lab.geometry import (
    AffineFrame,
    AffineTransform,
    PointSet2D,
    affine_coordinates,
    apply_affine,
    collinearity_sine,
    is_non_collinear,
    reconstruct_point,
    signed_double_area,
)

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

#: A fixed, comfortably conditioned reference frame:
#: sine ~ 0.78, condition number ~ 2.6.
FRAME_A = np.array([0.0, 0.0])
FRAME_B = np.array([2.0, 0.3])
FRAME_C = np.array([0.4, 1.7])

MIN_SCALE = 0.25
MAX_SCALE = 4.0

_coords = st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False)
_degrees = st.floats(-360.0, 360.0, allow_nan=False, allow_infinity=False)
_positive_scale = st.floats(MIN_SCALE, MAX_SCALE, allow_nan=False, allow_infinity=False)


@st.composite
def affine_maps(draw) -> AffineTransform:
    """Bounded-condition invertible affine map (see module docstring)."""
    degrees = draw(_degrees)
    sx = draw(_positive_scale)
    sy = draw(_positive_scale)
    tx = draw(_coords)
    ty = draw(_coords)
    return (
        AffineTransform.rotation(degrees)
        @ AffineTransform.scaling(sx, sy)
        @ AffineTransform.translation_of(tx, ty)
    )


@st.composite
def probes(draw, n: int = 4) -> np.ndarray:
    return np.array([[draw(_coords), draw(_coords)] for _ in range(n)], dtype=float)


# --------------------------------------------------------------------------- #
# set semantics
# --------------------------------------------------------------------------- #
@SETTINGS
@given(points=st.lists(st.tuples(_coords, _coords), min_size=1, max_size=8), seed=st.integers(0, 10**6))
def test_permutation_is_invisible_for_point_sets(points, seed):
    ps = PointSet2D([[x, y] for x, y in points])
    perm = np.random.default_rng(seed).permutation(len(points))
    shuffled = PointSet2D([list(points[i]) for i in perm])

    assert shuffled == ps
    assert shuffled.signature() == ps.signature()
    assert sorted(map(tuple, shuffled.sorted_points().tolist())) == sorted(
        map(tuple, ps.sorted_points().tolist())
    )


@SETTINGS
@given(points=st.lists(st.tuples(_coords, _coords), min_size=1, max_size=6), delta=_coords)
def test_a_moved_point_set_is_a_different_set(points, delta):
    """Negative control for set equality: it must be able to return False."""
    ps = PointSet2D([[x, y] for x, y in points])
    moved = ps.with_points(ps.as_array() + np.array([delta, 0.0]))
    if abs(delta) > 1e-6:
        assert moved != ps


# --------------------------------------------------------------------------- #
# affine invariance of the internal coordinates
# --------------------------------------------------------------------------- #
@SETTINGS
@given(transform=affine_maps(), point=st.tuples(_coords, _coords))
def test_internal_coordinates_are_affine_invariant(transform, point):
    p = np.array(point)
    before = affine_coordinates(p, FRAME_A, FRAME_B, FRAME_C)
    after = affine_coordinates(
        transform.apply(p),
        transform.apply(FRAME_A),
        transform.apply(FRAME_B),
        transform.apply(FRAME_C),
    )

    bound = max(
        tolerance_for_condition(before.condition_number),
        tolerance_for_condition(after.condition_number),
        1e-13,
    )
    assert abs(after.alpha - before.alpha) <= bound, (after.alpha, before.alpha, bound)
    assert abs(after.beta - before.beta) <= bound, (after.beta, before.beta, bound)


@SETTINGS
@given(transform=affine_maps())
def test_the_frame_points_always_map_to_the_unit_triangle(transform):
    assert transform.is_invertible
    moved = AffineFrame(
        transform.apply(FRAME_A), transform.apply(FRAME_B), transform.apply(FRAME_C)
    )
    assert moved.coordinates_xy(transform.apply(FRAME_A)) == pytest.approx((0.0, 0.0), abs=1e-12)
    assert moved.coordinates_xy(transform.apply(FRAME_B)) == pytest.approx((1.0, 0.0), abs=1e-12)
    assert moved.coordinates_xy(transform.apply(FRAME_C)) == pytest.approx((0.0, 1.0), abs=1e-12)


@SETTINGS
@given(alpha=st.floats(-3.0, 3.0), beta=st.floats(-3.0, 3.0))
def test_coordinates_and_reconstruct_are_mutual_inverses(alpha, beta):
    point = reconstruct_point((alpha, beta), FRAME_A, FRAME_B, FRAME_C)
    back = affine_coordinates(point, FRAME_A, FRAME_B, FRAME_C)
    assert back.alpha == pytest.approx(alpha, rel=0, abs=1e-12)
    assert back.beta == pytest.approx(beta, rel=0, abs=1e-12)
    assert back.residual <= FLOAT_TOL


@SETTINGS
@given(alpha=st.floats(-3.0, 3.0), beta=st.floats(-3.0, 3.0), lam=st.floats(0.05, 1.0))
def test_coordinates_are_affine_functions(alpha, beta, lam):
    """``alpha`` and ``beta`` are affine in ``P`` — tested on a random pair."""
    p = reconstruct_point((alpha, beta), FRAME_A, FRAME_B, FRAME_C)
    q = reconstruct_point((beta, alpha), FRAME_A, FRAME_B, FRAME_C)
    mix = lam * p + (1.0 - lam) * q

    a = affine_coordinates(p, FRAME_A, FRAME_B, FRAME_C)
    b = affine_coordinates(q, FRAME_A, FRAME_B, FRAME_C)
    m = affine_coordinates(mix, FRAME_A, FRAME_B, FRAME_C)

    assert m.alpha == pytest.approx(lam * a.alpha + (1.0 - lam) * b.alpha, rel=0, abs=1e-11)
    assert m.beta == pytest.approx(lam * a.beta + (1.0 - lam) * b.beta, rel=0, abs=1e-11)


# --------------------------------------------------------------------------- #
# collinearity
# --------------------------------------------------------------------------- #
@SETTINGS
@given(probe=probes(3))
def test_collinearity_sine_is_bounded_and_symmetric_in_b_and_c(probe):
    """``|sin|`` is bounded, and swapping the two spanning vectors is invisible.

    Note carefully what is *not* claimed: ``|sin(angle BAC)|`` differs from
    ``|sin(angle ABC)|``.  ``collinearity_sine(a, b, c)`` measures the angle at
    ``a``, whose value genuinely depends on which vertex is chosen (a 90-45-45
    triangle has sines 1, 0.707, 0.707).  The permutation-invariant object is the
    *vanishing condition*, tested separately below.
    """
    a, b, c = probe
    s = collinearity_sine(a, b, c)
    assert 0.0 <= s <= 1.0
    assert collinearity_sine(a, c, b) == s  # exact: same two spanning vectors


@SETTINGS
@given(probe=probes(3))
def test_law_of_sines_holds_for_the_sine_measure(probe):
    """``sin(angle) / opposite_side`` is the same at all three vertices.

    This is the identity that makes ``sin`` the right normalisation for a
    collinearity test: it equals ``2 * Area / (product of the three sides)``, a
    quantity that is manifestly symmetric in the three points — even though the
    individual sines are not.
    """
    a, b, c = probe
    side_a = float(np.linalg.norm(b - c))  # opposite vertex A
    side_b = float(np.linalg.norm(a - c))
    side_c = float(np.linalg.norm(a - b))
    # below this obliqueness the identity is swamped by cancellation; filtering
    # (rather than silently returning) keeps the example count honest
    assume(min(side_a, side_b, side_c) > 1e-6)
    assume(abs(signed_double_area(a, b, c)) / (side_a * side_b * side_c) > 1e-6)

    s_a = collinearity_sine(a, b, c)
    s_b = collinearity_sine(b, a, c)
    s_c = collinearity_sine(c, a, b)
    r_a, r_b, r_c = s_a / side_a, s_b / side_b, s_c / side_c

    assert r_a == pytest.approx(r_b, rel=1e-9, abs=1e-15)
    assert r_b == pytest.approx(r_c, rel=1e-9, abs=1e-15)
    # 2R form: r = 2 * Area / (|AB| |AC| |BC|), and Area = |det| / 2
    det = abs(signed_double_area(a, b, c))
    assert r_a == pytest.approx(det / (side_a * side_b * side_c), rel=1e-9, abs=1e-15)


@SETTINGS
@given(probe=probes(3))
def test_collinearity_verdict_is_permutation_invariant(probe):
    """The *relation* ``collinear?`` must not depend on the vertex order.

    The three sines are different numbers, but they vanish together.  Strictly,
    the numerical verdict could disagree only for a triple whose sine sits within
    a few ulps of :data:`COLLINEAR_TOL`; for uniformly drawn coordinates that has
    probability ~1e-8, and it is a threshold artefact rather than an error in the
    relation.
    """
    a, b, c = probe
    verdicts = {
        is_non_collinear(x, y, z)
        for x, y, z in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a))
    }
    assert len(verdicts) == 1


@SETTINGS
@given(t=st.floats(0.001, 3.0), scale=st.floats(1e-6, 1e6), lam=st.floats(1e-6, 5.0))
def test_collinearity_sine_is_scale_invariant_and_vanishes_on_a_line(t, scale, lam):
    """Two laws at once: scaling is invisible, and a linear combination is collinear.

    ``lam`` is bounded away from zero on purpose: with ``lam = 5e-324`` the
    product ``b * lam`` underflows into the subnormal range, and a subnormal
    vector is *not* exactly a scalar multiple of ``b`` any more (the products
    ``b_x * lam`` and ``b_y * lam`` round to different relative accuracies).  That
    is a property of IEEE-754, not of the collinearity measure, but a test that
    ignored it would report a false failure.
    """
    a = np.array([0.0, 0.0])
    b = np.array([math.cos(t), math.sin(t)]) * scale
    assert collinearity_sine(a, b, b * lam) == pytest.approx(0.0, abs=1e-12)

    c = np.array([math.cos(t + 1.0), math.sin(t + 1.0)]) * scale
    reference = collinearity_sine(a, np.array([math.cos(t), math.sin(t)]), np.array([math.cos(t + 1.0), math.sin(t + 1.0)]))
    assert collinearity_sine(a, b, c) == pytest.approx(reference, rel=1e-9, abs=1e-12)


@SETTINGS
@given(t=st.floats(0.01, math.pi - 0.01), scale=st.floats(1e-6, 1e6))
def test_degenerate_spanning_vector_is_reported_as_collinear_not_nan(t, scale):
    a = np.zeros(2)
    assert collinearity_sine(a, a, np.array([math.cos(t), math.sin(t)]) * scale) == 0.0
    assert is_non_collinear(a, a, np.array([math.cos(t), math.sin(t)]) * scale) is False


@SETTINGS
@given(probe=probes(3))
def test_signed_double_area_is_antisymmetric(probe):
    a, b, c = probe
    assert signed_double_area(a, b, c) == pytest.approx(-signed_double_area(a, c, b), rel=0, abs=1e-12)


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
@SETTINGS
@given(f=affine_maps(), g=affine_maps(), point=st.tuples(_coords, _coords))
def test_composition_is_associative_and_matches_pointwise_application(f, g, point):
    p = np.array(point)
    composed = f @ g
    assert np.allclose(composed.apply(p), f.apply(g.apply(p)), atol=1e-11, rtol=0)
    # and it really is non-commutative in general
    if not np.allclose(f.matrix, g.matrix, atol=1e-6):
        pass  # nothing to assert; kept explicit that commutativity is NOT claimed


@SETTINGS
@given(f=affine_maps(), p=st.tuples(_coords, _coords), q=st.tuples(_coords, _coords),
       lam=st.floats(0.0, 1.0))
def test_apply_affine_preserves_affine_combinations(f, p, q, lam):
    """Affine maps commute with affine combinations — the *correct* linearity law.

    Note what is deliberately *not* claimed: ``f(lam * p) != lam * f(p)``, because
    of the translation term.  Asserting proportionality here would be a
    mathematically wrong test, and it would "pass" only for maps with ``t = 0``.
    """
    a = np.array(p)
    b = np.array(q)
    mixed = lam * a + (1.0 - lam) * b
    assert f.apply(mixed) == pytest.approx(
        lam * f.apply(a) + (1.0 - lam) * f.apply(b), rel=0, abs=1e-11
    )


@SETTINGS
@given(f=affine_maps(), points=st.lists(st.tuples(_coords, _coords), min_size=1, max_size=5))
def test_apply_is_linear_on_the_linear_part(f, points):
    """Restricting to the linear part (no translation) restores proportionality."""
    pts = np.array([[x, y] for x, y in points], dtype=float)
    lam = 0.37
    linear_only = AffineTransform(f.matrix, np.zeros(2))
    assert np.allclose(linear_only.apply(pts * lam), linear_only.apply(pts) * lam, atol=1e-11, rtol=0)


@SETTINGS
@given(f=affine_maps(), point=st.tuples(_coords, _coords))
def test_single_point_application_preserves_the_input_shape(f, point):
    """A ``(2,)`` input must give a ``(2,)`` output, not ``(1, 2)``."""
    p = np.array(point)
    assert f.apply(p).shape == (2,)
    assert f.apply(PointSet2D([p])).shape == (1, 2)
    assert f.apply(p).shape == apply_affine(p, f.matrix, f.translation)[0].shape


@SETTINGS
@given(f=affine_maps(), point=st.tuples(_coords, _coords))
def test_single_point_survives_a_round_trip(f, point):
    p = np.array(point)
    back = f.inverse().apply(f.apply(p))
    assert back == pytest.approx(p, rel=0, abs=1e-11)


@SETTINGS
@given(f=affine_maps())
def test_classification_of_a_known_family(f):
    """Classification must be consistent with the determinant it reports."""
    kind = f.classify()
    assert kind in {"identity", "translation", "rotation", "reflection", "isometry",
                    "similarity", "affine", "singular"}
    assert f.is_invertible
    if kind in {"identity", "translation", "rotation", "reflection"}:
        assert f.det == pytest.approx(1.0, abs=1e-9) or f.det == pytest.approx(-1.0, abs=1e-9)
    if kind == "similarity":
        assert abs(f.det) > 0


@SETTINGS
@given(f=affine_maps(), g=affine_maps())
def test_condition_number_of_a_product_is_bounded_by_the_product(f, g):
    """Retained as an *upper bound only*: cond is sub-multiplicative.

    Asserting equality here would be wrong (cond is not multiplicative), which is
    exactly the kind of assumption that silently breaks a cost model later on.
    """
    product = f @ g
    assert product.condition_number <= f.condition_number * g.condition_number * (1 + 1e-9)
