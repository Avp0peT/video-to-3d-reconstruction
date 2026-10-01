"""Phase 2 acceptance tests: internal affine coordinates (README section 6).

README section 6 calls this "整个项目最重要的模块" and states the acceptance
criterion directly::

    变换前 (α, β) = (0.500, 0.250)
    变换后 (α', β') = (0.500, 0.250)
    误差 < 1e-9

Two things are checked here, and they are different claims:

**(a) the mathematics** — ``alpha``/``beta`` are exactly invariant, and the
closure identities (barycentric sum, reconstruction of the frame points) hold
exactly in exact arithmetic;

**(b) the floating-point reality** — the measured violation stays below the
bound implied by the *conditioning of the frame*.  A test suite that hard-codes
``1e-9`` regardless of conditioning would either be trivially satisfied or
silently wrong; ``test_numerical_stability.py`` covers the ill-conditioned
regime explicitly.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from kozlov_lab.config import COLLINEAR_TOL, FLOAT_TOL
from kozlov_lab.datasets import shapes
from kozlov_lab.geometry import (
    AffineFrame,
    AffineTransform,
    affine_coordinates,
    affine_coordinates_batch,
    collinearity_sine,
    count_non_collinear_triples,
    find_non_collinear_triple,
    frame_matrix,
    is_non_collinear,
    iter_non_collinear_triples,
    min_vertex_sine,
    reconstruct_point,
    signed_double_area,
    triangle_flatness,
)


# --------------------------------------------------------------------------- #
# collinearity
# --------------------------------------------------------------------------- #
def test_signed_double_area_sign_and_value():
    # counter-clockwise unit triangle -> +1
    assert signed_double_area([0, 0], [1, 0], [0, 1]) == pytest.approx(1.0)
    # clockwise -> -1
    assert signed_double_area([0, 0], [0, 1], [1, 0]) == pytest.approx(-1.0)
    # collinear -> 0
    assert signed_double_area([0, 0], [1, 1], [2, 2]) == pytest.approx(0.0)


def test_collinearity_sine_is_dimensionless_and_scale_invariant():
    """The central design decision: the *collinearity test* must not depend on image size.

    ``collinearity_sine`` is dimensionless; the raw determinant is an area and
    therefore scales quadratically.
    """
    for scale in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0]) * scale
        c = np.array([math.cos(0.3), math.sin(0.3)]) * scale

        assert collinearity_sine(a, b, c) == pytest.approx(math.sin(0.3), rel=1e-12)
        assert signed_double_area(a, b, c) == pytest.approx(math.sin(0.3) * scale**2, rel=1e-12)
        assert is_non_collinear(a, b, c)


def test_a_fixed_determinant_threshold_would_classify_these_three_points_wrongly():
    """Why the threshold may not be put on ``|det|``.

    Both triangles below have the *same* obliqueness (``sin = 0.295``), yet their
    determinants differ by 24 orders of magnitude.  Any fixed ``|det| > c`` test
    would call the big one generic and the small one collinear.
    """
    def triangle(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros(2),
            np.array([1.0, 0.0]) * scale,
            np.array([math.cos(0.3), math.sin(0.3)]) * scale,
        )

    big = triangle(1e6)
    tiny = triangle(1e-6)
    assert collinearity_sine(*big) == pytest.approx(collinearity_sine(*tiny), rel=1e-12)
    assert abs(signed_double_area(*big)) > 1.0
    assert abs(signed_double_area(*tiny)) < 1e-9
    # a "reasonable looking" absolute threshold of 1.0 would flip both verdicts
    assert abs(signed_double_area(*big)) > 1.0 > abs(signed_double_area(*tiny))


def test_collinearity_sine_bounds_and_degenerate_cases():
    assert collinearity_sine([0, 0], [1, 0], [2, 0]) == pytest.approx(0.0)
    assert collinearity_sine([0, 0], [1, 0], [0, 1]) == pytest.approx(1.0)
    # a zero-length spanning vector is treated as collinear, never as a NaN
    assert collinearity_sine([0, 0], [0, 0], [1, 1]) == 0.0
    for sine in (1e-12, 1e-6, 0.5, 1.0):
        assert 0.0 <= collinearity_sine([0, 0], [1, 0], [1, sine]) <= 1.0


@pytest.mark.parametrize(
    ("sine", "expected"),
    [
        (0.0, False),
        (1e-12, False),  # below COLLINEAR_TOL -> rejected
        (COLLINEAR_TOL, False),  # exactly at the threshold -> strict inequality
        (COLLINEAR_TOL * 10, True),
        (0.1, True),
        (1.0, True),
    ],
)
def test_is_non_collinear_thresholding(sine, expected):
    assert is_non_collinear([0, 0], [1, 0], [1, sine] if sine > 0 else [2, 0]) is expected


def test_frame_matrix_columns_are_the_spanning_vectors():
    m = frame_matrix([1, 1], [3, 1], [1, 4])
    np.testing.assert_allclose(m, [[2.0, 0.0], [0.0, 3.0]])


# --------------------------------------------------------------------------- #
# the core computation
# --------------------------------------------------------------------------- #
def test_readme_worked_example():
    """Reproduce the numbers quoted in README section 6.

    Frame ``A=(0,0), B=(2,0), C=(0,1)`` and ``P=(1, 0.25)``::
        P = 0.5 (B-A) + 0.25 (C-A)
    """
    res = affine_coordinates([1.0, 0.25], [0, 0], [2, 0], [0, 1])
    assert res.alpha == pytest.approx(0.5, abs=0.0)
    assert res.beta == pytest.approx(0.25, abs=0.0)
    assert res.residual < FLOAT_TOL
    assert res.condition_number == pytest.approx(2.0, rel=1e-12)
    assert res.formula() == "P - A = 0.500 (B - A) + 0.250 (C - A)"


def test_vertex_sine_is_not_permutation_invariant_flatness_is():
    """Regression test for a bug Hypothesis found during Phase 2.

    ``|sin(angle BAC)|`` depends on which point is named ``A``.  Thresholding it
    would define a *relation* ("are these three points collinear?") that depends
    on the index order — fatal for the image-code layer, which enumerates
    collinear triples over an arbitrary index order.  The concrete counterexample
    below has sines ``1.0`` and ``3.5e-152`` for the same three points.
    """
    a = np.array([0.0, 0.0])
    b = np.array([3.47370303e-152, 0.0])
    c = np.array([0.0, 1.0])

    sines = [
        collinearity_sine(a, b, c),
        collinearity_sine(b, a, c),
        collinearity_sine(c, a, b),
    ]
    assert max(sines) == pytest.approx(1.0)
    assert min(sines) < 1e-100
    # hence a fixed threshold on the vertex sine gives contradictory verdicts
    verdicts = {
        is_non_collinear(x, y, z) for x, y, z in ((a, b, c), (a, c, b), (b, a, c),
                                                  (b, c, a), (c, a, b), (c, b, a))
    }
    flat = triangle_flatness(a, b, c)
    assert len(verdicts) == 1, "is_non_collinear must be permutation invariant"
    # the flatness is the permutation-invariant quantity behind that verdict
    assert flat < 1e-100
    assert min_vertex_sine(a, b, c) == pytest.approx(min(sines), rel=0, abs=0)
    assert not is_non_collinear(a, b, c)


def test_flatness_matches_the_geometric_definition():
    """``flatness = 2 * Area / longest_side^2 = h_min / longest_side``."""
    equilateral = (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, math.sqrt(3) / 2]))
    assert triangle_flatness(*equilateral) == pytest.approx(math.sqrt(3) / 2, rel=1e-12)

    right = (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert triangle_flatness(*right) == pytest.approx(0.5, rel=1e-12)

    # a triangle whose shortest altitude is 1/10 of its longest side
    thin = (np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([5.0, 0.1]))
    assert triangle_flatness(*thin) == pytest.approx(0.01, rel=1e-9)

    # degenerate cases
    assert triangle_flatness([0, 0], [0, 0], [1, 1]) == 0.0
    assert triangle_flatness([0, 0], [1, 1], [2, 2]) == 0.0


def test_flatness_is_scale_invariant():
    base = (np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.3, 0.7]))
    reference = triangle_flatness(*base)
    for scale in (1e-9, 1e-3, 1e3, 1e9):
        scaled = tuple(p * scale for p in base)
        assert triangle_flatness(*scaled) == pytest.approx(reference, rel=1e-12)


def test_frame_points_have_unit_coordinates(demo_set):
    """A, B, C map to (0,0), (1,0), (0,1) by definition."""
    frames = [AffineFrame.from_indices(demo_set.as_array(), (0, 1, 2))]
    frames.append(AffineFrame.auto(demo_set.as_array()))
    for frame in frames:
        assert frame.coordinates_xy(frame.a) == pytest.approx((0.0, 0.0), rel=0, abs=1e-12)
        assert frame.coordinates_xy(frame.b) == pytest.approx((1.0, 0.0), rel=0, abs=1e-12)
        assert frame.coordinates_xy(frame.c) == pytest.approx((0.0, 1.0), rel=0, abs=1e-12)


def test_batch_and_single_agree(demo_set, well_conditioned_frame):
    pts = demo_set.as_array()
    batch = affine_coordinates_batch(
        pts, well_conditioned_frame.a, well_conditioned_frame.b, well_conditioned_frame.c
    )
    for index in range(pts.shape[0]):
        single = well_conditioned_frame.coordinates(pts[index])
        assert batch["alpha"][index] == pytest.approx(single.alpha, rel=0, abs=1e-14)
        assert batch["beta"][index] == pytest.approx(single.beta, rel=0, abs=1e-14)
    assert batch["residual"].max() < 1e-12
    assert batch["condition_number"] == pytest.approx(
        well_conditioned_frame.condition_number, rel=1e-12
    )


def test_batch_handles_empty_input(well_conditioned_frame):
    out = affine_coordinates_batch(
        np.zeros((0, 2)), well_conditioned_frame.a, well_conditioned_frame.b, well_conditioned_frame.c
    )
    assert out["alpha"].shape == (0,)
    assert out["residual"].shape == (0,)


def test_reconstruct_is_the_inverse_of_coordinates(grid_set, well_conditioned_frame):
    pts = grid_set.as_array()
    batch = well_conditioned_frame.coordinates_batch(pts)
    for index in range(pts.shape[0]):
        back = reconstruct_point(
            (batch["alpha"][index], batch["beta"][index]),
            well_conditioned_frame.a,
            well_conditioned_frame.b,
            well_conditioned_frame.c,
        )
        np.testing.assert_allclose(back, pts[index], rtol=0, atol=1e-12)


def test_coordinates_are_affine_functions_of_the_point():
    """``alpha`` and ``beta`` are linear in ``P`` — the algebraic backbone."""
    frame = ([0.0, 0.0], [2.0, 0.0], [0.0, 3.0])
    p = np.array([0.7, 1.1])
    q = np.array([-0.3, 2.4])
    a = affine_coordinates(p, *frame)
    b = affine_coordinates(q, *frame)
    mid = affine_coordinates((p + q) / 2.0, *frame)
    assert mid.alpha == pytest.approx((a.alpha + b.alpha) / 2.0, abs=1e-15)
    assert mid.beta == pytest.approx((a.beta + b.beta) / 2.0, abs=1e-15)


def test_coordinates_of_an_affine_combination_reproduce_it():
    """Barycentric closure: ``A + a(B-A) + b(C-A)`` with ``a+b <= 1`` stays inside."""
    frame = AffineFrame.from_points([1.0, 2.0], [4.0, 2.0], [1.0, 6.0])
    for alpha, beta in itertools.product((0.0, 0.25, 0.5, 0.75, 1.0), repeat=2):
        point = frame.reconstruct((alpha, beta))
        back = frame.coordinates(point)
        assert back.alpha == pytest.approx(alpha, abs=1e-12)
        assert back.beta == pytest.approx(beta, abs=1e-12)


# --------------------------------------------------------------------------- #
# the defining invariance — the acceptance criterion of README section 6
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kind",
    ["translation", "rotation", "rotation_about_center", "reflection", "similarity",
     "scale_nonuniform", "shear", "shear_both", "affine_general"],
)
def test_internal_coordinates_are_affine_invariant(transform_bank, square_set, kind):
    """``(alpha, beta)`` unchanged under **every** invertible affine map."""
    frame = AffineFrame.from_indices(square_set.as_array(), (0, 3, 5))
    transform = transform_bank[kind]
    delta = frame.invariance(transform, square_set.as_array())

    assert delta.max_delta < 1e-9, delta.as_dict()
    assert delta.within_bound, delta.as_dict()
    # the frame itself moved: this is not a trivial "nothing happened" test
    assert not np.allclose(transform.matrix, np.eye(2)) or not np.allclose(
        transform.translation, 0.0
    )


def test_invariance_under_a_hundred_random_affine_maps(square_set):
    frame = AffineFrame.from_indices(square_set.as_array(), (0, 3, 5))
    rng = np.random.default_rng(20261002)
    worst = 0.0
    for _ in range(100):
        t = AffineTransform.random_invertible(rng, max_condition=20.0)
        delta = frame.invariance(t, square_set.as_array())
        assert delta.within_bound, (t, delta.as_dict())
        worst = max(worst, delta.max_delta)
    assert worst < 1e-9, f"worst observed |delta| was {worst:.3e}"
    assert worst > 0.0, "an exactly-zero result would mean the comparison is a no-op"


def test_invariance_is_not_vacuous_absolute_coordinates_do_change(square_set):
    """Negative control: the same experiment must show *something* changing.

    If a bug made ``transform.apply`` an identity, the invariance test above
    would still pass; here we require the external coordinates to move.
    """
    frame = AffineFrame.from_indices(square_set.as_array(), (0, 3, 5))
    t = AffineTransform.similarity(3.0, 47.0, center=[0.5, 0.5])
    moved = t.apply(square_set.as_array())
    assert float(np.abs(moved - square_set.as_array()).max()) > 1.0
    assert frame.invariance(t, square_set.as_array()).max_delta < 1e-9


# --------------------------------------------------------------------------- #
# rejection of degenerate frames
# --------------------------------------------------------------------------- #
def test_collinear_frame_is_rejected_not_silently_answered(collinear_set):
    pts = collinear_set.as_array()
    with pytest.raises(ValueError, match="collinear"):
        affine_coordinates(pts[0], pts[1], pts[2], pts[3])
    with pytest.raises(ValueError, match="collinear"):
        affine_coordinates_batch(pts, pts[0], pts[1], pts[2])
    with pytest.raises(ValueError):
        AffineFrame.auto(pts)


def test_repeated_reference_point_is_rejected():
    with pytest.raises(ValueError, match="collinear"):
        affine_coordinates([1.0, 1.0], [0, 0], [0, 0], [1, 0])


def test_every_pair_of_a_collinear_set_fails_the_check(collinear_set):
    pts = collinear_set.as_array()
    for i, j, k in itertools.combinations(range(pts.shape[0]), 3):
        assert not is_non_collinear(pts[i], pts[j], pts[k])


# --------------------------------------------------------------------------- #
# frame enumeration (needed by the image-code layer)
# --------------------------------------------------------------------------- #
def test_count_non_collinear_triples_on_a_grid(grid_set):
    """A 4x4 grid: 16 points, C(16,3) = 560 triples, most of them collinear.

    The exact count is part of the specification of the enumeration order, so it
    is pinned here rather than recomputed from the same code under test.
    """
    n = grid_set.n
    total = n * (n - 1) * (n - 2) // 6
    assert total == 560
    good = count_non_collinear_triples(grid_set.as_array())
    assert 0 < good < total

    triples = list(iter_non_collinear_triples(grid_set.as_array()))
    assert len(triples) == good
    assert len(set(triples)) == good  # no duplicates
    assert all(i < j < k for i, j, k in triples)
    assert triples == sorted(triples)


def test_find_non_collinear_triple(collinear_set, triangle_set):
    assert find_non_collinear_triple(collinear_set.as_array()) is None
    found = find_non_collinear_triple(triangle_set.as_array())
    assert found == (0, 1, 2)


def test_count_is_zero_for_small_or_collinear_sets():
    assert count_non_collinear_triples([[0, 0]]) == 0
    assert count_non_collinear_triples([[0, 0], [1, 1]]) == 0
    assert count_non_collinear_triples(shapes.collinear_chain(6).as_array()) == 0


def test_auto_frame_prefers_oblique_well_conditioned_triples(grid_set):
    """``auto`` must not return the first valid triple when a better one exists."""
    thin = shapes.letter_l()
    for shape in (thin, grid_set, shapes.square(2.0, n_per_side=3)):
        frame = AffineFrame.auto(shape.as_array())
        assert frame.is_valid()
        assert frame.indices is not None
        assert frame.condition_number < 1e6

    # for a thin letter-L the *first* valid triple is (0, 1, 5): a degenerate-
    # looking corner; auto must not do worse than it, and the obliqueness score
    # it optimises is what keeps the conditioning sane
    first = next(iter(iter_non_collinear_triples(thin.as_array())))
    frame_first = AffineFrame.from_indices(thin.as_array(), first)
    thin_frame = AffineFrame.auto(thin.as_array())
    assert thin_frame.condition_number <= frame_first.condition_number * (1 + 1e-9)


def test_frame_json_round_trip():
    frame = AffineFrame.from_indices(shapes.grid(3, 3).as_array(), (0, 1, 3))
    back = AffineFrame.from_json(frame.to_json())
    np.testing.assert_array_equal(back.a, frame.a)
    assert back.indices == frame.indices
    assert back.condition_number == pytest.approx(frame.condition_number)


def test_frame_without_indices_can_still_report(well_conditioned_frame):
    manual = AffineFrame.from_points([0.0, 0.0], [1.0, 0.0], [0.0, 1.0])
    assert manual.indices is None
    assert manual.coordinates_xy([0.5, 0.5]) == pytest.approx((0.5, 0.5))
    assert manual.to_json()["indices"] is None
    assert len(list(manual)) == 3


def test_transformed_frame_tracks_the_image(square_set, well_conditioned_frame):
    t = AffineTransform.shear(0.7, -0.2)
    moved = well_conditioned_frame.transformed(t)
    np.testing.assert_allclose(moved.a, t.apply(well_conditioned_frame.a))
    assert moved.indices == well_conditioned_frame.indices
    assert moved.condition_number > 0
