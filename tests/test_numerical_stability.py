"""Numerical-stability test matrix (README section 19).

The README fixes the cases that must be covered::

    正常三角形 / 很扁的三角形 / 大坐标 / 小坐标 / 重复点 / 共线点 / 近共线点

and requires that the near-collinear problem be *checked*, not assumed away.

The organising idea
-------------------
``|det[B-A, C-A]|`` is an **area**, so a fixed threshold on it is meaningless.
Everything here is expressed either

* in the dimensionless collinearity sine ``|sin(angle BAC)| = det/(|B-A||C-A|)``, or
* relative to the image scale ``max_i ||x_i||``,

so that the same statement holds for an image of size ``1e-6`` and one of size
``1e6``.

The second idea, stated explicitly because it is the honest version of the
README's ``error < 1e-9`` requirement: **the achievable accuracy of the
internal coordinates is ``cond([B-A, C-A]) * eps``, not a constant.**  A flat
triangle with ``sine = 1e-8`` has ``cond ~ 1e8`` and *cannot* deliver ``1e-9``
in float64.  Instead of silently failing (or silently lying), the code reports
``AffineCoordinates.accuracy_bound`` and every test here asserts
``measured <= reported_bound`` rather than a hard-coded constant.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kozlov_lab.config import COLLINEAR_TOL, FLOAT_TOL, SIGNATURE_DECIMALS, tolerance_for_condition
from kozlov_lab.datasets import shapes
from kozlov_lab.geometry import (
    AffineFrame,
    AffineTransform,
    PointSet2D,
    affine_coordinates,
    collinearity_sine,
    is_non_collinear,
    report_for_transform,
)


# --------------------------------------------------------------------------- #
# the tolerance helper itself
# --------------------------------------------------------------------------- #
def test_tolerance_for_condition_is_a_floor_then_grows():
    assert tolerance_for_condition(1.0) == FLOAT_TOL
    assert tolerance_for_condition(10.0) == FLOAT_TOL
    # cond * eps must eventually exceed the floor, otherwise the helper is inert
    assert tolerance_for_condition(1e9) > 1e-9
    assert tolerance_for_condition(1e6) < tolerance_for_condition(1e9)
    # an infinite condition number degrades to the floor, never to NaN
    assert tolerance_for_condition(float("inf")) == FLOAT_TOL
    assert tolerance_for_condition(float("nan")) == FLOAT_TOL


def test_tolerance_for_condition_is_informative_not_constant():
    """A helper that returned a constant would make every assertion below vacuous."""
    values = {tolerance_for_condition(c) for c in (1.0, 1e3, 1e6, 1e9, 1e12)}
    assert len(values) >= 3


# --------------------------------------------------------------------------- #
# 正常三角形 / 很扁的三角形
# --------------------------------------------------------------------------- #
def test_normal_triangle_hits_machine_precision():
    """A comfortable frame must be far better than the 1e-9 requirement."""
    x = shapes.triangle().as_array()
    frame = AffineFrame.from_indices(x, (0, 1, 2))
    assert frame.condition_number < 5.0
    t = AffineTransform.random_invertible(1, max_condition=5.0)
    delta = frame.invariance(t, x)
    assert delta.max_delta < 1e-12, delta.as_dict()
    assert delta.accuracy_bound == FLOAT_TOL


@pytest.mark.parametrize("sine", [1e-2, 1e-4, 1e-6, 1e-8])
def test_flat_triangle_reports_an_honest_bound_instead_of_pretending(sine):
    """Very flat triangle: the error grows, and the code says so in advance.

    The assertion is *measured <= reported bound*, not ``< 1e-9``.  Writing
    ``< 1e-9`` here would be a test that fails on correct code — or, worse, a
    test that passes only because the tolerance was loosened until it did.
    """
    shape = shapes.near_collinear_triangle(sine, scale=1.0)
    x = shape.as_array()
    frame = AffineFrame.from_indices(x, (0, 1, 2))
    assert frame.is_valid()

    cond = frame.condition_number
    # for A=(0,0), B=(1,0), C=(cos t, sin t) the frame matrix is [[1, 1], [0, sin t]],
    # whose singular values are sqrt(2) and sin(t)/sqrt(2): cond = 2 / sin(t).
    assert cond == pytest.approx(2.0 / sine, rel=0.25), "fixture must really be ill conditioned"

    t = AffineTransform.rotation(17.0)  # an isometry: no extra conditioning introduced
    delta = frame.invariance(t, x)

    assert delta.accuracy_bound == pytest.approx(tolerance_for_condition(cond), rel=1e-9)
    assert delta.max_delta <= max(delta.accuracy_bound, FLOAT_TOL), (
        f"sine={sine:g}: measured {delta.max_delta:.3e} exceeded the reported bound "
        f"{delta.accuracy_bound:.3e}"
    )


def test_flat_triangle_makes_the_1e9_requirement_unreachable_by_construction():
    """Proves the 1e-9 threshold is *falsifiable* rather than self-fulfilling.

    With ``sine = 1e-8`` the accuracy bound reported by the library is already
    above ``1e-9``, so a blanket ``error < 1e-9`` assertion would be false here.
    This test therefore documents the *limit of the claim*: the README's 1e-9
    holds for well-conditioned frames, and the library tells you when it does not.
    """
    shape = shapes.near_collinear_triangle(1e-8, scale=1.0)
    frame = AffineFrame.from_indices(shape.as_array(), (0, 1, 2))

    bound = tolerance_for_condition(frame.condition_number)
    assert bound > 1e-9, f"bound {bound:.3e} should exceed 1e-9 for a sine=1e-8 frame"
    assert frame.condition_number > 1e7
    # and the frame is flagged, so a caller can refuse to make a 1e-9 claim
    assert frame.coordinates(shape.as_array()[0]).is_well_conditioned is False


# --------------------------------------------------------------------------- #
# 大坐标 / 小坐标
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e3, 1e6, 1e8])
def test_scale_equivariance_of_the_invariance_claim(scale):
    """The invariance is *scale-equivariant*: the transform must be scaled too.

    Applying a fixed absolute translation of 3.0 to an image of size ``1e-6``
    would make the coordinates ``O(1)`` and destroy the relative precision of
    the *differences* ``P - A`` — that is a property of float64, not a bug.  The
    experiment therefore uses a transform whose translation term is
    commensurate with the image, which is what "affine invariance" means
    geometrically.
    """
    base = shapes.square(2.0, n_per_side=3).as_array()
    x = base * scale
    frame = AffineFrame.from_indices(x, (0, 3, 5))

    centre = x.mean(axis=0)
    t = AffineTransform.rotation(31.0, center=centre) @ AffineTransform.translation_of(
        0.37 * scale, -0.21 * scale
    )
    delta = frame.invariance(t, x)

    image_scale = float(np.abs(x).max())
    assert image_scale == pytest.approx(2.0 * scale, rel=1e-12)
    assert delta.max_delta <= max(delta.accuracy_bound, FLOAT_TOL), (scale, delta.as_dict())


@pytest.mark.parametrize("magnitude", [1e-9, 1e-6, 1.0, 1e6, 1e9])
def test_collinearity_sine_is_stable_over_many_orders_of_magnitude(magnitude):
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0]) * magnitude
    c = np.array([math.cos(0.7), math.sin(0.7)]) * magnitude
    assert collinearity_sine(a, b, c) == pytest.approx(math.sin(0.7), rel=1e-12)
    assert is_non_collinear(a, b, c)


def test_large_coordinates_shift_the_frame_but_not_the_coordinates():
    """A pure translation by ``1e6`` is invisible to ``(alpha, beta)``."""
    base = shapes.square(2.0, n_per_side=3).as_array() + np.array([1e6, -1e6])
    frame = AffineFrame.from_indices(base, (0, 3, 5))
    t = AffineTransform.translation_of(-1e6, 1e6)
    delta = frame.invariance(t, base)
    # absolute error here is ~1e6 * eps / frame size ~ 1e-10, far below any
    # meaningful threshold relative to the image scale
    assert delta.max_delta <= max(delta.accuracy_bound, 1e-9)


def test_known_limitation_translation_magnitude_beats_frame_scale():
    """A *measured* limitation of the invariance, stated as a law.

    The reader's intuition says a translation cannot matter — it is the identity
    on the linear part.  Numerically it matters a lot, because a coordinate of
    magnitude ``|t|`` is only resolved to ``ulp(|t|)``, and ``alpha`` divides a
    difference of such coordinates by the frame's size ``|B - A|``.  The
    reproducible law is::

        max|delta alpha|  ~  ulp(max|coordinate|) / |B - A|  =  eps * |t| / |B - A|

    measured here over a grid of ``(image scale, translation magnitude)``.  The
    observed ratio to that prediction was at most ~2.2 across the grid, so the
    test uses a safety factor.

    Two consequences that matter for the report:

    1. ``accuracy_bound`` (derived from the frame's conditioning alone) does not
       capture this failure mode — the information is lost in the *input*, before
       the solve.  Callers comparing coordinates across very different coordinate
       magnitudes must add ``eps * |t| / frame_scale`` to the tolerance.
    2. The measured delta is sometimes ~1e-16 even in a badly scaled case (lucky
       exact cancellation over a small point set).  A single measurement can
       therefore look fine; only the max over many configurations is meaningful.
    """
    base = shapes.square(2.0, n_per_side=3).as_array()
    eps = float(np.finfo(np.float64).eps)
    worst_ratio = 0.0
    lucky_cases = 0

    for scale in (1.0, 1e-3, 1e-6, 1e-9, 1e-12):
        x = base * scale
        frame = AffineFrame.from_indices(x, (0, 3, 5))
        frame_size = float(np.linalg.norm(frame.b - frame.a))
        for translation in (1.0, 1e3, 1e6):
            predicted = eps * translation / frame_size
            if predicted > 0.1:
                # outside the valid regime: the translated image no longer
                # resolves its own structure.  Covered by the next test.
                continue
            t = AffineTransform.translation_of(translation, -translation)
            delta = frame.invariance(t, x)
            assert delta.max_delta <= predicted * 8.0, (
                f"scale={scale:g}, |t|={translation:g}: measured {delta.max_delta:.3e}, "
                f"law predicts {predicted:.3e}"
            )
            if delta.max_delta < predicted * 1e-3:
                lucky_cases += 1
            worst_ratio = max(worst_ratio, delta.max_delta / predicted)

    # the law is tight, not a vacuous upper bound sitting 8x away from reality
    assert 0.1 < worst_ratio < 8.0, f"worst measured/predicted ratio was {worst_ratio:.3g}"
    # and the "lucky exact cancellation" phenomenon really does occur
    assert lucky_cases >= 1, "expected at least one accidentally-exact configuration"


def test_image_structure_is_destroyed_once_ulp_exceeds_the_frame_scale():
    """The point of destruction, and that the library *refuses* there.

    At ``scale = 1e-12`` and ``|t| = 1e6`` the coordinate grid spacing near
    ``1e6`` (``ulp ~ 1.2e-10``) is larger than the whole image (``2e-12``): every
    point of the translated image collapses onto the same float.  The transformed
    frame is then degenerate and ``invariance`` raises instead of returning
    plausible-looking noise.

    This is the boundary of the ``eps * |t| / frame_scale`` law above; beyond it
    the transform is simply not injective in floating point.
    """
    scale, translation = 1e-12, 1e6
    x = shapes.square(2.0, n_per_side=3).as_array() * scale
    frame = AffineFrame.from_indices(x, (0, 3, 5))
    frame_size = float(np.linalg.norm(frame.b - frame.a))
    eps = float(np.finfo(np.float64).eps)

    # precondition: we really are beyond the resolvable regime
    assert eps * translation > frame_size
    assert np.spacing(translation) > frame_size

    t = AffineTransform.translation_of(translation, -translation)
    moved = t.apply(x)
    distinct = np.unique(moved, axis=0)
    assert distinct.shape[0] < x.shape[0], "structure should have collapsed"

    with pytest.raises(ValueError):
        frame.invariance(t, x)


def test_translation_only_degrades_when_it_dominates_the_image_scale():
    """The regime boundary of the law above, stated positively.

    When ``|t|`` is commensurate with the image scale, ``eps * |t| / frame_scale``
    is ``O(eps)`` and the invariance holds at machine precision.
    """
    base = shapes.square(2.0, n_per_side=3).as_array()
    for scale in (1e-9, 1e-3, 1.0, 1e3):
        x = base * scale
        frame = AffineFrame.from_indices(x, (0, 3, 5))
        t = AffineTransform.translation_of(0.5 * scale, -0.25 * scale)
        delta = frame.invariance(t, x)
        assert delta.max_delta < 1e-12, (scale, delta.as_dict())


# --------------------------------------------------------------------------- #
# 重复点 / 共线点 / 近共线点
# --------------------------------------------------------------------------- #
def test_duplicate_reference_points_are_rejected():
    pts = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match="collinear"):
        affine_coordinates(pts[3], pts[0], pts[1], pts[2])


def test_auto_frame_skips_duplicate_points_instead_of_failing():
    """A duplicate *elsewhere* in the image is normal after quantisation.

    ``auto`` must walk past the degenerate triples and still return a usable
    frame — failing here would make the GUI refuse perfectly ordinary images.
    """
    pts = [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    frame = AffineFrame.auto(pts)
    assert frame.is_valid()
    assert frame.indices == (0, 2, 3)


def test_duplicate_non_reference_points_are_harmless():
    """Repeats elsewhere in the image must not affect ``A, B, C`` at all."""
    x = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 0.5], [1.0, 0.5]])
    with_dupes = PointSet2D(x)
    assert with_dupes.has_duplicates
    clean = with_dupes.deduplicated().as_array()

    frame_clean = AffineFrame.from_indices(clean, (0, 1, 2))
    frame_dupes = AffineFrame.from_indices(x, (0, 1, 2))
    assert frame_clean.coordinates_xy([1.0, 0.5]) == pytest.approx(
        frame_dupes.coordinates_xy([1.0, 0.5]), rel=0, abs=1e-15
    )


@pytest.mark.parametrize("n", [3, 4, 6, 9])
def test_fully_collinear_set_has_no_frame_at_all(n):
    x = shapes.collinear_chain(n).as_array()
    assert not is_non_collinear(x[0], x[1], x[2])
    with pytest.raises(ValueError):
        AffineFrame.auto(x)
    report_pair = report_for_transform(x, AffineTransform.rotation(11.0))[1]
    assert report_pair.get("affine_coordinates").verdict == "vacuous"
    assert report_pair.get("collinearity").verdict in {"preserved", "vacuous"}


def test_near_collinear_below_the_threshold_is_treated_as_collinear():
    """``sine = COLLINEAR_TOL / 10`` must be refused, not answered with noise."""
    shape = shapes.near_collinear_triangle(COLLINEAR_TOL / 10.0)
    x = shape.as_array()
    assert not is_non_collinear(x[0], x[1], x[2])
    with pytest.raises(ValueError, match="collinear"):
        affine_coordinates(x[2], x[0], x[1], x[2])


def test_just_above_the_threshold_is_accepted_and_flagged_ill_conditioned():
    shape = shapes.near_collinear_triangle(COLLINEAR_TOL * 10.0)
    x = shape.as_array()
    assert is_non_collinear(x[0], x[1], x[2])
    frame = AffineFrame.from_indices(x, (0, 1, 2))
    res = frame.coordinates(x[0])
    assert res.is_well_conditioned is False
    assert res.accuracy_bound > 1e-9


# --------------------------------------------------------------------------- #
# conditioning of the *transform* (a different failure mode)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cond", [1, 10, 100, 1000])
def test_residual_stays_tiny_even_for_ill_conditioned_transforms(cond):
    """The algebraic residual is a backward-error indicator: it stays small.

    A large ``cond`` limits the *forward* accuracy of ``(alpha, beta)`` but must
    not produce a large residual — if it did, the solver itself would be wrong.
    """
    x = shapes.square(2.0, n_per_side=3).as_array()
    frame = AffineFrame.from_indices(x, (0, 3, 5))
    t = AffineTransform.random_invertible(
        cond, scale_range=(1.0, 1.0 + cond), max_condition=float(cond)
    )
    assert t.condition_number <= cond * (1 + 1e-9)

    moved_frame = frame.transformed(t)
    moved = t.apply(x)
    batch = moved_frame.coordinates_batch(moved)

    image_scale = float(np.abs(moved).max())
    assert batch["residual"].max() <= 1e-12 * max(image_scale, 1.0), batch["residual"].max()


def test_signature_decimals_are_configurable_and_documented():
    ps = PointSet2D([[0.1234567891, 0.0], [1.0, 1.0]])
    assert SIGNATURE_DECIMALS == 9
    assert ps.signature(decimals=3) == frozenset({(123, 0), (1000, 1000)})
    # rounding at 9 decimals merges the two representations of the same point
    other = PointSet2D([[0.12345678914, 0.0], [1.0, 1.0]])
    assert ps.signature() == other.signature()
