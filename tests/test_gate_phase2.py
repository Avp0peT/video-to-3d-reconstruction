"""The Phase 2 acceptance gate (README sections 14 and 24).

README section 24 declares::

    Phase 2 没有通过性质测试，不允许进入 Phase 3。

so this file is the machine-checkable form of that sentence.  It runs
**Experiment A** from README section 14:

    随机生成 1000 个 T(x) = Ax + t，验证内部坐标误差
    max|Δα|, max|Δβ| < 1e-9

and adds the three things a bare "< 1e-9" assertion cannot give you:

1. **a negative control per property** — the run must produce a *non-zero*
   error somewhere, and a deliberately ill-conditioned frame must *break* the
   1e-9 threshold.  Otherwise the threshold is unfalsifiable and the gate is
   decoration;
2. **the theoretical comparison** — the measured error is checked against
   ``cond(M) * eps``, so the number is understood rather than merely tolerated;
3. **order independence** — the whole point of Phase 1 is that a point set has
   no order; the experiment therefore re-runs with shuffled points and requires
   the per-point coordinates to follow the shuffle exactly rather than
   accidentally agreeing.

Runtime: ~10 s.  Marked ``gate`` so it can be selected with ``-m gate``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from kozlov_lab.config import FLOAT_TOL, tolerance_for_condition
from kozlov_lab.datasets import shapes
from kozlov_lab.geometry import (
    AffineFrame,
    AffineTransform,
    affine_coordinates_batch,
)

pytestmark = pytest.mark.gate

#: README section 14, Experiment A.
N_TRANSFORMS = 1000

#: README section 6 acceptance criterion.
REQUIRED_ACCURACY = 1e-9


@pytest.fixture(scope="module")
def experiment_a_data():
    """Fixed image + a fixed, well-conditioned frame; transform varies."""
    x = shapes.square(2.0, n_per_side=3).as_array()
    frame = AffineFrame.from_indices(x, (0, 3, 5))
    assert frame.condition_number < 10.0
    return x, frame


def _run(x: np.ndarray, frame: AffineFrame, n: int, *, max_condition: float):
    """Apply ``n`` random affine maps, return per-run ``(max|Δα|, max|Δβ|, cond)``."""
    rng = np.random.default_rng(20261001)
    before = frame.coordinates_batch(x)
    rows = []
    for _ in range(n):
        t = AffineTransform.random_invertible(rng, max_condition=max_condition)
        moved_frame = frame.transformed(t)
        after = moved_frame.coordinates_batch(t.apply(x))
        da = float(np.abs(after["alpha"] - before["alpha"]).max())
        db = float(np.abs(after["beta"] - before["beta"]).max())
        rows.append((da, db, float(before["condition_number"]), float(after["condition_number"])))
    return rows


# --------------------------------------------------------------------------- #
# the gate itself
# --------------------------------------------------------------------------- #
def test_experiment_a_internal_coordinates_are_affine_invariant(experiment_a_data):
    x, frame = experiment_a_data
    started = time.perf_counter()
    rows = _run(x, frame, N_TRANSFORMS, max_condition=50.0)
    elapsed = time.perf_counter() - started

    worst_alpha = max(r[0] for r in rows)
    worst_beta = max(r[1] for r in rows)
    worst = max(worst_alpha, worst_beta)

    print(
        f"\n[Experiment A] n={N_TRANSFORMS} transforms, {elapsed:.2f}s\n"
        f"  max|d alpha| = {worst_alpha:.3e}\n"
        f"  max|d beta|  = {worst_beta:.3e}\n"
        f"  required     = {REQUIRED_ACCURACY:.0e}"
    )

    assert len(rows) == N_TRANSFORMS
    assert worst < REQUIRED_ACCURACY, (
        f"GATE FAILED: max|delta| = {worst:.3e} >= {REQUIRED_ACCURACY:.0e}"
    )


def test_gate_is_not_vacuous_the_measured_error_is_non_zero(experiment_a_data):
    """A run that measured nothing would pass the gate above for free.

    ``max|Δ| == 0`` exactly would mean the "after" coordinates were copied from
    the "before" ones (e.g. a transform that silently became the identity).
    """
    x, frame = experiment_a_data
    rows = _run(x, frame, N_TRANSFORMS, max_condition=50.0)
    worst = max(max(r[0], r[1]) for r in rows)
    assert worst > 0.0, "the experiment observed a bit-exact zero everywhere: it is not measuring"
    # and the interesting magnitudes are float64-noise level, not larger
    assert worst < 1e-12


def test_gate_threshold_is_falsifiable_an_ill_conditioned_frame_breaks_it():
    """Proves that ``< 1e-9`` can fail.

    The same experiment is repeated with a frame whose condition number is
    ``~4e8`` (``sine = 5e-9``), transformed by **isometries** — the maps that
    change the conditioning least.  There the measured error exceeds ``1e-9``,
    so the gate's threshold demonstrably discriminates.  If this test ever
    passed with a tiny error, the gate above would be measuring nothing.

    Isometries are used on purpose: a general affine map can make the transformed
    frame *itself* collinear (``sine`` shrinks by up to ``cond(A)^2``), and then
    the rejection is about the frame, not about the accuracy claim.
    """
    shape = shapes.near_collinear_triangle(5e-9, scale=1.0)
    x = shape.as_array()
    frame = AffineFrame.from_indices(x, (0, 1, 2))
    assert frame.collinearity_sine > 1e-9
    assert frame.condition_number > 1e7

    rng = np.random.default_rng(777)
    before = frame.coordinates_batch(x)
    worst = 0.0
    conds = []
    for _ in range(500):
        t = AffineTransform.rotation(float(rng.uniform(0.0, 360.0))) @ AffineTransform.translation_of(
            float(rng.uniform(-1.0, 1.0)), float(rng.uniform(-1.0, 1.0))
        )
        moved_frame = frame.transformed(t)
        conds.append(moved_frame.condition_number)
        after = moved_frame.coordinates_batch(t.apply(x))
        worst = max(
            worst,
            float(np.abs(after["alpha"] - before["alpha"]).max()),
            float(np.abs(after["beta"] - before["beta"]).max()),
        )

    reported_bound = tolerance_for_condition(frame.condition_number)
    assert max(conds) == pytest.approx(frame.condition_number, rel=1e-6), "isometry must not re-condition"
    assert worst > REQUIRED_ACCURACY, (
        f"expected the ill-conditioned frame to exceed {REQUIRED_ACCURACY:.0e}, got {worst:.3e}"
    )
    # ... and the library had already told the caller not to expect 1e-9
    assert reported_bound > REQUIRED_ACCURACY
    assert worst <= reported_bound * 10.0, (
        f"measured {worst:.3e} is far above the reported bound {reported_bound:.3e}; "
        "either the bound is wrong or the solver is"
    )


def test_measured_error_matches_the_theoretical_conditioning_bound(experiment_a_data):
    """``|Δ| <= cond * eps * C`` **row by row**.

    This is the check that turns "it happens to be below 1e-9" into "we
    understand why it is below 1e-9".

    The bound is evaluated per run against *that run's* frame conditioning.  Using
    the global worst condition number against the global worst error would be
    wrong: they come from different runs, and the comparison would be a
    coincidence rather than a bound.
    """
    x, frame = experiment_a_data
    rows = _run(x, frame, N_TRANSFORMS, max_condition=50.0)
    eps = float(np.finfo(np.float64).eps)

    worst_error = 0.0
    worst_ratio = 0.0
    worst_cond = 0.0
    for da, db, cond_before, cond_after in rows:
        cond = max(cond_before, cond_after)
        bound = cond * eps * 64.0
        error = max(da, db)
        assert error <= bound, (
            f"measured {error:.3e} exceeds cond*eps*64 = {bound:.3e} for cond = {cond:.4f}"
        )
        worst_error = max(worst_error, error)
        worst_cond = max(worst_cond, cond)
        worst_ratio = max(worst_ratio, error / bound)

    # the bound must be tight enough to be informative, not a blanket 1e3 margin
    assert 1e-4 < worst_ratio <= 1.0, f"worst error/bound ratio was {worst_ratio:.3g}"
    assert worst_error < REQUIRED_ACCURACY
    # the moved frame's conditioning is bounded by cond(frame) * cond(transform)
    assert worst_cond < frame.condition_number * 50.0 * (1 + 1e-6)


# --------------------------------------------------------------------------- #
# order independence (Phase 1 property, re-checked at the Phase 2 boundary)
# --------------------------------------------------------------------------- #
def test_shuffling_the_points_does_not_change_any_internal_coordinate(experiment_a_data):
    """Set semantics must survive into the coordinate layer.

    Two separate claims are checked:

    * ``coordinates(x[perm])[k] == coordinates(x)[perm[k]]`` — the coordinates
      follow the permutation instead of accidentally agreeing;
    * the invariance still holds after shuffling, with the same bound.
    """
    x, frame = experiment_a_data
    rng = np.random.default_rng(31337)
    before = frame.coordinates_batch(x)
    t = AffineTransform.random_invertible(rng, max_condition=20.0)
    moved_frame = frame.transformed(t)
    moved = t.apply(x)
    after_reference = moved_frame.coordinates_batch(moved)

    for _ in range(25):
        perm = rng.permutation(x.shape[0])

        shuffled = frame.coordinates_batch(x[perm])
        np.testing.assert_allclose(shuffled["alpha"], before["alpha"][perm], rtol=0, atol=1e-15)
        np.testing.assert_allclose(shuffled["beta"], before["beta"][perm], rtol=0, atol=1e-15)

        after_shuffled = moved_frame.coordinates_batch(moved[perm])
        np.testing.assert_allclose(
            after_shuffled["alpha"], after_reference["alpha"][perm], rtol=0, atol=1e-15
        )
        # and the invariance itself: shifted coordinates equal unshifted ones
        worst = max(
            float(np.abs(after_shuffled["alpha"] - before["alpha"][perm]).max()),
            float(np.abs(after_shuffled["beta"] - before["beta"][perm]).max()),
        )
        assert worst < REQUIRED_ACCURACY


def test_shuffling_does_not_change_the_frame_diagnostics(experiment_a_data):
    x, frame = experiment_a_data
    rng = np.random.default_rng(5)
    reference = frame.coordinates_batch(x)["condition_number"]
    for _ in range(10):
        perm = rng.permutation(x.shape[0])
        shuffled_frame = AffineFrame.from_indices(x[perm], tuple(
            int(np.where(perm == i)[0][0]) for i in frame.indices  # type: ignore[union-attr]
        ))
        assert shuffled_frame.condition_number == pytest.approx(reference, rel=1e-12)
        assert shuffled_frame.collinearity_sine == pytest.approx(frame.collinearity_sine, rel=1e-12)


# --------------------------------------------------------------------------- #
# acceptance checklist from README section 25 (MVP, items 1-10)
# --------------------------------------------------------------------------- #
def test_mvp_walkthrough_items_1_to_7():
    """The live demo of README section 25 must be reproducible as a script.

    Steps 1-7 of the MVP completion standard: create a >=5-point figure, apply a
    random invertible affine map, show that external coordinates changed, pick
    three non-collinear reference points, compute all internal coordinates, and
    prove numerically that they did not change.  Steps 8-10 (image code and
    recognition) belong to Phases 3-4 and are **not** claimed here.
    """
    # 1. a figure with at least 5 points
    image = shapes.square(2.0, n_per_side=3)
    x = image.as_array()
    assert image.n >= 5

    # 2-3. a random invertible affine map
    t = AffineTransform.random_invertible(2026, max_condition=20.0)
    y = t.apply(x)
    assert t.is_invertible

    # 4. the external coordinates really moved
    assert float(np.abs(y - x).max()) > 1e-3

    # 5. three non-collinear reference points
    frame = AffineFrame.auto(x)
    assert frame.is_valid()

    # 6. internal coordinates of every point
    before = frame.coordinates_batch(x)
    after = frame.transformed(t).coordinates_batch(y)
    assert before["alpha"].shape == (image.n,)

    # 7. they are numerically unchanged
    delta = float(
        max(
            np.abs(after["alpha"] - before["alpha"]).max(),
            np.abs(after["beta"] - before["beta"]).max(),
        )
    )
    assert delta < REQUIRED_ACCURACY, f"step 7 failed with delta = {delta:.3e}"

    # the residual is at the level of FLOAT_TOL, i.e. the solve closed exactly
    assert before["residual"].max() < FLOAT_TOL
