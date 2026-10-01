"""Phase 2 tests for the invariant measurements / Invariant Explorer (M6).

README section 9 requires the table to be *verified numerically*, not printed.
The two things this file enforces beyond the plain numbers are:

1. **No vacuous green.** A relation (collinearity, parallelism) reported as
   "preserved" is only accepted when the relation was non-empty in the first
   image, and there is an explicit test that a shape with no such relation is
   reported as ``"vacuous"`` rather than ``"preserved"``.
2. **Falsifiability.** For every property there is at least one experiment in
   which it is demonstrably violated, so a bug that always answers "preserved"
   cannot pass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from kozlov_lab.datasets import shapes
from kozlov_lab.geometry import (
    AffineTransform,
    absolute_coordinate_change,
    affine_coordinate_change,
    angle_change,
    collinearity_relation_change,
    distance_change,
    invariant_report,
    length_ratio_change,
    parallelism_relation_change,
    report_for_transform,
)

PRESERVED = "preserved"
VIOLATED = "violated"
VACUOUS = "vacuous"


# --------------------------------------------------------------------------- #
# the invariant table, per transformation family
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (
            "identity",
            dict(
                absolute_coordinates=PRESERVED,
                distances=PRESERVED,
                angles=PRESERVED,
                length_ratios=PRESERVED,
                collinearity=PRESERVED,
                parallelism=PRESERVED,
                affine_coordinates=PRESERVED,
            ),
        ),
        (
            "translation",
            dict(
                absolute_coordinates=VIOLATED,
                distances=PRESERVED,
                angles=PRESERVED,
                length_ratios=PRESERVED,
                collinearity=PRESERVED,
                parallelism=PRESERVED,
                affine_coordinates=PRESERVED,
            ),
        ),
        (
            "rotation",
            dict(
                absolute_coordinates=VIOLATED,
                distances=PRESERVED,
                angles=PRESERVED,
                length_ratios=PRESERVED,
                collinearity=PRESERVED,
                parallelism=PRESERVED,
                affine_coordinates=PRESERVED,
            ),
        ),
        (
            "similarity",
            dict(
                absolute_coordinates=VIOLATED,
                distances=VIOLATED,
                angles=PRESERVED,
                length_ratios=PRESERVED,
                collinearity=PRESERVED,
                parallelism=PRESERVED,
                affine_coordinates=PRESERVED,
            ),
        ),
        (
            "scale_nonuniform",
            dict(
                absolute_coordinates=VIOLATED,
                distances=VIOLATED,
                angles=VIOLATED,
                length_ratios=VIOLATED,
                collinearity=PRESERVED,
                parallelism=PRESERVED,
                affine_coordinates=PRESERVED,
            ),
        ),
        (
            "shear",
            dict(
                absolute_coordinates=VIOLATED,
                distances=VIOLATED,
                angles=VIOLATED,
                length_ratios=VIOLATED,
                collinearity=PRESERVED,
                parallelism=PRESERVED,
                affine_coordinates=PRESERVED,
            ),
        ),
        (
            "affine_general",
            dict(
                absolute_coordinates=VIOLATED,
                distances=VIOLATED,
                angles=VIOLATED,
                length_ratios=VIOLATED,
                collinearity=PRESERVED,
                parallelism=PRESERVED,
                affine_coordinates=PRESERVED,
            ),
        ),
    ],
)
def test_invariant_table_matches_the_theory(transform_bank, square_set, kind, expected):
    x = square_set.as_array()
    y, report = report_for_transform(x, transform_bank[kind])
    got = report.verdicts()

    assert set(got) == set(expected)
    for key, want in expected.items():
        assert got[key] == want, (
            f"[{kind}] property {key}: expected {want}, got {got[key]} "
            f"(delta={report.get(key).delta:.3e}, tol={report.get(key).tolerance:.3e})"
        )
    # the report must also state which transform it was, not just the verdicts
    assert report.transform_kind == transform_bank[kind].classify()
    assert report.n_points == square_set.n
    np.testing.assert_allclose(y, transform_bank[kind].apply(x), rtol=0, atol=0)


def test_collinearity_and_parallelism_rows_are_not_vacuous_on_the_grid_fixture(square_set):
    """Guards the fixture: if the square lost its collinear triples, the table above
    would still be green while proving nothing.

    ``value_before`` is the size of the relation in ``X``; a zero there would turn
    the row into ``vacuous`` and quietly weaken every preservation assertion.
    """
    x = square_set.as_array()
    y, report = report_for_transform(x, AffineTransform.shear(0.8, 0.0))
    assert report.get("collinearity").value_before > 0
    assert report.get("parallelism").value_before > 0
    assert report.get("collinearity").verdict == PRESERVED
    assert report.get("parallelism").verdict == PRESERVED


# --------------------------------------------------------------------------- #
# vacuity: "preserved" must carry information
# --------------------------------------------------------------------------- #
def test_shape_without_collinear_triples_is_reported_vacuous_not_preserved():
    """A regular polygon has no three collinear vertices.

    The collinearity row must therefore be ``vacuous``: reporting ``preserved``
    would be technically true but carry no information, and would hide a broken
    implementation.
    """
    x = shapes.circle(7).as_array()
    only_x, only_y, n_rel = collinearity_relation_change(x, x)
    assert n_rel == 0, "regular 7-gon must have no collinear triples"
    assert only_x == frozenset() and only_y == frozenset()

    y, report = report_for_transform(x, AffineTransform.rotation(23.0))
    assert report.get("collinearity").verdict == VACUOUS
    assert report.get("collinearity").note is not None


def test_two_point_image_is_reported_vacuous_where_undefined():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    y = np.array([[0.0, 0.0], [2.0, 0.0]])
    report = invariant_report(x, y, kind="affine", invariant_group="affine")
    assert report.get("angles").verdict == VACUOUS
    assert report.get("collinearity").verdict == VACUOUS
    assert report.get("parallelism").verdict == VACUOUS
    assert report.get("affine_coordinates").verdict == VACUOUS
    # but the quantities that *are* defined still get a verdict
    assert report.get("distances").verdict == VIOLATED
    assert report.get("absolute_coordinates").verdict == VIOLATED


def test_vacuous_and_preserved_are_distinguishable_in_the_table():
    """The symbol used in the rendered table must differ for the two verdicts."""
    x = shapes.circle(7).as_array()
    y, report = report_for_transform(x, AffineTransform.rotation(23.0))
    table = report.markdown_table()
    assert "| 共线性 | – |" in table  # vacuous
    assert "| 仿射坐标 | ✓ |" in table  # preserved


# --------------------------------------------------------------------------- #
# falsifiability
# --------------------------------------------------------------------------- #
def test_unrelated_images_break_every_relation():
    """Negative control for the whole table.

    An implementation that always answers "preserved" (or an empty witness set)
    would pass every table test above; here nothing may be preserved except the
    quantities that are trivially equal.
    """
    x = shapes.square(2.0, n_per_side=3).as_array()
    y = shapes.random_points(x.shape[0], seed=4242).as_array()

    only_x, only_y, n_rel = collinearity_relation_change(x, y)
    assert n_rel > 0
    assert len(only_x) + len(only_y) > 0

    report = invariant_report(x, y, kind="unknown", invariant_group="unknown")
    assert report.get("collinearity").verdict == VIOLATED
    assert report.get("distances").verdict == VIOLATED
    assert report.get("affine_coordinates").verdict == VIOLATED
    assert report.transform_kind == "unknown"


def test_angle_change_detects_a_known_rotation_of_45_degrees():
    x = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    t = AffineTransform.rotation(45.0)
    delta, _, n_tested = angle_change(x, t.apply(x))
    assert n_tested == 3
    assert delta < 1e-12

    # and the *same* measure reports a large delta for a shear
    delta_shear, witnesses, _ = angle_change(x, AffineTransform.shear(1.0, 0.0).apply(x))
    assert delta_shear == pytest.approx(math.pi / 4, abs=1e-12)
    assert witnesses, "a violated property must name its witness"


# --------------------------------------------------------------------------- #
# individual measurements
# --------------------------------------------------------------------------- #
def test_absolute_coordinate_change_reports_the_witness_index():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0, 0.0], [1.0, 5.0]])
    delta, count, witness = absolute_coordinate_change(x, y)
    assert delta == pytest.approx(4.0)
    assert witness == "点 #1"


def test_distance_change_is_relative_to_the_image_scale():
    """The same experiment at two scales must give the same *relative* verdict."""
    x_small = np.array([[0.0, 0.0], [1e-6, 0.0], [0.0, 1e-6]])
    x_large = x_small * 1e6
    t = AffineTransform.translation_of(3.0, -4.0)

    for x in (x_small, x_large):
        delta, _, scale = distance_change(x, t.apply(x))
        assert delta <= 1e-12 * scale, (delta, scale)


def test_length_ratio_change_is_scale_free():
    x = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    t = AffineTransform.similarity(1000.0, 13.0)
    delta, _, n = length_ratio_change(x, t.apply(x))
    assert n > 0
    assert delta < 1e-12


def test_parallelism_relation_finds_the_square_sides(square_set):
    x = square_set.as_array()
    only_x, only_y, n_rel = parallelism_relation_change(x, x)
    assert n_rel > 0
    witness = sorted(only_x | only_y)
    assert witness == []  # identical images -> no witness at all

    # a non-affine distortion must break it
    distorted = x.copy()
    distorted[-1] = distorted[-1] + np.array([1.0, -1.5])
    only_x, only_y, _ = parallelism_relation_change(x, distorted)
    assert len(only_x) + len(only_y) > 0


def test_affine_coordinate_change_returns_both_deltas(grid_set):
    x = grid_set.as_array()
    t = AffineTransform.random_invertible(9, max_condition=10.0)
    da, db, bound, stacked = affine_coordinate_change(x, t.apply(x), (0, 1, 5))
    assert da < 1e-9 and db < 1e-9
    assert bound <= 1e-9
    assert stacked.shape == (2, x.shape[0])


def test_affine_coordinate_change_rejects_a_collinear_frame(collinear_set):
    pts = collinear_set.as_array()
    with pytest.raises(ValueError):
        affine_coordinate_change(pts, pts, (0, 1, 2))


def test_collinearity_change_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        collinearity_relation_change([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0]])


def test_parallelism_guard_refuses_an_explosive_point_count():
    x = shapes.random_points(30, seed=1).as_array()
    with pytest.raises(ValueError, match="24 points"):
        parallelism_relation_change(x, x)


# --------------------------------------------------------------------------- #
# transform inference (used by the GUI's manual editing mode)
# --------------------------------------------------------------------------- #
def test_kind_is_inferred_when_not_supplied(grid_set):
    t = AffineTransform.shear(0.4, 0.1)
    y = t.apply(grid_set.as_array())
    report = invariant_report(grid_set.as_array(), y)
    assert report.transform_kind == "affine"
    assert report.invariant_group == "affine"


def test_inference_reports_unknown_for_unrelated_images():
    x = shapes.random_points(8, seed=1).as_array()
    y = shapes.random_points(8, seed=2).as_array()
    report = invariant_report(x, y)
    assert report.transform_kind == "unknown"


def test_explicit_kind_wins_over_inference():
    x = shapes.grid(3, 3).as_array()
    t = AffineTransform.translation_of(2.0, 2.0)
    report = invariant_report(x, t.apply(x), kind="translation", invariant_group="isometry")
    assert report.transform_kind == "translation"
    assert report.get("distances").verdict == PRESERVED


# --------------------------------------------------------------------------- #
# report plumbing
# --------------------------------------------------------------------------- #
def test_report_rows_and_json_are_self_consistent(square_set):
    _, report = report_for_transform(square_set.as_array(), AffineTransform.rotation(15.0))
    rows = report.as_rows()
    assert len(rows) == 7
    assert [row["key"] for row in rows] == list(report.verdicts())
    payload = report.to_json()
    assert payload["n_points"] == square_set.n
    assert len(payload["rows"]) == 7
    assert report.markdown_table().count("\n") == 8  # header + separator + 7 rows


def test_frame_indices_are_respected(grid_set):
    x = grid_set.as_array()
    t = AffineTransform.shear(0.5, 0.0)
    report = invariant_report(x, t.apply(x), kind="affine", invariant_group="affine",
                              frame_indices=(0, 1, 5))
    assert "frame=(0, 1, 5)" in (report.get("affine_coordinates").witness or "")
    assert report.get("affine_coordinates").verdict == PRESERVED
