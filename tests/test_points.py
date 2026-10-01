"""Phase 1 acceptance tests for the finite point-set model (README section 4).

Acceptance criteria from the README, restated as executable assertions:

1. any finite non-empty point set can be saved and restored exactly;
2. reordering the points yields the *same mathematical object*;
3. the GUI and the mathematical object stay in sync (tested as: every mutation
   returns a new object and never changes the old one).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from kozlov_lab.datasets import shapes
from kozlov_lab.geometry import Point2D, PointSet2D


# --------------------------------------------------------------------------- #
# construction and validation
# --------------------------------------------------------------------------- #
def test_construct_from_various_inputs_gives_same_object():
    pairs = [[0, 0], [2, 0], [0, 1], [1, 0.5]]
    expected = PointSet2D(pairs)

    assert PointSet2D(np.array(pairs, dtype=float)) == expected
    assert PointSet2D([Point2D(x, y) for x, y in pairs]) == expected
    assert PointSet2D(expected) == expected
    assert PointSet2D({"points": pairs}) == expected


def test_single_point_is_legal_but_empty_is_not():
    one = PointSet2D([[1.0, 2.0]])
    assert one.n == 1
    with pytest.raises(ValueError):
        PointSet2D([])


@pytest.mark.parametrize(
    "bad",
    [
        [[0, 0], [1, float("nan")]],
        [[0, 0], [1, float("inf")]],
        [[0, 0, 0], [1, 1, 1]],
        [[0], [1]],
        5,
        "not points",
    ],
)
def test_invalid_inputs_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        PointSet2D(bad)


def test_non_finite_points_are_rejected_at_point_level():
    with pytest.raises(ValueError):
        Point2D(float("nan"), 0.0)


# --------------------------------------------------------------------------- #
# acceptance 1: save / restore round trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape_name",
    ["triangle_demo", "square_3x", "pentagon", "grid_5x5", "star5", "letter_L", "random_8"],
)
def test_json_round_trip_is_exact(shape_name, tmp_path):
    original = shapes.build_shape(shape_name)
    payload = original.to_json()

    from_dict = PointSet2D.from_json(payload)
    assert from_dict == original
    # not just "equal as a set": bit-exact coordinate equality
    np.testing.assert_array_equal(from_dict.as_array(), original.as_array())
    assert from_dict.name == original.name

    path = original.save(tmp_path / "shape.json")
    from_disk = PointSet2D.load(path)
    np.testing.assert_array_equal(from_disk.as_array(), original.as_array())

    # the file is valid JSON and carries the documented keys
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw) >= {"name", "points"}
    assert len(raw["points"]) == original.n


def test_readme_json_example_is_loadable():
    """The literal JSON snippet printed in README section 4."""
    snippet = """
    {
      "name": "triangle_demo",
      "points": [
        [0, 0],
        [2, 0],
        [0, 1],
        [1, 0.5]
      ]
    }
    """
    loaded = PointSet2D.from_json(snippet)
    assert loaded.name == "triangle_demo"
    assert loaded.n == 4
    np.testing.assert_allclose(
        loaded.as_array(), np.array([[0, 0], [2, 0], [0, 1], [1, 0.5]], dtype=float)
    )


# --------------------------------------------------------------------------- #
# acceptance 2: set semantics
# --------------------------------------------------------------------------- #
def test_reordering_points_preserves_identity(demo_set):
    arr = demo_set.as_array()
    rng = np.random.default_rng(0)
    for _ in range(50):
        shuffled = PointSet2D(arr[rng.permutation(len(arr))], name=demo_set.name)
        assert shuffled == demo_set
        assert shuffled.signature() == demo_set.signature()
        assert hash(shuffled) == hash(demo_set)


def test_equal_size_is_not_enough_shifted_set_must_differ(demo_set):
    """A *negative* control for set equality: a shifted copy is a different set.

    Without this, ``__eq__`` could be implemented as ``True`` and every other
    test in this file would still pass.
    """
    shifted = demo_set.with_points(demo_set.as_array() + np.array([1e-3, 0.0]))
    assert shifted != demo_set


def test_equality_across_types_returns_notimplemented():
    assert (PointSet2D([[0.0, 0.0]]) == [[0.0, 0.0]]) is False


def test_sorted_points_are_deterministic_and_order_independent(demo_set):
    arr = demo_set.as_array()
    rng = np.random.default_rng(7)
    reference = demo_set.sorted_points()
    for _ in range(20):
        shuffled = PointSet2D(arr[rng.permutation(len(arr))])
        np.testing.assert_array_equal(shuffled.sorted_points(), reference)


# --------------------------------------------------------------------------- #
# duplicates (README section 19 test matrix: "重复点")
# --------------------------------------------------------------------------- #
def test_duplicates_are_detected_and_removable():
    with_dupes = PointSet2D([[0, 0], [1, 1], [0, 0], [2, 2], [1, 1]])
    assert with_dupes.n == 5
    assert with_dupes.has_duplicates
    assert with_dupes.duplicate_count() == 2

    clean = with_dupes.deduplicated()
    assert clean.n == 3
    assert not clean.has_duplicates
    # first-occurrence order is preserved
    np.testing.assert_array_equal(clean.as_array(), np.array([[0, 0], [1, 1], [2, 2]], dtype=float))


def test_set_semantics_ignore_multiplicity_but_multiset_signature_does_not():
    """Two different notions of identity, both needed, deliberately separated.

    * ``signature`` / ``__eq__`` implement **set** semantics (README section 4:
      the order must not matter, and a set is determined by its elements);
    * ``multiset_signature`` keeps the multiplicity, which Phase 4 recognition
      needs in order to tell "the same shape" from "the same shape with a
      duplicated / missing point" (README section 13).
    """
    a = PointSet2D([[0, 0], [1, 1]])
    b = PointSet2D([[1, 1], [0, 0]])
    c = PointSet2D([[0, 0], [1, 1], [1, 1]])

    # set semantics: order and multiplicity are both invisible
    assert a.signature() == b.signature() == c.signature()
    assert a == b == c
    assert a.duplicate_count() == 0 and c.duplicate_count() == 1

    # multiset semantics: only the order is invisible
    assert a.multiset_signature() == b.multiset_signature()
    assert a.multiset_signature() != c.multiset_signature()
    assert a.is_same_multiset(b)
    assert not a.is_same_multiset(c)
    assert a.is_same_multiset("not a point set") is NotImplemented


# --------------------------------------------------------------------------- #
# acceptance 3: immutability ("GUI stays in sync")
# --------------------------------------------------------------------------- #
def test_mutating_helpers_never_touch_the_original(square_set):
    before = square_set.as_array().copy()
    moved = square_set.with_points(square_set.as_array() * 3.0)
    smaller = square_set[1:4]

    np.testing.assert_array_equal(square_set.as_array(), before)
    assert moved.n == square_set.n
    assert not np.array_equal(moved.as_array(), square_set.as_array())
    assert isinstance(smaller, PointSet2D) and smaller.n == 3


def test_as_array_returns_a_copy():
    ps = PointSet2D([[0, 0], [1, 1]])
    arr = ps.as_array()
    arr[0, 0] = 999.0
    assert ps[0].x == 0.0


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #
def test_basic_measurements(demo_set):
    assert demo_set.centroid().as_tuple() == pytest.approx((0.75, 0.375))
    assert demo_set.bounds() == pytest.approx((0.0, 0.0, 2.0, 1.0))
    assert demo_set.bounding_box_size() == pytest.approx((2.0, 1.0))
    assert demo_set.diameter() == pytest.approx(math.hypot(2.0, 1.0))
    assert demo_set.distance(0, 1) == pytest.approx(2.0)
    assert demo_set.distance_matrix().shape == (4, 4)
    assert demo_set.pairwise_distances().shape == (6,)
    assert demo_set.characteristic_scale() == pytest.approx(math.hypot(2.0, 1.0))


def test_single_point_has_zero_scale_but_no_division_by_zero():
    ps = PointSet2D([[3.0, 4.0]])
    assert ps.diameter() == 0.0
    assert ps.characteristic_scale() == 1.0
    assert ps.is_degenerate()


def test_hausdorff_distance_is_symmetric_and_order_independent():
    a = PointSet2D([[0, 0], [1, 0], [0, 1]])
    b = PointSet2D([[0, 0], [1, 0], [0, 1]])
    assert a.hausdorff_distance(b) == 0.0
    assert a.hausdorff_distance(b) == b.hausdorff_distance(a)

    shifted = a.with_points(a.as_array() + np.array([0.25, 0.0]))
    assert a.hausdorff_distance(shifted) == pytest.approx(0.25)

    permuted = PointSet2D([[0, 1], [0, 0], [1, 0]])
    assert a.hausdorff_distance(permuted) == 0.0
    assert permuted == a


def test_is_one_dimensional(collinear_set, triangle_set, grid_set):
    assert collinear_set.is_one_dimensional()
    assert not triangle_set.is_one_dimensional()
    assert not grid_set.is_one_dimensional()


def test_is_one_dimensional_is_scale_invariant():
    tiny = shapes.collinear_chain(4).with_points(shapes.collinear_chain(4).as_array() * 1e-6)
    huge = shapes.collinear_chain(4).with_points(shapes.collinear_chain(4).as_array() * 1e6)
    assert tiny.is_one_dimensional()
    assert huge.is_one_dimensional()


# --------------------------------------------------------------------------- #
# Point2D
# --------------------------------------------------------------------------- #
def test_point2d_vector_algebra():
    p = Point2D(3.0, 4.0)
    assert p.norm() == pytest.approx(5.0)
    assert p.distance_to(Point2D(0.0, 0.0)) == pytest.approx(5.0)
    assert (p - Point2D(1.0, 1.0)).as_tuple() == pytest.approx((2.0, 3.0))
    assert (p + [1.0, -1.0]).as_tuple() == pytest.approx((4.0, 3.0))
    assert Point2D(1.0, 0.0).cross(Point2D(0.0, 1.0)) == pytest.approx(1.0)
    assert Point2D(1.0, 0.0).dot(Point2D(0.0, 1.0)) == pytest.approx(0.0)
    assert tuple(Point2D(2.0, 5.0)) == (2.0, 5.0)
    assert Point2D(1, 2) == Point2D(1.0, 2.0)
    assert hash(Point2D(1, 2)) == hash(Point2D(1.0, 2.0))


def test_shapes_registry_is_buildable():
    for name in shapes.shape_names():
        built = shapes.build_shape(name)
        assert built.n >= 1
        assert built.name == name
    with pytest.raises(KeyError):
        shapes.build_shape("does_not_exist")
