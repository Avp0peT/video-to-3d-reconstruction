"""Code equivalence: the bijection ψ, and an independent route that must agree.

Two decision routes are implemented on purpose:

* the **code route**, which is the definition (search for :math:`\\psi : M_A \\to M_B`
  preserving every indexed ratio);
* the **geometric route**, which knows nothing about codes and instead solves for an
  affine map in exact rational arithmetic.

Theorem 1 predicts they agree wherever it applies, so comparing them tests the
implementation rather than restating it.  The tests below assert the agreement *and*
pin the cases where a route legitimately abstains.
"""

from __future__ import annotations

from fractions import Fraction
import itertools

import numpy as np
import pytest

from kozlov_lab.coding import (
    DEFAULT_SEARCH_BUDGET,
    KozlovImageCode,
    affine_equivalent_exact,
    build_kozlov_image_code_exact,
    build_kozlov_image_code_numeric,
    codes_equivalent_exact,
    codes_equivalent_numeric,
    cross_check_exact,
    find_bijection,
    images_equivalent_by_code_exact,
)

IMAGE = np.array([[0, 0], [2, 0], [0, 2], [1, 1], [3, 5]])


# --------------------------------------------------------------------------- #
# the renumbering relation
# --------------------------------------------------------------------------- #
def test_relabelling_is_an_equivalence():
    """Two images that differ only by the numbering of their points are equivalent."""
    reference = build_kozlov_image_code_exact(IMAGE)
    for permutation in itertools.islice(itertools.permutations(range(IMAGE.shape[0])), 12):
        moved = IMAGE[list(permutation)]
        result = images_equivalent_by_code_exact(IMAGE, moved)
        assert result.equivalent, (permutation, result.reason)
        assert result.bijection is not None
        # the bijection really is a bijection M_A -> M_B
        assert sorted(result.bijection) == list(range(IMAGE.shape[0]))
        assert sorted(result.bijection.values()) == list(range(IMAGE.shape[0]))
        # and it maps the indexing correctly: entry (num, den) -> (psi(num), psi(den))
        other = build_kozlov_image_code_exact(moved)
        psi = result.bijection
        for entry in reference.entries:
            mapped = (frozenset(psi[i] for i in entry.numerator), frozenset(psi[i] for i in entry.denominator))
            assert any(
                (candidate.numerator, candidate.denominator) == mapped and candidate.value == entry.value
                for candidate in other.entries
            ), mapped


def test_the_bijection_is_necessary_not_just_the_value_multiset():
    """A code comparison based on the value multiset would be *wrong*, not merely slower.

    The value multiset is a valid necessary condition, but on its own it accepts
    renumberings that scramble which triples produced which ratio — and the definition is
    about the indexed structure.  The test constructs a relabelling whose *values* match
    and checks that the returned ψ still respects the indexing, which is the property a
    multiset comparison could not certify.
    """
    reference = build_kozlov_image_code_exact(IMAGE)
    perm = [4, 3, 2, 1, 0]  # reversal
    moved = IMAGE[perm]
    other = build_kozlov_image_code_exact(moved)
    assert reference.value_multiset() == other.value_multiset()
    assert reference.signature_profile() == other.signature_profile()

    result = codes_equivalent_exact(reference, other)
    assert result.equivalent and result.bijection is not None
    assert result.bijection == {index: perm[index] for index in range(len(perm))}


def test_different_images_of_equal_size_are_not_equivalent():
    other = np.array([[0, 0], [5, 0], [0, 5], [2, 2], [7, 3]])
    result = images_equivalent_by_code_exact(IMAGE, other)
    assert not result.equivalent
    assert not result.undecided


def test_cardinality_mismatch_is_reported_plainly():
    result = images_equivalent_by_code_exact(IMAGE, IMAGE[:-1])
    assert not result.equivalent
    assert "label set sizes differ" in result.reason


def test_domain_class_mismatch_refutes_equivalence():
    """The domain class is itself an affine invariant, so it is a legitimate prune."""
    general = [[0, 0], [2, 0], [0, 2], [1, 1], [3, 5]]
    # Exact rationals preserving the parallel bases: upper_b - upper_a = (2, 1/4) is
    # exactly half of lower_b - lower_a = (4, 1/2).  Rounding 1.8625 to 37/20 would
    # destroy that, which is precisely the "non-representable coordinate" trap.
    from fractions import Fraction as F

    rational = [
        (F(0), F(0)),
        (F(4), F(1, 2)),
        (F(11, 4), F(41, 20)),
        (F(3, 4), F(9, 5)),
        (F(5, 4), F(149, 80)),
    ]
    first = build_kozlov_image_code_exact(general)
    second = build_kozlov_image_code_exact(rational)
    assert second.domain == "two_parallel_lines", second.domain
    assert first.domain != second.domain
    result = codes_equivalent_exact(first, second)
    assert not result.equivalent
    assert "domain class" in result.reason


# --------------------------------------------------------------------------- #
# affine invariance of the code (same labelling)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "matrix, translation",
    [
        ((1, 0, 0, 1), (7, -3)),          # translation
        ((0, -1, 1, 0), (0, 0)),          # rotation by 90 degrees
        ((2, 0, 0, 2), (1, 1)),           # similarity
        ((2, 0, 0, 3), (0, 0)),           # non-uniform scaling
        ((1, 1, 0, 2), (0, 0)),           # shear, det = 2
        ((1, 2, 3, 5), (-4, 6)),          # generic, det = -1
    ],
)
def test_affine_maps_preserve_the_indexed_entries_exactly(matrix, translation):
    """For an integer affine map the indexed entries must match *exactly*.

    Not "up to tolerance": with integer coordinates every doubled area is an integer, and
    ``det L`` is an integer, so the ratios are equal as rationals.  An exact assertion is
    therefore the right one here, and it is much stronger than ``pytest.approx``.
    """
    a, b, c, d = matrix
    tx, ty = translation
    moved = np.array(
        [[a * x + b * y + tx, c * x + d * y + ty] for x, y in IMAGE],
        dtype=object,
    ).astype(np.int64)
    reference = build_kozlov_image_code_exact(IMAGE)
    image = build_kozlov_image_code_exact(moved)
    assert image.entries == reference.entries
    assert image.canonical_text() == reference.canonical_text()


# --------------------------------------------------------------------------- #
# the two routes must agree
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "second_image, expect",
    [
        (IMAGE, True),
        (IMAGE[[2, 0, 4, 1, 3]], True),
        (np.array([[0, 0], [5, 0], [0, 5], [2, 2], [7, 3]]), False),
        (np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 3]]), False),
    ],
)
def test_code_route_and_geometry_route_agree(second_image, expect):
    """Theorem 1 predicts agreement; a disagreement would mean one route is wrong."""
    report = cross_check_exact(IMAGE, second_image)
    assert report["routes_agree"] is True, report
    assert report["code_route"]["equivalent"] is expect
    assert report["geometry_route"]["equivalent"] is expect


def test_the_geometry_route_recovers_a_known_map():
    """The geometric route is not a black box: it returns the map it found."""
    matrix = ((2, 1), (0, 3))
    translation = (5, -7)
    moved = np.array(
        [[matrix[0][0] * x + matrix[0][1] * y + translation[0],
          matrix[1][0] * x + matrix[1][1] * y + translation[1]] for x, y in IMAGE]
    )
    result = affine_equivalent_exact(IMAGE, moved)
    assert result.equivalent
    assert result.matrix == ((Fraction(2), Fraction(1)), (Fraction(0), Fraction(3)))
    assert result.translation == (Fraction(5), Fraction(-7))
    assert result.point_mapping is not None


def test_geometric_route_abstains_on_a_degenerate_image():
    """A method that needs a control triple must say so rather than guess."""
    collinear = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    result = affine_equivalent_exact(collinear, collinear)
    assert result.undecided is True
    assert "collinear" in result.reason


# --------------------------------------------------------------------------- #
# honesty about what was and was not decided
# --------------------------------------------------------------------------- #
def test_search_budget_exhaustion_reports_undecided_not_false():
    """"I could not decide" and "they are not equivalent" are different answers.

    A budget limit that silently returned ``False`` would turn a resource bound into a
    false negative — the failure mode the project's discipline exists to prevent.
    """
    result = images_equivalent_by_code_exact(IMAGE, IMAGE.copy(), budget=0)
    assert result.undecided is True
    assert result.equivalent is False, "no equivalence was *established*"
    assert "budget" in result.reason


def test_the_digest_is_an_index_not_the_invariant():
    """A relabelled image has a *different* digest and is *equivalent*.

    This is the README's "hash is not the mathematical invariant" rule, made
    demonstrable: if the digest were used to decide equivalence it would report a false
    negative here.
    """
    reference = build_kozlov_image_code_exact(IMAGE)
    relabelled = build_kozlov_image_code_exact(IMAGE[[3, 1, 4, 0, 2]])
    assert reference.digest() != relabelled.digest()
    assert reference.canonical_text() != relabelled.canonical_text(), (
        "the serialised presentations differ, which is exactly why the digest cannot be "
        "used as the equivalence test"
    )
    assert codes_equivalent_exact(reference, relabelled).equivalent, (
        "yet the codes *are* equivalent; only their presentation differs"
    )
    # the relabelling-invariant summary *is* shared, and is a cheap necessary condition
    assert reference.signature_profile() == relabelled.signature_profile()


def test_numeric_codes_compare_with_a_stated_tolerance():
    first = build_kozlov_image_code_numeric(IMAGE, flatness_tol=1e-9)
    second = build_kozlov_image_code_numeric(IMAGE, flatness_tol=1e-9)
    assert codes_equivalent_numeric(first, second, tolerance=1e-12).equivalent

    relabelled = build_kozlov_image_code_numeric(IMAGE[[1, 0, 3, 2, 4]], flatness_tol=1e-9)
    result = codes_equivalent_numeric(first, relabelled, tolerance=1e-9)
    assert result.equivalent, result.reason
    assert result.tolerance == 1e-9


def test_the_two_layers_cannot_be_mixed_by_accident():
    exact = build_kozlov_image_code_exact(IMAGE)
    numeric = build_kozlov_image_code_numeric(IMAGE)
    with pytest.raises(TypeError, match="exact codes"):
        codes_equivalent_exact(numeric, numeric)
    with pytest.raises(TypeError, match="float codes"):
        codes_equivalent_numeric(exact, exact, tolerance=1e-9)


def test_comparison_describes_itself_for_a_report():
    result = images_equivalent_by_code_exact(IMAGE, IMAGE.copy())
    text = result.describe()
    assert "代码等价" in text
    assert "tolerance" in text or "exact" in text
    payload = result.as_dict()
    assert payload["equivalent"] is True and payload["n_entries_a"] == payload["n_entries_b"]


def test_equivalence_is_reflexive_and_symmetric():
    first = build_kozlov_image_code_exact(IMAGE)
    second = build_kozlov_image_code_exact(IMAGE[[4, 2, 0, 3, 1]])
    forward = codes_equivalent_exact(first, second)
    backward = codes_equivalent_exact(second, first)
    assert forward.equivalent == backward.equivalent is True
    # the returned bijections must be mutual inverses
    assert {v: k for k, v in forward.bijection.items()} == dict(backward.bijection)


def test_a_one_point_change_breaks_equivalence():
    """Falsifiability: the comparison must be able to say "no"."""
    perturbed = IMAGE.copy()
    perturbed[3] = perturbed[3] + np.array([1, 0])
    result = images_equivalent_by_code_exact(IMAGE, perturbed)
    assert not result.equivalent


def test_default_budget_is_finite():
    assert 0 < DEFAULT_SEARCH_BUDGET < 10**9
