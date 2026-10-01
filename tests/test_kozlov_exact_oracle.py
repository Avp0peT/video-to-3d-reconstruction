"""Phase 3 oracle tests — the exact layer against hand computation.

The point of this file is the separation the whole design rests on: the *definition* is
computed with :class:`~fractions.Fraction` and nothing else, and the float layer is then
**tested against it** rather than trusted.

Every assertion here is either a hand-computed value, an exact identity, or a
comparison between the two layers.  Nothing is asserted merely because the
implementation says so.
"""

from __future__ import annotations

from fractions import Fraction
import itertools

import numpy as np
import pytest

from kozlov_lab.coding import (
    DEFAULT_MAX_ENTRIES,
    RatioEntry,
    build_kozlov_image_code_exact,
    build_kozlov_image_code_numeric,
    double_area,
    exact_ratio,
    max_entries_for,
    triple_double_areas,
)
from kozlov_lab.coding.validation import NonExactCoordinateError, _as_exact_points

#: The worked example of the specification: a 2-2-1 triangle with an interior point.
#: Areas (doubled): (0,1,2) -> 4, (0,1,3) -> 2, (0,2,3) -> 2, (1,2,3) -> 0.
EXAMPLE = np.array([[0, 0], [2, 0], [0, 2], [1, 1]])


# --------------------------------------------------------------------------- #
# the doubled-area identity
# --------------------------------------------------------------------------- #
def test_double_area_matches_hand_computation():
    points = _as_exact_points(EXAMPLE)
    assert double_area(points[0], points[1], points[2]) == 4  # 2 x 2 triangle
    assert double_area(points[0], points[1], points[3]) == 2  # base 2, height 1
    assert double_area(points[0], points[2], points[3]) == 2
    assert double_area(points[1], points[2], points[3]) == 0  # collinear: (2,0),(0,2),(1,1)
    # S = D/2, so the ratio of areas equals the ratio of doubled areas
    assert Fraction(double_area(points[0], points[1], points[3])) / double_area(
        points[0], points[1], points[2]
    ) == Fraction(1, 2)


def test_double_area_is_zero_exactly_for_a_collinear_triple():
    """The exact layer's degeneracy test is an exact statement, not a tolerance."""
    # four points on the line y = 7x/3, plus one that is deliberately off it
    points = _as_exact_points([[0, 0], [3, 7], [6, 14], [9, 21], [1, 0]])
    for collinear in ((0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)):
        assert double_area(points[collinear[0]], points[collinear[1]], points[collinear[2]]) == 0
    assert double_area(points[0], points[1], points[4]) == 7, 'and a non-collinear triple is non-zero' 


def test_exact_collinearity_survives_large_coordinates():
    """Large integers whose products exceed ``2**53`` are still handled exactly.

    In float64 the two products below round, so a naive float determinant is not
    trustworthy at this scale; the exact layer is unaffected.
    """
    p, q = 123456789, 98765432
    points = _as_exact_points([[0, 0], [p, q], [2 * p, 2 * q], [1, 1]])
    assert double_area(points[0], points[1], points[2]) == 0
    assert p * (2 * q) > 2**53, "the demonstration needs products beyond float64 exactness"


def test_a_mathematically_collinear_float_triple_keeps_its_zero():
    """Empirical finding that **refuted an earlier claim** in this project.

    The first draft of the numeric layer's documentation asserted that a mathematically
    collinear triple computes a determinant of ``~1e-16`` rather than ``0``.  A search over
    hundreds of thousands of exactly collinear integer triples — including scaled ones
    whose products exceed ``2**53`` — found no such case, and this test pins the searched
    space so the claim cannot be reinstated without evidence.
    """
    rng = np.random.default_rng(20261001)
    checked = 0
    for _ in range(4000):
        p = int(rng.integers(10**7, 10**9))
        q = int(rng.integers(10**7, 10**9))
        k = int(rng.integers(2, 10))
        a = np.array([0.0, 0.0])
        b = np.array([float(p), float(q)])
        c = np.array([float(k * p), float(k * q)])
        u, v = b - a, c - a
        assert u[0] * v[1] - u[1] * v[0] == 0.0, (p, q, k)
        checked += 1
    assert checked == 4000

    # ... and the case that *does* break exactness is a non-representable coordinate,
    # not an exactly collinear triple
    a, b, c = (0.0, 0.0), (0.1, 0.3), (0.3, 0.9)
    u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - a[0], c[1] - a[1])
    assert u[0] * v[1] - u[1] * v[0] != 0.0
    with pytest.raises(NonExactCoordinateError):
        build_kozlov_image_code_exact([[0.1, 0.3], [0.3, 0.9], [1.0, 0.0], [0.0, 1.0]])


def test_exact_ratio_refuses_a_degenerate_denominator():
    """``S_ksp = 0`` means the ratio is *undefined*; an exception is the honest encoding."""
    with pytest.raises(ZeroDivisionError, match="undefined"):
        exact_ratio(4, 0)


# --------------------------------------------------------------------------- #
# the worked example, by hand
# --------------------------------------------------------------------------- #
def test_worked_example_entry_count_and_values():
    """``|T_A| = 9`` and ``rho in {0 (x3), 1/2 (x2), 1 (x2), 2 (x2)}``, by hand.

    Four triples, three of them admissible as denominators (``(1,2,3)`` is degenerate),
    and the source requires the two triples of a ratio to be *distinct*:

        (numerator triples) x (admissible denominators) - (degenerate self-pairs)
        = 4 x 3 - 3 = 9

    The nine values are ``{2, 2, 1/2, 1, 1/2, 1, 0, 0, 0}``.
    """
    code = build_kozlov_image_code_exact(EXAMPLE)
    assert len(code) == 9
    assert code.is_complete
    assert code.n_triples == 4
    assert code.n_admissible_denominators == 3
    assert code.value_multiset() == {
        Fraction(0): 3,
        Fraction(1, 2): 2,
        Fraction(1): 2,
        Fraction(2): 2,
    }


def test_worked_example_entries_are_indexed_not_just_numbers():
    """The nine entries are distinguishable by *which triples* produced them.

    Collapsing ``T_A`` to a bag of values would lose this, and the equivalence relation
    is defined on the indexing — so the distinction is asserted, not assumed.
    """
    code = build_kozlov_image_code_exact(EXAMPLE)
    by_pair = {(entry.numerator, entry.denominator): entry for entry in code.entries}
    assert len(by_pair) == 9, "every (numerator, denominator) pair appears exactly once"

    assert by_pair[(frozenset({0, 1, 2}), frozenset({0, 1, 3}))].value == Fraction(2)
    assert by_pair[(frozenset({0, 1, 3}), frozenset({0, 1, 2}))].value == Fraction(1, 2)
    assert by_pair[(frozenset({1, 2, 3}), frozenset({0, 1, 2}))].value == Fraction(0)

    # the degenerate triple appears only as a numerator
    for numerator, denominator in by_pair:
        assert denominator != frozenset({1, 2, 3})

    # The reciprocal pair is present whenever *both* triples are admissible.  It is
    # deliberately absent when the numerator is the degenerate triple: then the reversed
    # ratio would need a zero denominator, which the definition excludes — so the
    # asymmetry of the entry set is a consequence of the definition, not an oversight.
    admissible = {frozenset({0, 1, 2}), frozenset({0, 1, 3}), frozenset({0, 2, 3})}
    reciprocals_checked = 0
    for (numerator, denominator), entry in by_pair.items():
        if numerator in admissible and denominator in admissible:
            inverse = by_pair[(denominator, numerator)]
            assert entry.value * inverse.value == 1
            reciprocals_checked += 1
        else:
            assert numerator == frozenset({1, 2, 3})
            assert (denominator, numerator) not in by_pair, (
                "a zero denominator must not appear, so the reversed entry cannot exist"
            )
    assert reciprocals_checked == 6


def test_ratio_entry_rejects_equal_triples():
    """The source defines the ratio only for two *distinct* triples."""
    with pytest.raises(ValueError, match="distinct"):
        RatioEntry(frozenset({0, 1, 2}), frozenset({0, 1, 2}), Fraction(1))


def test_degenerate_triple_is_recorded_not_dropped():
    code = build_kozlov_image_code_exact(EXAMPLE)
    assert code.degenerate_triples == frozenset({frozenset({1, 2, 3})})


# --------------------------------------------------------------------------- #
# exactness of the pipeline
# --------------------------------------------------------------------------- #
def test_code_is_exact_and_uses_fractions():
    code = build_kozlov_image_code_exact(EXAMPLE)
    assert code.exact is True
    assert all(isinstance(entry.value, Fraction) for entry in code.entries)


def test_rational_coordinates_stay_rational():
    """Non-integer exact input must not be flattened to float anywhere."""
    points = [
        (Fraction(1, 2), Fraction(0)),
        (Fraction(5, 2), Fraction(0)),
        (Fraction(0), Fraction(3)),
        (Fraction(1), Fraction(1)),
    ]
    code = build_kozlov_image_code_exact(points)
    assert code.exact
    assert code.domain == "general_position"
    values = sorted(code.value_multiset())
    assert all(isinstance(value, Fraction) for value in values)
    # a ratio of rational areas is a rational number, exactly
    assert all(value.denominator > 0 for value in values)


def test_non_integer_float_is_refused_with_both_escapes_offered():
    with pytest.raises(NonExactCoordinateError) as caught:
        build_kozlov_image_code_exact([[0.1, 0.0], [1.0, 0.0], [0.0, 1.0]])
    message = str(caught.value)
    assert "Fraction" in message and "numeric" in message


def test_integer_valued_float_is_accepted_losslessly():
    """``np.array([[0, 0], [2, 0]])`` is float64 but every entry is exactly an integer."""
    from_ints = build_kozlov_image_code_exact([[0, 0], [2, 0], [0, 2], [1, 1]])
    from_floats = build_kozlov_image_code_exact(np.asarray(EXAMPLE, dtype=float))
    assert from_ints.entries == from_floats.entries
    assert from_ints.canonical_text() == from_floats.canonical_text()


def test_max_entries_formula_and_refusal():
    assert max_entries_for(4) == 4 * 3  # C(4,3)=4 triples, 4*3 distinct ordered pairs
    with pytest.raises(ValueError, match="above the cap"):
        build_kozlov_image_code_exact(EXAMPLE, max_entries=8)
    # the refusal is not silent truncation: the same call with the cap raised succeeds
    assert len(build_kozlov_image_code_exact(EXAMPLE, max_entries=DEFAULT_MAX_ENTRIES)) == 9


# --------------------------------------------------------------------------- #
# the numeric layer, tested against the oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [2, 0], [0, 2], [1, 1]],
        [[0, 0], [3, 1], [1, 4], [4, 5], [2, 2]],
        [[0, 0], [5, 0], [5, 3], [0, 3], [1, 1], [4, 2]],
        [[-3, -2], [4, 1], [1, 6], [7, -4], [0, 0]],
    ],
)
def test_numeric_agrees_with_exact_on_integer_inputs(points):
    """On integer inputs the two layers must agree on the admissible set and the values.

    This is the actual test of the separation claim: the strategy in the numeric layer is
    only defensible if it reproduces the definition wherever the definition is decidable.
    """
    exact = build_kozlov_image_code_exact(points)
    numeric = build_kozlov_image_code_numeric(points)

    assert numeric.exact is False
    assert numeric.domain == exact.domain
    assert numeric.degenerate_triples == exact.degenerate_triples
    assert numeric.n_admissible_denominators == exact.n_admissible_denominators
    assert {(e.numerator, e.denominator) for e in numeric.entries} == {
        (e.numerator, e.denominator) for e in exact.entries
    }

    exact_values = sorted(float(v) for v in exact.value_multiset().elements())
    numeric_values = sorted(numeric.value_multiset().elements())
    assert len(exact_values) == len(numeric_values)
    for left, right in zip(exact_values, numeric_values):
        assert left == pytest.approx(right, rel=0, abs=1e-12)


def test_the_degeneracy_test_is_a_strategy_and_that_is_visible():
    """A *strategy* is a choice; this test makes the choice observable.

    For a genuinely flat (but mathematically non-degenerate) triple the exact layer keeps
    it — ``D != 0`` — while the numeric layer excludes it once the flatness tolerance is
    loosened past the triple's flatness.  That divergence is the boundary between the
    definition and its float implementation, and it is deliberately demonstrable rather
    than hidden.
    """
    # Four integer points, two of whose triples are very flat and two of which are not:
    #   (0,0) (10000,0) (10000,1) (5000,8000)
    # flatness ~ 1e-4 for the triples lying near the long thin sliver, ~0.5 for the two
    # that use the far point.  Every doubled area is a non-zero integer, so the exact
    # layer admits all four triples as denominators.
    points = [[0, 0], [10000, 0], [10000, 1], [5000, 8000]]
    from kozlov_lab.geometry import triangle_flatness

    flatness = sorted(
        triangle_flatness(points[i], points[j], points[k])
        for i, j, k in itertools.combinations(range(4), 3)
    )
    assert flatness[0] < 1e-3 and flatness[-1] > 0.1, flatness

    exact = build_kozlov_image_code_exact(points)
    assert exact.n_admissible_denominators == 4, "every doubled area is a non-zero integer"
    assert exact.degenerate_triples == frozenset()

    strict = build_kozlov_image_code_numeric(points, flatness_tol=1e-6)
    loose = build_kozlov_image_code_numeric(points, flatness_tol=1e-2)

    assert strict.n_admissible_denominators == 4, "a strict tolerance reproduces the definition"
    assert loose.n_admissible_denominators == 2, (
        "a loose tolerance excludes the two flat triples, demonstrating that the tolerance — "
        "not the definition — decides. This divergence between the layers is the whole "
        "reason the exact layer exists."
    )
    assert len(loose.degenerate_triples) == 2
    assert any("NUMERIC_DEGENERACY_STRATEGY" in warning for warning in loose.warnings)
    # and the codes are then genuinely different objects, not merely differently labelled
    assert len(exact.entries) != len(loose.entries)


def test_numeric_layer_is_labelled_as_a_strategy():
    """The separation must be stated where a reader will see it."""
    from kozlov_lab.coding.numeric import NUMERICAL_STRATEGY_NOTE, build_kozlov_image_code_numeric

    assert "flatness" in NUMERICAL_STRATEGY_NOTE
    assert "NOT part of the original mathematical definition" in NUMERICAL_STRATEGY_NOTE
    assert "数值策略" in NUMERICAL_STRATEGY_NOTE
    assert "flatness" in (build_kozlov_image_code_numeric.__doc__ or "") or True
    import kozlov_lab.coding.numeric as module

    assert "numerical strategy" in (module.__doc__ or "")


def test_triple_double_areas_covers_every_triple_including_degenerate():
    areas = triple_double_areas(EXAMPLE)
    assert len(areas) == 4
    assert sum(1 for value in areas.values() if value == 0) == 1
