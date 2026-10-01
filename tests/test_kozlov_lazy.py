"""4F/4G — the lazy representation must be logically identical to the materialised one.

The value of a lazy representation rests entirely on this: if ``ratio(a, b)`` ever disagreed
with the corresponding :class:`~kozlov_lab.coding.model.RatioEntry`, the optimisation would
have changed the mathematical object rather than its memory profile.

The comparison is done **entry by entry over all legitimate pairs**, not on a sample and not
on an aggregate: a count or a value-multiset could agree while individual indexed entries
differ, and the indexing is what the equivalence relation is defined on.
"""

from __future__ import annotations

from fractions import Fraction
import itertools

import numpy as np
import pytest

from kozlov_lab.coding import (
    RatioEntry,
    build_kozlov_image_code_exact,
    build_kozlov_image_code_lazy_exact,
    build_kozlov_image_code_lazy_numeric,
    build_kozlov_image_code_numeric,
    entry_count_formula,
)

#: Images used for the equivalence check.  ``|T_A|`` grows as ``C(n,3)(C(n,3)-1)``, so the
#: largest case here (``n = 8``) is already 3 080 entries — enough that a disagreement in a
#: rare branch would be caught.
IMAGES = {
    4: [[0, 0], [2, 0], [0, 2], [1, 1]],
    5: [[0, 0], [3, 0], [0, 4], [1, 1], [5, 2]],
    6: [[0, 0], [5, 0], [0, 5], [2, 2], [6, 3], [3, 7]],
    7: [[0, 0], [7, 0], [0, 6], [2, 1], [5, 4], [1, 6], [8, 2]],
    8: [[0, 0], [9, 0], [0, 8], [3, 1], [6, 5], [1, 7], [10, 3], [4, 9]],
}


@pytest.mark.parametrize("n", sorted(IMAGES))
def test_lazy_and_materialized_codes_are_identical(n):
    """Every legitimate ``(numerator, denominator)`` pair must agree exactly."""
    points = IMAGES[n]
    materialized = build_kozlov_image_code_exact(points)
    lazy = build_kozlov_image_code_lazy_exact(points)

    assert len(lazy) == len(materialized), (len(lazy), len(materialized))
    assert lazy.view.n_triples == materialized.n_triples
    assert lazy.view.n_admissible_denominators == materialized.n_admissible_denominators
    assert frozenset(lazy.view.degenerate) == materialized.degenerate_triples

    by_pair = {(e.numerator, e.denominator): e.value for e in materialized.entries}
    seen = 0
    for numerator, denominator in itertools.product(lazy.view.triples, lazy.view.admissible_triples):
        if numerator == denominator:
            continue
        value = lazy.ratio(numerator, denominator)
        assert value == by_pair[(numerator, denominator)], (sorted(numerator), sorted(denominator))
        seen += 1
    assert seen == len(materialized.entries), (seen, len(materialized.entries))


@pytest.mark.parametrize("n", sorted(IMAGES))
def test_lazy_materialize_round_trip(n):
    """``lazy.materialize()`` reproduced the same entry set, with the same value type."""
    points = IMAGES[n]
    materialized = build_kozlov_image_code_exact(points)
    from_lazy = build_kozlov_image_code_lazy_exact(points).materialize()
    assert from_lazy.entries == materialized.entries
    assert all(isinstance(entry.value, Fraction) for entry in from_lazy.entries)


def test_lazy_entry_count_matches_the_formula():
    """``|T_A| = q(N-1)`` — the count the complexity document is built on."""
    for n, points in IMAGES.items():
        lazy = build_kozlov_image_code_lazy_exact(points)
        triples = lazy.view.n_triples
        admissible = lazy.view.n_admissible_denominators
        assert len(lazy) == entry_count_formula(triples, admissible)
        assert len(lazy) == admissible * (triples - 1)
        assert triples == n * (n - 1) * (n - 2) // 6


def test_lazy_ratio_refuses_the_two_undefined_cases():
    """A lazy API must refuse the same two things the definition leaves undefined."""
    lazy = build_kozlov_image_code_lazy_exact([[0, 0], [3, 7], [6, 14], [1, 0]])
    triple = frozenset({0, 1, 2})  # collinear: D = 0
    other = frozenset({0, 1, 3})

    with pytest.raises(ValueError, match="distinct"):
        lazy.ratio(triple, triple)
    with pytest.raises(ValueError, match="degenerate"):
        lazy.ratio(other, triple)


def test_lazy_numeric_agrees_with_materialized_numeric():
    points = [[0.5, 0.0], [3.5, 0.25], [0.0, 2.75], [1.5, 1.0], [4.0, 3.0]]
    materialized = build_kozlov_image_code_numeric(points)
    lazy = build_kozlov_image_code_lazy_numeric(points)
    assert len(lazy) == len(materialized)
    by_pair = {(e.numerator, e.denominator): e.value for e in materialized.entries}
    for numerator, denominator in itertools.product(lazy.view.triples, lazy.view.admissible_triples):
        if numerator == denominator:
            continue
        assert lazy.ratio(numerator, denominator) == by_pair[(numerator, denominator)]


def test_lazy_memory_footprint_is_cubic_in_n():
    """The whole point: the view holds ``O(n^3)`` ingredients while ``|T_A|`` is ``O(n^6)``."""
    for n in (6, 8):
        lazy = build_kozlov_image_code_lazy_exact(IMAGES[n])
        footprint = lazy.view.memory_footprint()
        assert footprint["n_triples"] == n * (n - 1) * (n - 2) // 6
        assert footprint["n_entries_if_materialized"] > footprint["n_triples"] ** 2 / 4


def test_lazy_view_is_not_frozen_and_documents_why():
    """The index is built once; a per-call rebuild would be the accidental O(N^2) constant."""
    import inspect

    from kozlov_lab.coding import lazy as lazy_module

    doc = lazy_module.IndexedRatioView.__doc__ or ""
    assert "built **once**" in doc or "built once" in doc
    source = inspect.getsource(lazy_module.IndexedRatioView.__init__)
    assert "_index" in source, "the index must be built in __init__, not per access"


def test_lazy_supports_the_empty_code_for_three_points():
    """``n = 3``: one triple, so no pair of *distinct* triples, so ``T_A`` is empty but valid."""
    lazy = build_kozlov_image_code_lazy_exact([[0, 0], [1, 0], [0, 1]])
    assert len(lazy) == 0
    assert lazy.view.n_triples == 1
    assert lazy.materialize().entries == frozenset()
