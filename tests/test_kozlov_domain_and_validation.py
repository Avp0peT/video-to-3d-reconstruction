"""The three-layer domain and the strict validation adapter (Phase 3).

The specification requires one thing above all others here: keep apart

.. math::

    \\boxed{\\text{code definable} \\ne \\text{theorem domain satisfied}
            \\ne \\text{numerically well-conditioned}}

because a UI that shows "code computed successfully" invites the reader to conclude
"the theorem guarantees complete recognition".  These tests exist so that the
distinction cannot be lost.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from kozlov_lab.coding import (
    DOMAIN_CODE_UNDEFINED,
    DOMAIN_GENERAL_POSITION,
    DOMAIN_TWO_PARALLEL_LINES,
    CollinearImageError,
    DuplicatePointError,
    NonExactCoordinateError,
    TooFewPointsError,
    TheoremDomainError,
    build_kozlov_image_code_exact,
    classify_domain_exact,
    classify_domain_numeric,
    validate_kozlov_image,
)
from kozlov_lab.coding.validation import noise_floor_for, _as_exact_points

#: A trapezoid with sampled bases: the two bases are parallel **exactly** (the upper is
#: the lower scaled by 0.5, a power of two), so all points lie on two parallel lines.
TWO_PARALLEL = [[0.0, 0.0], [4.0, 0.5], [2.75, 2.05], [0.75, 1.8], [1.25, 1.8625]]
#: The trapezoid above, but with rational coordinates so the exact layer accepts it.
TWO_PARALLEL_RATIONAL = [
    (Fraction(0), Fraction(0)),
    (Fraction(4), Fraction(1, 2)),
    (Fraction(11, 4), Fraction(41, 20)),
    (Fraction(3, 4), Fraction(9, 5)),
    (Fraction(5, 4), Fraction(149, 80)),
]
GENERAL_POSITION = [[0, 0], [2, 0], [0, 2], [1, 1], [3, 7]]


# --------------------------------------------------------------------------- #
# layer 1: is the code definable at all?
# --------------------------------------------------------------------------- #
def test_all_collinear_is_code_undefined():
    verdict, evidence = classify_domain_exact([[0, 0], [1, 1], [2, 2], [3, 3]])
    assert verdict == DOMAIN_CODE_UNDEFINED
    assert "collinear" in evidence["reason"] or "degenerate" in evidence["reason"]

    with pytest.raises(CollinearImageError):
        validate_kozlov_image([[0, 0], [1, 1], [2, 2], [3, 3]])


def test_too_few_points_gets_its_own_error():
    """``n < 3`` and "all collinear" are different facts and must not share a message."""
    with pytest.raises(TooFewPointsError):
        validate_kozlov_image([[0, 0], [1, 1]])
    assert not isinstance(TooFewPointsError(2), CollinearImageError)


# --------------------------------------------------------------------------- #
# layer 2: the theorem's own hypothesis
# --------------------------------------------------------------------------- #
def test_two_parallel_lines_is_computable_but_outside_the_theorem():
    """**The distinction the specification insists on.**

    The code exists and is a perfectly good object; what does *not* hold is theorem 1's
    iff-statement, because the theorem is stated for images whose points do not lie on one
    line or on two parallel lines.
    """
    verification = validate_kozlov_image(TWO_PARALLEL_RATIONAL)
    assert verification.domain == DOMAIN_TWO_PARALLEL_LINES
    assert verification.code_definable is True, "the code can be computed"
    assert verification.theorem_applicable is False, "but the theorem does not apply"
    assert any("THEOREM_DOMAIN_NOT_MET" in note for note in verification.notes)

    # the code really is built, and says the same thing
    code = build_kozlov_image_code_exact(TWO_PARALLEL_RATIONAL)
    assert len(code.entries) > 0
    assert code.theorem_applies is False


def test_general_position_is_inside_the_theorem():
    verification = validate_kozlov_image(GENERAL_POSITION)
    assert verification.domain == DOMAIN_GENERAL_POSITION
    assert verification.code_definable and verification.theorem_applicable


def test_require_theorem_domain_turns_the_gap_into_an_error():
    """The flag is how a caller who *intends to rely on the theorem* is protected."""
    validate_kozlov_image(TWO_PARALLEL_RATIONAL)  # tolerated by default
    with pytest.raises(TheoremDomainError) as caught:
        validate_kozlov_image(TWO_PARALLEL_RATIONAL, require_theorem_domain=True)
    assert caught.value.domain == DOMAIN_TWO_PARALLEL_LINES
    assert "iff" in str(caught.value)
    # the same call on a general-position image is fine
    validate_kozlov_image(GENERAL_POSITION, require_theorem_domain=True)


def test_the_three_states_are_mutually_distinguishable_on_the_same_input_family():
    """A single table, so the three cases can be compared at a glance."""
    cases = {
        "collinear": ([[0, 0], [1, 1], [2, 2], [5, 5]], DOMAIN_CODE_UNDEFINED, None),
        "two parallel lines": (TWO_PARALLEL_RATIONAL, DOMAIN_TWO_PARALLEL_LINES, True),
        "general position": (GENERAL_POSITION, DOMAIN_GENERAL_POSITION, True),
    }
    seen: set[str] = set()
    for label, (points, expected_domain, definable) in cases.items():
        try:
            verification = validate_kozlov_image(points)
        except CollinearImageError:
            assert expected_domain == DOMAIN_CODE_UNDEFINED, label
            assert definable is None, label
            seen.add(expected_domain)
            continue
        assert verification.domain == expected_domain, label
        assert verification.code_definable is definable, label
        seen.add(expected_domain)
    assert seen == {DOMAIN_CODE_UNDEFINED, DOMAIN_TWO_PARALLEL_LINES, DOMAIN_GENERAL_POSITION}


def test_a_three_point_image_is_always_in_the_two_parallel_lines_class():
    """A consequence of the definition that is worth pinning down.

    Any three points lie on at most two parallel lines: take the direction of any pair and
    the third point sits on a parallel line through itself.  So for ``n = 3`` theorem 1's
    hypothesis is **never** satisfied.

    This is harmless rather than alarming — the affine group acts transitively on
    non-collinear triples, so all such images are equivalent anyway — but it must be
    visible, because a reader who sees ``theorem_applicable = False`` on every 3-point
    image deserves an explanation rather than a mystery.
    """
    for points in ([[0, 0], [1, 0], [0, 1]], [[0, 0], [5, 1], [2, 7]], [[0, 0], [1, 0], [3, 4]]):
        verdict, _ = classify_domain_exact(points)
        assert verdict == DOMAIN_TWO_PARALLEL_LINES, points

    # and for n = 3 the code carries no information at all: a ratio needs two *distinct*
    # triples, and there is only one.  All non-collinear triangles therefore share it.
    first = build_kozlov_image_code_exact([[0, 0], [1, 0], [0, 1]])
    second = build_kozlov_image_code_exact([[0, 0], [5, 1], [2, 7]])
    assert len(first.entries) == 0 and len(second.entries) == 0
    assert first.is_complete and second.is_complete


def test_collinear_triples_are_excluded_from_the_denominators():
    """The definition needs ``S_ksp != 0``, so a degenerate triple may only be a numerator."""
    verification = validate_kozlov_image([[0, 0], [3, 7], [6, 14], [1, 0]])
    assert frozenset({0, 1, 2}) in verification.degenerate_triples
    code = build_kozlov_image_code_exact([[0, 0], [3, 7], [6, 14], [1, 0]])
    for entry in code.entries:
        assert entry.denominator not in code.degenerate_triples
    assert any(entry.numerator in code.degenerate_triples for entry in code.entries), (
        "degenerate triples must still appear as numerators, giving rho = 0"
    )


# --------------------------------------------------------------------------- #
# layer 3: numerical conditioning
# --------------------------------------------------------------------------- #
def test_noise_dominated_triples_are_detected():
    """Admissible by definition, uninformative in practice — and reported as such.

    ``(0,0), (10**7, 3), (3333333, 1)`` has a determinantal area of exactly **1** while the
    image diameter is ``10**7``.  The definition admits the triple; the ratios it produces
    reach ``S_max / 1 ~ 1e13`` and say nothing about the image.
    """
    points = [[0, 0], [10**7, 3], [3333333, 1], [5, 5]]
    exact = _as_exact_points(points)
    floor = noise_floor_for(exact)
    assert floor > 1, floor

    verification = validate_kozlov_image(points)
    assert verification.code_definable
    assert verification.numerically_well_conditioned is False
    assert verification.conditioning_verdict == "noise_dominated"
    assert frozenset({0, 1, 2}) in verification.noise_dominated_triples
    assert any("NOISE_DOMINATED_TRIPLES" in note for note in verification.notes)


def test_integer_images_are_well_conditioned():
    """The mirror control: without it, a detector that always fires would look fine."""
    verification = validate_kozlov_image([[0, 0], [2, 0], [0, 2], [1, 1], [3, 5]])
    assert verification.numerically_well_conditioned
    assert verification.noise_dominated_triples == ()
    assert verification.conditioning_verdict == "well_conditioned"


def test_the_noise_floor_is_dimensionally_an_area():
    """``D`` is an area, so the floor must be one too — otherwise it scales wrongly."""
    small = noise_floor_for(_as_exact_points([[0, 0], [1, 0], [0, 1]]))
    large = noise_floor_for(_as_exact_points([[0, 0], [1000, 0], [0, 1000]]))
    assert large / small == pytest.approx(1000**2, rel=1e-9)


# --------------------------------------------------------------------------- #
# the strict adapter: duplicates are refused, never merged
# --------------------------------------------------------------------------- #
def test_duplicate_points_are_refused_not_deduplicated():
    """The formal object is a *set*; silently merging would change the user's input.

    This is the A8 ruling: the encoding layer refuses, and deduplication — if wanted — is
    an explicit step whose point-count change is visible.
    """
    points = [[0, 0], [1, 0], [1, 0], [0, 1]]
    with pytest.raises(DuplicatePointError) as caught:
        validate_kozlov_image(points)
    error = caught.value
    assert error.duplicates
    assert "silent" in str(error).lower() or "explicitly" in str(error).lower()

    with pytest.raises(DuplicatePointError):
        build_kozlov_image_code_exact(points)

    # the explicit route: deduplicate, and show what changed
    from kozlov_lab.geometry import PointSet2D

    original = PointSet2D(points)
    deduplicated = original.deduplicated()
    assert original.n == 4 and deduplicated.n == 3, "the caller must be able to see the change"
    assert validate_kozlov_image(deduplicated.as_array()).code_definable


def test_allow_duplicates_is_available_but_recorded():
    """A caller who knowingly passes a multiset gets a note, not silence."""
    verification = validate_kozlov_image(
        [[0, 0], [1, 0], [1, 0], [0, 1]], allow_duplicates=True
    )
    assert verification.duplicate_count == 1
    assert any("DUPLICATES_ALLOWED" in note for note in verification.notes)


def test_non_integer_float_is_refused_by_the_exact_layer():
    with pytest.raises(NonExactCoordinateError):
        validate_kozlov_image([[0.1, 0.0], [1.0, 0.0], [0.0, 1.0]])


# --------------------------------------------------------------------------- #
# the numeric validator mirrors the exact one
# --------------------------------------------------------------------------- #
def test_numeric_validator_agrees_on_the_domain_classes():
    verdict, _ = classify_domain_numeric(np.array(TWO_PARALLEL, dtype=float))
    assert verdict == DOMAIN_TWO_PARALLEL_LINES
    verdict, _ = classify_domain_numeric(np.array(GENERAL_POSITION, dtype=float))
    assert verdict == DOMAIN_GENERAL_POSITION
    verdict, _ = classify_domain_numeric(np.array([[0, 0], [1, 1], [2, 2], [4, 4]], dtype=float))
    assert verdict == DOMAIN_CODE_UNDEFINED


def test_numeric_validator_refuses_duplicates_too():
    from kozlov_lab.coding import validate_kozlov_image_numeric

    with pytest.raises(DuplicatePointError):
        validate_kozlov_image_numeric([[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])


def test_validation_report_serialises_every_layer():
    payload = validate_kozlov_image(TWO_PARALLEL_RATIONAL).as_dict()
    for key in (
        "domain",
        "code_definable",
        "theorem_applicable",
        "conditioning_verdict",
        "numerically_well_conditioned",
        "noise_dominated_triples",
        "evidence",
    ):
        assert key in payload, key
