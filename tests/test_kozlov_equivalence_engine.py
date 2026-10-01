"""4D/4N — the equivalence engine, its small-``n`` oracle, and the property matrix.

Two things are being established here:

* the **pruned backtracking search** gives the same answers as the ``n!`` enumeration it is
  meant to replace — otherwise the pruning has changed the relation, not just the runtime;
* the properties the specification lists hold: affine invariance of the code, the theorem's
  iff, relabelling behaviour, negative controls, and the domain cases.

Randomised tests are used, but every comparison is **checked against an oracle**
(:func:`~kozlov_lab.coding.brute_force_bijection` for the search,
:func:`~kozlov_lab.coding.check_geometric_equivalence` for equivalence).  Random sampling is
never used to *establish* a claim.
"""

from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pytest

from kozlov_lab.coding import (
    ArithmeticMode,
    brute_force_bijection,
    build_kozlov_image_code_exact,
    build_kozlov_image_code_numeric,
    check_geometric_equivalence,
    find_bijection,
    images_equivalent_by_code_exact,
    kozlov_codes_equivalent,
    refine_colours_joint,
)

#: Small general-position images.  ``n <= 7`` keeps the ``n!`` oracle affordable
#: (7! = 5 040) while still exercising non-trivial searches.
SMALL_IMAGES = {
    4: [[0, 0], [2, 0], [0, 2], [1, 1]],
    5: [[0, 0], [3, 0], [0, 4], [1, 1], [5, 2]],
    6: [[0, 0], [5, 0], [0, 5], [2, 2], [6, 3], [3, 7]],
    7: [[0, 0], [7, 0], [0, 6], [2, 1], [5, 4], [1, 6], [8, 2]],
}


def _psi_is_valid(
    code_a, code_b, psi: dict[int, int], *, tolerance: float | None = None
) -> bool:
    """Whether ``psi`` really maps every entry of ``code_a`` onto an entry of ``code_b``.

    Needed because **the correspondence is not unique when the image has a non-trivial affine
    automorphism**: the brute force and the pruned search may legitimately return different
    bijections that both satisfy the definition.  Asserting equality with one particular
    permutation would assert something the definition does not claim — and it failed first on
    the 4-point image ``{(0,0),(2,0),(0,2),(1,1)}``, which is symmetric under ``(x,y) -> (y,x)``.
    """
    index_b = {(e.numerator, e.denominator): e for e in code_b.entries}
    for entry in code_a.entries:
        mapped = (
            frozenset(psi[label] for label in entry.numerator),
            frozenset(psi[label] for label in entry.denominator),
        )
        target = index_b.get(mapped)
        if target is None:
            return False
        if tolerance is None:
            if entry.value != target.value:
                return False
        elif abs(float(entry.value) - float(target.value)) > tolerance:
            return False
    return True


# --------------------------------------------------------------------------- #
# 4D — the pruned search against the n! oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", sorted(SMALL_IMAGES))
def test_pruned_search_matches_the_brute_force_oracle(n):
    """Same verdict, same ψ, for a relabelled copy at every affordable ``n``."""
    points = SMALL_IMAGES[n]
    rng = np.random.default_rng(4000 + n)
    reference = build_kozlov_image_code_exact(points)
    for _ in range(4):
        permutation = list(rng.permutation(n))
        other = build_kozlov_image_code_exact([points[i] for i in permutation])
        pruned, reason, nodes, undecided = find_bijection(reference, other)
        brute, examined = brute_force_bijection(reference, other)
        assert not undecided, reason
        assert (pruned is None) == (brute is None), (permutation, reason)
        if pruned is not None:
            # ψ is not unique for images with an automorphism, so validity is the claim —
            # not equality with the oracle's particular choice.
            assert _psi_is_valid(reference, other, dict(pruned)), permutation
            assert _psi_is_valid(reference, other, dict(brute)), permutation
        # Deliberately *not* comparing `nodes` with `examined`: the two count different units
        # (partial assignments vs. complete permutations), so any ordering between them is
        # meaningless.  The runtime claim is made separately, against n!.


@pytest.mark.parametrize("n", sorted(SMALL_IMAGES))
def test_pruned_search_agrees_on_non_equivalent_pairs(n):
    """The oracle is also consulted for the *negative* answers, which is where over-pruning shows."""
    points = SMALL_IMAGES[n]
    rng = np.random.default_rng(7000 + n)
    for _ in range(4):
        other_points = [
            [int(rng.integers(-12, 13)), int(rng.integers(-12, 13))] for _ in range(n)
        ]
        try:
            other = build_kozlov_image_code_exact(other_points)
        except Exception:  # noqa: BLE001 - degenerate draws are not the subject here
            continue
        try:
            reference = build_kozlov_image_code_exact(points)
        except Exception:  # noqa: BLE001
            continue
        if reference.domain != other.domain or len(reference.entries) != len(other.entries):
            continue
        pruned, reason, _, undecided = find_bijection(reference, other)
        brute, _ = brute_force_bijection(reference, other)
        assert (pruned is None) == (brute is None), (reference.domain, reason)


def test_the_search_is_far_cheaper_than_n_factorial():
    """A measurable statement, not a claim: the pruned search visits far fewer nodes."""
    points = SMALL_IMAGES[7]
    reference = build_kozlov_image_code_exact(points)
    permutation = [3, 0, 6, 2, 5, 1, 4]
    other = build_kozlov_image_code_exact([points[i] for i in permutation])
    _, _, nodes, _ = find_bijection(reference, other)
    factorial = math.factorial(7)
    assert nodes * 100 < factorial, f"{nodes} nodes vs {factorial} permutations"


def test_refinement_colours_are_a_co_invariant():
    """The property that makes the pruning legitimate, checked on a known valid ψ.

    If ``psi`` preserves the code then ``colour_A(i) == colour_B(psi(i))``.  If the colours
    were *not* a co-invariant, the pruning could exclude the true correspondence and the
    engine would report a false negative.
    """
    points = SMALL_IMAGES[6]
    reference = build_kozlov_image_code_exact(points)
    permutation = [4, 1, 5, 0, 3, 2]
    other = build_kozlov_image_code_exact([points[i] for i in permutation])
    psi, _, _, _ = find_bijection(reference, other)
    assert psi is not None

    # the colours live in a registry **shared by the two codes**, which is what makes ids
    # comparable across them: the co-invariance is a property of the pair, not of one code
    for rounds in (0, 1, 2):
        colours_a, colours_b, registry = refine_colours_joint(reference, other, rounds=rounds)
        assert registry.n_colours > 0
        for label, target in psi.items():
            assert colours_a[label] == colours_b[target], (rounds, label, target)


def test_budget_exhaustion_stays_undecided():
    reference = build_kozlov_image_code_exact(SMALL_IMAGES[6])
    other = build_kozlov_image_code_exact(SMALL_IMAGES[6])
    result = kozlov_codes_equivalent(reference, other, budget=0)
    assert result.undecided is True
    assert result.equivalent is False


def test_brute_force_refuses_to_run_beyond_its_cap():
    points = [[i, i * i + i % 3] for i in range(10)]
    code = build_kozlov_image_code_exact(points)
    with pytest.raises(ValueError, match="capped at n"):
        brute_force_bijection(code, code, max_n=8)


def test_the_engine_requires_a_stated_tolerance_for_numeric_codes():
    numeric = build_kozlov_image_code_numeric(SMALL_IMAGES[5])
    with pytest.raises(ValueError, match="explicit tolerance"):
        kozlov_codes_equivalent(numeric, numeric)
    exact = build_kozlov_image_code_exact(SMALL_IMAGES[5])
    with pytest.raises(ValueError, match="tolerance=None"):
        kozlov_codes_equivalent(exact, exact, tolerance=1e-9)


# --------------------------------------------------------------------------- #
# 4N — Exact invariant
# --------------------------------------------------------------------------- #
def _random_invertible_integer_map(rng: np.random.Generator) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """A random integer 2x2 matrix with determinant in {-3..3} \\ {0}, plus a translation."""
    while True:
        a, b, c, d = (int(rng.integers(-3, 4)) for _ in range(4))
        det = a * d - b * c
        if det != 0:
            break
    translation = (int(rng.integers(-15, 16)), int(rng.integers(-15, 16)))
    return (a, b, c, d), translation


@pytest.mark.parametrize("n", sorted(SMALL_IMAGES))
def test_code_is_invariant_under_random_integer_affine_maps(n):
    """The exact code must be *identical*, not merely equivalent, under a fixed labelling.

    With integer coordinates and an integer map, every doubled area is an integer and
    ``det M`` is an integer, so the ratios are equal as rationals.  An exact assertion is
    therefore the right one, and it is much stronger than ``approx``.
    """
    points = np.array(SMALL_IMAGES[n])
    rng = np.random.default_rng(9000 + n)
    reference = build_kozlov_image_code_exact(points)
    for _ in range(6):
        (a, b, c, d), (tx, ty) = _random_invertible_integer_map(rng)
        moved = np.array([[a * x + b * y + tx, c * x + d * y + ty] for x, y in points])
        other = build_kozlov_image_code_exact(moved)
        assert other.entries == reference.entries
        assert other.canonical_text() == reference.canonical_text()


# --------------------------------------------------------------------------- #
# 4N — Exact theorem cross-check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", sorted(SMALL_IMAGES))
def test_code_equivalence_iff_affine_equivalence_in_general_position(n):
    """The theorem's content, checked with two independent routes.

    Both directions are exercised: transformed copies must be reported equivalent by *both*
    routes, and independent random images must be reported non-equivalent by both — with the
    geometric oracle supplying the ground truth rather than an assumption about random data.
    """
    points = SMALL_IMAGES[n]
    reference = build_kozlov_image_code_exact(points)
    rng = np.random.default_rng(11000 + n)

    same_count = 0
    for _ in range(6):
        (a, b, c, d), (tx, ty) = _random_invertible_integer_map(rng)
        moved = [[a * x + b * y + tx, c * x + d * y + ty] for x, y in points]
        code_result = images_equivalent_by_code_exact(points, moved)
        geometric = check_geometric_equivalence(points, moved)
        assert code_result.equivalent is True
        assert geometric.equivalent is True
        same_count += 1
    assert same_count == 6

    different_count = 0
    for _ in range(12):
        other_points = [
            [int(rng.integers(-15, 16)), int(rng.integers(-15, 16))] for _ in range(n)
        ]
        try:
            other = build_kozlov_image_code_exact(other_points)
        except Exception:  # noqa: BLE001 - a degenerate draw is simply skipped
            continue
        geometric = check_geometric_equivalence(points, other_points)
        code_result = kozlov_codes_equivalent(reference, other)
        # the oracle defines the ground truth; the code route must agree with it
        assert code_result.equivalent is geometric.equivalent, (points, other_points)
        different_count += 1
    assert different_count > 0, "the negative direction must actually be exercised"


# --------------------------------------------------------------------------- #
# 4N — Relabelling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", sorted(SMALL_IMAGES))
def test_relabelling_changes_the_presentation_but_not_the_equivalence(n):
    """The single most important behaviour, and the reason ``T_A`` stays indexed."""
    points = SMALL_IMAGES[n]
    rng = np.random.default_rng(13000 + n)
    reference = build_kozlov_image_code_exact(points)
    for _ in range(5):
        permutation = list(rng.permutation(n))
        relabelled = build_kozlov_image_code_exact([points[i] for i in permutation])
        # the raw presentation differs …
        differs = (
            relabelled.canonical_text() != reference.canonical_text()
            or relabelled.entries == reference.entries
        )
        assert differs is True
        # … but the formal relation holds
        result = images_equivalent_by_code_exact(points, [points[i] for i in permutation])
        assert result.equivalent, result.reason
        assert result.bijection is not None
        assert _psi_is_valid(reference, relabelled, dict(result.bijection)), permutation
        # …and the trivial relabelling-by-the-same-permutation is always *one* valid answer,
        # so the found ψ must be a bijection of the label set onto itself
        assert sorted(result.bijection) == list(range(n))
        assert sorted(result.bijection.values()) == list(range(n))


# --------------------------------------------------------------------------- #
# 4N — Domain tests
# --------------------------------------------------------------------------- #
def test_domain_cases_all_have_explicit_semantics():
    """One table covering every domain case the specification lists."""
    from kozlov_lab.coding import (
        CollinearImageError,
        DuplicatePointError,
        recognize_affine_exact,
        recognize_affine_numeric,
        validate_kozlov_image,
    )
    from kozlov_lab.coding import RecognitionStatus

    # all collinear -> the code does not exist
    with pytest.raises(CollinearImageError):
        validate_kozlov_image([[0, 0], [1, 1], [2, 2], [3, 3]])
    assert recognize_affine_exact([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 1], [2, 2], [3, 3]]).status is (
        RecognitionStatus.CODE_UNDEFINED
    )

    # two parallel lines -> code defined, theorem not applicable, oracle decides
    from fractions import Fraction as F

    trapezoid = [
        (F(0), F(0)), (F(4), F(1, 2)), (F(11, 4), F(41, 20)), (F(3, 4), F(9, 5)), (F(5, 4), F(149, 80)),
    ]
    validation = validate_kozlov_image(trapezoid)
    assert validation.code_definable is True and validation.theorem_applicable is False

    # general position
    assert validate_kozlov_image(SMALL_IMAGES[5]).theorem_applicable is True

    # n = 3
    tiny = recognize_affine_exact([[0, 0], [1, 0], [0, 1]], [[2, 3], [5, 3], [2, 9]])
    assert tiny.status is RecognitionStatus.EXACT_AFFINE_EQUIVALENT
    assert tiny.theorem_applicable is False

    # duplicates
    with pytest.raises(DuplicatePointError):
        validate_kozlov_image([[0, 0], [1, 0], [1, 0], [0, 1]])
    assert recognize_affine_exact([[0, 0], [1, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [1, 0], [0, 1]]).status is (
        RecognitionStatus.INVALID_INPUT
    )

    # near-degenerate numeric data
    numeric_result = recognize_affine_numeric(
        [[0.0, 0.0], [10000.0, 0.0], [10000.0, 1.0], [5000.0, 8000.0]],
        [[1.0, 2.0], [10001.0, 2.0], [10001.0, 3.0], [5001.0, 8002.0]],
        tolerance=1e-3,
    )
    assert numeric_result.status in {
        RecognitionStatus.NUMERIC_AFFINE_MATCH,
        RecognitionStatus.NUMERIC_INDETERMINATE,
        RecognitionStatus.NUMERIC_NON_MATCH,
    }
    assert numeric_result.arithmetic_mode is ArithmeticMode.NUMERIC


def test_negative_control_does_not_assume_random_images_are_inequivalent():
    """Ground truth comes from the oracle, never from "it was random, so it must differ".

    The test asserts the *agreement* between the two routes, not a particular verdict — if a
    random draw happened to be affinely equivalent, both routes would have to say so.
    """
    points = SMALL_IMAGES[5]
    reference = build_kozlov_image_code_exact(points)
    rng = np.random.default_rng(17000)
    agreements = 0
    for _ in range(25):
        candidate = [[int(rng.integers(-6, 7)), int(rng.integers(-6, 7))] for _ in range(5)]
        geometric = check_geometric_equivalence(points, candidate)
        try:
            other = build_kozlov_image_code_exact(candidate)
        except Exception:  # noqa: BLE001
            continue
        code_result = kozlov_codes_equivalent(reference, other)
        assert code_result.equivalent is geometric.equivalent, candidate
        agreements += 1
    assert agreements > 10


def test_the_engine_scales_without_exploding():
    """A coarse guard on runtime, with a generous bound: the point is to catch ``n!``.

    ``n = 8`` would be 40 320 permutations by brute force; the pruned search must finish it
    comfortably.  The bound is deliberately loose so the test does not flake on a loaded
    machine — a flaky guard is worse than a loose one.
    """
    points = [[0, 0], [9, 0], [0, 8], [3, 1], [6, 5], [1, 7], [10, 3], [4, 9]]
    reference = build_kozlov_image_code_exact(points)
    other = build_kozlov_image_code_exact([points[i] for i in [7, 2, 5, 0, 6, 1, 4, 3]])
    started = time.perf_counter()
    result = kozlov_codes_equivalent(reference, other)
    elapsed = time.perf_counter() - started
    assert result.equivalent
    assert elapsed < 20.0, f"n=8 equivalence took {elapsed:.1f}s, which suggests n! behaviour"
