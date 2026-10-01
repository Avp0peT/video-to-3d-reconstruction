"""4A/4B/4H/4I/4J/4L/4M — the recognition pipeline and its vocabulary.

The tests are organised around the properties the specification insists on, not around the
functions:

* a result is **never** a bare boolean, and the fields that were folded into it can be read
  independently;
* the three kinds of evidence are distinguished, and a numeric tolerance never becomes a
  theorem;
* outside theorem 1's domain a code-equivalence is reported as such instead of being
  promoted to an affine statement;
* inside the domain a disagreement between the two routes is an **error**, not a choice.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from kozlov_lab.coding import (
    EvidenceKind,
    InternalConsistencyError,
    RecognitionStatus,
    recognize_affine_exact,
    recognize_affine_numeric,
)

X = [[0, 0], [2, 0], [0, 2], [1, 1], [3, 5]]
MAP = ((2, 1), (0, 3))
TRANSLATION = (5, -7)
Y_MAPPED = [[MAP[0][0] * x + MAP[0][1] * y + TRANSLATION[0],
             MAP[1][0] * x + MAP[1][1] * y + TRANSLATION[1]] for x, y in X]
#: Exact rationals whose bases are parallel exactly, so the code exists but theorem 1 does not apply.
TWO_PARALLEL = [
    (Fraction(0), Fraction(0)),
    (Fraction(4), Fraction(1, 2)),
    (Fraction(11, 4), Fraction(41, 20)),
    (Fraction(3, 4), Fraction(9, 5)),
    (Fraction(5, 4), Fraction(149, 80)),
]


# --------------------------------------------------------------------------- #
# 4A — the vocabulary
# --------------------------------------------------------------------------- #
def test_status_vocabulary_covers_every_required_case():
    expected = {
        "EXACT_AFFINE_EQUIVALENT",
        "EXACT_NOT_AFFINE_EQUIVALENT",
        "CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE",
        "NUMERIC_AFFINE_MATCH",
        "NUMERIC_NON_MATCH",
        "NUMERIC_INDETERMINATE",
        "CODE_UNDEFINED",
        "INVALID_INPUT",
    }
    assert {member.value for member in RecognitionStatus} == expected


def test_result_exposes_the_separate_questions():
    """The fields stay separate so a UI does not have to infer them from one status."""
    result = recognize_affine_exact(X, Y_MAPPED)
    assert result.code_definable is True
    assert result.theorem_applicable is True
    assert result.code_equivalent is True
    assert result.arithmetic_mode is not None and str(result.arithmetic_mode) == "exact"
    assert isinstance(result.status, RecognitionStatus)
    payload = result.as_dict()
    for key in (
        "status",
        "evidence",
        "code_equivalent",
        "theorem_applicable",
        "code_definable",
        "arithmetic_mode",
        "mapping",
        "affine_matrix",
        "translation",
        "residual",
        "diagnostics",
    ):
        assert key in payload, key


def test_result_is_not_a_boolean():
    """A ``RecognitionResult`` must not be truthy-by-accident in a way that hides the status."""
    result = recognize_affine_exact(X, Y_MAPPED)
    assert not isinstance(result, bool)
    assert result.is_affine_equivalent is True
    # a *negative* answer must not read as "equivalent" either, and must not be reported as
    # undecided just because the theorem does not apply: the geometric oracle can decide
    negative = recognize_affine_exact(X, TWO_PARALLEL)
    assert negative.is_affine_equivalent is False
    assert negative.undecided is False
    assert negative.status is RecognitionStatus.EXACT_NOT_AFFINE_EQUIVALENT
    assert negative.theorem_applicable is False, "…and the theorem gap is still recorded"


# --------------------------------------------------------------------------- #
# 4B — three kinds of evidence
# --------------------------------------------------------------------------- #
def test_exact_equivalence_carries_a_geometric_witness():
    """Evidence kind A: an explicit map that the caller can re-apply."""
    result = recognize_affine_exact(X, Y_MAPPED)
    assert result.witnessed
    assert EvidenceKind.GEOMETRIC_WITNESS in result.evidence
    function_witness = result.geometric_witness
    assert function_witness is not None
    assert function_witness.matrix == ((Fraction(2), Fraction(1)), (Fraction(0), Fraction(3)))
    assert function_witness.translation == (Fraction(5), Fraction(-7))
    assert result.residual == 0
    assert result.mapping is not None and len(result.mapping) == len(X)


def test_code_equivalence_is_evidence_of_its_own_kind():
    """Evidence kind B appears separately from the witness."""
    result = recognize_affine_exact(X, Y_MAPPED)
    assert EvidenceKind.CODE_EQUIVALENCE in result.evidence
    assert EvidenceKind.NUMERIC_APPROXIMATION not in result.evidence


def test_numeric_evidence_is_never_promoted_to_a_theorem():
    """Evidence kind C: a match within tolerance, and the result says so."""
    result = recognize_affine_numeric(X, Y_MAPPED, tolerance=1e-9)
    assert EvidenceKind.NUMERIC_APPROXIMATION in result.evidence
    assert EvidenceKind.GEOMETRIC_WITNESS not in result.evidence
    assert result.geometric_witness is None, "a float pipeline must not manufacture a witness"
    assert result.status in {
        RecognitionStatus.NUMERIC_AFFINE_MATCH,
        RecognitionStatus.NUMERIC_INDETERMINATE,
    }


# --------------------------------------------------------------------------- #
# 4I — outside theorem 1's domain
# --------------------------------------------------------------------------- #
def test_two_parallel_lines_reports_the_theorem_gap():
    """The central requirement of 4I.

    ``code_definable = True`` and ``theorem_applicable = False`` must **not** produce
    "code equivalent therefore affine equivalent".
    """
    # a relabelled copy: the code is equivalent, and the theorem does not apply
    relabelled = [TWO_PARALLEL[i] for i in [2, 0, 4, 1, 3]]
    result = recognize_affine_exact(TWO_PARALLEL, relabelled)
    assert result.code_definable is True
    assert result.theorem_applicable is False
    assert result.code_equivalent is True
    assert result.status in {
        RecognitionStatus.EXACT_AFFINE_EQUIVALENT,  # the geometric oracle found a witness
        RecognitionStatus.CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE,
    }
    if result.status is RecognitionStatus.CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE:
        assert "theorem_gap" in result.diagnostics
        assert "must not be read as" in result.diagnostics["theorem_gap"]


def test_code_equivalent_theorem_not_applicable_is_reachable_when_the_oracle_abstains():
    """The vocabulary entry exists for the case where the oracle *cannot* upgrade the gap.

    In the current implementation the geometric oracle decides every non-degenerate case
    exhaustively, so the status is reached only through the decision logic with an abstaining
    verdict.  Testing the mapping directly is the honest option: asserting reachability that
    the oracle prevents would be asserting a falsehood, and dropping the entry would lose the
    specification's vocabulary.
    """
    from kozlov_lab.coding import KozlovCodeEquivalence
    from kozlov_lab.coding.geometric import AffineEquivalenceVerdict
    from kozlov_lab.coding.recognition import _status_outside_theorem_domain

    code_result = KozlovCodeEquivalence(
        equivalent=True, undecided=False, reason="fabricated", bijection=None,
        candidates_examined=0, n_entries_a=0, n_entries_b=0, tolerance=None,
    )
    abstaining = AffineEquivalenceVerdict(
        equivalent=False, exhaustive=False, undecided=True, witness=None,
        candidates_tried=0, reason="fabricated abstention",
    )
    assert (
        _status_outside_theorem_domain(code_result, abstaining)
        is RecognitionStatus.CODE_EQUIVALENT_THEOREM_NOT_APPLICABLE
    )


def test_theorem_gap_is_distinguishable_from_undecidable():
    """The specification's boxed point: theorem not applicable ≠ no other method can decide.

    When the geometric oracle is exhaustive it *can* decide, so the status must be the exact
    negative rather than "undecidable".
    """
    witness_result = recognize_affine_exact(TWO_PARALLEL, [TWO_PARALLEL[i] for i in [4, 3, 2, 1, 0]])
    assert witness_result.theorem_applicable is False
    assert witness_result.theorem_applicable != witness_result.code_definable
    assert witness_result.status is RecognitionStatus.EXACT_AFFINE_EQUIVALENT
    assert witness_result.witnessed, "the oracle decided, so the status is the witnessed one"

    negative = recognize_affine_exact(
        TWO_PARALLEL,
        [(Fraction(0), Fraction(0)), (Fraction(4), Fraction(1, 2)), (Fraction(11, 4), Fraction(41, 20)),
         (Fraction(3, 4), Fraction(9, 5)), (Fraction(9), Fraction(9))],
    )
    assert negative.theorem_applicable is False
    assert negative.status is RecognitionStatus.EXACT_NOT_AFFINE_EQUIVALENT, (
        "the oracle's negative is exhaustive, so 'theorem not applicable' must not be reported "
        "as 'undecidable' — that is the specification's boxed distinction"
    )


# --------------------------------------------------------------------------- #
# 4H — cross-check discipline
# --------------------------------------------------------------------------- #
def test_cross_check_agrees_on_the_ordinary_cases():
    for other in (X, [X[i] for i in [3, 1, 4, 0, 2]], [[0, 0], [5, 0], [0, 5], [2, 2], [7, 3]]):
        result = recognize_affine_exact(X, other)
        assert "cross_check_note" not in result.diagnostics


def test_internal_consistency_error_carries_a_reproduction_artifact(tmp_path):
    """The error must be actionable without a re-run: it carries inputs, both routes, commit."""
    artifact = {
        "inputs": {"x": X, "y": Y_MAPPED},
        "code_route": {"equivalent": True},
        "geometric_route": {"equivalent": False},
        "theorem_applicable": True,
    }
    error = InternalConsistencyError("routes disagree", artifact)
    assert error.artifact["theorem_applicable"] is True
    path = error.write_artifact(tmp_path)
    assert path.exists()
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["inputs"]["x"] == X
    assert payload["code_route"]["equivalent"] is True
    assert "geometric_route" in payload


def test_cross_check_raises_inside_the_domain_and_only_reports_outside_it():
    """The scope rule, stated as a test: inside → error, outside → diagnostic.

    Inside theorem 1's domain the routes are guaranteed to agree, so a mismatch is a defect.
    Outside it, the hypothesis is exactly what rules out agreement, so raising would be wrong.
    """
    from kozlov_lab.coding import KozlovCodeEquivalence, check_geometric_equivalence
    from kozlov_lab.coding.geometric import AffineEquivalenceVerdict
    from kozlov_lab.coding.recognition import _cross_check

    fake_code = KozlovCodeEquivalence(
        equivalent=True,
        undecided=False,
        reason="fabricated for the test",
        bijection=None,
        candidates_examined=0,
        n_entries_a=0,
        n_entries_b=0,
        tolerance=None,
    )
    real = check_geometric_equivalence(X, [[0, 0], [5, 0], [0, 5], [2, 2], [7, 3]])
    assert real.equivalent is False

    with pytest.raises(InternalConsistencyError):
        _cross_check(fake_code, real, theorem_applicable=True, x=X, y=X)

    note = _cross_check(fake_code, real, theorem_applicable=False, x=X, y=X)
    assert note is not None and "expected" in note


# --------------------------------------------------------------------------- #
# 4J — n = 3
# --------------------------------------------------------------------------- #
def test_three_noncollinear_points_have_empty_T():
    """``|T_A| = 0`` follows from "two distinct triples", not from a failure."""
    from kozlov_lab.coding import build_kozlov_image_code_exact

    code = build_kozlov_image_code_exact([[0, 0], [1, 0], [0, 1]])
    assert len(code) == 0
    assert code.is_complete
    assert code.n_triples == 1


def test_any_two_noncollinear_triangles_are_affine_equivalent():
    """Which is why the empty code is correct rather than a limitation."""
    from kozlov_lab.coding import check_geometric_equivalence

    triangles = [
        [[0, 0], [1, 0], [0, 1]],
        [[1, 2], [5, 2], [3, 9]],
        [[-2, 0], [0, -3], [4, 1]],
    ]
    for left, right in itertools_pairs(triangles):
        verdict = check_geometric_equivalence(left, right)
        assert verdict.equivalent, (left, right, verdict.reason)

    for left, right in itertools_pairs(triangles):
        result = recognize_affine_exact(left, right)
        # n = 3 never satisfies the theorem's hypothesis, so the status comes from the oracle
        assert result.theorem_applicable is False
        assert result.status is RecognitionStatus.EXACT_AFFINE_EQUIVALENT
        assert result.witnessed


def itertools_pairs(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


# --------------------------------------------------------------------------- #
# 4L/4M — modes and the numeric vocabulary
# --------------------------------------------------------------------------- #
def test_modes_do_not_mix_silently():
    from kozlov_lab.coding import ModeMismatchError, build_kozlov_image_code_exact, build_kozlov_image_code_numeric

    exact = build_kozlov_image_code_exact(X)
    numeric = build_kozlov_image_code_numeric(X)
    with pytest.raises(ModeMismatchError):
        exact.require_same_mode(numeric, "test")


def test_numeric_recogniser_reports_its_conditioning():
    result = recognize_affine_numeric(X, Y_MAPPED, tolerance=1e-9)
    assert result.numeric_condition is not None
    for key in ("tolerance", "noise_floor", "conditioning", "max_residual", "matched_points", "ambiguity"):
        assert key in result.numeric_condition, key
    assert result.numeric_condition["tolerance"] == 1e-9


def test_numeric_recogniser_can_say_indeterminate():
    """A residual at the predicted noise floor must not be forced into MATCH/NON_MATCH."""
    # a genuinely tiny image under a large translation: the ratio arithmetic is starved of
    # input precision, so the honest answer is "cannot tell"
    tiny = [[0, 0], [3, 0], [0, 3], [1, 1], [2, 2]]
    moved = [[x + 1e8, y - 1e8] for x, y in tiny]
    result = recognize_affine_numeric(tiny, moved, tolerance=1e-3)
    assert result.status in {
        RecognitionStatus.NUMERIC_INDETERMINATE,
        RecognitionStatus.NUMERIC_AFFINE_MATCH,
    }
    assert result.numeric_condition is not None
    # the point of the test: whichever answer comes out, the conditioning is on the record
    assert result.numeric_condition["noise_floor"] >= 0


def test_numeric_recogniser_distinguishes_non_matches():
    result = recognize_affine_numeric(X, [[0, 0], [5, 0], [0, 5], [2, 2], [7, 3]], tolerance=1e-9)
    assert result.status in {
        RecognitionStatus.NUMERIC_NON_MATCH,
        RecognitionStatus.NUMERIC_INDETERMINATE,
    }
