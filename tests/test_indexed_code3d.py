"""Independent exact tests for indexed-body-code-v1; no geometric oracle.

All point sets below are synthetic test data.  The reference determinant is
the 24-term Leibniz expansion of a homogeneous 4 x 4 matrix, independently
of the implementation's translated 3 x 3 determinant or search shortcut.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction
from itertools import combinations, permutations
from math import factorial

import pytest

from kozlov_lab.stereo.official.indexed_code3d import (
    CodeBudgetError,
    CodeInputError,
    IndexedBodyCode,
    code_equivalence_for_mapping,
    find_code_equivalence,
    indexed_body_code,
)


TETRAHEDRON = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
IRREGULAR = TETRAHEDRON + ((2, 3, 5),)
ZERO_VOLUME = TETRAHEDRON + ((1, 1, 0),)
TWO_PLANES = (
    (0, 0, 0), (1, 0, 0), (0, 1, 0),
    (0, 0, 2), (2, 0, 2), (0, 3, 2),
)
MOMENT_CURVE = tuple((t, t * t, t * t * t) for t in range(7))


def _six_volume(points):
    matrix = [(Fraction(1), *(Fraction(x) for x in point)) for point in points]
    determinant = Fraction(0)
    for order in permutations(range(4)):
        inversions = sum(order[i] > order[j] for i in range(4) for j in range(i + 1, 4))
        term = Fraction(-1 if inversions % 2 else 1)
        for row, column in enumerate(order):
            term *= matrix[row][column]
        determinant += term
    return abs(determinant)


def _volumes(points, labels=None):
    labels = tuple(range(len(points))) if labels is None else tuple(labels)
    point_by_label = dict(zip(labels, points, strict=True))
    return {
        quad: _six_volume([point_by_label[label] for label in quad])
        for quad in combinations(sorted(labels), 4)
    }


def _ratios(points, labels=None):
    volumes = _volumes(points, labels)
    return {
        (numerator, denominator): numerator_volume / denominator_volume
        for numerator, numerator_volume in volumes.items()
        for denominator, denominator_volume in volumes.items()
        if numerator != denominator and denominator_volume != 0
    }


def _entry_dict(code):
    result = {(entry.numerator, entry.denominator): entry.value for entry in code.entries}
    assert len(result) == len(code.entries), "duplicate indexed records are not allowed"
    return result


def _oracle_mapping_matches(left_points, right_points, permutation):
    """Compare every indexed ratio, without fitting coordinates or factoring ratios."""
    left = _ratios(left_points)
    right = _ratios(right_points)
    if not any(_volumes(left_points).values()) or not any(_volumes(right_points).values()):
        return False
    mapped = {
        (tuple(sorted(permutation[i] for i in numerator)),
         tuple(sorted(permutation[i] for i in denominator))): value
        for (numerator, denominator), value in left.items()
    }
    return mapped == right


def _affine(points, reflection=1):
    """An exactly rational invertible map, with optional orientation reversal."""
    matrix = (
        (Fraction(2), Fraction(1, 3), Fraction(0)),
        (Fraction(0), Fraction(3, 2), Fraction(1, 5)),
        (Fraction(0), Fraction(0), Fraction(reflection, 7)),
    )
    offset = (Fraction(5, 11), Fraction(-7, 13), Fraction(2, 17))
    return tuple(
        tuple(sum(matrix[i][j] * point[j] for j in range(3)) + offset[i]
              for i in range(3))
        for point in points
    )


def _assert_result(result, status):
    assert result["status"] == status
    assert result["equivalent"] is {
        "EQUIVALENT": True, "NOT_EQUIVALENT": False, "UNKNOWN": None
    }[status]
    assert isinstance(result["reason"], str) and result["reason"]
    assert result["decision_complete"] is (status != "UNKNOWN")
    assert isinstance(result["search_exhausted"], bool)
    assert type(result["permutations_checked"]) is int
    assert result["permutations_checked"] >= 0
    assert result["affine_equivalence"] is None
    if status != "EQUIVALENT":
        assert result["mapping"] is None
        assert result["permutation"] is None


def _assert_witness(left, right, result):
    _assert_result(result, "EQUIVALENT")
    order = result["permutation"]
    assert sorted(order) == list(range(len(right.labels)))
    assert result["mapping"] == [
        [label, right.labels[order[i]]] for i, label in enumerate(left.labels)
    ]
    assert _oracle_mapping_matches(left.points, right.points, order)


def _decode_rational(value):
    assert set(value) == {"numerator_hex", "denominator_hex"}
    numerator = int(value["numerator_hex"], 16)
    denominator = int(value["denominator_hex"], 16)
    assert denominator > 0
    fraction = Fraction(numerator, denominator)
    assert value == {
        "numerator_hex": hex(fraction.numerator),
        "denominator_hex": hex(fraction.denominator),
    }
    return fraction


def _all_serialized_rationals(value):
    assert not isinstance(value, Fraction)
    if isinstance(value, dict):
        if "numerator_hex" in value or "denominator_hex" in value:
            yield _decode_rational(value)
        else:
            for child in value.values():
                yield from _all_serialized_rationals(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_serialized_rationals(child)


@pytest.mark.parametrize("points", [IRREGULAR, TWO_PLANES, MOMENT_CURVE])
def test_complete_code_matches_an_independent_homogeneous_determinant(points):
    code = indexed_body_code(points)
    expected_volumes = _volumes(points)
    nonzero_count = sum(value != 0 for value in expected_volumes.values())
    assert isinstance(code, IndexedBodyCode)
    assert code.status == "DEFINED"
    assert code.complete is True
    assert code.labels == tuple(range(len(points)))
    assert dict(code.quadruple_volumes) == expected_volumes
    assert len(code.entries) == nonzero_count * (len(expected_volumes) - 1)
    assert _entry_dict(code) == _ratios(points)


def test_labels_preserve_input_order_while_quadruples_use_sorted_real_labels():
    labels = (40, -3, 8, 90, 2)
    code = indexed_body_code(IRREGULAR, labels=labels)
    assert code.labels == labels
    assert code.points == tuple(tuple(Fraction(x) for x in point) for point in IRREGULAR)
    assert tuple(quad for quad, _ in code.quadruple_volumes) == tuple(combinations(sorted(labels), 4))
    assert dict(code.quadruple_volumes) == _volumes(IRREGULAR, labels)
    assert _entry_dict(code) == _ratios(IRREGULAR, labels)
    assert [(entry.numerator, entry.denominator) for entry in code.entries] == sorted(
        _ratios(IRREGULAR, labels)
    )


def test_equal_ratio_values_retain_every_distinct_index_pair():
    points = TETRAHEDRON + ((Fraction(1, 4),) * 3,)
    code = indexed_body_code(points)
    assert len(code.entries) == 20
    assert Counter(entry.value for entry in code.entries) == {
        Fraction(1): 12, Fraction(4): 4, Fraction(1, 4): 4
    }
    assert _entry_dict(code) == _ratios(points)


def test_zero_numerators_are_kept_and_zero_denominators_are_excluded():
    code = indexed_body_code(ZERO_VOLUME)
    volumes = _volumes(ZERO_VOLUME)
    zero_quads = {quad for quad, volume in volumes.items() if volume == 0}
    assert zero_quads == {(0, 1, 2, 4)}
    assert len(code.entries) == 16
    assert sum(entry.value == 0 for entry in code.entries) == 4
    assert all(entry.denominator not in zero_quads for entry in code.entries)
    assert _entry_dict(code) == _ratios(ZERO_VOLUME)


def test_distinct_quadruples_may_overlap_but_self_ratios_are_absent():
    code = indexed_body_code(IRREGULAR)
    entries = _entry_dict(code)
    numerator, denominator = (0, 1, 2, 3), (0, 1, 2, 4)
    assert len(set(numerator) & set(denominator)) == 3
    assert entries[numerator, denominator] == Fraction(1, 5)
    assert entries[denominator, numerator] == Fraction(5)
    assert all(entry.numerator != entry.denominator for entry in code.entries)


def test_four_nonplanar_points_have_a_defined_empty_code():
    left = indexed_body_code(TETRAHEDRON)
    right = indexed_body_code(_affine(TETRAHEDRON), labels=(11, 7, -2, 5))
    assert left.status == "DEFINED" and left.complete is True
    assert left.entries == ()
    assert left.quadruple_volumes == (((0, 1, 2, 3), Fraction(1)),)
    result = find_code_equivalence(left, right, max_permutations=1)
    _assert_witness(left, right, result)
    assert result["permutations_checked"] == 1


@pytest.mark.parametrize("points", [
    ((2, 3, 4),),
    ((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    ((0, 0, 0), (1, 0, 0), (0, 1, 0), (2, 3, 0), (-1, 4, 0)),
])
def test_planar_or_lower_rank_sets_are_undefined_even_when_entries_are_empty(points):
    code = indexed_body_code(points)
    assert code.status == "UNDEFINED_PLANAR"
    assert code.entries == ()
    assert dict(code.quadruple_volumes) == _volumes(points)
    assert all(volume == 0 for _, volume in code.quadruple_volumes)
    assert code.theorem_domain["theorem3_applies"] is False
    _assert_result(find_code_equivalence(code, code), "UNKNOWN")


@pytest.mark.parametrize("reflection", [1, -1])
def test_exact_rational_affine_maps_preserve_every_indexed_ratio(reflection):
    transformed = _affine(IRREGULAR, reflection)
    left, right = indexed_body_code(IRREGULAR), indexed_body_code(transformed)
    assert _entry_dict(left) == _entry_dict(right) == _ratios(transformed)
    assert all(right_volume == left_volume * Fraction(3, 7)
               for (_, left_volume), (_, right_volume)
               in zip(left.quadruple_volumes, right.quadruple_volumes, strict=True))
    _assert_witness(left, right, code_equivalence_for_mapping(
        left, right, dict(zip(left.labels, right.labels, strict=True))
    ))


def test_explicit_bijection_handles_unrelated_labels_and_reordered_rows():
    left_labels = (40, -3, 8, 90, 2)
    right_labels = (-9, 6, 50, 0, 31)
    order = (4, 2, 0, 3, 1)
    left = indexed_body_code(IRREGULAR, labels=left_labels)
    transformed = _affine(IRREGULAR)
    right = indexed_body_code(tuple(transformed[i] for i in order), labels=right_labels)
    mapping = {left_labels[source]: right_labels[row] for row, source in enumerate(order)}
    result = code_equivalence_for_mapping(left, right, mapping)
    _assert_witness(left, right, result)
    assert result["permutation"] == [order.index(i) for i in range(5)]


def test_a_wrong_global_mapping_fails_despite_equal_sorted_ratio_values():
    left = indexed_body_code(IRREGULAR)
    right = indexed_body_code(IRREGULAR)
    assert sorted(entry.value for entry in left.entries) == sorted(entry.value for entry in right.entries)
    wrong = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    assert not _oracle_mapping_matches(IRREGULAR, IRREGULAR, [wrong[i] for i in range(5)])
    mismatch = code_equivalence_for_mapping(left, right, wrong)
    _assert_result(mismatch, "NOT_EQUIVALENT")
    assert mismatch['comparison_scope'] == 'supplied_bijection'
    # Failure of that one mapping does not disprove a different global bijection.
    alternative = find_code_equivalence(left, right)
    _assert_result(alternative, 'EQUIVALENT')
    assert alternative['comparison_scope'] == 'existential_bijection'


def test_search_returns_a_single_globally_valid_label_witness():
    order = (1, 0, 2, 3, 4)
    left = indexed_body_code(IRREGULAR, labels=(10, -8, 4, 3, 90))
    right = indexed_body_code(tuple(_affine(IRREGULAR)[i] for i in order), labels=(1, 7, 11, -5, 2))
    result = find_code_equivalence(left, right, max_permutations=factorial(5))
    _assert_witness(left, right, result)
    assert 1 < result["permutations_checked"] <= factorial(5)
    again = find_code_equivalence(left, right, max_permutations=factorial(5))
    assert again == result


def test_budget_exhaustion_is_unknown_without_destroying_complete_codes():
    left = indexed_body_code(IRREGULAR)
    right_points = (IRREGULAR[1], IRREGULAR[0], *IRREGULAR[2:])
    right = indexed_body_code(right_points)
    assert not _oracle_mapping_matches(IRREGULAR, right_points, list(range(5)))
    result = find_code_equivalence(left, right, max_permutations=1)
    _assert_result(result, "UNKNOWN")
    assert result["permutations_checked"] == 1
    assert result["search_exhausted"] is False
    assert left.complete is True and right.complete is True


def test_a_full_negative_search_agrees_with_independent_all_ratio_enumeration():
    left_points = TETRAHEDRON + ((Fraction(1, 4),) * 3,)
    right_points = TETRAHEDRON + ((2, 2, 2),)
    # The complete volume multiplicities differ, independent of any hull argument.
    assert Counter(_volumes(left_points).values()) == {Fraction(1): 1, Fraction(1, 4): 4}
    assert Counter(_volumes(right_points).values()) == {Fraction(1): 1, Fraction(2): 3, Fraction(5): 1}
    assert not any(_oracle_mapping_matches(left_points, right_points, order)
                   for order in permutations(range(5)))
    result = find_code_equivalence(
        indexed_body_code(left_points), indexed_body_code(right_points),
        max_permutations=factorial(5),
    )
    _assert_result(result, "NOT_EQUIVALENT")
    assert result["search_exhausted"] is True
    assert result["permutations_checked"] == factorial(5)


def test_different_cardinality_proves_that_no_label_bijection_exists():
    _assert_result(find_code_equivalence(
        indexed_body_code(TETRAHEDRON), indexed_body_code(IRREGULAR)
    ), "NOT_EQUIVALENT")


def test_an_undefined_code_cannot_be_compared_as_an_empty_defined_code():
    planar = indexed_body_code(((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)))
    defined = indexed_body_code(TETRAHEDRON)
    identity = dict(enumerate(range(4)))
    assert planar.entries == defined.entries == ()
    for left, right in ((planar, planar), (planar, defined), (defined, planar)):
        _assert_result(code_equivalence_for_mapping(left, right, identity), "UNKNOWN")
        _assert_result(find_code_equivalence(left, right), "UNKNOWN")


@pytest.mark.parametrize("points", [TWO_PLANES, _affine(TWO_PLANES)])
def test_parallel_plane_witness_is_verified_by_exact_projection(points):
    labels = (31, -7, 42, 3, 100, -9)
    code = indexed_body_code(points, labels=labels)
    domain = code.theorem_domain
    assert domain["theorem3_applies"] is False
    assert isinstance(domain["reason"], str) and domain["reason"]
    witness = domain["parallel_planes_witness"]
    normal, offsets, groups = witness["normal"], witness["offsets"], witness["groups"]
    assert len(normal) == 3 and any(normal)
    assert all(isinstance(value, Fraction) for value in (*normal, *offsets))
    assert len(offsets) == len(groups) == 2 and offsets[0] != offsets[1]
    flattened_labels = [label for group in groups for label in group]
    assert len(flattened_labels) == len(set(flattened_labels)) == len(labels)
    assert set(flattened_labels) == set(labels)
    point_by_label = dict(zip(labels, code.points, strict=True))
    for offset, group in zip(offsets, groups, strict=True):
        assert group
        for label in group:
            assert sum(a * b for a, b in zip(normal, point_by_label[label], strict=True)) == offset


def test_seven_moment_curve_points_are_inside_the_stated_theorem_domain():
    # A plane intersects (t,t^2,t^3) at <= 3 distinct t values, since a nonzero
    # polynomial of degree <= 3 has <= 3 roots. Two planes cover <= 6 points.
    code = indexed_body_code(MOMENT_CURVE)
    assert all(value > 0 for value in _volumes(MOMENT_CURVE).values())
    domain = code.theorem_domain
    assert domain["theorem3_applies"] is True
    assert domain["parallel_planes_witness"] is None
    assert isinstance(domain["reason"], str) and domain["reason"]
    result = find_code_equivalence(code, code, max_permutations=1)
    _assert_witness(code, code, result)
    assert result["search_exhausted"] is False
    assert result["permutations_checked"] == 1


def test_serialization_round_trips_exact_points_ratios_and_theorem_witness():
    code = indexed_body_code(_affine(TWO_PLANES), labels=(8, -4, 30, 2, 99, 0))
    payload = code.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["status"] == code.status
    assert payload["complete"] is True
    assert payload["labels"] == list(code.labels)
    assert tuple(tuple(_decode_rational(x) for x in point) for point in payload["points"]) == code.points
    assert {
        (tuple(entry["numerator"]), tuple(entry["denominator"])): _decode_rational(entry["value"])
        for entry in payload["entries"]
    } == _entry_dict(code)
    witness = payload["theorem_domain"]["parallel_planes_witness"]
    assert tuple(_decode_rational(x) for x in witness["normal"]) == code.theorem_domain[
        "parallel_planes_witness"
    ]["normal"]
    assert tuple(_decode_rational(x) for x in witness["offsets"]) == code.theorem_domain[
        "parallel_planes_witness"
    ]["offsets"]
    assert list(_all_serialized_rationals(payload))


def test_legal_large_rationals_serialize_without_decimal_integer_conversion():
    # This Cauchy matrix has nonzero determinant and nine very large distinct
    # odd denominators. The resulting volume needs more than 4300 decimal
    # digits even though each input scalar is below the 2048-bit limit.
    base = 1 << 2040
    points = ((0, 0, 0),) + tuple(
        tuple(Fraction(1, base + 1 + 6 * row + 2 * column) for column in range(3))
        for row in range(3)
    )
    expected_volume = _six_volume(points)
    assert expected_volume > 0
    assert expected_volume.denominator.bit_length() > 14300
    code = indexed_body_code(points)
    payload = code.to_dict()
    assert expected_volume in list(_all_serialized_rationals(payload))
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_code_and_entries_are_frozen_and_nested_coordinates_are_tuples():
    code = indexed_body_code(IRREGULAR)
    assert isinstance(code.labels, tuple) and isinstance(code.points, tuple)
    assert isinstance(code.entries, tuple) and isinstance(code.quadruple_volumes, tuple)
    assert all(isinstance(point, tuple) for point in code.points)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        code.status = "UNDEFINED_PLANAR"
    with pytest.raises((FrozenInstanceError, AttributeError)):
        code.entries[0].value = Fraction(999)
    with pytest.raises(TypeError):
        code.points[0][0] = Fraction(999)
    with pytest.raises(TypeError):
        code._theorem_domain["theorem3_applies"] = True
    two_planes = indexed_body_code(TWO_PLANES)
    with pytest.raises(TypeError):
        two_planes._theorem_domain["parallel_planes_witness"]["normal"] = (Fraction(0),) * 3


def test_public_construction_and_replace_cannot_forge_a_complete_empty_code():
    left_points = TETRAHEDRON + ((1, 1, 1),)
    right_points = TETRAHEDRON + ((2, 2, 2),)
    left, right = indexed_body_code(left_points), indexed_body_code(right_points)
    assert not _oracle_mapping_matches(left_points, right_points, list(range(5)))
    _assert_result(code_equivalence_for_mapping(left, right, dict(enumerate(range(5)))), "NOT_EQUIVALENT")
    with pytest.raises(CodeInputError):
        replace(left, entries=())
    with pytest.raises(CodeInputError):
        IndexedBodyCode(
            labels=left.labels, points=left.points, quadruple_volumes=left.quadruple_volumes,
            entries=(), status="DEFINED", complete=True, _theorem_domain=left.theorem_domain,
        )


def test_caller_data_domain_and_serialized_copies_cannot_mutate_the_code():
    points = [list(point) for point in TWO_PLANES]
    labels = [8, -4, 30, 2, 99, 0]
    code = indexed_body_code(points, labels=labels)
    before = code.to_dict()
    points[0][0] = 999
    labels[0] = 999
    domain = code.theorem_domain
    domain["theorem3_applies"] = True
    domain["parallel_planes_witness"]["normal"] = (Fraction(0),) * 3
    payload = code.to_dict()
    payload["labels"][0] = 999
    payload["points"][0][0]["numerator_hex"] = "0x999"
    payload["entries"][0]["value"]["numerator_hex"] = "0x999"
    payload["theorem_domain"]["parallel_planes_witness"]["groups"][0].append(999)
    assert code.to_dict() == before


@pytest.mark.parametrize("invalid_scalars", [
    (True, "1", complex(1, 0), Decimal("1")),
    (float("nan"), float("inf"), float("-inf")),
    (1 << 2048, Fraction(1, 1 << 2048)),
], ids=["unsupported_types", "nonfinite", "coordinate_bit_budget"])
def test_invalid_coordinate_families_are_rejected(invalid_scalars):
    for scalar in invalid_scalars:
        points = [(scalar, 0, 0), *TETRAHEDRON[1:]]
        with pytest.raises((CodeInputError, CodeBudgetError)):
            indexed_body_code(points)


@pytest.mark.parametrize("invalid_point_sets", [
    ((), tuple((i, i * i, i * i * i) for i in range(13))),
    (((0, 0),), ((0, 0, 0, 0),), ((0, 0, 0), (1, 2)), (1, 2, 3)),
], ids=["cardinality", "shape"])
def test_invalid_point_shapes_and_cardinalities_are_rejected(invalid_point_sets):
    for points in invalid_point_sets:
        with pytest.raises((CodeInputError, CodeBudgetError)):
            indexed_body_code(points)


def test_duplicate_coordinates_are_rejected_after_exact_scalar_conversion():
    points = TETRAHEDRON + ((Fraction(1), 0.0, Fraction(0)),)
    with pytest.raises(CodeInputError):
        indexed_body_code(points)


@pytest.mark.parametrize("invalid_label_sets", [
    ((0, 1, 2, 2),),
    ((False, 1, 2, 3), (0.0, 1, 2, 3), ("0", 1, 2, 3)),
    ((-(1 << 63) - 1, 1, 2, 3), (1 << 63, 1, 2, 3)),
    ((0, 1, 2), (0, 1, 2, 3, 4)),
], ids=["duplicate", "type", "signed_int64_bounds", "length"])
def test_invalid_labels_are_rejected_without_silent_normalization(invalid_label_sets):
    for labels in invalid_label_sets:
        with pytest.raises(CodeInputError):
            indexed_body_code(TETRAHEDRON, labels=labels)


def test_signed_int64_label_endpoints_are_valid():
    labels = (-(1 << 63), (1 << 63) - 1, 0, -1)
    code = indexed_body_code(TETRAHEDRON, labels=labels)
    assert code.labels == labels
    assert dict(code.quadruple_volumes) == _volumes(TETRAHEDRON, labels)


def test_entry_budget_parameters_have_strict_integer_bounds():
    for budget in (0, -1, 250001, True, 1.0):
        with pytest.raises((CodeInputError, CodeBudgetError)):
            indexed_body_code(TETRAHEDRON, max_entries=budget)
    assert indexed_body_code(TETRAHEDRON, max_entries=1).complete is True
    assert indexed_body_code(TETRAHEDRON, max_entries=250000).complete is True


def test_entry_budget_uses_actual_nonzero_denominators_and_never_truncates():
    assert len(indexed_body_code(ZERO_VOLUME, max_entries=16).entries) == 16
    with pytest.raises(CodeBudgetError):
        indexed_body_code(ZERO_VOLUME, max_entries=15)
    assert len(indexed_body_code(IRREGULAR, max_entries=20).entries) == 20
    with pytest.raises(CodeBudgetError):
        indexed_body_code(IRREGULAR, max_entries=19)


def test_permutation_budget_parameters_have_strict_integer_bounds():
    code = indexed_body_code(TETRAHEDRON)
    for budget in (0, -1, 100001, True, 1.0):
        with pytest.raises((CodeInputError, CodeBudgetError)):
            find_code_equivalence(code, code, max_permutations=budget)
    _assert_witness(code, code, find_code_equivalence(code, code, max_permutations=100000))


def test_search_point_cap_does_not_reject_a_valid_explicit_mapping():
    points = tuple((t, t * t, t * t * t) for t in range(9))
    code = indexed_body_code(points, max_entries=15750)
    assert len(code.entries) == 126 * 125
    with pytest.raises(CodeBudgetError):
        find_code_equivalence(code, code)
    _assert_witness(code, code, code_equivalence_for_mapping(code, code, dict(enumerate(range(9)))))


@pytest.mark.parametrize("mapping", [
    {0: 0, 1: 1, 2: 2},
    {0: 0, 1: 1, 2: 2, 3: 2},
    {0: 0, 1: 1, 2: 2, 3: 99},
    {False: 0, 1: 1, 2: 2, 3: 3},
])
def test_mapping_must_be_a_total_type_correct_label_bijection(mapping):
    code = indexed_body_code(TETRAHEDRON)
    with pytest.raises(CodeInputError):
        code_equivalence_for_mapping(code, code, mapping)


def test_mapping_validation_is_not_skipped_for_undefined_codes():
    code = indexed_body_code(((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    with pytest.raises(CodeInputError):
        code_equivalence_for_mapping(code, code, {0: 0, 1: 1})


def test_2048_bit_coordinate_boundaries_preserve_exact_integer_and_fraction_values():
    huge = 1 << 2047
    points = ((huge, 0, 0), (0, Fraction(1, huge), 0), (0, 0, 1), (0, 0, 0))
    code = indexed_body_code(points)
    assert code.points[0][0] == huge
    assert code.points[1][1] == Fraction(1, huge)
    assert dict(code.quadruple_volumes) == _volumes(points)
    assert code.status == "DEFINED"


def test_binary64_inputs_mean_the_stored_float_value_not_a_decimal_approximation():
    points = ((0.1, 0, 0), *TETRAHEDRON[1:])
    code = indexed_body_code(points)
    assert code.points[0][0] == Fraction.from_float(0.1)
    assert code.points[0][0] != Fraction(1, 10)
    assert dict(code.quadruple_volumes) == _volumes(points)
