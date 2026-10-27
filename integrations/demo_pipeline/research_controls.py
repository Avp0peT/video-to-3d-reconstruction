"""Automatic controlled experiment, not reconstruction or unknown correspondence.

S-1996 §2 (325-326), Lemma 3: affine-equivalent bodies have equal indexed
codes under one global label mapping. Experiment recipe is project-defined.
Independent oracle uses a 4x4 Leibniz determinant, not the product's 3x3 solver.
"""
from fractions import Fraction
import itertools


def rational(value):
    return {"numerator_hex": hex(value.numerator), "denominator_hex": hex(value.denominator)}


def decode(value):
    return Fraction(int(value["numerator_hex"], 16), int(value["denominator_hex"], 16))


def det4(rows):
    result = Fraction(0)
    for permutation in itertools.permutations(range(4)):
        inversions = sum(permutation[i] > permutation[j] for i in range(4) for j in range(i+1, 4))
        term = Fraction((-1)**inversions)
        for row, column in enumerate(permutation):
            term *= rows[row][column]
        result += term
    return abs(result)


def independent_table(payload):
    """Recompute complete indexed table from the serialized rational coordinates."""
    if payload.get("status") != "DEFINED" or payload.get("complete") is not True:
        raise ValueError("independent oracle requires complete defined code")
    points = {label: tuple(decode(v) for v in row) for label, row in zip(payload["labels"], payload["points"])}
    volumes = {q: det4([(*points[i], Fraction(1)) for i in q])
               for q in itertools.combinations(sorted(points), 4)}
    expected = {(p, q): vp / vq for p, vp in volumes.items() for q, vq in volumes.items() if vq and p != q}
    actual = {(tuple(row["numerator"]), tuple(row["denominator"])): decode(row["value"])
              for row in payload["entries"]}
    if len(actual) != len(payload["entries"]) or actual != expected:
        raise ValueError("independent complete indexed table differs")
    if volumes != {tuple(row["quadruple"]): decode(row["six_volume"]) for row in payload["quadruple_volumes"]}:
        raise ValueError("independent tetrahedron volumes differ")
    return expected


def mapped_equal(left, right, mapping):
    return len(left) == len(right) and all(right.get((tuple(sorted(mapping[i] for i in p)),
        tuple(sorted(mapping[i] for i in q)))) == value for (p, q), value in left.items())


def affine_controls(source_code, *, max_permutations=720):
    """Return controls and two full codes; fixed recipe, no reselection/retry.

    Negative comparison answers existence only when the full 6!=720 domain is
    exhausted or necessary conditions reject. Budget exhaustion stays UNKNOWN.
    """
    from kozlov_lab.stereo.official.indexed_code3d import indexed_body_code, code_equivalence_for_mapping, find_code_equivalence
    if type(max_permutations) is not int or not 1 <= max_permutations <= 720:
        raise ValueError("control permutation budget must be 1..720")
    if len(source_code.labels) != 6 or source_code.complete is not True:
        raise ValueError("controls require exactly six points with defined complete code")
    A = ((Fraction(2), Fraction(1, 3), Fraction(0)),
         (Fraction(0), Fraction(-1), Fraction(1, 5)),
         (Fraction(0), Fraction(0), Fraction(3, 2)))
    translation = (Fraction(1, 7), Fraction(-2, 9), Fraction(3, 11))
    determinant = Fraction(-3)  # fixed upper triangular diagonal product
    transformed = [tuple(sum(A[i][j]*point[j] for j in range(3)) + translation[i] for i in range(3))
                   for point in source_code.points]
    order = (5, 2, 0, 4, 1, 3)
    target_labels = (101, 102, 103, 104, 105, 106)
    affine = indexed_body_code([transformed[i] for i in order], labels=target_labels)
    mapping = {source_code.labels[source]: target_labels[row] for row, source in enumerate(order)}
    broken = [list(point) for point in transformed]
    broken[5][2] += Fraction(1, 7)
    negative = indexed_body_code([broken[i] for i in order], labels=target_labels)
    source_payload, affine_payload, negative_payload = (code.to_dict() for code in (source_code, affine, negative))
    source_table = independent_table(source_payload)
    affine_table = independent_table(affine_payload)
    negative_table = independent_table(negative_payload)
    positive = code_equivalence_for_mapping(source_code, affine, mapping)
    independent_positive = mapped_equal(source_table, affine_table, mapping)
    negative_result = find_code_equivalence(source_code, negative, max_permutations=max_permutations)
    checked = 0
    found = False
    for permutation in itertools.permutations(negative.labels):
        if checked == max_permutations:
            break
        checked += 1
        if mapped_equal(source_table, negative_table, dict(zip(source_code.labels, permutation))):
            found = True
            break
    independent_negative = False if found else True if checked == 720 else None
    negative_established = negative_result["equivalent"] is False and independent_negative is True
    result = {
        "schema": "six-point-affine-controls-v1",
        "status": "PASS" if positive["equivalent"] is True and independent_positive and negative_established else "UNKNOWN" if independent_negative is None or negative_result["equivalent"] is None else "CONTROL_NOT_ESTABLISHED",
        "source": "S-1996 §2 / Lemma3, PDF p.6–7 / journal325–326",
        "recipe_origin": "project-defined controlled experiment, not an author reconstruction algorithm",
        "native_point_ids": list(source_code.labels), "points": 6,
        "affine_matrix": [[rational(v) for v in row] for row in A],
        "translation": [rational(v) for v in translation], "determinant": rational(determinant),
        "target_source_row_order": list(order), "target_labels": list(target_labels),
        "known_global_bijection": [[left, mapping[left]] for left in source_code.labels],
        "positive_supplied_mapping": positive, "independent_positive": independent_positive,
        "negative_perturbation": {"source_native_point_id": source_code.labels[5], "axis": "z", "delta": rational(Fraction(1, 7)), "applied_after_affine": True},
        "negative_existential_comparison": negative_result,
        "independent_negative_no_bijection": independent_negative,
        "independent_permutations_checked": checked, "max_permutations": max_permutations,
        "independent_oracle": "4x4 Fraction Leibniz determinant and every indexed ratio from serialized coordinates; exhaustive mappings only at budget720",
        "counts": {"source": source_payload["counts"], "affine": affine_payload["counts"], "negative": negative_payload["counts"]},
        "limits": ["Fixed six-point subset only; no whole object/cloud conclusion.",
                   "Exact supplied-coordinate arithmetic, not metric accuracy.",
                   "Known constructed correspondence, not recovery of unknown photographed correspondence.",
                   "No new reconstruction, no performance or accuracy gain inferred.",
                   "Code equivalence is reported without inferring affine equivalence."]}
    return result, affine_payload, negative_payload
