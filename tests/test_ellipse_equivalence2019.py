from copy import deepcopy
from fractions import Fraction as F
import json
import math

import pytest

from kozlov_lab.coding import ellipse_equivalence2019 as e
from kozlov_lab.kozlov2025.exact_center import encode_exact, decode_exact


P = {11: (0, 0), 22: (2, 0), 33: (0, 2), 44: (3, 3)}


def _transformed(points):
    labels = list(reversed(sorted(points)))
    return {100+i: (2*points[label][0]+points[label][1]+7,
                    -points[label][0]+3*points[label][1]-4)
            for i, label in enumerate(labels)}


def _negative():
    return ({0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)},
            {10: (0, 0), 11: (3, 0), 12: (0, 2), 13: (4, 4)})


def test_renumbered_affine_positive_keeps_complete_indices():
    target = _transformed(P)
    result = e.search_ellipse_equivalence2019(P, target)
    assert result['status'] == 'ELLIPSE_EQUIVALENT'
    assert result['theorem_domain'] is True
    assert result['theorem_conclusion'] == 'AFFINE_EQUIVALENT'
    assert result['affine_mapping_status'] == 'WITNESS_VERIFIED'
    c = result['certificate']
    assert len(c['mapping']) == 4 and len(c['ratio_ledger']) == 12
    assert all(row[2] == row[3] for row in c['ratio_ledger'])
    assert set(dict(c['mapping'])) == set(P)
    assert set(dict(c['mapping']).values()) == set(target)
    assert e.assess_ellipse_equivalence2019(P, target, c)['status'] == 'VERIFIED'


def test_empty_triangle_ratio_set_still_requires_whole_bijection():
    source = {-7: (0, 0), 8: (1, 0), 2**63-1: (0, 1)}
    target = {100: (F(1, 3), 0), 300: (2, 1), 200: (0, 4)}
    result = e.search_ellipse_equivalence2019(source, target)
    assert result['status'] == 'ELLIPSE_EQUIVALENT'
    assert len(result['certificate']['mapping']) == 3
    assert result['certificate']['ratio_ledger'] == ()
    assert result['theorem_domain'] is False
    assert result['affine_mapping_status'] == 'WITNESS_VERIFIED'


def test_domain_and_same_mapping_geometry_are_separate():
    source = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}
    target = {10: (0, 0), 11: (1, 0), 12: (1, 1), 13: (0, 1)}
    result = e.search_ellipse_equivalence2019(source, target)
    assert result['status'] == 'ELLIPSE_EQUIVALENT'
    assert result['theorem_domain'] is False
    assert result['theorem_conclusion'] == 'NOT_APPLICABLE'
    # This lexicographically first code bijection is not an affine mapping,
    # even though these square sets have other affine bijections.
    assert result['affine_mapping_status'] == 'NOT_ESTABLISHED'
    assert result['certificate']['affine_witness'] is None
    assert e.assess_ellipse_equivalence2019(source, target, result['certificate'])['status'] == 'VERIFIED'


def test_negative_has_all_permutation_inequality_witnesses():
    source, target = _negative()
    result = e.search_ellipse_equivalence2019(source, target)
    assert result['status'] == 'NOT_ELLIPSE_EQUIVALENT'
    c = result['certificate']
    rows = [row for chunk in c['rejection_chunks'] for row in chunk]
    assert len(rows) == math.factorial(4)
    assert len({row[0] for row in rows}) == 24
    assert all(row[1][2] != row[1][3] for row in rows)
    assert e.assess_ellipse_equivalence2019(source, target, c)['status'] == 'VERIFIED'


def test_unequal_cardinality_direct_negative():
    source = {i: (i, i*i) for i in range(3)}
    result = e.search_ellipse_equivalence2019(source, P)
    assert result['status'] == 'NOT_ELLIPSE_EQUIVALENT'
    assert result['certificate']['rejection_chunks'] == ()
    assert e.assess_ellipse_equivalence2019(source, P, result['certificate'])['status'] == 'VERIFIED'


def test_eight_points_no_sampling_and_exact_roundtrip():
    source = {i: (F(i, 7), F(i*i, 11)) for i in range(8)}
    # Keep identity label ordering so this checks the complete maximum-size
    # ratio object without requiring a deliberately long permutation search.
    target = {i: (2*x+y+1, x+3*y-2) for i, (x, y) in source.items()}
    result = e.search_ellipse_equivalence2019(source, target)
    assert result['status'] == 'ELLIPSE_EQUIVALENT'
    c = result['certificate']
    assert len(c['source_code']['ellipses']) == 56
    assert len(c['ratio_ledger']) == 3080
    envelope = e.encode_ellipse_equivalence2019(c)
    assert len(envelope['ratio_chunks']) == 13
    restored = e.decode_ellipse_equivalence2019(json.loads(json.dumps(envelope)))
    assert e.assess_ellipse_equivalence2019(source, target, restored)['status'] == 'VERIFIED'


@pytest.mark.parametrize('mutation', ['missing_ratio', 'ratio_bool', 'wrong_index', 'missing_mapping',
                                    'duplicate_target', 'bool_label', 'geometry', 'domain', 'source'])
def test_positive_forgery_rejected(mutation):
    target = _transformed(P)
    c = deepcopy(e.search_ellipse_equivalence2019(P, target)['certificate'])
    if mutation == 'missing_ratio':
        c['ratio_ledger'] = c['ratio_ledger'][:-1]
    elif mutation == 'ratio_bool':
        row = c['ratio_ledger'][0]
        c['ratio_ledger'] = ((row[0], row[1], True, row[3]),) + c['ratio_ledger'][1:]
    elif mutation == 'wrong_index':
        row = c['ratio_ledger'][0]
        c['ratio_ledger'] = ((row[0], '1,2,3/4,5,6', row[2], row[3]),) + c['ratio_ledger'][1:]
    elif mutation == 'missing_mapping':
        c['mapping'] = c['mapping'][:-1]
    elif mutation == 'duplicate_target':
        c['mapping'] = tuple((i, c['mapping'][0][1]) for i, _ in c['mapping'])
    elif mutation == 'bool_label':
        c['mapping'] = ((True, c['mapping'][0][1]),) + c['mapping'][1:]
    elif mutation == 'geometry':
        c['affine_witness']['offset'] = (F(0), F(0))
    elif mutation == 'domain':
        c['theorem_domain'] = False
    else:
        c['source'] = 'S-1996'
    assert e.assess_ellipse_equivalence2019(P, target, c)['status'] == 'INVALID'


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'equal', 'order', 'verdict'])
def test_negative_exhaustion_forgery_rejected(mutation):
    source, target = _negative()
    c = deepcopy(e.search_ellipse_equivalence2019(source, target)['certificate'])
    rows = list(c['rejection_chunks'][0])
    if mutation == 'missing':
        rows.pop()
    elif mutation == 'duplicate':
        rows[-1] = rows[0]
    elif mutation == 'equal':
        image, proof = rows[0]
        rows[0] = (image, (proof[0], proof[1], proof[2], proof[2]))
    elif mutation == 'order':
        rows[0], rows[1] = rows[1], rows[0]
    else:
        c['verdict'] = 'ELLIPSE_EQUIVALENT'
    c['rejection_chunks'] = (tuple(rows),)
    assert e.assess_ellipse_equivalence2019(source, target, c)['status'] == 'INVALID'


@pytest.mark.parametrize('positive', [False, True])
def test_independent_verifier_never_calls_search_or_builders(monkeypatch, positive):
    source, target = (P, _transformed(P)) if positive else _negative()
    c = e.search_ellipse_equivalence2019(source, target)['certificate']
    def forbidden(*args, **kwargs):
        raise AssertionError('constructor called during readback')
    for name in ('search_ellipse_equivalence2019', 'build_ellipse_code2019',
                 '_comparison_rows', '_affine_witness'):
        monkeypatch.setattr(e, name, forbidden)
    assert e.assess_ellipse_equivalence2019(source, target, c)['status'] == 'VERIFIED'


def test_verifier_budget_returns_unknown():
    source, target = _negative()
    c = e.search_ellipse_equivalence2019(source, target)['certificate']
    assert e.assess_ellipse_equivalence2019(source, target, c, max_work=0)['status'] == 'UNKNOWN'


@pytest.mark.parametrize('arguments', [dict(max_work=0), dict(seconds=0)])
def test_budget_cannot_be_negative_verdict(arguments):
    source, target = _negative()
    result = e.search_ellipse_equivalence2019(source, target, **arguments)
    assert result['status'] == 'UNKNOWN'
    assert result['code_equivalent'] is None and result['certificate'] is None


def test_search_interruption_does_not_publish_partial_negative(monkeypatch):
    source, target = _negative()
    def exhausted(*args, **kwargs):
        raise e._Exhausted('frozen search budget')
    monkeypatch.setattr(e, '_comparison_rows', exhausted)
    result = e.search_ellipse_equivalence2019(source, target)
    assert result['status'] == 'UNKNOWN' and result['certificate'] is None


@pytest.mark.parametrize('invalid', [
    {0: (0, 0), 1: (1, 0), 2: (2, 0)},
    {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (2, 0)},
    {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (0, 0)},
    {0: (0.0, 0), 1: (1, 0), 2: (0, 1)},
    {True: (0, 0), 2: (1, 0), 3: (0, 1)},
    {i: (i, i*i) for i in range(9)},
])
def test_unsupported_inputs_are_not_negatives(invalid):
    result = e.search_ellipse_equivalence2019(invalid, P)
    assert result['status'] == 'UNSUPPORTED'
    assert result['code_equivalent'] is None


def test_certificate_binds_source_coordinates_and_labels():
    target = _transformed(P)
    c = e.search_ellipse_equivalence2019(P, target)['certificate']
    changed = dict(P)
    changed[44] = (4, 3)
    assert e.assess_ellipse_equivalence2019(changed, target, c)['status'] == 'INVALID'


def test_negative_codec_roundtrip_requires_separate_mathematical_assessment():
    source, target = _negative()
    c = e.search_ellipse_equivalence2019(source, target)['certificate']
    envelope = e.encode_ellipse_equivalence2019(c)
    restored = e.decode_ellipse_equivalence2019(json.loads(json.dumps(envelope)))
    assert restored == c
    assert e.assess_ellipse_equivalence2019(source, target, restored)['status'] == 'VERIFIED'
    # A missing inequality is still structurally decodable, but never VERIFIED.
    shortened = deepcopy(envelope)
    chunk = decode_exact(shortened['rejection_chunks'][0])
    shortened['rejection_chunks'][0] = encode_exact(chunk[:-1])
    decoded = e.decode_ellipse_equivalence2019(shortened)
    assert e.assess_ellipse_equivalence2019(source, target, decoded)['status'] == 'INVALID'


@pytest.mark.parametrize('mutation', ['extra_envelope', 'missing_core', 'extra_core',
    'ratio_chunks', 'rejection_chunks', 'chunk_size', 'bool_label', 'bool_ratio',
    'integer_domain', 'float', 'nonreduced', 'missing_source_code_field'])
def test_codec_structure_and_tamper_bounds(mutation):
    target = _transformed(P)
    envelope = e.encode_ellipse_equivalence2019(e.search_ellipse_equivalence2019(P, target)['certificate'])
    if mutation == 'extra_envelope':
        envelope['extra'] = None
    elif mutation == 'missing_core':
        del envelope['core']['verdict']
    elif mutation == 'extra_core':
        envelope['core']['extra'] = None
    elif mutation == 'ratio_chunks':
        envelope['ratio_chunks'] *= 14
    elif mutation == 'rejection_chunks':
        envelope['rejection_chunks'] = [encode_exact(())] * 159
    elif mutation == 'chunk_size':
        row = decode_exact(envelope['ratio_chunks'][0])[0]
        envelope['ratio_chunks'][0] = encode_exact((row,) * 257)
    elif mutation == 'bool_label':
        pairs = decode_exact(envelope['core']['mapping'])
        envelope['core']['mapping'] = encode_exact(((True, pairs[0][1]),) + pairs[1:])
    elif mutation == 'bool_ratio':
        rows = decode_exact(envelope['ratio_chunks'][0])
        row = rows[0]
        envelope['ratio_chunks'][0] = encode_exact(((row[0], row[1], True, row[3]),) + rows[1:])
    elif mutation == 'integer_domain':
        envelope['core']['theorem_domain'] = 1
    elif mutation == 'float':
        envelope['core']['mapping'] = 1.0
    elif mutation == 'nonreduced':
        envelope['core']['mapping'] = {'$exact': 'rational', 'numerator': '2', 'denominator': '2'}
    else:
        del envelope['core']['source_code']['ellipses']
    with pytest.raises(ValueError):
        e.decode_ellipse_equivalence2019(envelope)


def test_codec_budget_and_native_boolean_rejection():
    target = _transformed(P)
    c = e.search_ellipse_equivalence2019(P, target)['certificate']
    envelope = e.encode_ellipse_equivalence2019(c)
    with pytest.raises(e._Exhausted):
        e.encode_ellipse_equivalence2019(c, max_work=0)
    with pytest.raises(e._Exhausted):
        e.decode_ellipse_equivalence2019(envelope, max_work=0)
    bad = deepcopy(c)
    bad['affine_witness']['offset'] = (False, F(0))
    with pytest.raises(ValueError, match='boolean'):
        e.encode_ellipse_equivalence2019(bad)


def test_codec_does_not_invoke_mathematical_verifier(monkeypatch):
    target = _transformed(P)
    c = e.search_ellipse_equivalence2019(P, target)['certificate']
    def forbidden(*args, **kwargs):
        raise AssertionError('codec invoked mathematical verifier')
    monkeypatch.setattr(e, '_verify', forbidden)
    restored = e.decode_ellipse_equivalence2019(e.encode_ellipse_equivalence2019(c))
    assert restored == c
