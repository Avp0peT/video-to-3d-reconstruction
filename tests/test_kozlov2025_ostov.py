import copy
import itertools
import math
from fractions import Fraction as F

import pytest
from kozlov_lab.kozlov2025.ostov import enumerate_collinear_m_ostovs_2025 as solve
from kozlov_lab.kozlov2025.ostov import verify_collinear_ostov_result as verify


@pytest.mark.parametrize('n', range(1, 8))
@pytest.mark.parametrize('direction', [(1, 0), (0, 1), (F(2, 3), F(-4, 5))])
def test_complete_cut_oracle(n, direction):
    points = {100-3*i: (F(i*i)*direction[0]+7, F(i*i)*direction[1]-4) for i in range(n)}
    result = solve(points)
    assert result['status'] == 'COMPLETE'
    assert result['complete'] and verify(points, result)
    assert len(result['ostovs']) == 2**(n-1)
    assert result['counts_by_m'] == {m: math.comb(n-1, m-1) for m in range(1, n+1)}
    order = [100-3*i for i in range(n)]
    expected = set()
    for cuts in itertools.product([0, 1], repeat=n-1):
        groups = [[order[0]]]
        for ident, cut in zip(order[1:], cuts):
            if cut:
                groups.append([])
            groups[-1].append(ident)
        expected.add(frozenset(frozenset(g) for g in groups))
    assert {frozenset(frozenset(g) for g in r['parts']) for r in result['ostovs']} == expected


def test_midpoint_not_mean_and_radical():
    points = {0: (0, 0), 1: (1, 1), 2: (10, 10)}
    row = next(r for r in solve(points)['ostovs'] if r['m'] == 1)
    assert row['centers'] == [(F(5), F(5))]
    assert row['cover_size'] == {'squared': F(200), 'nonnegative_root': True}


def test_rational_affine_covariance():
    points = {i: (F(i*i, 3), F(2*i*i, 3)) for i in range(7)}
    transform = lambda p: (2*p[0]+F(1, 3)*p[1]+7, -p[0]+3*p[1]-F(2, 5))
    first, second = solve(points), solve({i: transform(p) for i, p in points.items()})
    for a, b in zip(first['ostovs'], second['ostovs']):
        assert a['parts'] == b['parts']
        assert [transform(p) for p in a['centers']] == b['centers']
    assert first['ostovs'][0]['cover_size'] != second['ostovs'][0]['cover_size']


@pytest.mark.parametrize('points', [{}, {True: (0, 0)}, {1: (False, 0)},
    {1: (float('inf'), 0)}, {1: (0, 0), 2: (0, 0)}, {i: (i, 0) for i in range(8)}])
def test_invalid(points):
    r = solve(points)
    assert r['status'] == 'UNSUPPORTED_INPUT' and not r['complete']


@pytest.mark.parametrize('points', [{0: (0, 0), 1: (1, 0), 2: (2, F(1, 10**30))},
                                   {0: (0, 0), 1: (1, 0), 2: (0, 1)}])
def test_noncollinear_refusal(points):
    r = solve(points)
    assert r['status'] == 'UNSUPPORTED_DOMAIN' and r['ostovs'] == []


@pytest.mark.parametrize('budget', [{'seconds': 0}, {'total_max_work': 0}, {'max_candidates': 1},
                                  {'total_max_work': 100}])
def test_incomplete_budget(budget):
    r = solve({i: (i, 0) for i in range(7)}, **budget)
    assert not r['complete'] and r['status'] == 'PARTIAL_UNKNOWN'
    if r['ostovs']:
        assert verify({i: (i, 0) for i in range(7)}, r)


@pytest.mark.parametrize('budget', [{'seconds': True}, {'total_max_work': True},
    {'max_candidates': True}, {'seconds': float('nan')}, {'seconds': 31}])
def test_invalid_budget(budget):
    assert solve({0: (0, 0)}, **budget)['status'] == 'UNSUPPORTED_INPUT'


@pytest.mark.parametrize('field', ['center', 'endpoint', 'size', 'labels', 'hash', 'complete', 'candidate'])
def test_tamper_rejected(field):
    points = {0: (0, 0), 1: (1, 0), 2: (10, 0)}
    r = copy.deepcopy(solve(points))
    p = r['ostovs'][0]['pieces'][0]
    if field == 'center': p['center'] = (999, 999)
    elif field == 'endpoint': p['endpoint_labels'] = [0, 0]
    elif field == 'size': p['piece_size']['squared'] = 123
    elif field == 'labels': p['labels'] = [99]
    elif field == 'hash': r['source_coordinate_sha256'] = 'broken'
    elif field == 'complete': r['complete'] = 'true'
    elif field == 'candidate': r['ostovs'][0]['candidate_index'] = -1
    assert not verify(points, r)


def test_stored_binary64_and_labels():
    points = {17: (0.1, 0.0), -9: (0.3, 0.0)}
    r = solve(points)
    assert verify(points, r)
    assert r['ostovs'][0]['centers'][0][0] == (F(0.1)+F(0.3))/2
    assert len(r['ostovs']) == 2  # all covers remain separate witnesses
