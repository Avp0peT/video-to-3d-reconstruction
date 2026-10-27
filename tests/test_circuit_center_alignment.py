from copy import deepcopy
from fractions import Fraction as F
from math import factorial
import pytest
from kozlov_lab.kozlov2025 import circuit_center_alignment as module
from kozlov_lab.kozlov2025.exact_four_center_alignment import align_four_centers


def example(n):
    y = dict(enumerate(((0, 0), (4, 0), (0, 3), (5, 4), (1, 1), (2, 1))[:n]))
    signs = (1, -1, -1, 1, 0, 0)
    x = {i: (F(p[0])+F(signs[i], 10), p[1]) for i, p in y.items()}
    return x, y


@pytest.mark.parametrize('n', (4, 5, 6))
def test_complete_nonzero(n):
    x, y = example(n)
    r = module.align_circuit_centers(x, y)
    assert r['status'] == module.SUCCESS
    c = r['certificate']
    assert c['error_squared'] > 0
    if n > 4:
        assert c['error_squared'] == F(1, 100)
    assert len(c['ledger']) == factorial(n)
    assert len(c['alpha']) == n
    assert r['permutations_completed'] == factorial(n)
    assert c['epsilon'] > 0 and c['epsilon_point_displacement'] > 0
    assert module.assess_circuit_alignment_certificate(x, y, c)['status'] == 'VERIFIED'
    if n == 4:
        assert align_four_centers(x, y)['certificate']['error_squared'] == c['error_squared']


def test_zero_and_independent_verifier(monkeypatch):
    _, y = example(5)
    c = module.align_circuit_centers(y, y)['certificate']
    def forbidden(*args, **kwargs):
        raise AssertionError('producer/solver called')
    monkeypatch.setattr(module, 'align_circuit_centers', forbidden)
    monkeypatch.setattr(module, '_solve', forbidden)
    assert module.assess_circuit_alignment_certificate(y, y, c)['status'] == 'VERIFIED'
    assert c['alpha'] == (F(0),)*5


@pytest.mark.parametrize('field', ('ledger', 'duplicate', 'vector', 'circuit', 'matrix', 'alpha', 'epsilon', 'chosen', 'binding'))
def test_tamper(field):
    x, y = example(5); c = deepcopy(module.align_circuit_centers(x, y)['certificate'])
    if field == 'ledger':
        c['ledger'] = c['ledger'][:-1]
    elif field == 'duplicate':
        c['ledger'] = (c['ledger'][1],)+c['ledger'][1:]
    elif field == 'vector':
        c['ledger'][0]['vector'] = (F(0), F(0))
    elif field == 'circuit':
        c['ledger'][0]['circuit'] = (0, 1, 2, 2)
    elif field == 'matrix':
        c['matrix'] = ((F(0), F(0)), (F(0), F(0)))
    elif field == 'alpha':
        c['alpha'] = (F(2),)*5
    elif field == 'epsilon':
        c['epsilon'] = F(1)
    elif field == 'binding':
        c['source'] = ((True, c['source'][0][1]),)+c['source'][1:]
    else:
        c['chosen'] = True
    assert module.assess_circuit_alignment_certificate(x, y, c)['status'] == 'INVALID'


def test_budget_and_domain():
    x, y = example(6)
    r = module.align_circuit_centers(x, y, max_work=1000)
    assert r['status'] == 'UNKNOWN' and 'certificate' not in r
    assert r['circuits_evaluated'] > 0
    assert module.align_circuit_centers(x, y, seconds=0)['status'] == 'UNKNOWN'
    y[0] = (True, 0)
    assert module.align_circuit_centers(x, y)['status'] == 'UNSUPPORTED'


def test_no_circuit_and_unknown_no_nonexistence_claim():
    y = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (0, 1)}
    assert module.align_circuit_centers(y, y)['status'] == 'UNSUPPORTED'
    x, y = example(5); x[4] = (F(1), F(11, 10))
    r = module.align_circuit_centers(x, y)
    assert r['status'] == 'UNKNOWN' and 'certificate' not in r
