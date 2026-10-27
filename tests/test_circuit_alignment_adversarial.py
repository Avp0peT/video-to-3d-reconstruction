"""Independent adversarial checks; product modules are not modified."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations, permutations
from math import factorial
import pytest
from kozlov_lab.kozlov2025 import circuit_center_alignment as m


def points(n=5):
    y = dict(enumerate(((0, 0), (10, 0), (0, 20), (10, 20), (31, 47), (53, 61), (79, 83))[:n]))
    signs = (-1, 1, 1, -1, 0, 0, 0)
    x = {i: (F(p[0])+F(signs[i], 10), F(p[1])) for i, p in y.items()}
    return x, y


def det(a, b):
    return a[0]*b[1]-a[1]*b[0]


def sub(a, b):
    return tuple(F(a[j])-F(b[j]) for j in range(2))


def independent_bound(x, y, p, ids):
    coeff = []
    for j in range(4):
        a, b, c = [y[i] for k, i in enumerate(ids) if k != j]
        coeff.append((-1)**j*det(sub(b, a), sub(c, a)))
    if not all(coeff):
        return None
    s = sum(abs(q) for q in coeff)
    v = tuple(-sum(coeff[j]*x[p[i]][k] for j, i in enumerate(ids))/s for k in range(2))
    return sum(q*q for q in v)


def test_complete_subsets_and_permutation_bound_independent():
    x, y = points()
    result = m.align_circuit_centers(x, y)
    assert result['status'] == m.SUCCESS
    cert = result['certificate']
    expected = []
    for p in permutations(range(5)):
        energies = [independent_bound(x, y, p, ids) for ids in combinations(range(5), 4)]
        expected.append(max(e for e in energies if e is not None))
    assert tuple(z['lower_bound_squared'] for z in cert['ledger']) == tuple(expected)
    assert cert['error_squared'] == min(expected) == F(1, 100)
    assert result['circuits_evaluated'] == factorial(5)*5


def test_nonattaining_circuit_bound_unknown_not_negative_fits():
    x, y = points()
    x[4] = (31, 49)
    answer = m.align_circuit_centers(x, y)
    assert answer['status'] == 'UNKNOWN'
    assert 'certificate' not in answer
    assert 'nonexist' not in answer['reason'].lower()


def test_verifier_checks_fifth_residual_even_after_input_rebinding():
    x, y = points()
    c = deepcopy(m.align_circuit_centers(x, y)['certificate'])
    x[4] = (31, 49)
    c['target'] = tuple((i, tuple(F(q) for q in x[i])) for i in sorted(x))
    # Recompute every lower-bound datum to isolate the all-point witness check.
    for z in c['ledger']:
        relation = m._relation(c['source'], z['circuit'])
        z['vector'] = m._vector(c['target'], z['permutation'], z['circuit'], relation)
        z['lower_bound_squared'] = m._energy(z['vector'])
    assert m.assess_circuit_alignment_certificate(x, y, c)['status'] == 'INVALID'


def test_verifier_never_constructs_affine_candidate(monkeypatch):
    x, y = points()
    c = m.align_circuit_centers(x, y)['certificate']
    def forbidden(*args, **kwargs):
        raise AssertionError('solver or producer entered')
    monkeypatch.setattr(m, '_solve', forbidden)
    monkeypatch.setattr(m, 'align_circuit_centers', forbidden)
    assert m.assess_circuit_alignment_certificate(x, y, c)['status'] == 'VERIFIED'


def test_seven_points_budget_is_unknown_with_no_partial_certificate():
    x, y = points(7)
    a = m.align_circuit_centers(x, y, max_work=10000, seconds=30)
    assert a['status'] == 'UNKNOWN' and 'certificate' not in a
    assert 0 < a['permutations_completed'] < 5040
    assert a['circuits_evaluated'] > a['permutations_completed']


def test_tree_accepts_5040_ledger_bound_rejects_5041():
    from kozlov_lab.kozlov2025.zero_error_fits import _budget
    m._tree(tuple(F(0) for _ in range(5040)), _budget(10000, 30))
    with pytest.raises(ValueError, match='tuple bound'):
        m._tree(tuple(F(0) for _ in range(5041)), _budget(10000, 30))


def test_point_radius_recomputed_independently_and_fixed_eps1():
    x, y = points()
    c = m.align_circuit_centers(x, y)['certificate']
    a, b, d = x[0], x[1], x[2]
    u, v = sub(b, a), sub(d, a)
    determinant = det(u, v)
    inv = ((v[1]/determinant, -v[0]/determinant), (-u[1]/determinant, u[0]/determinant))
    k = max(sum(abs(inv[i][j]) for i in range(2)) for j in range(2))
    assert c['epsilon_point_displacement'] == c['epsilon']/(4*max(F(1), k))
    assert 2*c['epsilon_point_displacement']*k < c['epsilon']
    assert c['epsilon1'] == F(1)
    c['epsilon1'] = F(2)
    assert m.assess_circuit_alignment_certificate(x, y, c)['status'] == 'INVALID'


@pytest.mark.parametrize('field', ('epsilon', 'epsilon1', 'epsilon_point_displacement', 'error_squared', 'classification', 'schema'))
def test_boolean_and_status_field_tamper(field):
    x, y = points()
    c = deepcopy(m.align_circuit_centers(x, y)['certificate'])
    c[field] = True
    assert m.assess_circuit_alignment_certificate(x, y, c)['status'] == 'INVALID'


def test_missing_required_field_is_invalid():
    x, y = points()
    c = deepcopy(m.align_circuit_centers(x, y)['certificate'])
    del c['epsilon_point_displacement']
    assert m.assess_circuit_alignment_certificate(x, y, c)['status'] == 'INVALID'


@pytest.mark.parametrize('location', ('top', 'ledger'))
def test_extra_schema_field_is_invalid(location):
    x, y = points()
    c = deepcopy(m.align_circuit_centers(x, y)['certificate'])
    node = c if location == 'top' else c['ledger'][0]
    node['unexpected_extension'] = F(0)
    assert m.assess_circuit_alignment_certificate(x, y, c)['status'] == 'INVALID'
