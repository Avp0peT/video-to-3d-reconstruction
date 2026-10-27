from fractions import Fraction as F
import copy
import pytest
from kozlov_lab.kozlov2025.exact_triangle_ellipse import exact_triangle_enclosing_ellipse as solve, verify_exact_triangle_ellipse as verify


def test_reference_triangle_and_interior():
    p = {0:(0,0),1:(3,0),2:(0,3),3:(1,1),4:(F(1,2),0)}
    r = solve(p)
    assert r['status'] == 'EXACT_OPTIMUM'
    z = r['certificate']
    assert z['center'] == (1,1)
    assert z['shape_matrix'] == ((4,-2),(-2,4))
    assert z['determinant_shape'] == 12
    assert z['semiaxis_squared']['trace'] == 8
    assert z['semiaxis_squared']['discriminant'] == 16
    assert verify(p,z)


@pytest.mark.parametrize('matrix,shift', [(((2,1),(-1,3)),(4,-2)), (((F(1,7),0),(0,F(2,11))),(F(2,3),F(-1,9))), (((1,2),(3,1)),(-10,5))])
def test_independent_affine_truth(matrix,shift):
    # Known reference ellipse shape for (0,0),(3,0),(0,3), then Q'=M Q M^T.
    base = ((0,0),(3,0),(0,3),(1,1),(F(1,2),F(1,2)))
    p = {i:tuple(sum(matrix[j][k]*v[k] for k in (0,1))+shift[j] for j in (0,1)) for i,v in enumerate(base)}
    z = solve(p)['certificate']
    truth = ((4,-2),(-2,4))
    expected = tuple(tuple(sum(matrix[j][a]*truth[a][b]*matrix[k][b] for a in (0,1) for b in (0,1)) for k in (0,1)) for j in (0,1))
    assert z['shape_matrix'] == expected
    assert z['center'] == tuple(sum(matrix[j])+shift[j] for j in (0,1))
    assert verify(p,z)


@pytest.mark.parametrize('p', [{0:(0,0)}, {0:(0,0),1:(1,0),2:(2,0)}, {0:(0,0),1:(1,0),2:(1,1),3:(0,1)}])
def test_unsupported_full_hull(p):
    r = solve(p)
    assert r['status'] == 'UNSUPPORTED_DOMAIN'
    assert not r['exact_optimum'] and not r['complete']


@pytest.mark.parametrize('bad', [True, '1',float('nan'),float('inf'),complex(1,2), F(1,2**2049)])
def test_input_semantics(bad):
    assert solve({0:(0,0),1:(1,0),2:(0,bad)})['status'] == 'UNSUPPORTED_INPUT'


def test_binary64_exact_and_near_collinear():
    p = {0:(0.0,0.0),1:(0.1,0.0),2:(0.0,2.0**-1000)}
    r = solve(p)
    assert r['status'] == 'EXACT_OPTIMUM'
    assert r['center'][0] == F(0.1)/3
    assert verify(p,r['certificate'])


def test_tampering_and_complete_binding():
    p = {0:(0,0),1:(3,0),2:(0,3),3:(1,1)}
    z = solve(p)['certificate']
    for field,value in [('center',(0,0)),('shape_matrix',((5,-2),(-2,4))),('contact_weights',(F(1,2),F(1,4),F(1,4))),('area_squared_over_pi_squared',13),('barycentric',{0:(1,0,0)}),('exact_optimum',False),('inverse_shape_matrix',((1,0),(0,1)))]:
        changed = copy.deepcopy(z)
        changed[field] = value
        assert not verify(p,changed)
    assert not verify(dict(p,extra=(10,10)),z)
    assert not verify({i:v for i,v in p.items() if i != 3},z)
    assert solve(dict(p,extra=(10,10)))['status'] == 'UNSUPPORTED_INPUT'
    assert solve(dict(p,**{}),max_work=True)['status'] == 'UNSUPPORTED_INPUT'


def test_budget_and_duplicate():
    p = {0:(0,0),1:(1,0),2:(0,1)}
    assert solve(p,max_work=0)['status'] == 'PARTIAL_UNKNOWN'
    assert solve(p,seconds=0)['status'] == 'PARTIAL_UNKNOWN'
    assert solve(p,seconds=True)['status'] == 'UNSUPPORTED_INPUT'
    assert solve(dict(p,**{}),seconds=31)['status'] == 'UNSUPPORTED_INPUT'
    assert solve({0:(0,0),1:(0,0),2:(0,1)})['status'] == 'UNSUPPORTED_INPUT'


def test_fourth_extreme_cannot_be_dropped():
    p = {0:(0,0),1:(3,0),2:(0,3),3:(3,3),4:(1,1),5:(2,1),6:(1,2)}
    assert solve(p)['status'] == 'UNSUPPORTED_DOMAIN'
    assert solve({i:p[i] for i in (0,1,2)})['status'] == 'EXACT_OPTIMUM'
    assert not verify(p,solve({i:p[i] for i in (0,1,2)})['certificate'])
