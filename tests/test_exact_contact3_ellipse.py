from fractions import Fraction as F
import copy
import pytest
from kozlov_lab.kozlov2025.exact_contact3_ellipse import complete_input_contact3_ellipse as solve,verify_complete_input_contact3_ellipse as verify


P={0:(0,0),1:(3,0),2:(0,3),3:(F(7,4),F(7,4)),4:(1,1)}


def test_four_extremes_outside_support_hull_complete_optimum():
    r=solve(P);z=r['certificate']
    assert r['status']=='EXACT_OPTIMUM_COMPLETE_INPUT'
    assert r['exact_optimum'] and r['complete']
    assert z['support_labels']==[0,1,2]
    assert z['declared_labels']==[0,1,2,3,4]
    assert z['center']==(1,1) and z['shape_matrix']==((4,-2),(-2,4))
    assert z['all_input_quadratic_values'][3]==F(9,16)
    assert z['area_squared_over_pi_squared']==12
    assert z['major_axis_length']==dict(expression='sqrt(2*(trace+sqrt(discriminant)))',trace=8,discriminant=16)
    assert verify(P,z)


@pytest.mark.parametrize('m,t',[(((2,1),(-1,3)),(10,4)),(((F(1,7),0),(0,F(1,11))),(F(1,5),F(2,3)))])
def test_independent_affine_truth(m,t):
    transformed={i:tuple(sum(m[j][k]*p[k] for k in (0,1))+t[j] for j in (0,1)) for i,p in P.items()}
    z=solve(transformed)['certificate'];q=((4,-2),(-2,4))
    truth=tuple(tuple(sum(m[j][a]*q[a][b]*m[k][b] for a in (0,1) for b in (0,1)) for k in (0,1)) for j in (0,1))
    assert z['shape_matrix']==truth
    assert z['center']==tuple(sum(m[j])+t[j] for j in (0,1))
    assert verify(transformed,z)


def test_square_no_certificate_is_unsupported_not_nonexistence():
    r=solve({0:(0,0),1:(1,0),2:(1,1),3:(0,1)})
    assert r['status']=='UNSUPPORTED_CONTACT_DOMAIN' and r['visited_triples']==4
    assert not r['complete'] and not r['exact_optimum'] and 'certificate' not in r
    assert 'does not disprove' in r['reason']


@pytest.mark.parametrize('p',[{0:(0,0)},{0:(0,0),1:(1,0)},{0:(0,0),1:(1,0),2:(2,0)}])
def test_degenerate_profile_unsupported(p):
    assert solve(p)['status']=='UNSUPPORTED_CONTACT_DOMAIN'


def test_complete_binding_and_outside_point():
    z=solve(P)['certificate']
    assert not verify({i:p for i,p in P.items() if i!=4},z)
    assert not verify(dict(P,**{}),dict(z,declared_labels=[0,1,2]))
    assert not verify({**P,5:(100,100)},z)
    changed={**P,3:(100,100)}
    assert not verify(changed,z)


def test_tampering():
    z=solve(P)['certificate']
    for key,value in [('support_labels',[False,1,2]),('support_labels',[0,1,3]),('declared_labels',[False,1,2,3,4]),('center',(True,1)),('shape_matrix',((5,-2),(-2,4))),('area_squared_over_pi_squared',13),('exact_optimum',False),('proof','fake')]:
        bad=copy.deepcopy(z);bad[key]=value
        assert not verify(P,bad)
    bad=copy.deepcopy(z);bad['all_input_quadratic_values'][3]=1
    assert not verify(P,bad)
    bad=copy.deepcopy(z);bad['all_input_quadratic_values'][0]=True
    assert not verify(P,bad)
    bad=copy.deepcopy(z);bad['support_triangle_certificate']['contact_weights']=(F(1,2),F(1,4),F(1,4))
    assert not verify(P,bad)


def test_verifier_no_solver_hull_or_enumeration(monkeypatch):
    import kozlov_lab.kozlov2025.exact_contact3_ellipse as mod
    import kozlov_lab.kozlov2025.independent_cover as cover
    z=solve(P)['certificate']
    def forbidden(*a,**kw):raise AssertionError('verifier invoked solver or hull')
    monkeypatch.setattr(mod,'complete_input_contact3_ellipse',forbidden)
    monkeypatch.setattr(mod,'exact_triangle_enclosing_ellipse',forbidden)
    monkeypatch.setattr(mod,'combinations',forbidden)
    monkeypatch.setattr(cover,'hull',forbidden)
    assert verify(P,z)


@pytest.mark.parametrize('bad',[True,'1',float('nan'),float('inf'),complex(1,1),F(1,2**2049)])
def test_invalid_coordinates(bad):
    assert solve({0:(0,0),1:(1,0),2:(0,bad)})['status']=='UNSUPPORTED_INPUT'


@pytest.mark.parametrize('kw',[{'max_work':True},{'seconds':True},{'seconds':31},{'max_work':2000001}])
def test_invalid_budget(kw):
    assert solve(P,**kw)['status']=='UNSUPPORTED_INPUT'


@pytest.mark.parametrize('kw',[{'max_work':0},{'seconds':0},{'max_work':10}])
def test_budget_failure_no_success(kw):
    r=solve(P,**kw)
    assert r['status']=='PARTIAL_UNKNOWN' and not r['complete'] and not r['exact_optimum']


def test_bounds_duplicate_bool_label():
    assert solve({i:(i,i*i) for i in range(8)})['status']=='UNSUPPORTED_INPUT'
    assert solve({0:(0,0),1:(0,0),2:(0,1)})['status']=='UNSUPPORTED_INPUT'
    assert solve({False:(0,0),1:(1,0),2:(0,1)})['status']=='UNSUPPORTED_INPUT'


def test_binary64_exact():
    p={0:(0.,0.),1:(0.1,0.),2:(0.,2.**-1000)}
    r=solve(p)
    assert r['status']=='EXACT_OPTIMUM_COMPLETE_INPUT'
    assert r['center']==(F(0.1)/3,F(2.**-1000)/3)
    assert verify(p,r['certificate'])


def test_shared_subcall_budget(monkeypatch):
    import kozlov_lab.kozlov2025.exact_contact3_ellipse as mod
    original=mod.exact_triangle_enclosing_ellipse;budgets=[]
    def wrapped(*a,**kw):
        budgets.append((kw['max_work'],kw['seconds']))
        return original(*a,**kw)
    monkeypatch.setattr(mod,'exact_triangle_enclosing_ellipse',wrapped)
    r=solve({0:(0,0),1:(1,0),2:(1,1),3:(0,1)})
    assert r['visited_triples']==4 and len(budgets)==4
    assert all(budgets[i+1][0]<budgets[i][0] and budgets[i+1][1]<=budgets[i][1] for i in range(3))
