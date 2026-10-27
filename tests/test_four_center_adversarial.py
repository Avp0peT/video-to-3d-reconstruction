"""Independent rational oracles and adversarial complete-certificate checks."""
from copy import deepcopy
from fractions import Fraction as Q
from itertools import permutations, product
import pytest
from kozlov_lab.kozlov2025 import exact_four_center_alignment as mod

SOURCE=dict(enumerate(((0,0),(1,0),(0,1),(1,1))))
STABLE=dict(enumerate(((0,0),(3,0),(0,2),(4,2))))
TIES=dict(enumerate(((0,0),(2,0),(0,2),(-1,-1))))

def independent_energy(target, perm):
    # Square source dependence (-1,1,1,-1), normalized sumabs = 4.
    c=(-1,1,1,-1)
    v=tuple(-sum(c[i]*Q(target[perm[i]][k]) for i in range(4))/4 for k in range(2))
    return v, sum(q*q for q in v)

def move(points,h,offset=(Q(7),Q(-11))):
    return {i:tuple(sum(h[k][j]*Q(p[j]) for j in range(2))+offset[k] for k in range(2)) for i,p in points.items()}

@pytest.mark.parametrize('target,expected',[(STABLE,Q(1,16)),(TIES,Q(5,8))])
def test_independent_all24_lower_bound_and_residual_oracle(target,expected):
    c=mod.align_four_centers(target,SOURCE)['certificate']
    energies=[]
    for p,row in zip(permutations(range(4)),c['ledger']):
        v,e=independent_energy(target,p)
        assert row['vector']==tuple(-q for q in v)  # product canonical c0=1
        assert row['error_squared']==e
        energies.append(e)
        for i,b in SOURCE.items():
            r=tuple(sum(row['matrix'][k][j]*Q(b[j]) for j in range(2))+row['translation'][k]-target[p[i]][k] for k in range(2))
            assert r==tuple((-1 if i in (0,3) else 1)*q for q in v)
    assert min(energies)==expected==c['error_squared']

def test_assessor_forbidden_producer_and_solver(monkeypatch):
    c=mod.align_four_centers(STABLE,SOURCE)['certificate']
    def forbidden(*args,**kwargs): raise AssertionError('assessor called fit solver')
    monkeypatch.setattr(mod,'align_four_centers',forbidden)
    monkeypatch.setattr(mod,'_solve',forbidden)
    assert mod.assess_four_center_alignment_certificate(STABLE,SOURCE,c)['status']=='VERIFIED'

@pytest.mark.parametrize('row',range(24))
def test_every_matrix_row_is_checked(row):
    c=deepcopy(mod.align_four_centers(STABLE,SOURCE)['certificate'])
    m=c['ledger'][row]['matrix']
    c['ledger'][row]['matrix']=((m[0][0]+Q(1,97),m[0][1]),m[1])
    assert mod.assess_four_center_alignment_certificate(STABLE,SOURCE,c)['status']=='INVALID'

def test_stability_through_all_sign_matrix_perturbations():
    c=mod.align_four_centers(STABLE,SOURCE)['certificate']
    d=c['epsilon']/2
    for signs in product((-1,1),repeat=4):
        h=((1+signs[0]*d,signs[1]*d),(signs[2]*d,1+signs[3]*d))
        out=mod.align_four_centers(move(STABLE,h),move(SOURCE,h))['certificate']
        assert out['ledger'][c['chosen']]['error_squared']==out['error_squared']

def test_extremely_small_strict_gap_remains_exact():
    target=dict(TIES); target[3]=(-1,-1+Q(1,1<<100))
    r=mod.align_four_centers(target,SOURCE); c=r['certificate']
    assert r['status']=='GLOBAL_OPTIMAL_STABLE'
    assert 0<c['epsilon']<Q(1,1<<90)
    d=c['epsilon']/2; h=((1+d,0),(0,1-d))
    # Multiplying the tiny-gap certificate by input fractions can exceed the
    # product's 256-bit input profile; use the unrestricted Fraction oracle.
    moved=move(target,h)
    energies=[independent_energy(moved,p)[1] for p in permutations(range(4))]
    assert energies[c['chosen']]==min(energies)

def test_nonparallel_ties_break_global_stability_but_keep_unresolved_label():
    c=mod.align_four_centers(TIES,SOURCE)['certificate']
    assert c['classification']=='GLOBAL_OPTIMAL_STABILITY_UNRESOLVED'
    v=c['ledger'][c['chosen']]['vector']; t=Q(1,1<<60)
    # Choose the sign that worsens the selected family's larger coordinate.
    t=t if abs(v[0])>abs(v[1]) else -t
    h=((1+t,0),(0,1))
    changed=mod.align_four_centers(move(TIES,h),move(SOURCE,h))['certificate']
    assert changed['ledger'][c['chosen']]['error_squared']>changed['error_squared']

@pytest.mark.parametrize('field',['chosen','epsilon','error_squared','weight_sum'])
def test_bool_is_not_exact_number(field):
    c=deepcopy(mod.align_four_centers(STABLE,SOURCE)['certificate']); c[field]=True
    assert mod.assess_four_center_alignment_certificate(STABLE,SOURCE,c)['status']=='INVALID'

def test_giant_number_input_and_certificate_have_explicit_outcomes():
    bad=dict(SOURCE); bad[0]=(1<<257,0)
    assert mod.align_four_centers(STABLE,bad)['status']=='UNSUPPORTED'
    c=deepcopy(mod.align_four_centers(STABLE,SOURCE)['certificate']); c['epsilon']=Q(1<<16385)
    assert mod.assess_four_center_alignment_certificate(STABLE,SOURCE,c)['status']=='UNKNOWN'

def test_full_rank_target_has_unattained_global_infimum():
    target=dict(enumerate(((0,0),(10,0),(2,1),(8,1))))
    r=mod.align_four_centers(target,SOURCE)
    assert r['status']=='GLOBAL_INFIMUM_UNATTAINED'
    c=r['certificate']; assert c['error_squared']==Q(1,4)
    winners=[z for z in c['ledger'] if z['error_squared']==c['error_squared']]
    assert all(z['matrix'][0][0]*z['matrix'][1][1]-z['matrix'][0][1]*z['matrix'][1][0]==0 for z in winners)
    assert mod.assess_four_center_alignment_certificate(target,SOURCE,c)['status']=='VERIFIED'

def test_source_definition_bridge_fields_and_tampering():
    c=mod.align_four_centers(STABLE,SOURCE)['certificate']
    assert c['epsilon1']==Q(1) and 0<c['epsilon_point_displacement']<=c['epsilon']/4
    for name in ('epsilon1','epsilon_point_displacement'):
        bad=deepcopy(c); bad[name]+=Q(1,1000)
        assert mod.assess_four_center_alignment_certificate(STABLE,SOURCE,bad)['status']=='INVALID'
