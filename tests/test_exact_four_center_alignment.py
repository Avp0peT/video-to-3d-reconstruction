from copy import deepcopy
from fractions import Fraction as F
import pytest
from kozlov_lab.kozlov2025 import exact_four_center_alignment as a

S=dict(enumerate(((0,0),(2,0),(0,2),(2,2))))
def target(points): return dict(enumerate(points))
STABLE=target(((0,0),(0,1),(0,2),(1,1)))
TIED=target(((0,0),(0,1),(0,2),(1,0)))

@pytest.mark.parametrize('x,status,error',[(STABLE,'GLOBAL_OPTIMAL_STABLE',F(1,16)),(TIED,'GLOBAL_OPTIMAL_STABILITY_UNRESOLVED',F(1,8)),(S,'GLOBAL_OPTIMAL_STABLE',F(0))])
def test_complete_global_and_stability(x,status,error):
    r=a.align_four_centers(x,S)
    assert r['status']==status
    c=r['certificate']
    assert c['error_squared']==error and len(c['ledger'])==24
    assert a.assess_four_center_alignment_certificate(x,S,c)['status']=='VERIFIED'
    assert a.zdet(c['ledger'][c['chosen']]['matrix'])
    assert (c['epsilon']>0)==(status=='GLOBAL_OPTIMAL_STABLE')


def test_verifier_does_not_solve_or_produce(monkeypatch):
    c=a.align_four_centers(STABLE,S)['certificate']
    def forbidden(*args,**kwargs): raise AssertionError('solver called')
    monkeypatch.setattr(a,'_solve',forbidden)
    monkeypatch.setattr(a,'align_four_centers',forbidden)
    assert a.assess_four_center_alignment_certificate(STABLE,S,c)['status']=='VERIFIED'

@pytest.mark.parametrize('field,value',[('epsilon',F(1)),('chosen',True),('error_squared',False),('relation',(F(1),F(1),F(1),F(1)))])
def test_tampered_summary(field,value):
    c=deepcopy(a.align_four_centers(STABLE,S)['certificate']); c[field]=value
    assert a.assess_four_center_alignment_certificate(STABLE,S,c)['status']=='INVALID'


def test_ledger_completeness_and_matrix_tampering():
    c=a.align_four_centers(STABLE,S)['certificate']
    for change in ('missing','duplicate','matrix','vector','input'):
        d=deepcopy(c)
        if change=='missing': d['ledger']=d['ledger'][:-1]
        if change=='duplicate': d['ledger']=(d['ledger'][1],)+d['ledger'][1:]
        if change=='matrix': d['ledger'][0]['matrix']=((F(0),F(0)),(F(0),F(0)))
        if change=='vector': d['ledger'][0]['vector']=(F(7),F(0))
        if change=='input': d['source']=d['target']
        assert a.assess_four_center_alignment_certificate(STABLE,S,d)['status']=='INVALID'


def test_singular_candidates_are_retained_and_infimum_summary():
    c=a.align_four_centers(S,S)['certificate']
    assert any(not a.zdet(z['matrix']) for z in c['ledger'])
    # The classification branch must never turn an all-singular minimum into success.
    singular=tuple(dict(z,error_squared=F(0)) for z in c['ledger'] if not a.zdet(z['matrix']))
    assert a._summary(singular)[3]=='GLOBAL_INFIMUM_UNATTAINED'
    assert a._summary(singular)[2]==0

@pytest.mark.parametrize('kwargs',[dict(max_work=0),dict(max_work=8000),dict(seconds=0)])
def test_shared_budget(kwargs):
    assert a.align_four_centers(STABLE,S,**kwargs)['status']=='UNKNOWN'


def test_limits_and_strict_inputs():
    c=a.align_four_centers(STABLE,S)['certificate']; c['epsilon']=F(1<<16385)
    assert a.assess_four_center_alignment_certificate(STABLE,S,c)['status']=='UNKNOWN'
    bad=dict(S); bad[0]=(True,0)
    assert a.align_four_centers(STABLE,bad)['status']=='UNSUPPORTED'
    bad=dict(S); bad[3]=(1,1)
    assert a.align_four_centers(STABLE,bad)['status']=='UNSUPPORTED'
    assert a.align_four_centers(target(((0,0),(1,0),(2,0),(3,0))),S)['status']=='UNSUPPORTED'


def test_common_rational_perturbation_preserves_strong_winner():
    c=a.align_four_centers(STABLE,S)['certificate']; e=c['epsilon']/2
    h=((F(1)+e,e),(-e,F(1)))
    # Translation cancels from the relation, and source affine relation is invariant.
    move=lambda points:{i:tuple(a._linear(h,p)[k]+F(k+7) for k in range(2)) for i,p in points.items()}
    d=a.align_four_centers(move(STABLE),move(S))['certificate']
    chosen=c['chosen']; assert d['ledger'][chosen]['error_squared']==d['error_squared']
    assert d['ledger'][chosen]['vector']==a._linear(h,c['ledger'][chosen]['vector'])


@pytest.mark.parametrize('x,status,error',[
    (target(((0,0),(3,0),(0,2),(4,2))),'GLOBAL_OPTIMAL_STABLE',F(1,16)),
    (target(((0,0),(2,0),(0,2),(-1,-1))),'GLOBAL_OPTIMAL_STABILITY_UNRESOLVED',F(5,8))])
def test_general_position_nonzero_oracles(x,status,error):
    r=a.align_four_centers(x,S); assert r['status']==status
    assert r['certificate']['error_squared']==error
    assert a.assess_four_center_alignment_certificate(x,S,r['certificate'])['status']=='VERIFIED'
    energies={z['error_squared'] for z in r['certificate']['ledger']}
    assert len(energies)>1


def test_all_24_matrix_rows_independently_checked():
    c=a.align_four_centers(STABLE,S)['certificate']
    for i in range(24):
        d=deepcopy(c)
        t=d['ledger'][i]['translation']
        d['ledger'][i]['translation']=(t[0]+1,t[1])
        assert a.assess_four_center_alignment_certificate(STABLE,S,d)['status']=='INVALID'


def test_codec_roundtrip_and_verification_budget():
    from kozlov_lab.kozlov2025.exact_center import encode_exact,decode_exact
    c=a.align_four_centers(STABLE,S)['certificate']
    assert a.assess_four_center_alignment_certificate(STABLE,S,decode_exact(encode_exact(c)))['status']=='VERIFIED'
    assert a.assess_four_center_alignment_certificate(STABLE,S,c,max_work=0)['status']=='UNKNOWN'
    assert a.assess_four_center_alignment_certificate(STABLE,S,c,seconds=0)['status']=='UNKNOWN'
    assert a.align_four_centers(STABLE,S,max_work=True)['status']=='UNSUPPORTED'

@pytest.mark.parametrize('field,value',[('epsilon1',F(2)),('epsilon1',True),('epsilon_point_displacement',F(1)),('epsilon_point_displacement',False)])
def test_point_radius_tampering(field,value):
    c=a.align_four_centers(STABLE,S)['certificate']; c[field]=value
    assert a.assess_four_center_alignment_certificate(STABLE,S,c)['status']=='INVALID'


def test_small_strict_gap_keeps_positive_point_radius():
    # An asymmetric target approaches the nonparallel-tie fixture with a rational tiny displacement.
    tiny=F(1,10**12)
    x=target(((0,0),(2,0),(0,2),(-1-tiny,-1)))
    r=a.align_four_centers(x,S); c=r['certificate']
    assert r['status']=='GLOBAL_OPTIMAL_STABLE'
    assert F(0)<c['epsilon']<F(1,10**10)
    assert F(0)<c['epsilon_point_displacement']<=c['epsilon']/4
    assert c['epsilon1']==F(1)
    assert a.assess_four_center_alignment_certificate(x,S,c)['status']=='VERIFIED'
    tied=a.align_four_centers(TIED,S)['certificate']
    assert tied['epsilon_point_displacement']==0


def test_point_displacement_bridge_uses_first_noncollinear_triple():
    # STABLE starts with three collinear targets; the first affine basis comes later.
    c=a.align_four_centers(STABLE,S)['certificate']
    assert c['epsilon_point_displacement']==c['epsilon']/8
    assert c['epsilon1']==1
    # Compressing the target by 1/100 increases K and reduces the displacement radius.
    small={i:tuple(F(q,100) for q in p) for i,p in STABLE.items()}
    d=a.align_four_centers(small,S)['certificate']
    assert d['epsilon_point_displacement']==d['epsilon']/800
