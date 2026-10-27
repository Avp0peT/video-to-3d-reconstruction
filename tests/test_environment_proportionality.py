from copy import deepcopy
from fractions import Fraction as F
import pytest
from kozlov_lab.coding.environment_proportionality import (
    paper_proportionality_given_mapping as solve,assess_proportionality_certificate as assess,
)


def affine_case():
    a={-4:(0,0),7:(1,0),20:(0,1),90:(2,3)}
    m={i:i+100 for i in a}
    env=dict(a)
    env.update({m[i]:(10+2*p[0],20+3*p[1]) for i,p in a.items()})
    return env,tuple(a),tuple(m.values()),m


def test_nonunit_affine_ratio_global_indices():
    env,a,b,m=affine_case();out=solve(env,a,b,m)
    assert out['status']=='PROPORTIONAL_GIVEN_MAPPING'
    assert out['full_area_proportionality'] and out['degeneracy_preserved'] and out['theorem_eligible']
    assert out['affine_matrix_constructed'] is False and out['unknown_mapping_decided'] is False
    c=out['certificate'];assert c['constant']==F(1,6) and len(c['ledger'])==4
    assert c['source_labels']==tuple(sorted(a))
    for row in c['ledger']:
        assert row['environment_ratio_reference']==row['pair']
        assert c['environment']['ratios'][row['pair']]==F(1,6)
    assert assess(env,a,b,m,c)['status']=='VERIFIED'


@pytest.mark.parametrize('env',[
    {0:(0,0),1:(1,0),2:(0,1)},
    {0:(0,0),1:(2,0),2:(0,2),3:(1,1),4:(3,1)},
])
def test_identical_overlapping_self_pair_not_fabricated(env):
    ids=tuple(env);m={i:i for i in ids};out=solve(env,ids,ids,m);c=out['certificate']
    assert out['status']=='PROPORTIONAL_GIVEN_MAPPING' and c['constant']==1
    rows=[r for r in c['ledger'] if r['source_double_area']]
    assert rows
    for r in rows:
        assert r['environment_ratio_reference']=='SELF_PAIR_EXCLUDED'
        assert r['ratio']==1 and r['pair'] not in c['environment']['ratios']
    assert assess(env,ids,ids,m,c)['status']=='VERIFIED'
    if len(env)==3:assert c['environment']['ratios']=={} and not out['theorem_eligible']


def test_one_way_source_definition_is_not_full_degeneracy_gate():
    env={0:(0,0),1:(1,0),2:(2,0),3:(0,1),4:(1,1)}
    a=(0,1,2,3);b=(0,4,2,3);m={0:0,1:4,2:2,3:3}
    out=solve(env,a,b,m)
    assert out['status']=='PROPORTIONAL_GIVEN_MAPPING'
    assert out['paper_proportionality_given_mapping'] is True
    assert out['degeneracy_preserved'] is out['full_area_proportionality'] is False
    assert out['theorem_eligible'] is False
    row=out['certificate']['ledger'][0]
    assert row['source_double_area']==0 and row['target_double_area']==2
    assert row['ratio']=='NOT_QUANTIFIED'
    assert assess(env,a,b,m,out['certificate'])['status']=='VERIFIED'


def test_target_zero_is_not_divided_or_filtered():
    env={0:(0,0),1:(1,0),2:(0,1),3:(10,0),4:(11,0),5:(12,0)}
    a=(0,1,2);b=(3,4,5);m=dict(zip(a,b));out=solve(env,a,b,m)
    assert out['status']=='NOT_PROPORTIONAL_GIVEN_MAPPING'
    c=out['certificate'];row=c['ledger'][0]
    assert row['ratio']=='UNDEFINED' and row['environment_ratio_reference']=='TARGET_ZERO_DENOMINATOR'
    assert row['pair'] not in c['environment']['ratios']
    assert c['constant']=='NO_POSITIVE_RATIO' and not out['theorem_eligible']
    assert assess(env,a,b,m,c)['status']=='VERIFIED'


def test_inconsistent_ratios_only_refute_given_mapping():
    env,a,b,m=affine_case();env[m[90]]=(15,30)
    out=solve(env,a,b,m)
    assert out['status']=='NOT_PROPORTIONAL_GIVEN_MAPPING'
    assert not out['unknown_mapping_decided']
    assert assess(env,a,b,m,out['certificate'])['status']=='VERIFIED'


def test_no_anchor_is_not_vacuous_proof():
    env={0:(0,0),1:(1,0),2:(2,0)};m={i:i for i in env}
    for ids in ((0,),(0,1),(0,1,2)):
        out=solve(env,ids,ids,{i:i for i in ids})
        assert out['status']=='UNSUPPORTED' and out['certificate'] is None


def test_verifier_independent_of_constructors(monkeypatch):
    from kozlov_lab.coding import environment_proportionality as mod
    env,a,b,m=affine_case();c=solve(env,a,b,m)['certificate']
    def forbidden(*args,**kwargs):raise AssertionError('constructor invoked')
    monkeypatch.setattr(mod,'_construct_ledger',forbidden)
    monkeypatch.setattr(mod,'paper_proportionality_given_mapping',forbidden)
    monkeypatch.setattr(mod.ec,'build_environment_code',forbidden)
    monkeypatch.setattr(mod.ec,'_build',forbidden)
    assert assess(env,a,b,m,c)['status']=='VERIFIED'


@pytest.mark.parametrize('field,value',[
    ('constant',F(1)),('constant',True),('ledger',()),('mapping',((1,2),)),
    ('theorem_eligible',1),('degeneracy_preserved',False),('full_area_proportionality',False),
    ('source_domain','collinear'),('verdict','NOT_PROPORTIONAL_GIVEN_MAPPING'),
])
def test_certificate_tampering(field,value):
    env,a,b,m=affine_case();c=deepcopy(solve(env,a,b,m)['certificate']);c[field]=value
    assert assess(env,a,b,m,c)['status']=='INVALID'


def test_environment_and_ledger_binding():
    env,a,b,m=affine_case();c=solve(env,a,b,m)['certificate']
    mutated=deepcopy(c);mutated['ledger'][0]['pair']='fake'
    assert assess(env,a,b,m,mutated)['status']=='INVALID'
    mutated=deepcopy(c);mutated['environment']['ratios'].pop(next(iter(mutated['environment']['ratios'])))
    assert assess(env,a,b,m,mutated)['status']=='INVALID'
    changed=dict(env);changed[-4]=(F(1,2),0)
    assert assess(changed,a,b,m,c)['status']=='INVALID'
    assert assess(env,a,b,m,c,max_work=0)['status']=='UNKNOWN'
    assert solve(env,a,b,m,max_work=0)['status']=='UNKNOWN'
    assert solve(env,a,b,m,seconds=0)['status']=='UNKNOWN'


def test_input_refusal_and_canonical_subset_order():
    env,a,b,m=affine_case();c=solve(env,a,b,m)['certificate']
    assert solve(env,a[::-1],b[::-1],m)['certificate']==c
    for mapping in (None,{},dict(m,**{'bad':1}),{i:True for i in a}):
        assert solve(env,a,b,mapping)['status']=='UNSUPPORTED'
    assert solve(env,a+(a[0],),b,m)['status']=='UNSUPPORTED'
    bad=dict(env);bad[-4]=(False,0)
    assert solve(bad,a,b,m)['status']=='UNSUPPORTED'
    bad=dict(env);bad[2**63]=(30,40)
    assert solve(bad,a,b,m)['status']=='UNSUPPORTED'


def test_lossless_roundtrip():
    from kozlov_lab.kozlov2025.exact_center import encode_exact,decode_exact
    import json
    env,a,b,m=affine_case();c=solve(env,a,b,m)['certificate']
    restored=decode_exact(json.loads(json.dumps(encode_exact(c))))
    assert restored==c and assess(env,a,b,m,restored)['status']=='VERIFIED'
