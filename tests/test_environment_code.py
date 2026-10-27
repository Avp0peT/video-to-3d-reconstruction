"""Complete global index certificates, degeneracies and literal restriction."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import json
import pytest
from kozlov_lab.coding import environment_code as e
from kozlov_lab.kozlov2025.exact_center import encode_exact, decode_exact

P={-4:(0,0),7:(1,0),20:(2,0),50:(0,1),60:(2,2)}


def test_all_areas_ratios_zero_numerator_and_self_exclusion():
    r=e.build_environment_code(P);c=r['certificate']
    assert r['status']=='EXACT_ENVIRONMENT_CODE'
    assert c['source']=='S-2019-CODES:57-58'
    assert c['labels']==(-4,7,20,50,60)
    assert len(c['double_areas'])==10
    assert c['double_areas']['-4,7,20']==0
    assert '-4,7,20' in c['zero_denominators']
    assert c['ratios']['-4,7,20/-4,7,50']==0
    assert all(key.split('/')[0]!=key.split('/')[1] for key in c['ratios'])
    assert all(key.split('/')[1] not in c['zero_denominators'] for key in c['ratios'])
    assert len(c['ratios'])==len(c['double_areas'])*(10-len(c['zero_denominators']))-(10-len(c['zero_denominators']))
    assert e.assess_environment_code(P,c)['status']=='VERIFIED'
    assert r['affine_equivalence_decided'] is False


def test_literal_restriction_and_codec_preserve_global_labels():
    labels=(50,-4,7,20)
    r=e.restrict_environment_code(P,labels);c=r['certificate'];sub=c['restricted']
    assert r['status']=='EXACT_RESTRICTED_CODE'
    assert sub['labels']==(-4,7,20,50)
    want={k:v for k,v in c['environment']['ratios'].items() if all(int(x) in labels for part in k.split('/') for x in part.split(','))}
    assert sub['ratios']==want
    assert sub['double_areas']==e.build_environment_code({i:P[i] for i in labels})['certificate']['double_areas']
    restored=decode_exact(json.loads(json.dumps(encode_exact(c))))
    assert e.assess_restricted_code(P,labels,restored)['status']=='VERIFIED'


def test_n8_complete_codec_even_full_restriction():
    p={-100+i:(i,i*i) for i in range(8)}
    c=e.build_environment_code(p)['certificate']
    assert len(c['double_areas'])==56 and len(c['ratios'])==3080
    full=e.restrict_environment_code(p,tuple(p))['certificate']
    encoded=encode_exact(full);restored=decode_exact(json.loads(json.dumps(encoded)))
    assert e.assess_restricted_code(p,tuple(p),restored)['status']=='VERIFIED'


@pytest.mark.parametrize('points,defined,domain',[
    ({0:(0,0)},False,'collinear'),({0:(0,0),1:(1,0)},False,'collinear'),
    ({0:(0,0),1:(1,0),2:(2,0)},False,'collinear'),
    ({0:(0,0),1:(1,0),2:(0,1)},True,'two_parallel_lines'),
    ({0:(0,0),1:(1,0),2:(0,1),3:(1,1)},True,'two_parallel_lines'),
    ({i:(i,i*i) for i in range(5)},True,'plane')])
def test_exact_domains_are_separate_from_code_definition(points,defined,domain):
    c=e.build_environment_code(points)['certificate']
    assert c['code_defined'] is defined and c['domain']==domain
    assert c['plane_domain'] is (domain=='plane')
    assert e.assess_environment_code(points,c)['status']=='VERIFIED'


def test_domain_and_ratios_preserved_by_invertible_affine_map_and_label_order():
    p={i:(i,i*i) for i in [-3,-1,0,2,5]}
    q={i:(2*x+y+3,x+3*y-4) for i,(x,y) in reversed(tuple(p.items()))}
    a=e.build_environment_code(p)['certificate'];b=e.build_environment_code(q)['certificate']
    assert a['ratios']==b['ratios'] and a['plane_domain']==b['plane_domain']==True
    assert all(b['double_areas'][k]==5*v for k,v in a['double_areas'].items())


@pytest.mark.parametrize('bad',[{}, {0:(0,0),1:(0,0)}, {True:(0,0)}, {2**63:(0,0)}, {0:(0.0,0)}, {0:(False,0)}, {0:(F(1,2**257),0)}, {i:(i,0) for i in range(9)}])
def test_input_rejected_without_partial_code(bad):
    r=e.build_environment_code(bad)
    assert r['status']=='UNSUPPORTED' and r['certificate'] is None


@pytest.mark.parametrize('labels',[(),(True,),(999,),(7,7),{7}])
def test_subset_labels_strictly_typed(labels):
    assert e.restrict_environment_code(P,labels)['status']=='UNSUPPORTED'


@pytest.mark.parametrize('kind',['area','missing_area','missing_ratio','extra_ratio','self_pair','zero_denominator','bool_value','bool_label','bool_predicate','domain','noncanonical','extra','binding'])
def test_environment_tampering_rejected(kind):
    c=deepcopy(e.build_environment_code(P)['certificate'])
    if kind=='area':c['double_areas']['-4,7,20']=F(1)
    elif kind=='missing_area':c['double_areas'].pop('-4,7,20')
    elif kind=='missing_ratio':c['ratios'].pop(next(iter(c['ratios'])))
    elif kind=='extra_ratio':c['ratios']['-4,7,50/-4,7,20']=F(0)
    elif kind=='self_pair':c['ratios']['-4,7,50/-4,7,50']=F(1)
    elif kind=='zero_denominator':c['zero_denominators']=()
    elif kind=='bool_value':c['ratios']['-4,7,20/-4,7,50']=False
    elif kind=='bool_label':c['labels']=(False,)+c['labels'][1:]
    elif kind=='bool_predicate':c['code_defined']=1
    elif kind=='domain':c['plane_domain']=not c['plane_domain']
    elif kind=='noncanonical':c['ratios']['-04,7,20/-4,7,50']=c['ratios'].pop('-4,7,20/-4,7,50')
    elif kind=='extra':c['affine_equivalent']=True
    else:c['environment_points']=c['environment_points'][:-1]
    assert e.assess_environment_code(P,c)['status']=='INVALID'


@pytest.mark.parametrize('kind',['subset','restricted','environment','extra','small_fake_environment'])
def test_restriction_tampering_rejected(kind):
    labels=(-4,7,20,50);c=deepcopy(e.restrict_environment_code(P,labels)['certificate'])
    if kind=='subset':c['subset_labels']=(False,7,20,50)
    elif kind=='restricted':c['restricted']['ratios'].pop(next(iter(c['restricted']['ratios'])))
    elif kind=='environment':c['environment']['double_areas'].pop(next(iter(c['environment']['double_areas'])))
    elif kind=='small_fake_environment':c['environment']=e.build_environment_code({i:P[i] for i in labels})['certificate']
    else:c['extra']=True
    assert e.assess_restricted_code(P,labels,c)['status']=='INVALID'


def test_verifier_independent_of_all_constructors(monkeypatch):
    env=e.build_environment_code(P)['certificate'];sub=e.restrict_environment_code(P,(-4,7,50))['certificate']
    def forbidden(*args,**kwargs):raise AssertionError('constructor called')
    for name in ('_build','_projection','build_environment_code','restrict_environment_code'):
        monkeypatch.setattr(e,name,forbidden)
    assert e.assess_environment_code(P,env)['status']=='VERIFIED'
    assert e.assess_restricted_code(P,(-4,7,50),sub)['status']=='VERIFIED'


def test_budget_exhaustion_and_invalid_configuration():
    env=e.build_environment_code(P)['certificate']
    assert e.build_environment_code(P,max_work=0)['status']=='UNKNOWN'
    assert e.restrict_environment_code(P,(-4,7),seconds=0)['status']=='UNKNOWN'
    assert e.assess_environment_code(P,env,max_work=1)['status']=='UNKNOWN'
    for bad in (True,-1,2000001):
        assert e.build_environment_code(P,max_work=bad)['status']=='UNSUPPORTED'
        assert e.assess_environment_code(P,env,max_work=bad)['status']=='INVALID'
