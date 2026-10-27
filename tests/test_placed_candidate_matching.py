from copy import deepcopy
from fractions import Fraction as F
from itertools import permutations
import json,random
import pytest
from kozlov_lab.registration import candidate_matching as m
from kozlov_lab.kozlov2025.exact_center import encode_exact,decode_exact

A={0:(0,0),1:(F(1,10),0),2:(10,0)}
B={10:(0,0),11:(F(99,10),0),12:(F(101,10),0)}

def test_all_have_neighbors_but_no_bijection():
    r=m.candidate_acceptance_exact(A,B,F(1,25))
    assert r['status']=='EXACT_PLACED_CANDIDATE' and r['acceptable'] and not r['narrow_acceptable']
    assert r['certificate']['obstruction']==dict(kind='hall_deficiency',source_subset=(0,1),target_neighbors=(10,))
    assert m.assess_candidate_acceptance_certificate(A,B,r['certificate'])['status']=='VERIFIED'

def test_fixed_placement_bottleneck_and_roundtrip():
    r=m.placed_bottleneck_exact(A,B)
    assert r['status']=='EXACT_PLACED_BOTTLENECK' and r['distance_squared']==F(2401,25)
    assert r['global_registration_decided'] is False
    c=decode_exact(json.loads(json.dumps(encode_exact(r['certificate']))))
    assert m.assess_placed_bottleneck_certificate(A,B,c)['status']=='VERIFIED'

def test_independent_verifiers_never_search(monkeypatch):
    c=m.placed_bottleneck_exact(A,B)['certificate']
    def forbidden(*a,**k):raise AssertionError('search called')
    for name in ('_matching','_certificate','_hall','placed_bottleneck_exact','candidate_acceptance_exact'):
        monkeypatch.setattr(m,name,forbidden)
    assert m.assess_placed_bottleneck_certificate(A,B,c)['status']=='VERIFIED'

def test_unequal_cardinality_ordinary_acceptance():
    a={0:(0,0)};b={10:(0,0),11:(1,0)}
    r=m.candidate_acceptance_exact(a,b,1)
    assert r['acceptable'] and not r['narrow_acceptable']
    assert m.assess_candidate_acceptance_certificate(a,b,r['certificate'])['status']=='VERIFIED'
    assert m.placed_bottleneck_exact(a,b)['status']=='UNSUPPORTED'

@pytest.mark.parametrize('a,b,r2,expected',[
    ({0:(0,0)},{1:(0,0)},0,True),({0:(0,0)},{1:(1,0)},0,False),
    ({0:(0,0)},{1:(1,0)},1,True),({0:(0,0)},{1:(1,0)},F(999,1000),False)])
def test_exact_threshold_boundary(a,b,r2,expected):
    c=m.candidate_acceptance_exact(a,b,r2)['certificate']
    assert c['narrow_acceptable'] is expected and c['acceptable'] is expected
    assert m.assess_candidate_acceptance_certificate(a,b,c)['status']=='VERIFIED'

@pytest.mark.parametrize('kind',['edge_drop','hall_bool','hall_drop','claim_positive','foreign','extra','radius','binding'])
def test_negative_certificate_tampering(kind):
    c=deepcopy(m.candidate_acceptance_exact(A,B,F(1,25))['certificate'])
    if kind=='edge_drop':c['edges']=c['edges'][:-1]
    elif kind=='hall_bool':c['obstruction']['source_subset']=(False,1)
    elif kind=='hall_drop':c['obstruction']['target_neighbors']=()
    elif kind=='claim_positive':c['narrow_acceptable']=True
    elif kind=='foreign':c['obstruction']['source_subset']=(999,)
    elif kind=='extra':c['general_nonexistence']=True
    elif kind=='radius':c['radius_squared']=True
    else:c['source_points']=c['source_points'][:-1]
    assert m.assess_candidate_acceptance_certificate(A,B,c)['status']=='INVALID'

@pytest.mark.parametrize('kind',['optimum','lower_radius','drop_lower','false_label','extra'])
def test_bottleneck_optimality_tampering(kind):
    c=deepcopy(m.placed_bottleneck_exact(A,B)['certificate'])
    if kind=='optimum':c['distance_squared']+=1
    elif kind=='lower_radius':c['lower']['radius_squared']=0
    elif kind=='drop_lower':c['lower']=None
    elif kind=='false_label':
        mapping=c['upper']['matching'];c['upper']['matching']=((False,mapping[0][1]),)+mapping[1:]
    else:c['global_rotation_optimal']=True
    assert m.assess_placed_bottleneck_certificate(A,B,c)['status']=='INVALID'

def test_bounded_inputs_and_unknown():
    assert m.placed_bottleneck_exact(A,B,max_work=0)['status']=='UNKNOWN'
    assert m.candidate_acceptance_exact(A,B,1,seconds=0)['status']=='UNKNOWN'
    for bad in (True,0.1,-1,F(1,2**300)):
        assert m.candidate_acceptance_exact(A,B,bad)['status']=='UNSUPPORTED'
    assert m.placed_bottleneck_exact({False:(0,0)},{0:(0,0)})['status']=='UNSUPPORTED'
    assert m.placed_bottleneck_exact({0:(0,0),1:(0,0)},{0:(0,0),1:(1,0)})['status']=='UNSUPPORTED'
    c=m.placed_bottleneck_exact(A,B)['certificate']
    assert m.assess_placed_bottleneck_certificate(A,B,c,max_work=0)['status']=='UNKNOWN'

def test_exhaustive_permutation_oracle_seeded_complete_inputs():
    rng=random.Random(2009)
    for n in range(1,7):
        for trial in range(12):
            a=dict(enumerate(rng.sample([(i,j) for i in range(-4,5) for j in range(-3,4)],n)))
            b={i+10:p for i,p in enumerate(rng.sample([(i,j) for i in range(-4,5) for j in range(-3,4)],n))}
            expected=min(max((a[i][0]-b[j][0])**2+(a[i][1]-b[j][1])**2 for i,j in zip(a,perm)) for perm in permutations(b))
            r=m.placed_bottleneck_exact(a,b)
            assert r['status']=='EXACT_PLACED_BOTTLENECK' and r['distance_squared']==expected,(n,trial,r)
            assert m.assess_placed_bottleneck_certificate(a,b,r['certificate'])['status']=='VERIFIED'

def test_augmenting_reassignment_needed():
    # source0 can match both, source1 only target10; greedy without augmentation fails.
    a={0:(1,0),1:(0,0)};b={10:(0,0),11:(2,0)}
    c=m.candidate_acceptance_exact(a,b,1)['certificate']
    assert c['narrow_acceptable'] and c['matching']==((0,11),(1,10))
