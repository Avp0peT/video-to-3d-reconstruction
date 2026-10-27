"""Independent bounded adversarial review of placed acceptance and bottleneck."""
from copy import deepcopy
from itertools import combinations, permutations
from fractions import Fraction as F
import pytest
from kozlov_lab.registration import candidate_matching as m


def test_all_small_line_sets_against_bijection_oracle():
    lattice=[(0,0),(1,0),(3,0),(7,0)]
    for n in (1,2,3):
        for ap in combinations(lattice,n):
            for bp in combinations(lattice,n):
                a=dict(enumerate(ap)); b={j+10:p for j,p in enumerate(bp)}
                values=sorted({(p[0]-q[0])**2 for p in ap for q in bp})
                optimum=min(max((a[i][0]-b[j][0])**2 for i,j in zip(a,perm)) for perm in permutations(b))
                r=m.placed_bottleneck_exact(a,b)
                assert r['status']=='EXACT_PLACED_BOTTLENECK'
                assert r['distance_squared']==optimum
                assert m.assess_placed_bottleneck_certificate(a,b,r['certificate'])['status']=='VERIFIED'
                for threshold in values:
                    got=m.candidate_acceptance_exact(a,b,threshold)
                    ordinary=all(any((p[0]-q[0])**2<=threshold for q in bp) for p in ap) and all(any((p[0]-q[0])**2<=threshold for p in ap) for q in bp)
                    narrow=any(all((a[i][0]-b[j][0])**2<=threshold for i,j in zip(a,perm)) for perm in permutations(b))
                    assert got['acceptable']==ordinary
                    assert got['narrow_acceptable']==narrow


def test_feasible_matching_at_larger_threshold_is_not_an_optimum_certificate():
    a={0:(0,0),1:(3,0)};b={10:(0,0),11:(3,0)}
    c=deepcopy(m.placed_bottleneck_exact(a,b)['certificate'])
    c['distance_squared']=F(9)
    c['upper']=m.candidate_acceptance_exact(a,b,9)['certificate']
    # A perfectly feasible upper bound does not suffice: preceding threshold is feasible too.
    c['lower']=m.candidate_acceptance_exact(a,b,0)['certificate']
    assert m.assess_placed_bottleneck_certificate(a,b,c)['status']=='INVALID'
    c['lower']['narrow_acceptable']=False
    c['lower']['matching']=None
    c['lower']['obstruction']={'kind':'hall_deficiency','source_subset':(0,1),'target_neighbors':()}
    assert m.assess_placed_bottleneck_certificate(a,b,c)['status']=='INVALID'


@pytest.mark.parametrize('a,b',[({},{}),({},{0:(0,0)}),({0:(0,0)},{}),({False:(0,0)},{0:(0,0)}),({0:(0.,0)},{0:(0,0)})])
def test_empty_and_untyped_input_rejected(a,b):
    assert m.candidate_acceptance_exact(a,b,0)['status']=='UNSUPPORTED'
    assert m.placed_bottleneck_exact(a,b)['status']=='UNSUPPORTED'
    assert m.assess_placed_bottleneck_certificate(a,b,{})['status']=='INVALID'


@pytest.mark.parametrize('mutation',['bool_count','duplicate_hall','hall_no_deficiency','bad_matching','numeric_bool','float','extra'])
def test_forged_certificates_rejected(mutation):
    a={0:(0,0),1:(1,0)};b={10:(0,0)}
    c=deepcopy(m.candidate_acceptance_exact(a,b,1)['certificate'])
    if mutation=='bool_count':c['obstruction']['target_count']=True
    elif mutation=='duplicate_hall':
        b={10:(0,0),11:(10,0)};c=m.candidate_acceptance_exact(a,b,1)['certificate'];c['obstruction']['source_subset']=(0,0)
    elif mutation=='hall_no_deficiency':
        b={10:(0,0),11:(10,0)};c=m.candidate_acceptance_exact(a,b,1)['certificate'];c['obstruction']['source_subset']=(0,)
    elif mutation=='bad_matching':c['narrow_acceptable']=True;c['matching']=((0,10),(1,10));c['obstruction']=None
    elif mutation=='numeric_bool':c['acceptable']=1
    elif mutation=='float':c['radius_squared']=1.
    else:c['invented_global_optimality']=True
    assert m.assess_candidate_acceptance_certificate(a,b,c)['status']=='INVALID'


def test_certificate_depth_work_and_time_bounds_fail_closed():
    a={0:(0,0)};b={10:(1,0)}
    c=m.placed_bottleneck_exact(a,b)['certificate']
    assert m.assess_placed_bottleneck_certificate(a,b,c,max_work=1)['status']=='UNKNOWN'
    assert m.assess_placed_bottleneck_certificate(a,b,c,seconds=0)['status']=='UNKNOWN'
    for budget in (True,-1,2000001):
        assert m.assess_placed_bottleneck_certificate(a,b,c,max_work=budget)['status']=='INVALID'
    corrupted=deepcopy(c);v=None
    for _ in range(22):v=[v]
    corrupted['lower']=v
    assert m.assess_placed_bottleneck_certificate(a,b,corrupted)['status']=='INVALID'
