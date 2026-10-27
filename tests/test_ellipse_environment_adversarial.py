from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import pytest
from kozlov_lab.coding import ellipse_code2019 as el
from kozlov_lab.coding import environment_code as ec
from kozlov_lab.coding import environment_proportionality as ep

P={-9:(0,0),3:(2,0),17:(0,3),22:(4,5)}

def test_reflection_certificate_has_independent_area_lower_bound():
    p={i:(-F(x,7)+F(y,11),F(y,13)) for i,(x,y) in P.items()}
    r=el.build_ellipse_code2019(p)
    assert r['status']=='EXACT_ELLIPSE_CODE'
    for ids in combinations(sorted(p),3):
        g=r['certificate']['ellipses'][','.join(map(str,ids))]
        a,b,c=(p[i] for i in ids)
        D=abs(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))
        # Independent closed-form det identity; no constructor geometry reused.
        assert g['area_squared_over_pi_squared']==F(4,27)*D*D
        assert sum(g['contact_weights'])==1
    assert el.assess_ellipse_code2019(p,r['certificate'])['status']=='VERIFIED'

@pytest.mark.parametrize('mutation',['cycle','depth','bits','list_center','wrong_source_flags','consistent_scaled_geometry'])
def test_adversarial_geometry_or_tree_fails_closed(mutation):
    c=deepcopy(el.build_ellipse_code2019(P)['certificate'])
    g=next(iter(c['ellipses'].values()))
    if mutation=='cycle':c['ellipses']['loop']=c
    elif mutation=='depth':
        x=0
        for _ in range(26):x=(x,)
        g['contact_weights']=x
    elif mutation=='bits':g['center']=(F(1,2**16385),0)
    elif mutation=='list_center':g['center']=list(g['center'])
    elif mutation=='wrong_source_flags':
        c['area_code']['domain']='plane';c['area_code']['plane_domain']=True;c['area_code']['code_defined']=False
    else:
        g['shape_matrix']=tuple(tuple(2*x for x in row) for row in g['shape_matrix'])
        g['inverse_shape_matrix']=tuple(tuple(x/2 for x in row) for row in g['inverse_shape_matrix'])
        g['area_squared_over_pi_squared']*=4
    assert el.assess_ellipse_code2019(P,c)['status']=='INVALID'

@pytest.mark.parametrize('kind',['environment','restricted','ellipse','proportionality'])
def test_zero_budget_never_validates(kind):
    if kind=='environment':
        c=ec.build_environment_code(P)['certificate'];r=ec.assess_environment_code(P,c,max_work=0)
    elif kind=='restricted':
        c=ec.restrict_environment_code(P,[-9,3,17])['certificate'];r=ec.assess_restricted_code(P,[-9,3,17],c,max_work=0)
    elif kind=='ellipse':
        c=el.build_ellipse_code2019(P)['certificate'];r=el.assess_ellipse_code2019(P,c,max_work=0)
    else:
        ids=list(P);m={i:i for i in ids};c=ep.proportionality_given_mapping(P,ids,ids,m)['certificate']
        r=ep.assess_proportionality_certificate(P,ids,ids,m,c,max_work=0)
    assert r['status']=='UNKNOWN'

def test_overlapping_fixed_triple_is_geometry_not_environment_element():
    p={0:(0,0),1:(1,0),2:(0,1)};m={i:i for i in p}
    r=ep.proportionality_given_mapping(p,list(p),list(p),m)
    assert r['status']=='PROPORTIONAL_GIVEN_MAPPING'
    c=r['certificate'];row=c['ledger'][0]
    assert c['environment']['ratios']=={}
    assert row['ratio']==1 and row['environment_ratio_reference']=='SELF_PAIR_EXCLUDED'
    assert row['pair'] not in c['environment']['ratios']
    assert not r['theorem_eligible'] and r['full_area_proportionality']

def test_literal_restriction_preserves_global_labels_and_degenerate_zero_numerators():
    p={-5:(0,0),2:(1,0),9:(2,0),14:(0,1),18:(3,4)}
    r=ec.restrict_environment_code(p,[14,9,2,-5]);c=r['certificate']
    assert c['restricted']['zero_denominators']==('-5,2,9',)
    assert c['restricted']['ratios']['-5,2,9/-5,2,14']==0
    assert '-5,2,14/-5,2,9' not in c['restricted']['ratios']
    assert ec.assess_restricted_code(p,[-5,2,9,14],c)['status']=='VERIFIED'
    changed=deepcopy(c);changed['restricted']['ratios']['-5,2,14/-5,2,9']=F(1)
    assert ec.assess_restricted_code(p,[-5,2,9,14],changed)['status']=='INVALID'

def test_source_degeneracy_changed_is_not_full_area_proportionality():
    # Shared points retain shared global IDs; middle point lifts in target.
    p={0:(0,0),1:(1,0),2:(2,0),3:(0,1),4:(1,1)}
    r=ep.proportionality_given_mapping(p,[0,1,2,3],[0,4,2,3],{0:0,1:4,2:2,3:3})
    assert r['paper_proportionality_given_mapping'] is True
    assert r['degeneracy_preserved'] is False and r['full_area_proportionality'] is False
    assert not r['theorem_eligible']
    c=r['certificate'];c['full_area_proportionality']=True
    assert ep.assess_proportionality_certificate(p,[0,1,2,3],[0,4,2,3],{0:0,1:4,2:2,3:3},c)['status']=='INVALID'

def test_nonzero_source_to_zero_target_is_not_filtered_out():
    p={0:(0,0),1:(1,0),2:(2,0),3:(0,1)}
    r=ep.proportionality_given_mapping(p,[0,1,3],[0,1,2],{0:0,1:1,3:2})
    assert r['status']=='NOT_PROPORTIONAL_GIVEN_MAPPING'
    assert r['certificate']['ledger'][0]['ratio']=='UNDEFINED'
    assert r['certificate']['ledger'][0]['environment_ratio_reference']=='TARGET_ZERO_DENOMINATOR'
