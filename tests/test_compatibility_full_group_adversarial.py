from copy import deepcopy
from unittest.mock import patch
import pytest
from kozlov_lab.kozlov2025 import finite_compatibility_search as search
from kozlov_lab.kozlov2025 import full_group_certificate as full

ENV7={0:(0,0),1:(1,0),2:(3,0),3:(0,10),4:(1,10),5:(2,10),6:(99,99)}
GROUP3=((0,1,2),(3,4,5))
ENV6={0:(0,0),1:(10,0),2:(0,20),3:(10,20),4:(50,50),5:(60,50)}

def exact(env,group):
    r=search.search_finite_compatibility(env,group)
    assert r['status']=='EXACT_GRANULARITY_CERTIFIED',r
    return r['certificate']

@pytest.fixture(scope='module')
def full_case():
    c=exact(ENV7,GROUP3)
    r=full.certify_full_group(ENV7,GROUP3,c['granularity_certificate'])
    assert r['status']=='FULL_GROUP_CERTIFIED'
    return c,r['certificate']

def test_full_three_point_nontrivial_and_forbidden_producers(full_case):
    _,c=full_case
    with patch.object(full,'certify_full_group',side_effect=AssertionError('producer')), patch.object(search,'search_finite_compatibility',side_effect=AssertionError('search')):
        assert full.assess_full_group_certificate(ENV7,GROUP3,c)['status']=='VERIFIED'
    assert c['granularity']==3 and c['unused_labels']==(6,)

@pytest.mark.parametrize('level',('search','full','granularity','compatibility'))
def test_bool_equal_group_labels_rejected(full_case,level):
    wrapper,fc=map(deepcopy,full_case)
    if level=='search': c=wrapper
    elif level=='full': c=fc
    elif level=='granularity': c=fc['granularity_certificate']
    else: c=fc['granularity_certificate']['compatible_witness']
    c['group']=((False,True,2),(3,4,5))
    assess=search.assess_finite_compatibility_certificate if level=='search' else full.assess_full_group_certificate
    assert assess(ENV7,GROUP3,wrapper if level=='search' else fc)['status']=='INVALID'

@pytest.mark.parametrize('level',('wrapper','granularity','compatibility'))
def test_three_layer_exact_m_bool_rejected(full_case,level):
    _,c=map(deepcopy,full_case)
    if level=='wrapper': c['granularity']=True
    elif level=='granularity': c['granularity_certificate']['granularity']=True
    else: c['granularity_certificate']['compatible_witness']['m']=True
    assert full.assess_full_group_certificate(ENV7,GROUP3,c)['status']=='INVALID'

def test_bool_reference_rejected(full_case):
    _,c=map(deepcopy,full_case)
    c['granularity_certificate']['compatible_witness']['reference_index']=False
    assert full.assess_full_group_certificate(ENV7,GROUP3,c)['status']=='INVALID'

@pytest.mark.parametrize('unused',((),(6,6),(0,6),(True,)))
def test_missing_duplicate_or_false_unused_complement(full_case,unused):
    _,c=map(deepcopy,full_case); c['unused_labels']=unused
    assert full.assess_full_group_certificate(ENV7,GROUP3,c)['status']=='INVALID'

def test_unused_coordinate_drift_rejects_both_wrappers(full_case):
    sc,fc=full_case; changed={**ENV7,6:(100,99)}
    assert search.assess_finite_compatibility_certificate(changed,GROUP3,sc)['status']=='INVALID'
    assert full.assess_full_group_certificate(changed,GROUP3,fc)['status']=='INVALID'

def test_legal_given_same_m_extension():
    group=((0,1),); ext=((0,1),(2,3))
    gc=exact(ENV6,group)['granularity_certificate']; ec=exact(ENV6,ext)['granularity_certificate']
    assert full.certify_full_group(ENV6,group,gc)['status']=='UNKNOWN'
    r=full.certify_full_group(ENV6,group,gc,extension_certificate=ec)
    assert r['status']=='NOT_FULL_GIVEN_EXTENSION'
    assert full.assess_full_group_certificate(ENV6,group,r['certificate'])['status']=='VERIFIED'
    assert full.certify_full_group(ENV6,group,gc,extension_certificate=gc)['status']=='UNKNOWN'

def test_point_union_superset_is_not_member_set_extension():
    group=((0,1),); replacement=((0,2),(1,3))
    assert set(group[0]) < set().union(*map(set,replacement))
    gc=exact(ENV6,group)['granularity_certificate']; ec=exact(ENV6,replacement)['granularity_certificate']
    assert full.certify_full_group(ENV6,group,gc,extension_certificate=ec)['status']=='UNKNOWN'
    valid=exact(ENV6,((0,1),(2,3)))['granularity_certificate']
    forged=full.certify_full_group(ENV6,group,gc,extension_certificate=valid)['certificate']
    forged['extension_certificate']=ec
    assert full.assess_full_group_certificate(ENV6,group,forged)['status']=='INVALID'

def test_exhaustion_keeps_unknown_and_all_source_partitions():
    no=dict(status='NO_SUPPORTED_WITNESS',work_used=3)
    with patch.object(search,'search_zero_error_fits',return_value=no), patch.object(search,'collinear_fits',return_value=no):
        r=search.search_finite_compatibility(ENV7,GROUP3)
    assert r['status']=='UNKNOWN' and r['certificate'] is None and not r['general_compatibility_decided']
    rows=r['visited_candidates']; sources=[x for x in rows if x['kind']=='source']
    assert any(x['parts']==((0,2),(1,)) for x in sources)
    assert len({(x['kind'],x['id']) for x in rows})==len(rows)
    assert r['counters']['source_partitions']==len(sources)
    assert r['counters']['zero_calls']==r['counters']['collinear_calls']==len(sources)
    assert r['work_used']>=6*len(sources)

def test_all_members_same_f_and_reference(full_case):
    sc,_=map(deepcopy,full_case)
    w=sc['compatibility_certificate']; fit=w['member_certificates'][1]
    fit['reference_points']=tuple((i,ENV7[i]) for i in GROUP3[1])
    assert search.assess_finite_compatibility_certificate(ENV7,GROUP3,sc)['status']=='INVALID'
    sc,_=map(deepcopy,full_case)
    pc=sc['compatibility_certificate']['member_certificates'][1]['primary_certificate']
    pc['radius_squared']+=1
    assert search.assess_finite_compatibility_certificate(ENV7,GROUP3,sc)['status']=='INVALID'

@pytest.mark.parametrize('budget',(0,50,1000))
def test_no_tight_or_full_claim_after_nested_budget_exhaustion(budget):
    r=search.search_finite_compatibility(ENV7,GROUP3,max_work=budget)
    assert r['status']=='UNKNOWN' and r['certificate'] is None and r['granularity'] is None
    assert r['work_used']>=0 and not r['general_compatibility_decided']
