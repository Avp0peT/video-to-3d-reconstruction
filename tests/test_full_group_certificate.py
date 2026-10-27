import copy
from unittest.mock import patch
from kozlov_lab.kozlov2025.finite_compatibility_search import search_finite_compatibility
from kozlov_lab.kozlov2025.full_group_certificate import certify_full_group, assess_full_group_certificate

def exact(env,group):
    r=search_finite_compatibility(env,group)
    assert r['status']=='EXACT_GRANULARITY_CERTIFIED',r
    return r['certificate']['granularity_certificate']

def test_nontrivial_unused_label_full_and_no_search_verification():
    env={0:(0,0),1:(1,0),2:(3,0),3:(0,10),4:(1,10),5:(2,10),6:(99,99)}
    group=((0,1,2),(3,4,5))
    gc=exact(env,group)
    r=certify_full_group(env,group,gc)
    assert r['status']=='FULL_GROUP_CERTIFIED',r
    assert r['certificate']['unused_labels']==(6,)
    with patch('kozlov_lab.kozlov2025.full_group_certificate.certify_full_group',side_effect=AssertionError), patch('kozlov_lab.kozlov2025.finite_compatibility_search.search_finite_compatibility',side_effect=AssertionError),patch('kozlov_lab.kozlov2025.exact_center.solve_exact_center',side_effect=AssertionError):
        assert assess_full_group_certificate(env,group,r['certificate'])['status']=='VERIFIED'
    for field,value in [('granularity',True),('unused_labels',(True,)),('claim','NOT_FULL_GIVEN_EXTENSION'),('proof','wrong')]:
        bad=copy.deepcopy(r['certificate']);bad[field]=value
        assert assess_full_group_certificate(env,group,bad)['status']=='INVALID'
    changed={**env,6:(98,99)}
    assert assess_full_group_certificate(changed,group,r['certificate'])['status']=='INVALID'

def test_unknown_and_given_proper_same_m_extension():
    env={0:(0,0),1:(10,0),2:(0,20),3:(10,20),4:(50,50),5:(60,50)}
    group=((0,1),);extension=((0,1),(2,3))
    gc=exact(env,group);ec=exact(env,extension)
    assert certify_full_group(env,group,gc)['status']=='UNKNOWN'
    r=certify_full_group(env,group,gc,extension_certificate=ec)
    assert r['status']=='NOT_FULL_GIVEN_EXTENSION',r
    assert assess_full_group_certificate(env,group,r['certificate'])['status']=='VERIFIED'
    assert certify_full_group(env,group,gc,extension_certificate=gc)['status']=='UNKNOWN'
    bad=copy.deepcopy(r['certificate']);bad['claim']='FULL_GROUP_CERTIFIED'
    assert assess_full_group_certificate(env,group,bad)['status']=='INVALID'
    bad=copy.deepcopy(r['certificate']);bad['extension_certificate']['granularity']=True
    assert assess_full_group_certificate(env,group,bad)['status']=='INVALID'

def test_budget_invalid_granularity_and_complete_identity():
    env={0:(0,0),1:(10,0)};group=((0,1),);gc=exact(env,group)
    assert certify_full_group(env,group,gc,max_work=0)['status']=='UNKNOWN'
    assert certify_full_group(env,group,gc,seconds=0)['status']=='UNKNOWN'
    assert certify_full_group(env,group,gc,max_work=True)['status']=='UNSUPPORTED_INPUT'
    bad=copy.deepcopy(gc);bad['granularity']=1
    assert certify_full_group(env,group,bad)['status']=='UNKNOWN'
    c=certify_full_group(env,group,gc)['certificate']
    assert assess_full_group_certificate({**env,2:(20,20)},group,c)['status']=='INVALID'
    assert assess_full_group_certificate(env,group,c,max_work=0)['status']=='UNKNOWN'
    bad=copy.deepcopy(c);bad['closed']=True
    assert assess_full_group_certificate(env,group,bad)['status']=='INVALID'
