from fractions import Fraction as F
from copy import deepcopy
import json
import pytest
from kozlov_lab.kozlov2025 import circuit_fits as m
from kozlov_lab.kozlov2025.exact_center import encode_exact,decode_exact

A={0:(F(-1,10),0),1:(F(101,10),0),2:(F(1,10),20),3:(F(99,10),20),4:(31,47)}
B={10:(0,0),11:(10,0),12:(0,20),13:(10,20),14:(31,47)}
AP=tuple((i,) for i in A);BP=tuple((i,) for i in B)

@pytest.fixture(scope='module')
def result():
    r=m.circuit_fits(A,AP,B,BP)
    assert r['status']=='CIRCUIT_FITS_SUFFICIENT_WITNESS',r
    return r

def test_five_center_whole_fits_roundtrip(result):
    c=decode_exact(json.loads(json.dumps(encode_exact(result['certificate']))))
    assert m.assess_circuit_fits_certificate(A,B,c)['status']=='VERIFIED'
    assert len(c['alignment_certificate']['ledger'])==120
    assert c['alignment_certificate']['error_squared']==F(1,100)
    assert len(c['point_checks'])==5

def test_seven_source_points_not_replaced_by_five_centers():
    b={**B,15:(F(-1,100),0),16:(F(1,100),0)}
    parts=((10,15,16),(11,),(12,),(13,),(14,))
    r=m.circuit_fits(A,AP,b,parts)
    assert r['status']=='CIRCUIT_FITS_SUFFICIENT_WITNESS',r
    assert len(r['certificate']['point_checks'])==len(b)==7
    assert m.assess_circuit_fits_certificate(A,b,r['certificate'])['status']=='VERIFIED'
    for _,_,target,_,distances in r['certificate']['point_checks']:
        assert distances[target]<r['certificate']['primary_certificate']['radius_squared']
        assert all(distances[target]<d for i,d in enumerate(distances) if i!=target)

def test_selected_center_optimum_does_not_imply_whole_block_fits():
    b={**B,15:(-1000,0),16:(1000,0)}
    r=m.circuit_fits(A,AP,b,((10,15,16),(11,),(12,),(13,),(14,)))
    assert r['status']=='NO_SUPPORTED_WITNESS',r
    assert r['general_fits_decided'] is False
    assert 'other transforms not excluded' in r['reason']

def test_verifier_constructs_nothing(result,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('construction forbidden')
    for name in ('circuit_fits','align_circuit_centers','solve_exact_center','assess_primary_futlyar'):
        monkeypatch.setattr(m,name,forbidden)
    assert m.assess_circuit_fits_certificate(A,B,result['certificate'])['status']=='VERIFIED'

@pytest.mark.parametrize('kind',('group_bool','cache_bool','drop_point','source_drift','extra_field','bounds','alpha','no_stability'))
def test_altered_whole_certificate_rejected(result,kind):
    c=deepcopy(result['certificate'])
    if kind=='group_bool':c['reference_parts']=((False,),)+AP[1:]
    elif kind=='cache_bool':
        row=c['point_checks'][0];c['point_checks']=((row[0],False,*row[2:]),)+c['point_checks'][1:]
    elif kind=='drop_point':c['point_checks']=c['point_checks'][:-1]
    elif kind=='source_drift':c['source_points']=c['source_points'][:-1]
    elif kind=='extra_field':c['general_complete']=True
    elif kind=='bounds':c['alignment_certificate']['ledger']=c['alignment_certificate']['ledger'][:-1]
    elif kind=='alpha':c['alignment_certificate']['alpha']=(F(2),)*5
    else:c['alignment_certificate']['epsilon_point_displacement']=F(0)
    assert m.assess_circuit_fits_certificate(A,B,c)['status']=='INVALID'

def test_budget_unknown_and_oversize_not_sampled(result):
    assert m.circuit_fits(A,AP,B,BP,max_work=0)['status']=='UNKNOWN'
    assert m.assess_circuit_fits_certificate(A,B,result['certificate'],seconds=0)['status']=='UNKNOWN'
    assert m.circuit_fits(A,AP,B,BP,max_work=True)['status']=='UNSUPPORTED'
    assert m.circuit_fits(A,AP,{**B,15:(100,100),16:(101,100),17:(102,100)},BP)['status']=='UNSUPPORTED'

def test_given_m_and_finite_search_dispatch(monkeypatch):
    from kozlov_lab.kozlov2025.given_m_compatibility import given_m_compatibility,assess_given_m_compatibility_certificate
    from kozlov_lab.kozlov2025 import finite_compatibility_search as search
    group=(tuple(A),)
    c=m.circuit_fits(A,AP,A,AP)['certificate']
    r=given_m_compatibility(A,group,0,AP,[c])
    assert r['status']=='M_COMPATIBLE_SUFFICIENT_WITNESS'
    assert assess_given_m_compatibility_certificate(A,r['certificate'])['status']=='VERIFIED'
    # Only force prior sufficient solver to abstain to test actual new dispatch.
    monkeypatch.setattr(search,'search_zero_error_fits',lambda *a,**k:dict(status='UNKNOWN',work_used=0))
    r=search.search_finite_compatibility(A,group)
    assert r['status']=='EXACT_GRANULARITY_CERTIFIED' and r['granularity']==5,r
    assert r['counters']['circuit_calls']==1
    assert r['certificate']['compatibility_certificate']['member_certificates'][0]['schema']=='circuit-fits-v1'
    assert search.assess_finite_compatibility_certificate(A,group,r['certificate'])['status']=='VERIFIED'
