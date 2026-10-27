from fractions import Fraction as F
from copy import deepcopy
import pytest
from kozlov_lab.kozlov2025 import four_center_fits as m
from kozlov_lab.kozlov2025.exact_center import encode_exact,decode_exact
import json

A={0:(0,0),1:(3,0),2:(0,2),3:(4,2)}
B={10:(0,0),11:(1,0),12:(0,1),13:(1,1)}
AP=((0,),(1,),(2,),(3,));BP=((10,),(11,),(12,),(13,))

@pytest.fixture(scope='module')
def result():
    r=m.four_center_fits(A,AP,B,BP)
    assert r['status']=='FOUR_CENTER_FITS_SUFFICIENT_WITNESS',r
    return r

def test_noncollinear_nonzero_full_fits_and_readback(result):
    c=result['certificate']
    assert c['alignment_certificate']['error_squared']==F(1,16)
    assert c['alignment_certificate']['epsilon_point_displacement']>0
    assert len(c['point_checks'])==4
    assert m.assess_four_center_fits_certificate(A,B,c)['status']=='VERIFIED'
    restored=decode_exact(json.loads(json.dumps(encode_exact(c))))
    assert m.assess_four_center_fits_certificate(A,B,restored)['status']=='VERIFIED'

def test_verifier_never_constructs_or_searches(result,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('construction in independent verifier')
    monkeypatch.setattr(m,'align_four_centers',forbidden)
    monkeypatch.setattr(m,'solve_exact_center',forbidden)
    monkeypatch.setattr(m,'assess_primary_futlyar',forbidden)
    monkeypatch.setattr(m,'four_center_fits',forbidden)
    assert m.assess_four_center_fits_certificate(A,B,result['certificate'])['status']=='VERIFIED'

@pytest.mark.parametrize('mutation',('index_bool','identity','bounds','cache_bool','partition','nonminimal','no_stability'))
def test_tamper_rejected(result,mutation):
    c=deepcopy(result['certificate'])
    if mutation=='index_bool':c['selected_index']=False
    elif mutation=='identity':c['source_points']=c['source_points'][:-1]
    elif mutation=='bounds':c['alignment_certificate']['ledger']=c['alignment_certificate']['ledger'][:-1]
    elif mutation=='cache_bool':
        row=list(c['point_checks'][0]);row[1]=False;c['point_checks']=(tuple(row),)+c['point_checks'][1:]
    elif mutation=='partition':c['source_parts']=tuple(reversed(c['source_parts']))
    elif mutation=='nonminimal':
        ac=c['alignment_certificate'];c['selected_index']=next(i for i,r in enumerate(ac['ledger']) if r['error_squared']>ac['error_squared'])
    else:c['alignment_certificate']['epsilon_point_displacement']=F(0)
    assert m.assess_four_center_fits_certificate(A,B,c)['status']=='INVALID'

def test_complete_large_block_fails_not_general_false():
    b={**B,14:(-100,0),15:(100,0)}
    bp=((10,14,15),(11,),(12,),(13,))
    r=m.four_center_fits(A,AP,b,bp)
    assert r['status']=='NO_SUPPORTED_WITNESS'
    assert r['general_fits_decided'] is False
    assert len(r['tried_ledger_indices'])>1 # all invertible winning correspondences tried

def test_unresolved_ties_and_unattained_infimum_not_used_as_fits():
    for target,classification in (({0:(0,0),1:(2,0),2:(0,2),3:(-1,-1)},'GLOBAL_OPTIMAL_STABILITY_UNRESOLVED'),
                                  ({0:(0,0),1:(10,0),2:(2,1),3:(8,1)},'GLOBAL_INFIMUM_UNATTAINED')):
        r=m.four_center_fits(target,AP,B,BP)
        assert r['status']=='NO_SUPPORTED_WITNESS' and r['alignment_status']==classification
        assert r['general_fits_decided'] is False

def test_budget_unknown_and_complete_input_binding(result):
    assert m.four_center_fits(A,AP,B,BP,max_work=0)['status']=='UNKNOWN'
    assert m.assess_four_center_fits_certificate(A,B,result['certificate'],max_work=1)['status']=='UNKNOWN'
    assert m.assess_four_center_fits_certificate(A,{**B,14:(9,9)},result['certificate'])['status']=='INVALID'
    assert m.four_center_fits(A,AP,{**B,10:(True,0)},BP)['status']=='UNSUPPORTED'


def test_given_m_and_automatic_search_accept_independent_four_center_proof(monkeypatch):
    from kozlov_lab.kozlov2025.given_m_compatibility import given_m_compatibility,assess_given_m_compatibility_certificate
    from kozlov_lab.kozlov2025 import finite_compatibility_search as search
    c=m.four_center_fits(A,AP,A,AP)['certificate']
    r=given_m_compatibility(A,((0,1,2,3),),0,AP,[c])
    assert r['status']=='M_COMPATIBLE_SUFFICIENT_WITNESS'
    assert assess_given_m_compatibility_certificate(A,r['certificate'])['status']=='VERIFIED'
    # Exercise the actual four-centre dispatch instead of borrowing the zero
    # solver's self-fit. This controlled no-witness does not disprove zero fit.
    monkeypatch.setattr(search,'search_zero_error_fits',lambda *a,**k:dict(status='NO_SUPPORTED_WITNESS',work_used=0))
    r=search.search_finite_compatibility(A,((0,1,2,3),))
    assert r['status']=='EXACT_GRANULARITY_CERTIFIED' and r['granularity']==4,r
    assert r['counters']['four_center_calls']==1
    assert r['certificate']['compatibility_certificate']['member_certificates'][0]['schema']=='four-center-fits-v1'
    assert search.assess_finite_compatibility_certificate(A,((0,1,2,3),),r['certificate'])['status']=='VERIFIED'
