from fractions import Fraction as F
from copy import deepcopy
import pytest
from kozlov_lab.kozlov2025 import given_m_compatibility as m
from kozlov_lab.kozlov2025.zero_error_fits import given_zero_error_fits
from kozlov_lab.kozlov2025.collinear_fits import collinear_fits


@pytest.fixture(scope='module')
def witness():
    a={0:(0,0),1:(1,0),2:(3,0)}
    b={3:(10,0),4:(11,0),5:(12,0)}
    ap=[[0],[1],[2]];bp=[[3],[4],[5]]
    self_fit=given_zero_error_fits(a,ap,a,ap,((1,0),(0,1)),(0,0),{0:0,1:1,2:2})
    other=collinear_fits(a,ap,b,bp)
    assert other['status']=='COLLINEAR_FITS_SUFFICIENT_WITNESS'
    assert other['certificate']['alignment_certificate']['error_squared']==F(1,16)
    env={**a,**b};group=[[0,1,2],[3,4,5]]
    result=m.given_m_compatibility(env,group,0,ap,[self_fit['certificate'],other['certificate']])
    assert result['status']=='M_COMPATIBLE_SUFFICIENT_WITNESS',result
    return env,result['certificate']


def test_mixed_zero_and_nonzero_fits_share_one_futlyar(witness):
    env,z=witness
    assert z['m']==3
    assert [c['schema'] for c in z['member_certificates']]==['zero-error-fits-v1','collinear-fits-v1']
    assert m.assess_given_m_compatibility_certificate(env,z)['status']=='VERIFIED'


def test_mixed_verifier_has_no_search_or_constructor(witness,monkeypatch):
    from kozlov_lab.kozlov2025 import collinear_center_alignment as alignment
    from kozlov_lab.kozlov2025 import primary_futlyar as primary
    from kozlov_lab.kozlov2025 import exact_center
    def forbidden(*a,**k):raise AssertionError('producer called')
    monkeypatch.setattr(alignment,'align_collinear_centers',forbidden)
    monkeypatch.setattr(primary,'assess_primary_futlyar',forbidden)
    monkeypatch.setattr(exact_center,'solve_exact_center',forbidden)
    env,z=witness
    assert m.assess_given_m_compatibility_certificate(env,z)['status']=='VERIFIED'


def test_mixed_dual_failure_cannot_hide_in_group(witness):
    env,z=witness
    bad=deepcopy(z)
    bad['member_certificates'][1]['alignment_certificate']['branches']=()
    assert m.assess_given_m_compatibility_certificate(env,bad)['status']=='INVALID'
    bad=deepcopy(z);bad['member_certificates'][1]['schema']='fits-claimed'
    assert m.assess_given_m_compatibility_certificate(env,bad)['status']=='INVALID'
    assert m.assess_given_m_compatibility_certificate(env,z,max_work=1)['status']=='UNKNOWN'
