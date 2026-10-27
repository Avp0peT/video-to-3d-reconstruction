from copy import deepcopy
import pytest
from kozlov_lab.kozlov2025 import certified_granularity as g
from kozlov_lab.kozlov2025.zero_error_fits import given_zero_error_fits
from kozlov_lab.kozlov2025.given_m_compatibility import given_m_compatibility


def make_witness(partition):
    a={0:(0,0),1:(1,0),2:(4,0)};b={3:(10,0),4:(11,0),5:(14,0)}
    bp=[[i+3 for i in part] for part in partition]
    mapping={i:i for i in range(len(partition))}
    first=given_zero_error_fits(a,partition,a,partition,((1,0),(0,1)),(0,0),mapping)
    second=given_zero_error_fits(a,partition,b,bp,((1,0),(0,1)),(-10,0),mapping)
    env={**a,**b};group=[[0,1,2],[3,4,5]]
    answer=given_m_compatibility(env,group,0,partition,[first['certificate'],second['certificate']])
    assert answer['status']=='M_COMPATIBLE_SUFFICIENT_WITNESS'
    return env,group,answer['certificate']


def test_witness_at_cardinality_upper_bound_proves_exact_maximum():
    env,group,w=make_witness([[0],[1],[2]])
    answer=g.granularity_from_witnesses(env,group,[w])
    assert answer['status']=='EXACT_GRANULARITY_CERTIFIED'
    assert answer['granularity']==answer['lower_bound']==answer['upper_bound']==3
    # No false claim to have decided all smaller compatibility predicates.
    assert answer['undecided_m']==[1,2]
    assert g.assess_granularity_certificate(env,answer['certificate'])['status']=='VERIFIED'


def test_lower_witness_is_not_maximum():
    env,group,w=make_witness([[0,1],[2]])
    answer=g.granularity_from_witnesses(env,group,[w])
    assert answer['status']=='BOUNDS_ONLY' and answer['granularity'] is None
    assert answer['lower_bound']==2 and answer['upper_bound']==3
    assert 'certificate' not in answer
    assert g.granularity_from_witnesses(env,group,[])['lower_bound'] is None


def test_independent_upper_bound_and_witness_tampering(monkeypatch):
    env,group,w=make_witness([[0],[1],[2]])
    z=g.granularity_from_witnesses(env,group,[w])['certificate']
    def forbidden(*a,**k):raise AssertionError('producer called')
    monkeypatch.setattr(g,'granularity_from_witnesses',forbidden)
    assert g.assess_granularity_certificate(env,z)['status']=='VERIFIED'
    for field,value in [('granularity',2),('upper_bound',4),('granularity',True)]:
        bad=deepcopy(z);bad[field]=value
        assert g.assess_granularity_certificate(env,bad)['status']=='INVALID'
    bad=deepcopy(z);bad['compatible_witness']['member_certificates']=()
    assert g.assess_granularity_certificate(env,bad)['status']=='INVALID'
    assert g.assess_granularity_certificate(env,z,max_work=1)['status']=='UNKNOWN'
