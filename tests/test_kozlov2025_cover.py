"""Exact geometric truth fixtures and independent witness corruption checks; stdlib only."""
from fractions import Fraction as F
import copy
import pytest
from kozlov_lab.kozlov2025.independent_cover import verify_independent_cover_2d as solve, verify_cover_certificate as verify


@pytest.mark.parametrize("points,parts,status", [
    ({0:(0,0),1:(1,0),2:(0,1),3:(3,0),4:(4,0),5:(3,1)},[[0,1,2],[3,4,5]],"VALID_INDEPENDENT_COVER"),
    ({0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)},[[0,1],[2,3]],"NOT_INDEPENDENT"),
    ({0:(0,0),1:(2,0),2:(1,0)},[[0,1],[2]],"NOT_INDEPENDENT"),
    ({0:(0,0),1:(4,0),2:(0,4),3:(1,1)},[[0,1,2],[3]],"NOT_INDEPENDENT"),
    ({0:(0,0),1:(1,1),2:(3,3),3:(4,4)},[[0,1],[2,3]],"VALID_INDEPENDENT_COVER"),
    ({0:(0,0),1:(1,0),2:(0,1)},[[0,1,2]],"VALID_INDEPENDENT_COVER"),
    ({0:(0,0),1:(1,0),2:(0,1)},[[0],[1],[2]],"VALID_INDEPENDENT_COVER"),
    ({0:(-3,-1),1:(3,-1),2:(3,1),3:(-3,1),4:(-1,-3),5:(1,-3),6:(1,3),7:(-1,3)},[[0,1,2,3],[4,5,6,7]],"NOT_INDEPENDENT"),
])
def test_analytic_truth_and_known_invertible_affine_transformation(points,parts,status):
    result=solve(points,parts)
    assert result['status']==status and verify(points,parts,result)
    transformed={100+i:(2*F(x)+F(1,3)*y+F(1,7),-F(y)+F(2,9)) for i,(x,y) in points.items()}
    renamed=[[100+i for i in reversed(part)] for part in reversed(parts)]
    moved=solve(transformed,renamed)
    assert moved['status']==status and verify(transformed,renamed,moved)


@pytest.mark.parametrize("points,parts", [
    (None,[]),([],[]),({0:0},[[0]]),({0:(0,)},[[0]]),({True:(0,0)},[[True]]),
    ({0:(True,0)},[[0]]),({0:(float('nan'),0)},[[0]]),({0:(float('inf'),0)},[[0]]),
    ({0:(float('-inf'),0)},[[0]]),({0:('0',0)},[[0]]),({0:(0,0)},None),
    ({0:(0,0)},[False]),({0:(0,0)},[[False]]),({0:(0,0)},[[0,0]]),
    ({0:(0,0),1:(0,0)},[[0],[1]]),({0:(0,0)},[[1]]),
    ({0:(1<<2049,0)},[[0]]),
])
def test_malformed_input_fails_closed_without_accidental_exceptions(points,parts):
    result=solve(points,parts)
    assert result['status']=='UNSUPPORTED_INPUT'
    assert verify(points,parts,result) is False


def test_incomplete_cover_and_resource_unknown():
    points={0:(0,0),1:(1,0),2:(3,0),3:(4,0)}
    assert solve(points,[[0],[1]])['status']=='INCOMPLETE_COVER'
    result=solve(points,[[0,1],[2,3]],max_work=0)
    assert result['status']=='UNKNOWN_BUDGET' and result['decision_complete'] is False
    assert not verify(points,[[0,1],[2,3]],result)
    for kwargs in ({'max_work':True},{'seconds':True},{'seconds':float('nan')},{'seconds':float('inf')},{'seconds':10**10000}):
        assert solve(points,[[0,1],[2,3]],**kwargs)['status']=='UNSUPPORTED_INPUT'


@pytest.mark.parametrize("change", ['result','part_bool','axis_nan','axis_inf','axis_bool','axis_huge','axis_shape','cert_shape','unknown_label','missing_pair'])
def test_independent_verifier_rejects_malformed_or_polluted_separation(change):
    points={0:(0,0),1:(1,0),2:(3,0),3:(4,0)};parts=[[0,1],[2,3]]
    result=solve(points,parts)
    if change=='result':result=[]
    elif change=='part_bool':result['pair_certificates'][0]['left_part']=False
    elif change=='axis_nan':result['pair_certificates'][0]['axis']=(float('nan'),0)
    elif change=='axis_inf':result['pair_certificates'][0]['axis']=(float('inf'),0)
    elif change=='axis_bool':result['pair_certificates'][0]['axis']=(True,0)
    elif change=='axis_huge':result['pair_certificates'][0]['axis']=(F(1<<8193),0)
    elif change=='axis_shape':result['pair_certificates'][0]['axis']=5
    elif change=='cert_shape':result['pair_certificates']=[5]
    elif change=='unknown_label':parts=[[0,7],[2,3]]
    elif change=='missing_pair':result['pair_certificates']=[]
    assert verify(points,parts,result) is False


def test_independent_verifier_uses_every_raw_point_and_not_solver_hull(monkeypatch):
    import kozlov_lab.kozlov2025.independent_cover as module
    points={0:(0,0),1:(1,0),2:(3,0),3:(4,0)};parts=[[0,1],[2,3]]
    result=solve(points,parts)
    monkeypatch.setattr(module,'hull',lambda *args:pytest.fail('independent verifier called hull'))
    monkeypatch.setattr(module,'pair_certificate',lambda *args:pytest.fail('independent verifier called SAT'))
    assert verify(points,parts,result)
    points[4]=(5,0);parts[0].append(4)
    assert not verify(points,parts,result)


@pytest.mark.parametrize("change", ['weight','bool_label','nan_point','short_point','weights_shape'])
def test_negative_convex_combination_corruption(change):
    points={0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)};parts=[[0,1],[2,3]]
    result=solve(points,parts);witness=result['pair_certificates'][0]
    if change=='weight':witness['left_weights'][0]=(0,F(7,2))
    elif change=='bool_label':witness['left_weights'][0]=(False,F(1,2))
    elif change=='nan_point':witness['point']=(float('nan'),0)
    elif change=='short_point':witness['point']=(0,)
    elif change=='weights_shape':witness['left_weights']=5
    assert verify(points,parts,result) is False


def test_tiny_exact_gap_is_not_epsilon_tolerance():
    gap=F(1,2**200)
    points={0:(0,0),1:(1,0),2:(1+gap,0),3:(2,0)};parts=[[0,1],[2,3]]
    result=solve(points,parts)
    assert result['status']=='VALID_INDEPENDENT_COVER' and verify(points,parts,result)

@pytest.mark.parametrize('points,parts', [({0:(0,0)},[[0]]),({0:(0,0),1:(1,0)},[[0]]),({0:(0,0),1:(1,0)},[[0],[1]])])
def test_zero_time_always_unknown_before_geometry_decision(points,parts):
    result=solve(points,parts,seconds=0,max_work=0)
    assert result['status']=='UNKNOWN_BUDGET'
    assert result['decision_complete'] is False and result['work_used']==0
    assert not verify(points,parts,result)


def test_zero_work_vacuous_singleton_cover_with_positive_time():
    points={0:(0,0)};parts=[[0]]
    result=solve(points,parts,max_work=0,seconds=1)
    assert result['status']=='VALID_INDEPENDENT_COVER' and result['work_used']==0
    assert verify(points,parts,result)


@pytest.mark.parametrize('status', ['VALID_INDEPENDENT_COVER','NOT_INDEPENDENT'])
@pytest.mark.parametrize('key,value', [('schema','wrong'),('contract','wrong'),('coverage_complete',False),('coverage_complete',1),('decision_complete',1),('pairwise_independence',None),('missing_labels',[0])])
def test_certificate_metadata_cannot_contradict_witness(status,key,value):
    points={0:(0,0),1:(2,0),2:(1,0) if status=='NOT_INDEPENDENT' else (3,0)}
    parts=[[0,1],[2]];result=solve(points,parts)
    assert result['status']==status and verify(points,parts,result)
    result[key]=value
    assert not verify(points,parts,result)


def test_hull_labels_are_explicitly_untrusted_diagnostics():
    points={0:(0,0),1:(1,0),2:(3,0)};parts=[[0,1],[2]]
    result=solve(points,parts)
    result['hull_labels']='untrusted diagnostic, not a geometric witness'
    assert verify(points,parts,result)
    result['coverage_complete']=False
    assert not verify(points,parts,result)
