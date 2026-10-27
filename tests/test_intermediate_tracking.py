"""Deterministic numerical discrimination; no model or native process runs."""
from copy import deepcopy
import numpy as np
import pytest
from integrations.demo_pipeline.intermediate_tracking import compare_edge


class FakeCV:
    TERM_CRITERIA_COUNT=1
    TERM_CRITERIA_EPS=2

    def __init__(self, mode='translation'):
        self.mode=mode;self.calls=[]

    def calcOpticalFlowPyrLK(self,a,b,xy,unused,**kwargs):
        i=int(a[0,0]);j=int(b[0,0]);self.calls.append((i,j,xy.copy(),kwargs))
        delta=float(j-i)
        if self.mode=='drift':
            # Local closure is .4 px; cumulative reverse drifts 1.2 px.
            delta=delta if j>i else delta+.4
        out=xy.copy();out[0,0,0]+=delta
        st=np.ones((1,1),np.uint8);err=np.zeros((1,1),np.float32)
        if self.mode=='largejump' and abs(j-i)>1:st[:]=0
        if self.mode=='status':st[:]=0
        if self.mode=='error':err[:]=26
        if self.mode=='nan':out[:]=np.nan
        return out,st,err


def inputs(gap=3):
    edge=dict(model_key='m',source_input_index=0,target_input_index=1,
              source_decode_index=0,target_decode_index=gap,decode_path=list(range(gap+1)),
              points=[dict(group_id='m:1:0',source_raw_xy=[10.5,10.5])])
    decoded={i:dict(gray=np.full((32,32),i,np.uint8),mask=np.full((32,32),255,np.uint8) if i in (0,gap) else None,
                   original_mask_status='OK' if i in (0,gap) else 'UNKNOWN_NOT_SEGMENTED',
                   width=32,height=32,path=f'frame{i}.png') for i in range(gap+1)}
    return edge,decoded


def test_large_jump_recovered_same_point_and_fixed_profile():
    e,d=inputs();cv=FakeCV('largejump');r=compare_edge(e,d,cv)
    row=r['rows'][0]
    assert row['outcome']=='RECOVERED' and r['counts']['RECOVERED']==1
    assert row['dense']['observed_raw_coordinates'][-1]['raw_xy']==[13.5,10.5]
    assert row['direct']['steps'][0]['forward']['status']==[[0]]
    assert r['flow_calls']==10
    assert all(c[3]==dict(winSize=(21,21),maxLevel=3,criteria=(3,30,.01)) for c in cv.calls)
    assert not r['identity_verified'] and not r['geometry_accepted'] and not r['intermediate_masks_verified']


def test_fullcycle_catches_cumulative_drift():
    e,d=inputs();r=compare_edge(e,d,FakeCV('drift'));row=r['rows'][0]
    assert row['direct']['status']=='PASSED'
    assert row['dense']['reasons']==['WHOLE_CYCLE_THRESHOLD']
    assert row['dense']['fullcycle']['error_px']>1
    assert all(s['fb_error_px']<1 for s in row['dense']['steps'])
    assert row['outcome']=='LOST'


@pytest.mark.parametrize('mode,reason',[('status','LK_STATUS_FAILED'),('error','LK_ERROR_THRESHOLD'),('nan','LK_XY_NONFINITE')])
def test_bad_lk_retains_outputs(mode,reason):
    e,d=inputs();r=compare_edge(e,d,FakeCV(mode));row=r['rows'][0]
    assert row['outcome']=='BOTH_FAIL'
    assert row['direct']['reasons']==[reason]
    assert row['direct']['steps'][0]['forward']['xy'] is not None


def test_endpoint_mask_and_unknown_intermediates():
    e,d=inputs();d[3]['mask'][10,13]=0
    r=compare_edge(e,d,FakeCV())
    assert r['rows'][0]['outcome']=='BOTH_FAIL'
    assert r['rows'][0]['dense']['reasons']==['POINT_OUTSIDE_ORIGINAL_MASK']
    assert r['rows'][0]['dense']['steps'][0]['forward']['destination_mask_status']=='UNKNOWN_NOT_SEGMENTED'


def test_reverse_uses_measured_endpoint_and_reversed_full_path_no_mutation():
    e,d=inputs();before=deepcopy((e,d));cv=FakeCV();r=compare_edge(e,d,cv)
    assert r['rows'][0]['outcome']=='BOTH_PASS'
    assert [(c[0],c[1]) for c in cv.calls[-3:]]==[(3,2),(2,1),(1,0)]
    assert cv.calls[-3][2][0,0].tolist()==[13.,10.]
    assert e==before[0]
    for i in d:
        assert np.array_equal(d[i]['gray'],before[1][i]['gray'])
        if d[i]['mask'] is not None:assert np.array_equal(d[i]['mask'],before[1][i]['mask'])


def test_budget_valueerror_propagates_with_partial_evidence():
    e,d=inputs();calls=0
    def budget():
        nonlocal calls
        calls+=1
        if calls==3:raise ValueError('budget-stop')
    with pytest.raises(ValueError,match='budget-stop') as exc:compare_edge(e,d,FakeCV(),budget=budget)
    partial=exc.value.partial_result
    assert partial['status']=='INTERRUPTED' and partial['flow_calls']==2
    assert len(partial['rows'][0]['direct']['steps'])==1
    assert partial['rows'][0]['direct']['fullcycle']['steps']==[{}]


@pytest.mark.parametrize('change',[lambda e,d:e.update(decode_path=[0,2,3]),
    lambda e,d:e['points'][0].update(source_raw_xy=[float('nan'),2]),
    lambda e,d:d[1].update(original_mask_status='OK'),
    lambda e,d:d[0].update(gray=np.zeros((31,32),np.uint8)),
    lambda e,d:e.update(points=e['points']*65)])
def test_invalid_inputs_fail_before_cv(change):
    e,d=inputs();change(e,d);cv=FakeCV()
    with pytest.raises(ValueError):compare_edge(e,d,cv)
    assert cv.calls==[]


def test_descending_path_and_reverse_cycle():
    e,d=inputs();e.update(source_decode_index=3,target_decode_index=0,decode_path=[3,2,1,0])
    cv=FakeCV();r=compare_edge(e,d,cv)
    assert r['rows'][0]['outcome']=='BOTH_PASS'
    assert r['rows'][0]['dense']['observed_raw_coordinates'][-1]['raw_xy']==[7.5,10.5]
    assert [(c[0],c[1]) for c in cv.calls[-3:]]==[(0,1),(1,2),(2,3)]


@pytest.mark.parametrize('path',[[0,1,0,3],[0,1,1,3],[3,2,3,0]])
def test_mixed_or_duplicate_path_rejected(path):
    e,d=inputs();e.update(source_decode_index=path[0],target_decode_index=path[-1],decode_path=path)
    cv=FakeCV()
    with pytest.raises(ValueError,match='NONCONSECUTIVE_DECODE_PATH'):compare_edge(e,d,cv)
    assert cv.calls==[]


def test_nonbinary_original_mask_rejected():
    e,d=inputs();d[0]['mask'][0,0]=1;cv=FakeCV()
    with pytest.raises(ValueError,match='NONBINARY_ORIGINAL_MASK'):compare_edge(e,d,cv)
    assert cv.calls==[]


def test_failure_progress_does_not_clobber_primary_budget_exception():
    e,d=inputs()
    primary=RuntimeError('primary-budget')
    def budget():raise primary
    def progress(event):raise ValueError('secondary-progress')
    with pytest.raises(RuntimeError,match='primary-budget') as exc:
        compare_edge(e,d,FakeCV(),budget=budget,progress=progress)
    assert exc.value is primary
    assert exc.value.partial_result['status']=='INTERRUPTED'
