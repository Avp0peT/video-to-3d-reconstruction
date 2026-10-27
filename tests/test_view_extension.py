"""Synthetic correspondence contracts only; fake LK, no native/model execution."""
from copy import deepcopy
import json

import numpy as np
import pytest

from integrations.demo_pipeline.view_extension import extend_views, PROFILE


class FakeCV:
    TERM_CRITERIA_COUNT = 1
    TERM_CRITERIA_EPS = 2

    def __init__(self, mutation=None):
        self.calls = []
        self.mutation = mutation

    def calcOpticalFlowPyrLK(self, a, b, p, initial, **kwargs):
        self.calls.append((a.copy(), b.copy(), p.copy(), initial, deepcopy(kwargs)))
        # The synthetic field translates .25 pixels per sequence position.
        delta = (float(b[0, 0]) - float(a[0, 0])) * .25
        q = p.copy()
        q[..., 0] += delta
        s = np.ones((len(p), 1), np.uint8)
        e = np.ones((len(p), 1), np.float32)
        return self.mutation(len(self.calls), q, s, e) if self.mutation else (q, s, e)


def fixture(n=7, points=1, models=1):
    frames = [dict(input_index=10 + 10*p, frame_id=f'f{p}', width=32, height=32,
                   original_mask_status='OK', path=f'fixture://image{p}', sha256='a'*64,
                   mask_binding=dict(path=f'fixture://mask{p}', sha256='b'*64),
                   time_seconds=p*.1) for p in range(n)]
    grays = [np.full((32, 32), p, np.uint8) for p in range(n)]
    masks = [np.full((32, 32), 255, np.uint8) for _ in frames]
    rows = []
    camera_models = []
    for model in range(models):
        cameras = {}
        for p in range(3):
            f = frames[p]
            cameras[f['input_index']] = dict(input_index=f['input_index'], frame_id=f['frame_id'],
                image_id=p+1, camera_id=1, camera_model='SIMPLE_RADIAL', params=[32.,16.,16.,0.],
                R=np.eye(3).tolist(), t=[p*.1,0.,0.], width=32, height=32)
        camera_models.append(dict(model_key=str(model), cameras=cameras, unsupported_cameras={}))
        for tid in range(points):
            sources = [dict(input_index=frames[p]['input_index'], frame_id=frames[p]['frame_id'],
                image_id=p+1, raw_pixels=[10.5+p*.25, 8.5+tid*.1],
                evidence_kind='ORIGINAL_STRONG_FLOW') for p in range(3)]
            rows.append(dict(group_id=f'{model}:{tid}:0', model_key=str(model), track_id=tid,
                identity_segment=0, tier='STRICT_CONDITIONAL', xyz=[tid*.1, float(model), 5.],
                sources=sources, source_indices=[s['input_index'] for s in sources],
                frame_ids=[s['frame_id'] for s in sources], branch_scope='MONOTONIC_RADIAL_CONDITIONAL',
                geometry_accepted=False, native_injected=False, identity_verified=False, scale_status='unknown'))
    classification = dict(status='ANCHOR_CANDIDATES_CLASSIFIED', rows=rows, geometry_accepted=False, native_injected=False,
                          identity_verified=False, scale_status='unknown')
    cameras = dict(models=camera_models, unsupported_models=[])
    return frames, grays, masks, classification, cameras


def run(data, cv=None, **kwargs):
    return extend_views(*data, cv or FakeCV(), **kwargs)


def query(result, index=40, model='0'):
    return next(r for r in result['query_rows'] if r['query_input_index']==index and r['model_key']==model)


def accepted(row):
    return [p for p in row['points'] if p['status']=='ACCEPTED_CONDITIONAL_CORRESPONDENCE']


def test_classifier_rights_schema_uses_accepted_not_acceptance():
    # Match anchor_candidate_gate.RIGHTS and its actual return status. The
    # adapter consumes an already recomputed classifier, not an invented gate.
    from integrations.demo_pipeline.anchor_candidate_gate import RIGHTS
    data=fixture()
    assert 'geometry_acceptance' not in RIGHTS
    assert 'geometry_acceptance' not in data[3]
    assert 'geometry_acceptance' not in data[3]['rows'][0]
    assert all(data[3][key]==value for key,value in RIGHTS.items())
    assert all(data[3]['rows'][0][key]==value for key,value in RIGHTS.items())
    assert accepted(query(run(data)))


def test_exact_offsets_reverse_measured_forward_and_two_route_agreement():
    data=fixture();cv=FakeCV();events=[]
    frozen=deepcopy((data[0],data[3],data[4]));arrays=[a.copy() for a in data[1]+data[2]]
    got=run(data,cv,progress=events.append);row=query(got);point=accepted(row)[0]
    assert got['status']=='VIEW_EXTENSION_LEDGER_COMPLETE'
    assert got['queries'] is got['query_rows']
    assert [r['source_input_index'] for r in point['routes']]==[30,20]
    assert [r['path_input_indices'] for r in point['routes']]==[[30,40],[20,30,40]]
    assert point['query_raw_xy']==[11.25,8.5]
    assert point['endpoint_agreement_px']==0.
    forward,reverse=cv.calls[:2]
    np.testing.assert_array_equal(forward[2],np.array([[[10.5,8.]]],np.float32))
    np.testing.assert_array_equal(reverse[2],np.array([[[10.75,8.]]],np.float32))
    assert forward[3] is None and forward[2].dtype==np.float32
    assert forward[4]==dict(winSize=(21,21),maxLevel=3,criteria=(3,30,.01))
    assert forward[0][0,0]==2 and forward[1][0,0]==3
    assert reverse[0][0,0]==3 and reverse[1][0,0]==2
    assert (data[0],data[3],data[4])==frozen
    for old,new in zip(arrays,data[1]+data[2]):np.testing.assert_array_equal(old,new)
    assert point['xyz']==data[3]['rows'][0]['xyz']
    assert row['summary']['registration_reasons']==['INTRINSICS_UNESTABLISHED','INSUFFICIENT_CORRESPONDENCES']
    assert row['summary']['pose_fits']==got['pose_fits']==got['new_cameras']==0
    for key in ('geometry_acceptance','geometry_accepted','native_injected','identity_verified','original_observations_modified'):
        assert got[key] is False and row[key] is False and point[key] is False
    assert events[0]['event']=='PLAN'
    json.dumps(got,allow_nan=False)


def test_different_source_routes_disagree_without_selecting_best_route():
    data=fixture();data[3]['rows'][0]['sources'][1]['raw_pixels'][0]+=2.
    got=run(data);point=query(got)['points'][0]
    assert all(r['status']=='PASSED' for r in point['routes'])
    assert point['status']=='REJECTED' and point['reasons']==['ENDPOINT_DISAGREEMENT']
    assert point['query_raw_xy'] is None


@pytest.mark.parametrize('kind',['masked_intermediate','unknown_source','unknown_target','far','time_gap'])
def test_route_and_original_mask_refusals(kind):
    data=fixture()
    if kind=='masked_intermediate':
        data[2][3][:]=0
    elif kind=='unknown_source':
        data[0][1]['original_mask_status']='NO_MATCH'
        data[0][2]['original_mask_status']='NO_MATCH'
    elif kind=='unknown_target':data[0][3]['original_mask_status']='NO_MATCH'
    elif kind=='time_gap':
        for p in range(3,len(data[0])):data[0][p]['time_seconds']+=2.
    if kind=='unknown_source':
        cv=FakeCV()
        with pytest.raises(ValueError,match='SOURCE_ORIGINAL_MASK_NOT_OK'):run(data,cv)
        assert cv.calls==[]
        return
    got=run(data);row=query(got,70 if kind=='far' else 50 if kind=='masked_intermediate' else 40)
    assert not accepted(row)
    if kind=='unknown_target':assert row['status']=='NO_MATCH' and row['points']==[]
    elif kind=='masked_intermediate':
        assert row['points'][0]['status']=='REJECTED'
        assert any(r['status']=='REJECTED' for r in row['points'][0]['routes'])
    else:assert row['status']=='NO_ANCHORS'


@pytest.mark.parametrize('kind',['forward_status','reverse_status','nan','infinity','bool_status','bool_error','shape','forward_error','reverse_error','fb'])
def test_bad_flow_is_finite_and_fail_closed(kind):
    def mutate(i,q,s,e):
        if kind=='forward_status' and i==1:s[:]=0
        elif kind=='reverse_status' and i==2:s[:]=0
        elif kind=='nan' and i==1:q[:]=np.nan
        elif kind=='infinity' and i==1:e[:]=np.inf
        elif kind=='bool_status':s=s.astype(bool)
        elif kind=='bool_error':e=e.astype(bool)
        elif kind=='shape':q=q.reshape(1,2)
        elif kind=='forward_error' and i==1:e[:]=26.
        elif kind=='reverse_error' and i==2:e[:]=26.
        elif kind=='fb' and i==2:q[...,0]+=2.
        return q,s,e
    got=run(fixture(),FakeCV(mutate));point=query(got)['points'][0]
    assert point['status']=='REJECTED' and point['query_raw_xy'] is None
    assert point['routes'][0]['steps'][0]['status']=='REJECTED'
    json.dumps(got,allow_nan=False)


@pytest.mark.parametrize('kind',['xyz','query'])
def test_duplicate_correspondences_refuse_all_conflicting_points(kind):
    data=fixture(points=2)
    if kind=='xyz':data[3]['rows'][1]['xyz']=deepcopy(data[3]['rows'][0]['xyz'])
    else:
        for a,b in zip(data[3]['rows'][0]['sources'],data[3]['rows'][1]['sources']):b['raw_pixels']=deepcopy(a['raw_pixels'])
    got=run(data);row=query(got)
    assert len(row['points'])==2 and accepted(row)==[]
    assert all(p['reasons']==['AMBIGUOUS_EXACT_QUERY_COORDINATE_OR_XYZ'] for p in row['points'])
    assert row['summary']['correspondence_count']==0


def test_models_never_mix_or_merge_gauges():
    data=fixture(models=2);got=run(data)
    for key in ('0','1'):
        row=query(got,model=key);point=accepted(row)[0]
        assert point['group_id']==f'{key}:0:0'
        assert point['xyz']==[0.,float(key),5.]
        assert len(row['points'])==1
    assert query(got,10)['status']=='ALREADY_REGISTERED'


def test_frozen_point_cap_has_no_failure_refill():
    data=fixture(points=66)
    # A failed first attempt still consumes its frozen slot.
    def mutate(i,q,s,e):
        if i==1:s[:]=0
        return q,s,e
    got=run(data,FakeCV(mutate));row=query(got)
    assert row['planned_point_attempts']==64
    assert row['points'][0]['status']=='REJECTED'
    assert [p['status'] for p in row['points'][64:]]==['NOT_EVALUATED_POINT_BUDGET']*2
    assert all(not p['routes'][0]['steps'] for p in row['points'][64:])
    again=run(data)
    assert got['plan']==again['plan']


def test_query_cap_is_model_round_robin_and_never_refilled():
    data=fixture(models=5);got=run(data)
    planned=[(r['model_key'],r['query_input_index']) for r in got['plan'] if r.get('planned_point_attempts')]
    assert len(planned)==PROFILE['max_queries']==8
    assert set(planned)=={(str(i),40) for i in range(5)}|{(str(i),50) for i in range(3)}
    assert all(not p['routes'][0]['steps'] for r in got['query_rows'] if r['status']=='NOT_EVALUATED_QUERY_BUDGET' for p in r['points'] if p['routes'])


def test_timeout_after_forward_preserves_raw_and_previous_points_and_exception():
    data=fixture(points=2);cv=FakeCV();events=[];failure=TimeoutError('synthetic frozen budget')
    # First point needs 6 calls (one-step and two-step routes); stop after
    # the second point's first forward measurement, before its reverse.
    def budget():
        if len(cv.calls)==7:raise failure
    with pytest.raises(TimeoutError) as exc:run(data,cv,budget=budget,progress=events.append)
    assert exc.value is failure and len(cv.calls)==7
    partial=next(e['partial'] for e in events if e['event']=='FAILED')
    row=query(partial);assert row['points'][0]['status']=='PROVISIONAL_ROUTE_AGREEMENT'
    step=row['points'][1]['routes'][0]['steps'][0]
    assert step['forward_xy'] is not None and 'backward_xy' not in step
    assert step['status']=='RUNNING' and partial['status']=='FAILED'
    assert row['points'][1]['xyz']==data[3]['rows'][1]['xyz']
    json.dumps(partial,allow_nan=False)


@pytest.mark.parametrize('kind',['classification_rights','row_rights','source_frame','source_image','source_index','source_indices','frame_ids','camera_binding','bad_xyz','image_hash','mask_hash','explicit_source_binding'])
def test_source_identity_and_rights_tamper_never_calls_flow(kind):
    data=fixture();c=data[3];r=c['rows'][0];s=r['sources'][0]
    if kind=='classification_rights':c['identity_verified']=True
    elif kind=='row_rights':r['native_injected']=True
    elif kind=='source_frame':s['frame_id']='wrong'
    elif kind=='source_image':s['image_id']=999
    elif kind=='source_index':s['input_index']=40
    elif kind=='source_indices':r['source_indices'].reverse()
    elif kind=='frame_ids':r['frame_ids'].reverse()
    elif kind=='camera_binding':data[4]['models'][0]['cameras'][10]['frame_id']='wrong'
    elif kind=='bad_xyz':r['xyz'][0]=float('nan')
    elif kind=='image_hash':data[0][0]['sha256']='bad'
    elif kind=='mask_hash':data[0][0]['mask_binding']['sha256']='bad'
    elif kind=='explicit_source_binding':s['mask_binding']=dict(path='fixture://wrong',sha256='b'*64)
    cv=FakeCV();events=[]
    with pytest.raises(ValueError):run(data,cv,progress=events.append)
    assert cv.calls==[] and events[-1]['event']=='FAILED'


def test_unexpected_flow_error_keeps_original_exception():
    failure=RuntimeError('synthetic backend failure')
    class Broken(FakeCV):
        def calcOpticalFlowPyrLK(self,*args,**kwargs):raise failure
    events=[]
    with pytest.raises(RuntimeError) as exc:run(fixture(),Broken(),progress=events.append)
    assert exc.value is failure
    assert events[-1]['partial']['error']==dict(type='RuntimeError',message=str(failure))


def test_point_events_stay_provisional_until_authoritative_query_close():
    events=[];got=run(fixture(points=2),progress=events.append)
    point_events=[e for e in events if e['event']=='POINT_END' and e['query_input_index']==40]
    assert len(point_events)==2
    assert all(e['point']['status']=='PROVISIONAL_ROUTE_AGREEMENT' for e in point_events)
    closed=next(e['query'] for e in events if e['event']=='QUERY_END' and e['query']['query_input_index']==40)
    assert len(accepted(closed))==2
    assert closed['points']==query(got)['points'] and closed['summary']==query(got)['summary']
    assert query(got)['committed'] is True


def test_interrupted_conflict_scan_never_publishes_accepted_duplicate_points():
    data=fixture(points=2)
    data[3]['rows'][1]['xyz']=deepcopy(data[3]['rows'][0]['xyz'])
    events=[];failure=TimeoutError('synthetic conflict scan interruption')
    def budget():
        if sum(e['event']=='POINT_END' and e['query_input_index']==40 for e in events)==2:
            raise failure
    with pytest.raises(TimeoutError) as exc:
        run(data,budget=budget,progress=events.append)
    assert exc.value is failure
    finished=[e['point'] for e in events if e['event']=='POINT_END']
    assert len(finished)==2
    assert all(p['status']=='PROVISIONAL_ROUTE_AGREEMENT' for p in finished)
    partial=events[-1]['partial'];row=query(partial)
    assert accepted(row)==[] and row['summary'] is None
    assert all(p['status']=='PROVISIONAL_ROUTE_AGREEMENT' for p in row['points'])
    assert not any(e['event']=='QUERY_END' for e in events)
    json.dumps(partial,allow_nan=False)


def test_failed_query_end_rolls_back_current_acceptance_but_retains_committed_query():
    events=[];failure=RuntimeError('synthetic checkpoint write failure')
    def progress(event):
        if event['event']=='QUERY_END' and event['query']['query_input_index']==50:
            raise failure
        events.append(event)
    with pytest.raises(RuntimeError) as exc:run(fixture(),progress=progress)
    assert exc.value is failure
    partial=events[-1]['partial']
    previous=query(partial,40);current=query(partial,50)
    assert previous['committed'] is True and len(accepted(previous))==1
    assert previous['summary']['correspondence_count']==1
    assert current.get('committed') is not True and current['summary'] is None
    assert accepted(current)==[]
    assert current['points'][0]['status']=='PROVISIONAL_ROUTE_AGREEMENT'
    assert current['points'][0]['reasons']==['QUERY_END_NOT_COMMITTED']
    assert [e['query']['query_input_index'] for e in events if e['event']=='QUERY_END']==[40]
    json.dumps(partial,allow_nan=False)


def test_point_end_callback_failure_preserves_provisional_measurement():
    events=[];failure=RuntimeError('synthetic point checkpoint failure')
    def progress(event):
        if event['event']=='POINT_END':
            assert event['point']['status']=='PROVISIONAL_ROUTE_AGREEMENT'
            raise failure
        events.append(event)
    with pytest.raises(RuntimeError) as exc:run(fixture(),progress=progress)
    assert exc.value is failure
    partial=events[-1]['partial'];point=query(partial)['points'][0]
    assert point['status']=='PROVISIONAL_ROUTE_AGREEMENT'
    assert point['query_raw_xy']==[11.25,8.5]
    assert accepted(query(partial))==[] and query(partial)['summary'] is None
