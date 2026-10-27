from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.anchor_route_tracking import plan_anchor_routes, measure_anchor_routes


class CV:
    TERM_CRITERIA_COUNT=1;TERM_CRITERIA_EPS=2
    def calcOpticalFlowPyrLK(self,a,b,xy,unused,**kw):
        out=xy.copy();out[0,0,0]+=int(b[0,0])-int(a[0,0])
        return out,np.ones((1,1),np.uint8),np.zeros((1,1),np.float32)


def fixture(n=3):
    frames=[dict(input_index=i,decode_index=i*2,frame_id=f'f{i}',width=32,height=32,original_mask_status='OK') for i in range(4)]
    cameras={i:dict(input_index=i,frame_id=f'f{i}',image_id=i+1,camera_model='SIMPLE_RADIAL',width=32,height=32,params=[20.,16.,16.,0.],R=np.eye(3).tolist(),t=[float(i),0.,0.]) for i in (1,2,3)}
    points=[];records=[]
    for i in range(n):
        pid=f'track:{i}:segment:0:source:1';p=dict(point_id=pid,group_id=pid,track_id=i,identity_segment=0,source_input_index=1,source_decode_index=2,source_raw_xy=[10.5,10.5])
        points.append(p);records.append(dict(tracklet_id=pid+':edge:0',point_id=pid,track_id=i,identity_segment=0,source_input_index=1,status='CANDIDATE_2D_TRACKLET',source_point=deepcopy(p)))
    pop=dict(edges=[dict(model_key='m',model_keys=['m'],source_input_index=1,target_input_index=0,source_decode_index=2,target_decode_index=0,decode_path=[2,1,0],points=points)],decode_indices=[0,1,2])
    decoded={i:dict(gray=np.full((32,32),i,np.uint8),mask=None,width=32,height=32,path=f'{i}.png',original_mask_status='UNKNOWN_NOT_SEGMENTED',endpoint_verified=i in (0,2,4,6)) for i in range(7)}
    masks=[np.full((32,32),255,np.uint8) for _ in frames]
    return pop,frames,dict(models=[dict(model_key='m',cameras=cameras)],unsupported_models=[]),dict(records=records),decoded,masks


def test_plan_opposite_direction_two_nearest_and_immutable():
    pop,frames,cameras,ledger,decoded,masks=fixture();before=deepcopy((pop,frames,cameras))
    plan=plan_anchor_routes(pop,frames,cameras)
    assert plan['routes'][0]['camera_input_indices']==[1,2,3]
    assert plan['routes'][0]['decode_path']==[2,3,4,5,6]
    assert plan['extra_decode_indices']==[3,4,5,6]
    assert (pop,frames,cameras)==before


def test_measured_actual_three_camera_endpoints_and_originals_unchanged(tmp_path):
    pop,frames,cameras,ledger,decoded,masks=fixture();before=deepcopy(ledger);refs=[]
    result=measure_anchor_routes(plan_anchor_routes(pop,frames,cameras),ledger,frames,masks,decoded,tmp_path/'run',refs,CV())
    assert result['counts']=={'MEASURED_THREE_VIEW_ANCHOR':3}
    row=result['records'][0]
    assert [o['input_index'] for o in row['observations']]==[1,2,3]
    assert [o['raw_xy'] for o in row['observations']]==[[10.5,10.5],[12.5,10.5],[14.5,10.5]]
    assert row['segment_measurements'][1]['rows'][0]['source_raw_xy']==[12.5,10.5]
    assert ledger==before and refs==result['source_bindings']
    assert all(decoded[i]['mask'] is None for i in decoded)
    assert json.loads((tmp_path/'run/RESULT.json').read_text())['counts']==result['counts']


def test_first_camera_mask_rejection_prevents_three_view(tmp_path):
    pop,frames,cameras,ledger,decoded,masks=fixture(1);masks[2][:]=0
    result=measure_anchor_routes(plan_anchor_routes(pop,frames,cameras),ledger,frames,masks,decoded,tmp_path/'run',[],CV())
    row=result['records'][0]
    assert row['status']=='BLOCKED' and len(row['observations'])==1
    assert row['reasons']==['POINT_OUTSIDE_ORIGINAL_MASK']


def test_source_coordinate_mismatch_fails_before_flow(tmp_path):
    pop,frames,cameras,ledger,decoded,masks=fixture();plan=plan_anchor_routes(pop,frames,cameras)
    ledger['records'][0]['source_point']['source_raw_xy']=[1.5,1.5]
    with pytest.raises(ValueError,match='COHORT_SOURCE_IDENTITY_MISMATCH'):
        measure_anchor_routes(plan,ledger,frames,masks,decoded,tmp_path/'run',[],CV())


def test_budget_preserves_failure_and_propagates(tmp_path):
    pop,frames,cameras,ledger,decoded,masks=fixture()
    def budget():raise TimeoutError('clock')
    with pytest.raises(TimeoutError,match='clock'):
        measure_anchor_routes(plan_anchor_routes(pop,frames,cameras),ledger,frames,masks,decoded,tmp_path/'run',[],CV(),budget=budget)
    saved=json.loads((tmp_path/'run/FAILURE.json').read_text())
    assert saved['status']=='INTERRUPTED' and saved['flow_calls']==0


def test_frozen_route64_bound_no_refill(tmp_path):
    pop,frames,cameras,ledger,decoded,masks=fixture(70)
    result=measure_anchor_routes(plan_anchor_routes(pop,frames,cameras),ledger,frames,masks,decoded,tmp_path/'run',[],CV())
    assert len(result['frozen_cohort'])==64 and len(result['omitted'])==6


def test_missing_camera_or_mask_blocks_plan():
    pop,frames,cameras,ledger,decoded,masks=fixture();frames[3]['original_mask_status']='NO_MATCH'
    plan=plan_anchor_routes(pop,frames,cameras)
    assert not plan['routes'] and plan['omitted_routes'][0]['reason']=='NO_TWO_NEARBY_OPPOSITE_CAMERAS'


def test_empty_plan_has_no_unused_decode_metadata_prerequisites():
    plan=plan_anchor_routes(dict(edges=[]),[dict(input_index=17,frame_id='unused')],{})
    assert plan['routes']==[] and plan['decode_indices']==[]
    assert not plan['geometry_accepted']


def test_duplicate_model_track_segment_other_source_omitted_before_measurement(tmp_path):
    pop,frames,cameras,ledger,decoded,masks=fixture(1)
    plan=plan_anchor_routes(pop,frames,cameras)
    other=deepcopy(plan['routes'][0]);other['route_id']='m:source:2';other['source_input_index']=2
    point=deepcopy(other['points'][0]);point.update(point_id='track:0:segment:0:source:2',group_id='track:0:segment:0:source:2',source_input_index=2,source_decode_index=4)
    other['points']=[point];other['point_ids']=[point['point_id']];plan['routes'].append(other)
    duplicate=deepcopy(ledger['records'][0]);duplicate.update(point_id=point['point_id'],source_input_index=2,tracklet_id=point['point_id']+':edge:1',source_point=point)
    ledger['records'].append(duplicate)
    result=measure_anchor_routes(plan,ledger,frames,masks,decoded,tmp_path/'run',[],CV())
    assert len(result['records'])==1 and len(result['frozen_cohort'])==1
    assert result['omitted']==[dict(route_id='m:source:2',tracklet_id=duplicate['tracklet_id'],reason='DUPLICATE_GROUP_ANOTHER_SOURCE')]
    frozen=json.loads((tmp_path/'run/PLAN.json').read_text())
    assert frozen['omitted']==result['omitted']


@pytest.mark.parametrize('corruption',['cycle_output','cycle_missing','forward_extra','cycle_status','cycle_error'])
def test_false_passed_status_cannot_promote_bad_raw_cycle(tmp_path,monkeypatch,corruption):
    import integrations.demo_pipeline.anchor_route_tracking as module
    actual=module.compare_edge
    def forged(*args,**kwargs):
        result=actual(*args,**kwargs);arm=result['rows'][0]['dense']
        assert arm['status']=='PASSED'
        if corruption=='cycle_output':
            # Keep cycle status and recorded error falsely passed, but change
            # actual last reverse output by two pixels: independent audit rejects.
            rec=arm['fullcycle']['steps'][-1]
            rec['xy'][0][0][0]+=2;rec['observed_raw_xy'][0]+=2
        elif corruption=='cycle_missing':arm['fullcycle']['steps'].pop()
        elif corruption=='forward_extra':arm['steps'].append(deepcopy(arm['steps'][-1]))
        elif corruption=='cycle_status':arm['fullcycle']['status']='NOT_COMPLETED'
        else:arm['fullcycle']['error_px']=.5
        return result
    monkeypatch.setattr(module,'compare_edge',forged)
    pop,frames,cameras,ledger,decoded,masks=fixture(1)
    with pytest.raises(ValueError,match='ANCHOR_'):
        measure_anchor_routes(plan_anchor_routes(pop,frames,cameras),ledger,frames,masks,decoded,tmp_path/'run',[],CV())
    saved=json.loads((tmp_path/'run/FAILURE.json').read_text())
    assert saved['records'][0]['status']=='BLOCKED'
    assert len(saved['records'][0]['observations'])==1


@pytest.mark.parametrize('field',['track_id','identity_segment','source_input_index'])
def test_boolean_candidate_identity_rejected(tmp_path,field):
    pop,frames,cameras,ledger,decoded,masks=fixture(1)
    ledger['records'][0][field]=True
    with pytest.raises(ValueError,match='INVALID_CANDIDATE_IDENTITY_TYPE'):
        measure_anchor_routes(plan_anchor_routes(pop,frames,cameras),ledger,frames,masks,decoded,tmp_path/'run',[],CV())


def test_same_route_duplicate_point_is_schema_error(tmp_path):
    pop,frames,cameras,ledger,decoded,masks=fixture(1)
    plan=plan_anchor_routes(pop,frames,cameras)
    plan['routes'][0]['points'].append(deepcopy(plan['routes'][0]['points'][0]))
    with pytest.raises(ValueError,match='DUPLICATE_ROUTE_POINT_ID'):
        measure_anchor_routes(plan,ledger,frames,masks,decoded,tmp_path/'run',[],CV())
