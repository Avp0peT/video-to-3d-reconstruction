from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.region_tracking import build_region_plan,measure_regions


class CV:
    TERM_CRITERIA_COUNT=1;TERM_CRITERIA_EPS=2
    def __init__(self,mode='translation'):self.mode=mode;self.calls=[]
    def calcOpticalFlowPyrLK(self,a,b,xy,unused,**kw):
        i=int(a[0,0]);j=int(b[0,0]);self.calls.append((i,j,xy.copy(),kw))
        out=xy.copy();out[:,:,0]+=j-i;status=np.ones((len(xy),1),np.uint8);err=np.zeros((len(xy),1),np.float32)
        if self.mode=='status':status[:]=0
        if self.mode=='occlusion' and j==2:status[xy[:,0,0]<65]=0
        if self.mode=='error':err[:]=26
        if self.mode=='nonfinite':out[:]=np.nan
        return out,status,err


def fixture(centers=((64.5,64.5),),known_query=False):
    records=[]
    for i,center in enumerate(centers):
        pid=f'track:{i}:segment:0:source:0'
        point=dict(point_id=pid,source_input_index=0,source_decode_index=0,source_raw_xy=list(center))
        records.append(dict(point_id=pid,tracklet_id=pid+':edge:0',edge_ordinal=0,track_id=i,identity_segment=0,source_input_index=0,status='CANDIDATE_2D_TRACKLET',verified_legs=3,source_point=point,
            observations=[dict(decode_index=j,raw_xy=[center[0]+j,center[1]]) for j in range(4)]))
    frames=[dict(input_index=0,decode_index=0,width=128,height=128,path='0.png',original_mask_status='OK')];masks=[np.full((128,128),255,np.uint8)]
    if known_query:
        frames.append(dict(input_index=1,decode_index=2,width=128,height=128,path='2.png',original_mask_status='OK'));masks.append(np.full((128,128),255,np.uint8))
    decoded={i:dict(width=128,height=128,path=f'{i}.png',gray=np.full((128,128),i,np.uint8),original_mask_status='UNKNOWN_NOT_SEGMENTED') for i in range(4)}
    return dict(records=records),frames,masks,decoded


def test_independent_grid_coordinates_center_control_and_immutability():
    ledger,frames,masks,decoded=fixture();before=deepcopy(ledger);maskbefore=masks[0].copy();graybefore=decoded[0]['gray'].copy()
    plan=build_region_plan(ledger,frames,masks,decoded);frozen=deepcopy(plan);cv=CV();result=measure_regions(plan,decoded,cv)
    region=result['regions'][0]
    assert region['center_baseline']['tracklets_2d']==1 and region['expanded_grid']['tracklets_2d']==80
    assert len({tuple(p['observations'][-1]['raw_xy']) for p in region['points']})==81
    assert all(p['observations'][-1]['raw_xy'][0]==p['source_raw_xy'][0]+3 for p in region['points'])
    assert all(s['ncc']['status']=='UNKNOWN_LOW_VARIANCE' for p in region['points'] for s in p['steps'])
    assert all(a['status']=='EVALUATED' and a['holdout']['max_px']<1e-8 for a in region['affine_diagnostics'])
    assert all(c[3]==dict(winSize=(21,21),maxLevel=3,criteria=(3,30,.01)) for c in cv.calls)
    assert result['flow_calls']==6 and plan==frozen and ledger==before
    assert np.array_equal(masks[0],maskbefore) and np.array_equal(decoded[0]['gray'],graybefore)
    json.dumps(result,allow_nan=False)


def test_source_mask_crossing_excluded_not_tracking_failure():
    ledger,frames,masks,decoded=fixture();masks[0][:,84:]=0
    plan=build_region_plan(ledger,frames,masks,decoded);points=plan['regions'][0]['points']
    assert any(p['status'].startswith('EXCLUDED_SOURCE_') for p in points)
    result=measure_regions(plan,decoded,CV());region=result['regions'][0]
    assert region['expanded_grid']['excluded']>0 and region['expanded_grid']['tracking_rejected']==0
    assert all(not p['steps'] for p in region['points'] if p['status'].startswith('EXCLUDED_SOURCE_'))


def test_known_query_mask_stops_but_unknown_does_not_claim_mask():
    ledger,frames,masks,decoded=fixture(known_query=True);masks[1][:,75:]=0
    result=measure_regions(build_region_plan(ledger,frames,masks,decoded),decoded,CV())
    failed=[p for p in result['regions'][0]['points'] if p['reason']=='WINDOW_OUTSIDE_ORIGINAL_FOREGROUND']
    assert failed and all(p['verified_legs']==1 and len(p['steps'])==2 for p in failed)
    assert all(p['steps'][0]['destination_mask_status']=='UNKNOWN_NOT_SEGMENTED' for p in failed)
    assert all(p['steps'][1]['destination_mask_status']=='OK' for p in failed)


@pytest.mark.parametrize('mode,reason',[('status','LK_STATUS_FAILED'),('error','LK_ERROR_THRESHOLD'),('nonfinite','LK_NONFINITE_XY')])
def test_failed_outputs_preserved_and_never_continued(mode,reason):
    ledger,frames,masks,decoded=fixture();result=measure_regions(build_region_plan(ledger,frames,masks,decoded),decoded,CV(mode))
    for p in result['regions'][0]['points']:
        assert p['status']=='REJECTED' and p['reason']==reason and p['verified_legs']==0 and len(p['steps'])==1
        assert p['steps'][0]['forward'] and p['steps'][0]['backward'] is None


def test_occluded_points_do_not_get_center_or_later_coordinates():
    ledger,frames,masks,decoded=fixture();result=measure_regions(build_region_plan(ledger,frames,masks,decoded),decoded,CV('occlusion'))
    points=result['regions'][0]['points'];failed=[p for p in points if p['status']=='REJECTED'];passed=[p for p in points if p['status']=='TRACKLET_2D']
    assert failed and passed and all(len(p['observations'])==2 for p in failed)
    assert all(p['verified_legs']==3 for p in passed)


def test_line_support_affine_is_explicitly_degenerate():
    ledger,frames,masks,decoded=fixture();plan=build_region_plan(ledger,frames,masks,decoded)
    class LineCV(CV):
        def calcOpticalFlowPyrLK(self,a,b,xy,unused,**kw):
            i=int(a[0,0]);j=int(b[0,0]);self.calls.append((i,j,xy.copy(),kw))
            if j>i:
                self.start=xy.copy();out=xy.copy();out[:,:,0]+=1;out[:,:,1]=64.
            else:out=self.start.copy()
            return out,np.ones((len(xy),1),np.uint8),np.zeros((len(xy),1),np.float32)
    result=measure_regions(plan,decoded,LineCV())
    diagnostics=result['regions'][0]['affine_diagnostics']
    assert diagnostics[0]['status']=='EVALUATED'
    assert all(a['status']=='BLOCKED' and a['reason']=='COLLINEAR_AFFINE_TRAINING_SUPPORT' for a in diagnostics[1:])


def test_frozen_center_spacing_and_region_budget():
    ledger,frames,masks,decoded=fixture(((40.5,40.5),(45.5,45.5),(88.5,88.5)))
    plan=build_region_plan(ledger,frames,masks,decoded)
    assert len(plan['regions'])==2
    assert any(o['reason']=='CENTER_SPACING_32PX' for o in plan['omitted_centers'])


@pytest.mark.parametrize('change',[lambda l:l['records'][0]['observations'][1].update(decode_index=6),lambda l:l['records'][0]['source_point'].update(source_raw_xy=[2.,3.]),lambda l:l['records'][0].update(source_input_index=True)])
def test_bad_identity_or_nonconsecutive_input_rejected(change):
    ledger,frames,masks,decoded=fixture();change(ledger)
    with pytest.raises(ValueError):build_region_plan(ledger,frames,masks,decoded)


def test_budget_after_native_propagates_with_partial():
    ledger,frames,masks,decoded=fixture();plan=build_region_plan(ledger,frames,masks,decoded);cv=CV()
    def budget():
        if cv.calls:raise TimeoutError('deadline after native')
    with pytest.raises(TimeoutError,match='deadline after native') as exc:measure_regions(plan,decoded,cv,budget=budget)
    assert exc.value.partial_result['status']=='INTERRUPTED' and exc.value.partial_result['flow_calls']==1


def test_empty_ledger_and_plan_need_no_cv_or_image_prerequisites():
    plan=build_region_plan(dict(records=[]),[],[],{})
    assert plan['regions']==[]
    result=measure_regions(plan,{},None)
    assert result['status']=='REGION_MEASUREMENTS_COMPLETE' and result['flow_calls']==0
    json.dumps(plan,allow_nan=False)


def test_reusable_growth_grid_has_independent_ids_and_same_source_exclusions():
    from integrations.demo_pipeline.region_tracking import make_region_points
    ledger,frames,masks,decoded=fixture();plan=build_region_plan(ledger,frames,masks,decoded)
    d=plan['frame_domains']['0'];points=make_region_points('child',[112.5,64.5],d)
    assert len(points)==81 and len({p['point_id'] for p in points})==81
    assert points[40]['source_raw_xy']==[112.5,64.5]
    assert any(p['status'].startswith('EXCLUDED_SOURCE_') for p in points)



@pytest.mark.parametrize('change',[lambda p:p['regions'][0].update(decode_path=[0,2,3]),lambda p:p['regions'][0]['points'][0].update(grid_ordinal=True),lambda p:p['regions'][0]['points'][0].update(source_raw_xy=[20.5,20.5]),lambda p:p['frame_domains']['0']['mask_rle'].__setitem__(0,[[0,999]]),lambda p:p['regions'].append(deepcopy(p['regions'][0]))])
def test_malformed_measurement_plan_rejected_before_cv(change):
    ledger,frames,masks,decoded=fixture();plan=build_region_plan(ledger,frames,masks,decoded);change(plan);cv=CV()
    with pytest.raises(ValueError):measure_regions(plan,decoded,cv)
    assert cv.calls==[]


def test_native_input_coordinates_and_quantization_are_explicit():
    ledger,frames,masks,decoded=fixture(((64.500001,64.500001),))
    result=measure_regions(build_region_plan(ledger,frames,masks,decoded),decoded,CV())
    point=result['regions'][0]['points'][40];step=point['steps'][0]
    assert step['forward']['input_raw_xy']==point['source_raw_xy']
    assert step['forward']['input_cv_xy']==(np.asarray(point['source_raw_xy'])-.5).astype(np.float32).astype(float).tolist()
    assert step['backward']['input_raw_xy']==step['forward']['observed_raw_xy']
    expected=float(np.linalg.norm(np.asarray(step['backward']['observed_raw_xy'])-point['source_raw_xy']))
    assert step['fb_error_px']==expected
