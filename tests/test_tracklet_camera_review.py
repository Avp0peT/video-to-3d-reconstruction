from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.tracklet_camera_review import assess_cameras


def fixture(n=12):
    rng=np.random.default_rng(819);world=rng.uniform([-2,-2,5],[2,2,9],(n,3))
    frames=[];cams={};sources=[]
    for i in range(3):
        frames.append(dict(input_index=10+i,decode_index=20+2*i,frame_id=f'f{i}',width=640,height=480,original_mask_status='OK'))
        cams[10+i]=dict(input_index=10+i,frame_id=f'f{i}',image_id=i+1,camera_model='SIMPLE_RADIAL',
                       params=[400,320,240,0],R=np.eye(3).tolist(),t=[-i,0,0],width=640,height=480)
        sources.append((world[:,:2]+np.array([-i,0]))/world[:,2,None]*400+[320,240])
    records=[];candidates=[]
    for i,X in enumerate(world):
        pid=f'track:{i}:segment:0:source:10';observations=[]
        for j in range(3):
            uv=(X[:2]+[-.25*j,0])/X[2]*400+[320,240]
            observations.append(dict(decode_index=20+j,raw_xy=uv.tolist(),width=640,height=480,original_mask_status='OK' if j!=1 else 'UNKNOWN_NOT_SEGMENTED'))
        point=dict(point_id=pid,track_id=i,identity_segment=0,source_input_index=10,source_decode_index=20,source_raw_xy=sources[0][i].tolist())
        records.append(dict(tracklet_id=pid+':edge:0',point_id=pid,track_id=i,identity_segment=0,source_input_index=10,
                            edge_ordinal=0,verified_legs=2,source_point=point,observations=observations,status='CANDIDATE_2D_TRACKLET'))
        candidates.append(dict(group_id=f'm:{i}:0',model_key='m',track_id=i,identity_segment=0,xyz=X.tolist(),
            tier='STRICT_CONDITIONAL' if i<9 else 'RELAXED_CONDITIONAL',geometry_accepted=False,native_injected=False,identity_verified=False,
            sources=[dict(input_index=10+j,frame_id=f'f{j}',image_id=j+1,raw_pixels=sources[j][i].tolist()) for j in range(3)]))
    return dict(records=records),dict(status='ANCHOR_CANDIDATES_CLASSIFIED',rows=candidates),frames,dict(unsupported_models=[dict(model_key='m',unsupported_cameras=cams)])


def test_exact_binding_strict_and_mixed_cohorts_are_separate():
    args=fixture();original=deepcopy(args);r=assess_cameras(*args)
    assert args==original and len(r['joins'])==12 and len(r['records'])==2
    a,b=r['records'];assert a['query_decode_index']==b['query_decode_index']==21
    assert (a['cohort'],len(a['points']))==('STRICT',9)
    assert (b['cohort'],len(b['points']))==('STRICT_PLUS_RELAXED',12)
    assert max(x['assessment']['heldout_fit']['holdout_reprojection']['max_px'] for x in r['records'])<1e-7
    assert not r['camera_pose_established'] and r['new_cameras']==0 and not r['intrinsics_borrowed']
    assert a['original_mask_status']=='UNKNOWN_NOT_SEGMENTED'
    json.dumps(r,allow_nan=False)


@pytest.mark.parametrize('mutation,reason',[
    (lambda a:a[0]['records'][0]['source_point']['source_raw_xy'].__setitem__(0,123.),'SOURCE_COORDINATE'),
    (lambda a:a[1]['rows'][0]['sources'][0]['raw_pixels'].__setitem__(0,123.),'SOURCE_COORDINATE'),
    (lambda a:a[1]['rows'][0]['sources'][0].update(frame_id='wrong'),'SOURCE_FRAME'),
    (lambda a:a[1]['rows'][0].update(identity_verified=True),'RIGHTS'),
    (lambda a:a[1]['rows'].append(deepcopy(a[1]['rows'][0])),'CANDIDATE_IDENTITY'),
    (lambda a:a[0]['records'].append(deepcopy(a[0]['records'][0])),'DUPLICATE_TRACKLET'),
    (lambda a:a[0]['records'][0].update(point_id='foreign'),'TRACKLET_IDENTITY'),
    (lambda a:a[0]['records'][0]['observations'][1].update(width=600),'QUERY_DOMAIN_MISMATCH'),
    (lambda a:a[0]['records'][0]['observations'][1].update(decode_index=900),'NONCONSECUTIVE'),
])
def test_foreign_identity_and_nearby_coordinate_never_join(mutation,reason):
    args=fixture();mutation(args)
    with pytest.raises(ValueError,match=reason):assess_cameras(*args)


def test_components_never_merge_xyz():
    args=fixture();model=deepcopy(args[3]['unsupported_models'][0]);model['model_key']='n'
    args[3]['unsupported_models'].append(model)
    for c in args[1]['rows'][6:]:c['model_key']='n';c['group_id']=f"n:{c['track_id']}:0"
    r=assess_cameras(*args)
    assert len(r['records'])==4 and all(len(row['points'])<=6 for row in r['records'])
    assert all(row['assessment']['heldout_fit'] is None for row in r['records'])


def test_different_segment_is_not_nearest_neighbour_matched():
    args=fixture()
    for c in args[1]['rows']:c['identity_segment']=1;c['group_id']=f"m:{c['track_id']}:1"
    assert assess_cameras(*args)['joins']==[]


def test_shared_budget_and_missing_classification():
    def budget():raise TimeoutError('common deadline')
    with pytest.raises(TimeoutError,match='common deadline'):assess_cameras(*fixture(),budget=budget)
    r=assess_cameras(dict(records=[]),None,[],{})
    assert r['status']=='NOT_AVAILABLE_NO_CURRENT_CLASSIFICATION' and r['numeric_fits']==0


def test_empty_prefix_has_no_camera_prerequisite():
    r=assess_cameras(dict(records=[]),dict(rows=[]),[],{})
    assert r['status']=='NOT_APPLICABLE_NO_SHORT_TRACKLETS' and r['numeric_fits']==0


def test_complete_classification_ledger_is_not_confused_with_numeric_cohort():
    args=fixture()
    for i in range(12,4012):
        args[1]['rows'].append(dict(group_id=f'm:{i}:0',model_key='m',track_id=i,identity_segment=0,
            tier='NOT_EVALUATED',xyz=None,sources=[],geometry_accepted=False,native_injected=False,identity_verified=False))
    r=assess_cameras(*args)
    assert len(r['joins'])==12 and len(r['records'])==2
    assert r['numeric_fits']==6
