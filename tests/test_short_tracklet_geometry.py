from copy import deepcopy
import numpy as np
import pytest
from integrations.demo_pipeline.short_tracklet_geometry import assess_tracklets


def fixture(n=3):
    frames=[];masks=[];cams={};observations=[]
    for i in range(3):
        frames.append(dict(input_index=10+i,decode_index=20+i,frame_id=f'f{i}',width=100,height=100,original_mask_status='OK'))
        masks.append(np.full((100,100),255,np.uint8))
        observations.append(dict(decode_index=20+i,raw_xy=[50-10*i,50.],width=100,height=100))
        if i<n:cams[10+i]=dict(input_index=10+i,frame_id=f'f{i}',image_id=i+1,camera_id=i+1,camera_model='SIMPLE_RADIAL',params=[100,50,50,0],R=np.eye(3).tolist(),t=[-i,0,0],width=100,height=100)
    ledger=dict(records=[dict(tracklet_id='t',status='CANDIDATE_2D_TRACKLET',observations=observations)])
    return ledger,frames,masks,dict(models=[],unsupported_models=[dict(model_key='a',unsupported_cameras=cams)])


def test_missing_poses_are_not_borrowed():
    args=fixture(1);r=assess_tracklets(*args);g=r['records'][0]['model_assessments'][0]
    assert g['camera_views']==1 and g['missing_camera_decode_indices']==[21,22]
    assert r['counts']=={'BLOCKED_MISSING_POSES':1} and r['numeric_groups']==0
    assert g['fit'] is None and r['pose_interpolation'] is False


def test_three_real_saved_views_use_existing_numeric_fit():
    args=fixture();original=deepcopy(args[0]);r=assess_tracklets(*args);g=r['records'][0]['model_assessments'][0]
    assert np.allclose(g['fit']['xyz'],[0,0,10],atol=1e-6)
    assert len(g['leave_one_view_out'])==3 and g['evidence_level']=='MULTIVIEW_CONDITIONAL_DIAGNOSTIC'
    assert r['numeric_groups']==1 and r['geometry_accepted'] is False and args[0]==original


def test_two_views_do_not_claim_three_view_validation():
    r=assess_tracklets(*fixture(2));g=r['records'][0]['model_assessments'][0]
    assert g['evidence_level']=='TWO_VIEW_PROVISIONAL_DIAGNOSTIC' and not g['leave_one_view_out']


def test_models_cannot_supply_one_camera_each():
    args=list(fixture());cams=args[3]['unsupported_models'][0]['unsupported_cameras']
    args[3]=dict(models=[dict(model_key=str(k),cameras={k:c}) for k,c in cams.items()])
    r=assess_tracklets(*args)
    assert r['numeric_groups']==0 and r['records'][0]['status']=='BLOCKED_MISSING_POSES'


def test_unknown_foreground_stays_unknown():
    args=fixture();args[2][1]=None;args[2][2]=None
    r=assess_tracklets(*args);g=r['records'][0]['model_assessments'][0]
    assert g['camera_views']==3 and g['usable_views']==1 and g['unknown_mask_decode_indices']==[21,22]
    assert g['status']=='BLOCKED_FOREGROUND_OR_CAMERA_MODEL' and r['numeric_groups']==0


def test_shared_budget_and_identity_rejections():
    def budget():raise TimeoutError('shared budget')
    with pytest.raises(TimeoutError):assess_tracklets(*fixture(),budget=budget)
    args=fixture();args[3]['unsupported_models'][0]['unsupported_cameras'][10]['frame_id']='wrong'
    with pytest.raises(ValueError,match='IDENTITY'):assess_tracklets(*args)


def test_no_candidates_need_no_camera_metadata():
    r=assess_tracklets(dict(records=[]),[],[],dict(models=[]))
    assert r['records']==[] and r['numeric_groups']==0
