from copy import deepcopy
import cv2
import numpy as np
from integrations.mvs_surface import temporal_local_geometry as t


def scene(baseline=.04,repeated=False):
    cv2.setNumThreads(2)
    rng=np.random.default_rng(19)
    image=rng.integers(0,256,(96,96,3),np.uint8)
    image=cv2.GaussianBlur(image,(3,3),.5)
    if repeated:
        y,x=np.mgrid[:96,:96]
        image=np.repeat(((((x//5)+(y//5))%2)*200+20).astype(np.uint8)[:,:,None],3,axis=2)
    views=[]
    for index in range(5):
        b=(index-2)*baseline
        shifted=cv2.warpAffine(image,np.float32([[1,0,-100*b/2],[0,1,0]]),(96,96),borderMode=cv2.BORDER_REFLECT)
        views.append(dict(view_id=index,decode_index=index*5,image=shifted,mask=np.ones((96,96),bool),validity=np.ones((96,96),bool),depth=np.zeros((96,96),np.float32),K=np.array([[100.,0,48],[0,100.,48],[0,0,1]]),R=np.eye(3),C=np.array([b,0.,0.])))
    roi=np.zeros((96,96),bool);roi[32:64,32:64]=True
    return views,roi


def test_correct_tracks_recover_metric_camera_plane_without_old_depth_or_mutation():
    views,roi=scene();before=deepcopy(views)
    result=t.recover_local_geometry(views,2,roi)
    assert result['accepted'].sum()>5
    np.testing.assert_allclose(result['points'][result['accepted'],2],2.,atol=.03)
    assert result['observation_mask'][result['accepted']].sum(axis=1).min()>=3
    assert result['holdout_error'][result['accepted']].max()<=2
    assert result['depth_used'] is False
    for view,old in zip(views,before):
        for key in ('image','mask','validity','depth','K','R','C'): np.testing.assert_array_equal(view[key],old[key])


def test_low_parallax_rejected():
    views,roi=scene(baseline=.001)
    result=t.recover_local_geometry(views,2,roi)
    assert not result['accepted'].any()
    assert (result['reasons']==1).any()


def test_occlusion_stops_direction_without_reactivation():
    views,roi=scene();views[3]['validity'][:]=False
    result=t.recover_local_geometry(views,2,roi)
    assert not result['observation_mask'][:,3:].any()
    assert result['attempted'][:,3].all()
    assert not result['attempted'][:,4].any()
    assert (result['track_reason'][:,3]==5).any()


def test_full_reverse_chain_failure_not_an_observation(monkeypatch):
    views,roi=scene();step=t._step
    def bad(a,b,xy,check):
        out,reason=step(a,b,xy,check)
        if b['view_id']==2:
            out=out+np.array([3.,0.],np.float32)
        return out,reason
    monkeypatch.setattr(t,'_step',bad)
    result=t.recover_local_geometry(views,2,roi)
    assert not result['accepted'].any()
    assert not result['observation_mask'][:,:2].any()
    assert (result['track_reason']==8).any()


def test_repetitive_texture_ambiguity_rejected_before_triangulation():
    views,roi=scene(repeated=True)
    result=t.recover_local_geometry(views,2,roi)
    assert not result['accepted'].any()
    assert result['counters']['rejected_repetitive_reference']>0


def test_uniform_and_unknown_reference_empty():
    views,roi=scene()
    for view in views: view['image'][:]=100
    result=t.recover_local_geometry(views,2,roi)
    assert not len(result['reference_xy'])
    views,roi=scene();views[2]['validity'][:]=False
    assert not len(t.recover_local_geometry(views,2,roi)['reference_xy'])


def test_inconsistent_camera_heldout_geometry_rejects_otherwise_closed_tracks():
    views,roi=scene()
    views[0]['C'][0] -= .5
    result=t.recover_local_geometry(list(reversed(views)),2,roi)
    assert result['view_ids']==[0,1,2,3,4]
    assert (result['observation_mask'].sum(axis=1)>=3).any()
    assert not result['accepted'].any()
    assert np.isin(result['reasons'],[3,4]).any()


def test_nan_backward_flow_with_nonzero_status_is_explicitly_rejected(monkeypatch):
    views,roi=scene()
    for v in views:
        v['gray']=cv2.cvtColor(v['image'],cv2.COLOR_RGB2GRAY)
        v['known']=np.ones((96,96),np.float32)
    calls=[]
    def fake(a,b,xy,nextpts,**kwargs):
        calls.append(1)
        out=xy.copy()
        if len(calls)==2: out[:]=np.nan
        return out,np.ones((len(xy),1),np.uint8),None
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',fake)
    _,reasons=t._step(views[2],views[3],np.array([[48.,48.]],np.float32),lambda:None)
    assert reasons.tolist()==[3]


def test_holdout_removal_train_low_parallax_cannot_be_rescued():
    # Two training images share one camera center. The farthest-time source has
    # baseline and is excluded from fitting, so total angle alone is insufficient.
    views,roi=scene()
    chosen=[deepcopy(views[2]),deepcopy(views[2]),deepcopy(views[4])]
    for i,v in enumerate(chosen):
        v['view_id']=i;v['decode_index']=i*5
    result=t.recover_local_geometry(chosen,0,roi)
    assert (result['observation_mask'].sum(axis=1)==3).any()
    assert not result['accepted'].any()
    assert (result['reasons']==7).any()
    assert (result['train_angle_degrees'][result['reasons']==7]<1).all()
    assert result['per_source']['0']['external_observed']==0
    assert result['counters']['external_observed']==int(result['observation_mask'][:,1:].sum())
