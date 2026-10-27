from copy import deepcopy
import cv2
import numpy as np
import pytest
from integrations.mvs_surface.silhouette_completion import infer_silhouette_completion as infer


def sphere_views():
    cv2.setNumThreads(2)
    k=np.array([[45.,0,32],[0,45.,32],[0,0,1]])
    y,x=np.mgrid[:65,:65];rays=np.stack(((x-32)/45,(y-32)/45,np.ones((65,65))),axis=-1)
    result=[]
    for i,center in enumerate(([0.,0.,-3.],[3.,0.,0.],[0.,3.,0.])):
        c=np.asarray(center);forward=-c/np.linalg.norm(c);up=np.array([0.,1.,0.])
        if abs(up@forward)>.9:up=np.array([0.,0.,1.])
        right=np.cross(up,forward);right/=np.linalg.norm(right);down=np.cross(forward,right)
        rotation=np.stack((right,down,forward));direction=rays@rotation
        dot=np.sum(direction*c,axis=-1);norm=np.sum(direction*direction,axis=-1)
        disc=dot*dot-norm*(c@c-.65**2)
        mask=(disc>=0)&(dot<0)
        result.append(dict(view_id=i,K=k.copy(),R=rotation,C=c,mask=mask,validity=np.ones((65,65),bool)))
    return result,np.array([[-1.,-1.,-1.],[1.,1.,1.]])


def test_strict_three_view_envelope_closed_inside_safe_known_negative_box_readonly():
    views,bounds=sphere_views();before=deepcopy(views)
    result=infer(views,bounds,max_axis=16,quantile=0.)
    assert len(result['faces'])>0
    assert result['boundary_known_negative']
    assert result['closed_mesh_topology']
    assert result['closed_envelope_with_safe_boundary']
    assert result['counts']['boundary_edges']==0
    assert result['counts']['nonmanifold_edges']==0
    assert (result['face_min_known']>=3).all()
    assert not result['geometry_accepted'] and not result['complete']
    assert (result['colors']==[255,150,30]).all()
    for a,b in zip(views,before):
        for key in ('K','R','C','mask','validity'):np.testing.assert_array_equal(a[key],b[key])


def test_quantile_lower_allows_one_of_eleven_but_not_one_of_three():
    original,bounds=sphere_views();views=[]
    for i in range(11):
        v=deepcopy(original[0]);v['view_id']=i;v['C'][0]=i*.005
        v['mask'][:]=i!=0;views.append(v)
    strict=infer(views,bounds,max_axis=8,quantile=0.)
    robust=infer(views,bounds,max_axis=8,quantile=.1)
    assert not (strict['field']>0).any()
    assert (robust['field']>0).all()
    assert not robust['boundary_known_negative']
    assert robust['status']=='UNKNOWN_VOLUME_BOUNDARY'
    assert not robust['closed_envelope_with_safe_boundary']
    three=infer(views[:3],bounds,max_axis=8,quantile=.1)
    assert not (three['field']>0).any()
    assert three['parameters']['quantile_method']=='lower'


def test_unknown_is_not_background_or_a_surface_cap():
    views,bounds=sphere_views();views[0]['validity'][:]=False
    result=infer(views,bounds,max_axis=8)
    assert not len(result['faces'])
    assert not result['trusted_mask'].any()
    assert result['known_count'].max()==2
    assert not result['boundary_known_negative']
    assert result['per_view']['0']['background_votes']==0
    assert result['per_view']['0']['known_lattice_votes']==0


def test_touching_box_does_not_get_fake_boundary_faces_or_closed_guarantee():
    views,_=sphere_views();bounds=np.array([[-.3,-.3,-.3],[.3,.3,.3]])
    result=infer(views,bounds,max_axis=8,quantile=0.)
    assert (result['field']>0).all()
    assert not len(result['faces'])
    assert result['status']=='UNKNOWN_VOLUME_BOUNDARY'
    assert not result['closed_envelope_with_safe_boundary']


def test_explicit_optional_sigma_and_camera_baseline_validation():
    views,bounds=sphere_views();r=infer(views,bounds,max_axis=8,smoothing_sigma=1.)
    assert r['parameters']['smoothing_sigma']==1.
    assert r['parameters']['unknown_caps'] is False
    for v in views:v['C']=views[0]['C'].copy()
    with pytest.raises(ValueError,match='distinct camera centers'):infer(views,bounds,max_axis=8)
    views,bounds=sphere_views()
    with pytest.raises(ValueError):infer(views,bounds,quantile=.2)
    with pytest.raises(ValueError):infer(views,bounds,smoothing_sigma=.5)
    with pytest.raises(ValueError):infer(views,bounds,max_axis=161)
    def stop():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):infer(views,bounds,check=stop)


def test_face_capacity_rejects_without_truncation(monkeypatch):
    from skimage import measure
    views,bounds=sphere_views()
    def huge(*args,**kwargs):
        vertices=np.array([[4.,4.,4.],[5.,4.,4.],[5.,5.,4.]],np.float32)
        faces=np.tile(np.array([[0,1,2]],np.int32),(400001,1))
        return vertices,faces,None,None
    monkeypatch.setattr(measure,'marching_cubes',huge)
    with pytest.raises(ValueError,match='400000 face limit; no truncation'):infer(views,bounds,max_axis=8)
