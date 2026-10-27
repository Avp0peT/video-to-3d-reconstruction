"""Synthetic projective TSDF only; no native reconstruction inputs."""
import copy
import numpy as np
import pytest
from integrations.mvs_surface.projective_tsdf import fuse_depth_views as fuse


def scene():
    shape=(65,65);views=[]
    for i,x in enumerate((-.08,.08)):
        views.append(dict(view_id=i,image=np.full((*shape,3),110,np.uint8),mask=np.ones(shape,bool),validity=np.ones(shape,bool),depth=np.full(shape,2.,np.float32),
            K=np.array([[50.,0,32],[0,50.,32],[0,0,1]]),R=np.eye(3),C=np.array([x,0.,0.]),depth_available=True))
    return views,np.array([[-.4,-.4,1.6],[.4,.4,2.4]])


def test_two_views_flat_plane_native_depth_and_no_mutation():
    views,bounds=scene();before=copy.deepcopy(views)
    result=fuse(views,bounds,max_axis=24)
    assert len(result['faces'])>0
    assert np.allclose(result['vertices'][:,2],2.,atol=1e-6)
    assert (result['face_min_weight']>=2).all() and (result['face_min_near_votes']>=2).all()
    assert (result['vertex_min_weight']>=2).all() and (result['vertex_min_near_votes']>=2).all()
    assert result['weight'].max()==2 and result['near_surface_votes'].max()==2
    assert np.isfinite(result['tsdf_vote_variance']).all() and (result['tsdf_vote_variance']>=0).all()
    assert np.allclose(result['tsdf_vote_variance'],0,atol=1e-7)
    np.testing.assert_array_equal(result['mask_background_votes'],result['mask_background_count'])
    assert not result['geometry_accepted'] and not result['complete']
    assert result['counts']['removed_untrusted_cell_faces']>0
    for old,new in zip(before,views):
        for key in old:
            if isinstance(old[key],np.ndarray):np.testing.assert_array_equal(old[key],new[key])


def test_face_winding_points_to_positive_free_space():
    views,bounds=scene();r=fuse(views,bounds,max_axis=16)
    tri=r['vertices'][r['faces']]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    assert len(normals)>0 and (np.linalg.norm(normals,axis=1)>0).all()
    # Cameras are below z=2. Free space has positive d-z TSDF, hence -z outward.
    assert (normals[:,2]<0).all()
    assert np.allclose(normals[:,:2],0,atol=1e-8)
    assert 'descent' in r['parameters']['face_winding']


def test_depth_unavailable_zero_and_single_support_never_form_surface():
    views,bounds=scene();views[1]['depth_available']=False
    r=fuse(views,bounds,max_axis=16)
    assert not len(r['faces']) and r['weight'].max()==1
    for v in views:v['depth'][:]=0
    r=fuse(views,bounds,max_axis=16)
    assert not len(r['faces']) and not r['weight'].any()
    assert (r['tsdf']==1).all()


def test_unknown_and_background_boundaries_not_capped():
    views,bounds=scene()
    for v in views:
        v['validity'][:,:32]=False
        v['mask'][40:,:]=False
    r=fuse(views,bounds,max_axis=24)
    assert len(r['faces'])>0 and r['mask_background_count'].max()==2
    assert np.allclose(r['vertices'][:,2],2.,atol=1e-6)
    triangles=r['vertices'][r['faces']]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    assert np.allclose(normals[:,:2],0,atol=1e-8)


def test_separated_depth_planes_do_not_bridge():
    views,_=scene()
    for v in views:v['depth'][:,33:]=3.
    bounds=np.array([[-.7,-.25,1.7],[.7,.25,3.3]])
    r=fuse(views,bounds,max_axis=48)
    assert len(r['faces'])>0
    z=r['vertices'][:,2]
    assert np.isclose(z,2,atol=1e-5).any() and np.isclose(z,3,atol=1e-5).any()
    assert (np.isclose(z,2,atol=1e-5)|np.isclose(z,3,atol=1e-5)).all()
    assert (np.ptp(r['vertices'][r['faces'],2],axis=1)<1e-5).all()


def test_camera_world_rotation_translation_convention():
    views,bounds=scene();a=fuse(views,bounds,max_axis=16)
    q=np.array([[0.,-1,0],[1,0,0],[0,0,1]]);t=np.array([2.,3.,4.])
    for v in views:v['C']=q@v['C']+t;v['R']=v['R']@q.T
    corners=np.array([[x,y,z] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])@q.T+t
    b=fuse(views,np.array([corners.min(0),corners.max(0)]),max_axis=16)
    assert len(a['faces'])==len(b['faces'])
    assert set(map(tuple,np.round(a['vertices']@q.T+t,6)))==set(map(tuple,np.round(b['vertices'],6)))


def test_positive_space_and_occluded_negative_samples():
    views,bounds=scene();r=fuse(views,bounds,max_axis=16,truncation_voxels=1)
    assert (r['weight'][:,:,0]==2).all() and (r['tsdf'][:,:,0]==1).all()
    assert (r['weight'][:,:,-1]==0).all()
    assert (r['near_surface_votes'][:,:,0]==0).all()
    assert all(s['occluded_samples']>0 for s in r['per_view'].values())


def test_duplicate_ids_bounds_budget_and_cancel():
    views,bounds=scene();views[1]['view_id']=0
    with pytest.raises(ValueError,match='duplicate'):fuse(views,bounds,max_axis=16)
    views,bounds=scene()
    with pytest.raises(ValueError):fuse(views,bounds,max_axis=193)
    with pytest.raises(ValueError):fuse(views,bounds,max_axis=True)
    with pytest.raises(ValueError):fuse(views,np.zeros((2,3)))
    with pytest.raises(ValueError):fuse(views,bounds,min_views=1)
    def stop():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):fuse(views,bounds,check=stop)


def test_extraction_face_budget_rejects_without_truncation(monkeypatch):
    from skimage import measure
    views,bounds=scene()
    def too_many(*a,**k):
        # A valid eligible plane triangle, repeated beyond the hard face limit.
        vertices=np.array([[8.,8.,8.],[9.,8.,8.],[9.,9.,8.]],np.float32)
        faces=np.tile(np.array([[0,1,2]],np.int32),(400001,1))
        return vertices,faces,None,None
    monkeypatch.setattr(measure,'marching_cubes',too_many)
    with pytest.raises(ValueError,match='face budget exceeded; no truncation'):
        fuse(views,bounds,max_axis=16)



def many_views(count):
    original,bounds=scene()
    views=[]
    for i in range(count):
        view=copy.deepcopy(original[i%2]);view['view_id']=1000+i
        views.append(view)
    return views,bounds


def test_seventeen_sources_are_all_counted_not_truncated_to_sixteen():
    views,bounds=many_views(17)
    r=fuse(views,bounds,max_axis=8)
    assert len(r['source_ids'])==17
    assert r['weight'].max()==17
    assert r['view_support_bits'].dtype==np.uint64
    assert r['near_view_support_bits'].dtype==np.uint64
    assert r['face_common_near_bits'].dtype==np.uint64
    assert (r['face_common_support_bits']==np.uint64((1<<17)-1)).all()
    assert r['parameters']['max_views']==64


def test_sixty_four_sources_preserve_all_common_near_bits_and_counts():
    views,bounds=many_views(64)
    r=fuse(views,bounds,max_axis=8,min_views=64)
    full=np.uint64((1<<64)-1)
    assert len(r['faces'])>0
    assert r['weight'].max()==64
    assert r['near_surface_votes'].max()==64
    assert r['tsdf'].dtype==np.float32
    assert (r['face_min_weight']==64).all()
    assert (r['face_min_near_votes']==64).all()
    assert (r['face_common_support_bits']==full).all()
    assert (r['face_common_near_bits']==full).all()
    np.testing.assert_allclose(r['vertices'][:,2],2.,atol=1e-5)


def test_high_bit63_without_low_bits_does_not_overflow_and_cap65_rejects():
    views,bounds=many_views(64)
    for view in views[:62]:view['depth_available']=False
    r=fuse(views,bounds,max_axis=8)
    high=np.uint64((1<<62)|(1<<63))
    assert len(r['faces'])>0
    assert r['weight'].max()==2
    assert (r['face_common_support_bits']==high).all()
    assert (r['face_common_near_bits']==high).all()
    assert np.bitwise_and(r['near_view_support_bits'],np.uint64(1<<63)).max()==np.uint64(1<<63)
    views,bounds=many_views(65)
    with pytest.raises(ValueError,match='sixty-four'):fuse(views,bounds,max_axis=8)
    with pytest.raises(ValueError,match='2..64'):fuse(views[:64],bounds,max_axis=8,min_views=65)
