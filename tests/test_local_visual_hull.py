"""Synthetic hull tests only; no real-image reconstruction."""
import copy
import numpy as np
import pytest
from integrations.mvs_surface.local_visual_hull import build_local_visual_hull as build


def scene():
    k=np.array([[100.,0,50],[0,100.,50],[0,0,1]])
    views=[]
    for i in range(3):
        mask=np.ones((101,101),bool)
        if i: mask[:,:45]=False; mask[:,56:]=False
        views.append(dict(view_id=i,K=k.copy(),R=np.eye(3),C=np.array([0.,0.,-10.]),mask=mask,validity=np.ones_like(mask)))
    return views,np.ones((101,101),bool),np.array([[-2.,-.25,0],[2.,.25,.5]])


def run(views,roi,bounds): return build(views,0,roi,bounds)


def test_synthetic_box_cross_section_and_open_volume_boundaries():
    views,roi,bounds=scene(); before=copy.deepcopy(views)
    result=run(views,roi,bounds)
    assert result['counts']['occupied']>0 and result['counts']['known_outside']>0
    assert result['counts']['faces']>0 and result['counts']['omitted_volume_boundary_adjacencies']>0
    assert not result['geometry_accepted'] and not result['surface_accepted']
    assert result['status']=='INFERRED_SILHOUETTE_BOUNDARY'
    vertices,faces=result['vertices'],result['faces']
    assert vertices.shape[1]==faces.shape[1]==3
    assert (faces>=0).all() and (faces<len(vertices)).all()
    triangles=vertices[faces]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    assert (np.linalg.norm(normals,axis=1)>0).all()
    # Perspective side planes staircase in z, but AABB ends are never capped.
    for axis in (1,2):
        for endpoint in bounds[:,axis]:
            assert not np.isclose(triangles[:,:,axis],endpoint).all(axis=1).any()
    xfaces=np.abs(normals[:,0])>0
    assert (normals[xfaces,0]*triangles[xfaces,:,0].mean(axis=1)>0).all()
    assert result['counts']['ambiguous_corner_triangles']==int(result['face_corner_ambiguous'].sum())
    assert result['face_corner_ambiguous'].shape==(len(faces),)
    for old,new in zip(before,views):
        for key in old:
            if isinstance(old[key],np.ndarray): np.testing.assert_array_equal(old[key],new[key])
    assert result['voxel_info']['dimensions'][0]==96
    assert 'NOT a bound' in result['voxel_info']['discretization']


def test_uniform_foreground_does_not_cap_volume():
    views,roi,bounds=scene()
    for v in views: v['mask'][:]=True
    result=run(views,roi,bounds)
    assert result['counts']['occupied']==result['voxel_info']['total_voxels']
    assert not len(result['faces'])
    assert result['counts']['omitted_volume_boundary_adjacencies']>0


def test_unknown_and_roi_cuts_are_not_closed():
    views,roi,bounds=scene()
    for v in views: v['mask'][:]=True
    views[2]['validity'][:50]=False
    result=run(views,roi,bounds)
    assert result['counts']['occupied']>0 and result['counts']['unknown']>0
    assert result['counts']['known_outside']==0 and not len(result['faces'])
    assert result['counts']['omitted_unknown_adjacencies']>0
    views[2]['validity'][:]=True
    roi[:,:50]=False
    result=run(views,roi,bounds)
    assert result['counts']['outside_reference_domain']>0
    assert result['counts']['occupied']>0 and not len(result['faces'])
    assert result['counts']['omitted_unknown_adjacencies']>0


def test_mask_gap_remains_known_outside_and_is_not_stitched():
    views,roi,bounds=scene()
    for v in views[1:]: v['mask'][:,50]=False
    result=run(views,roi,bounds)
    triangles=result['vertices'][result['faces']]
    assert len(triangles)>0
    # Faces may staircase with perspective depth, but none crosses the gap.
    assert ((triangles[:,:,0].max(axis=1)<=0) | (triangles[:,:,0].min(axis=1)>=0)).all()
    centers=bounds[0,0]+(np.arange(result['voxel_state'].shape[0])+.5)*result['voxel_info']['spacing'][0]
    gap=np.abs(centers)<.04
    assert (result['voxel_state'][gap]==2).all()


def test_rotated_world_camera_convention():
    views,roi,bounds=scene(); original=run(views,roi,bounds)
    q=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    t=np.array([3.,-2,1.])
    rotated=copy.deepcopy(views)
    for v in rotated:
        v['C']=q@v['C']+t; v['R']=v['R']@q.T
    corners=np.array([[a,b,c] for a in bounds[:,0] for b in bounds[:,1] for c in bounds[:,2]])@q.T+t
    result=run(rotated,roi,np.array([corners.min(axis=0),corners.max(axis=0)]))
    expected=np.round(original['vertices']@q.T+t,8)
    actual=np.round(result['vertices'],8)
    assert set(map(tuple,expected))==set(map(tuple,actual))
    assert len(result['faces'])==len(original['faces'])


def test_explicit_validation_and_cancellation():
    views,roi,bounds=scene()
    with pytest.raises(ValueError): run(views,roi,np.zeros((2,3)))
    bad=copy.deepcopy(views); bad[2]['view_id']=0
    with pytest.raises(ValueError): run(bad,roi,bounds)
    bad=copy.deepcopy(views); bad[2]['R'][0,0]=2
    with pytest.raises(ValueError): run(bad,roi,bounds)
    def stop(): raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'): build(views,0,roi,bounds,check=stop)


def test_face_budget_rejects_instead_of_truncating(monkeypatch):
    from integrations.mvs_surface import local_visual_hull as module
    views,roi,_=scene()
    def stripes(views,ref,roi,points,check):
        check()
        indices=np.floor(points[:,0]*96).astype(int)
        return np.where(indices%2,1,2).astype(np.uint8),np.ones(len(points),bool)
    monkeypatch.setattr(module,'_classify',stripes)
    with pytest.raises(ValueError,match='face budget exceeded; no truncation'):
        run(views,roi,np.array([[0.,0.,0.],[1.,1.,1.]]))
