from copy import deepcopy
import numpy as np
import pytest
from PIL import Image
from integrations.mvs_surface.depth_fusion_visuals import project_footprint,write_coverage_panel


def view():
    return dict(view_id=1,image=np.full((24,24,3),90,np.uint8),mask=np.ones((24,24),bool),validity=np.ones((24,24),bool),depth=np.full((24,24),2.,np.float32),depth_available=True,K=np.array([[10.,0,12],[0,10.,12],[0,0,1]]),R=np.eye(3),C=np.zeros(3))


def vertices_from_uv(uv,z=2.):
    xy=np.asarray(uv,float)
    return np.column_stack(((xy[:,0]-12)*z/10,(xy[:,1]-12)*z/10,np.full(len(xy),z)))


def test_triangle_union_is_not_projected_bounding_box_and_readonly():
    v=view();vertices=vertices_from_uv([[2,2],[20,2],[2,20]]);faces=np.array([[0,1,2]])
    before=deepcopy((v,vertices,faces))
    fp=project_footprint(vertices,faces,v)
    assert fp.dtype==bool and fp[5,5]
    assert not fp[19,19]
    np.testing.assert_array_equal(vertices,before[1]);np.testing.assert_array_equal(faces,before[2])
    for key in ('image','mask','validity','depth','K','R','C'):np.testing.assert_array_equal(v[key],before[0][key])


def test_outside_clipped_actual_triangle_and_behind_or_empty():
    v=view();faces=np.array([[0,1,2]])
    fp=project_footprint(vertices_from_uv([[-10,5],[5,5],[5,20]]),faces,v)
    assert fp[10,1]
    assert not fp[20,0]
    assert not project_footprint(vertices_from_uv([[40,40],[50,40],[40,50]]),faces,v).any()
    assert not project_footprint(vertices_from_uv([[2,2],[20,2],[2,20]],z=-2),faces,v).any()
    assert not project_footprint(np.empty((0,3)),np.empty((0,3),np.int64),v).any()


def test_near_plane_crossing_clips_without_unbounded_coordinates():
    v=view();vertices=np.array([[-.2,-.2,1.],[.2,-.2,1.],[0.,.3,-1.]])
    footprint=project_footprint(vertices,np.array([[0,1,2]]),v)
    assert footprint.shape==(24,24)
    assert footprint.any()


def test_background_crossing_panel_counts_and_no_visibility_claim(tmp_path):
    v=view();v['mask'][:,12:]=False;v['depth'][5:8,:12]=0;v['validity'][0]=False
    vertices=vertices_from_uv([[2,2],[20,2],[2,20]]);faces=np.array([[0,1,2]])
    old=np.zeros((24,24),bool);new=project_footprint(vertices,faces,v)
    assert (new&~v['mask']).any()
    coverage=np.zeros((24,24),np.uint8);coverage[v['mask']&v['validity']]=3
    coverage[v['mask']&v['validity']&(v['depth']==0)]=2;coverage[~v['validity']]=1
    path=tmp_path/'panel.png'
    result=write_coverage_panel(path,v,coverage,old,new)
    counts=result['counts']
    assert counts['foreground_without_depth']==36
    assert counts['valid_foreground_pixels']==276
    assert counts['foreground_native_depth']==240
    assert counts['old_foreground_covered']==counts['old_background_projection']==0
    assert counts['new_background_projection']>0
    assert counts['new_foreground_covered']==int((new&v['mask']&v['validity']).sum())
    assert 'not visibility' in result['semantics']
    assert result['bytes']==path.stat().st_size
    with Image.open(path) as image:assert image.width>=96 and image.height>24
    with pytest.raises(ValueError,match='fresh'):write_coverage_panel(path,v,coverage,old,new)


def test_invalid_indices_huge_coordinates_and_cancel(tmp_path):
    v=view();vertices=vertices_from_uv([[2,2],[20,2],[2,20]])
    with pytest.raises(ValueError):project_footprint(vertices,np.array([[0,1,8]]),v)
    huge=vertices.copy();huge[0,0]=1e20
    with pytest.raises(ValueError,match='magnitude'):project_footprint(huge,np.array([[0,1,2]]),v)
    def stop():raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError,match='cancelled'):project_footprint(vertices,np.array([[0,1,2]]),v,check=stop)
    with pytest.raises(ValueError):write_coverage_panel(tmp_path/'bad.png',v,np.full((24,24),9,np.uint8),np.zeros((24,24),bool),np.zeros((24,24),bool))
