from copy import deepcopy
import numpy as np
import pytest
from integrations.mvs_surface import strict_point_audit as s


def scene(centers=(-.1,0.,.1)):
    views=[]
    for i,c in enumerate(centers):
        views.append(dict(view_id=i,image=np.full((64,64,3),110,np.uint8),mask=np.ones((64,64),bool),validity=np.ones((64,64),bool),depth=np.full((64,64),2.,np.float32),depth_available=True,K=np.array([[100.,0,32],[0,100.,32],[0,0,1]]),R=np.eye(3),C=np.array([c,0.,0.])))
    vertices=np.array([[-.1,-.1,2.],[.1,-.1,2.],[0.,.1,2.]])
    faces=np.array([[0,1,2]],np.int64)
    return vertices,faces,views


def test_three_sources_strict_face_and_original_geometry_unchanged():
    vertices,faces,views=scene();before=deepcopy((vertices,faces,views))
    result=s.audit_native_mesh(vertices,faces,views)
    assert result['keep_vertices'].tolist()==[True,True,True]
    assert result['keep_faces'].tolist()==[True]
    assert result['face_common_support_count'].tolist()==[3]
    assert result['face_full_foreground_view_count'].tolist()==[3]
    assert result['vertex_strict_near_view_bits'].tolist()==[[7],[7],[7]]
    assert (result['keep_vertices']<=result['baseline_supported']).all()
    assert result['retained_face_ids'].tolist()==[0]
    assert result['original_vertex_ids'].tolist()==[0,1,2]
    np.testing.assert_array_equal(vertices,before[0]);np.testing.assert_array_equal(faces,before[1])
    for a,b in zip(views,before[2]):
        for key in ('image','mask','validity','depth','K','R','C'):np.testing.assert_array_equal(a[key],b[key])


def test_baseline_passes_two_views_but_strict_requires_three():
    vertices,faces,views=scene()
    result=s.audit_native_mesh(vertices,faces,views[:2])
    assert result['baseline_supported'].all()
    assert not result['keep_vertices'].any() and not result['keep_faces'].any()
    assert (result['vertex_reasons']==2).all()


def test_baseline_one_degree_passes_but_strict_two_degree_fails():
    vertices,faces,views=scene((-.02,0.,.02))
    result=s.audit_native_mesh(vertices,faces,views)
    assert result['baseline_supported'].all()
    assert (result['vertex_strict_support_count']==3).all()
    assert (result['vertex_reasons']==3).all()
    assert not result['keep_vertices'].any()


def test_one_percent_gate_not_silently_old_two_percent():
    vertices,faces,views=scene()
    for view in views:view['depth'][:]=2/1.015
    result=s.audit_native_mesh(vertices,faces,views)
    assert result['baseline_supported'].all()
    assert (result['vertex_strict_support_count']==0).all()
    assert not result['keep_vertices'].any()
    assert not result['vertex_strict_free_space_conflict'].any()


@pytest.mark.parametrize('mode',['stricter_free','positive_residual','occluded','missing'])
def test_extra_source_strict_free_space_and_unknown_occlusion_rules(mode):
    vertices,faces,views=scene((-.1,0.,.1,.15))
    if mode=='stricter_free':views[3]['depth'][:]=2/.985
    elif mode=='positive_residual':views[3]['depth'][:]=2/1.015
    elif mode=='occluded':views[3]['depth'][:]=1.
    else:views[3]['depth'][:]=0;views[3]['depth_available']=False
    result=s.audit_native_mesh(vertices,faces,views)
    assert result['baseline_supported'].all()
    if mode=='stricter_free':
        assert result['vertex_strict_free_space_conflict'].all()
        assert (result['vertex_reasons']==1).all()
        assert not result['keep_vertices'].any()
    else:
        assert not result['vertex_strict_free_space_conflict'].any()
        assert result['keep_vertices'].all() and result['keep_faces'].all()


def test_vertices_strict_but_only_two_common_sources_no_face_gate_call(monkeypatch):
    vertices,faces,views=scene((-.15,-.075,0.,.075,.15))
    allowed=[{0,1,2},{0,1,3},{0,1,4}]
    for j,view in enumerate(views):
        for i,point in enumerate(vertices):
            if j in allowed[i]:continue
            camera=point-view['C'];uv=camera@view['K'].T;uv=uv[:2]/camera[2]
            x,y=np.floor(uv).astype(int);view['depth'][y:y+2,x:x+2]=0
    def forbidden(*args,**kwargs):raise AssertionError('not a candidate')
    monkeypatch.setattr(s,'gate_surface',forbidden)
    result=s.audit_native_mesh(vertices,faces,views)
    assert result['keep_vertices'].all()
    assert result['face_common_support_count'].tolist()==[2]
    assert not result['face_candidate'].any()
    assert not result['keep_faces'].any()


@pytest.mark.parametrize('kind',['background','unknown'])
def test_full_triangle_footprint_blocks_interior_without_relaxing_vertex_support(kind):
    vertices,faces,views=scene()
    # In view1 center=(32,30) is inside the triangle and away from its vertices.
    if kind=='background':views[1]['mask'][30,32]=False
    else:views[1]['validity'][30,32]=False
    result=s.audit_native_mesh(vertices,faces,views)
    assert result['keep_vertices'].all()
    assert result['face_candidate'].all()
    assert not result['keep_faces'].any()
    assert result['face_reasons'].tolist()==([3] if kind=='background' else [4])
    if kind=='unknown':assert result['face_mask_gate_keep'].all()


def test_original_mask_conflict_label_preserved_and_empty_is_legal():
    vertices,faces,views=scene((-.1,0.,.1,.15));views[3]['mask'][:]=False
    result=s.audit_native_mesh(vertices,faces,views)
    assert (result['baseline_classification']==2).all()
    assert result['baseline_mask_background_diagnostic'].all()
    assert not result['baseline_supported'].any() and not result['keep_vertices'].any()
    vertices,faces,views=scene()
    empty=s.audit_native_mesh(np.empty((0,3)),np.empty((0,3),np.int64),views)
    assert empty['keep_vertices'].shape==(0,) and empty['keep_faces'].shape==(0,)
    assert empty['counts']['strict_supported_vertices']==0


def test_invalid_faces_fail_and_view_identity_not_counted_twice():
    vertices,faces,views=scene()
    with pytest.raises(ValueError):s.audit_native_mesh(vertices,np.array([[0,1,20]]),views)
    with pytest.raises(ValueError,match='duplicate'):s.audit_native_mesh(vertices,faces,[views[0],views[0],views[1]])
