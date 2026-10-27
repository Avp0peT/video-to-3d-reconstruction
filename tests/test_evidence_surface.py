import copy
import numpy as np
import pytest
from integrations.mvs_surface.evidence_surface import build_depth_surface, _covered_pixels


def test_integral_foreground_shortcut_is_sufficient_only():
    from integrations.mvs_surface.evidence_surface import _known_foreground_bbox
    rng=np.random.default_rng(1984)
    for index in range(150):
        triangle=rng.uniform(-1,17,(3,2))
        known=rng.random((18,18))>.02
        if index%3==0: known[:]=True
        integral=np.pad(known.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
        if _known_foreground_bbox(triangle,known.shape,integral):
            covered=_covered_pixels(triangle,known.shape)
            assert covered is not None
            cells,complete=covered
            assert complete and len(cells) and known[cells[:,1],cells[:,0]].all()


def test_integral_negative_does_not_veto_nonfootprint_background_or_unknown():
    from integrations.mvs_surface.evidence_surface import _known_foreground_bbox
    triangle=np.array([[1.,1.],[7.,1.],[7.,7.]])
    for mode in ('background','unknown'):
        mask=np.ones((10,10),bool);valid=mask.copy()
        if mode=='background':mask[6,1]=False
        else:valid[6,1]=False
        integral=np.pad((mask&valid).astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
        assert not _known_foreground_bbox(triangle,mask.shape,integral)
        cells,complete=_covered_pixels(triangle,mask.shape)
        assert complete and (mask&valid)[cells[:,1],cells[:,0]].all()


def test_fastpath_preserves_ordered_support_ids():
    views=scene();result=build_depth_surface(views)
    assert result['counts'].get('source_bbox_fast_accepts',0)>0
    assert len(result['faces'])>0
    for refid,ids in zip(result['sourceview'],result['face_support_view_ids']):
        assert ids==[v['view_id'] for v in views if v['view_id']!=refid]


def scene():
    views=[]; shape=(13,13)
    for i,c in enumerate([-.2,0.,.2]):
        views.append(dict(view_id=i,image=np.full((*shape,3),[80,120,160],np.uint8),mask=np.ones(shape,bool),validity=np.ones(shape,bool),depth=np.full(shape,2.,np.float32),K=np.array([[20.,0,6.],[0,20.,6.],[0,0,1.]]),R=np.eye(3),C=np.array([c,0.,0.])))
    return views


def test_plane_retains_real_depth_faces_with_provenance():
    views=scene(); result=build_depth_surface(views)
    assert len(result['faces'])>0
    assert np.isfinite(result['worldvertices']).all()
    assert np.all(result['worldvertices'][:,2]==2.)
    assert len(result['sourceview'])==len(result['faces'])
    for face,iid in zip(result['faces'],result['sourceview']):
        assert (result['vertex_sourceview'][face]==iid).all()
    assert all(len(ids)>=2 for ids in result['face_support_view_ids'])
    assert not result['geometry_accepted'] and not result['watertight']


def test_reference_thin_gap_no_bridge():
    views=scene()
    # A single native-pixel gap lies strictly between grid vertices x=3 and6.
    for view in views: view['depth'][:,4]=0
    result=build_depth_surface(views)
    for face,iid in zip(result['faces'],result['sourceview']):
        v=views[iid]; camera=(result['worldvertices'][face]-v['C'])@v['R'].T
        uv=camera@v['K'].T; uv=uv[:,:2]/camera[:,2,None]
        assert not (uv[:,0].min()<4<uv[:,0].max())
    assert result['counts']['ref_missing_or_background_cells']>0


def test_projected_gap_inside_triangle_not_only_vertices():
    views=scene()
    # Keep reference native depth only. Supporting camera has an interior stripe;
    # foreground vertices cannot override the all-camera background veto.
    for view in views[1:]: view['depth'][:]=0
    views[2]['mask'][:,4]=False
    result=build_depth_surface(views)
    assert result['counts'].get('background_veto_faces',0)>0
    source=views[2]
    for face in result['faces']:
        xyz=result['worldvertices'][face]; cam=(xyz-source['C'])@source['R'].T
        uv=cam@source['K'].T; uv=uv[:,:2]/cam[:,2,None]
        cells,_=_covered_pixels(uv,source['mask'].shape)
        assert source['mask'][cells[:,1],cells[:,0]].all()


def test_depth_jump_rejected():
    views=scene()
    for view in views: view['depth'][:,4:]=3.
    result=build_depth_surface(views)
    assert result['counts']['ref_depth_jump_cells']>0
    for face in result['faces']:
        z=result['worldvertices'][face,2]
        assert z.max()/z.min()<=1.02


def test_unknown_is_not_support():
    views=scene(); views[2]['validity'][:]=False
    result=build_depth_surface(views)
    assert not len(result['faces'])
    assert result['status']=='NO_EVIDENCE_SUPPORTED_FACES'


def test_image_boundary_not_support():
    views=scene(); views[2]['C'][0]=200
    result=build_depth_surface(views)
    assert not len(result['faces'])


def test_support_requires_spatial_separation():
    views=scene()
    for view in views: view['C'][:]=0
    result=build_depth_surface(views)
    assert not len(result['faces'])
    assert result['counts']['insufficient_view_spread_faces']>0


def test_krc_world_transform_and_no_input_mutation():
    views=scene(); before=copy.deepcopy(views)
    theta=.2; rotation=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    translation=np.array([2.,3.,4.])
    first=build_depth_surface(views)
    for view in views:
        view['R']=view['R']@rotation.T; view['C']=view['C']@rotation.T+translation
    transformed_before=copy.deepcopy(views); second=build_depth_surface(views)
    np.testing.assert_allclose(second['worldvertices'],first['worldvertices']@rotation.T+translation,atol=1e-12)
    np.testing.assert_array_equal(first['faces'],second['faces'])
    for old,new in zip(transformed_before,views):
        for key in old:
            if isinstance(old[key],np.ndarray): np.testing.assert_array_equal(old[key],new[key])
    assert all(np.array_equal(a['depth'],b['depth']) for a,b in zip(before,views))


def test_interior_pixel_coverage_and_no_duplicate_identity():
    cells,complete=_covered_pixels(np.array([[1.,1.],[7.,1.],[7.,7.]]),(10,10))
    assert complete and any(np.array_equal(p,[5,3]) for p in cells)
    assert any(np.array_equal(p,[4,1]) for p in cells)
    views=scene(); views[2]['view_id']=0
    with pytest.raises(ValueError,match='duplicate'): build_depth_surface(views)


def test_veto_short_circuits_but_all_references_remain_participating():
    views=scene()
    # Add an early source with foreground vertices but stripe background.
    views[1]['mask'][:,4]=False
    result=build_depth_surface(views)
    assert set(result['per_view'])=={'0','1','2'}
    assert result['counts']['source_checks_processed']>0
    assert result['counts']['source_checks_skipped_after_veto']>0
    assert result['counts']['background_veto_faces']>0
    # A processed source check is precisely foreground support, unknown, or veto.
    assert 'not classified as unknown' in result['source_check_count_semantics']


def test_eleven_reference_views_not_reduced_by_optimization():
    views=scene()
    views=[copy.deepcopy(views[i%3]) for i in range(11)]
    for i,view in enumerate(views): view['view_id']=i
    result=build_depth_surface(views)
    assert set(result['per_view'])=={str(i) for i in range(11)}
    assert (result['counts']['source_checks_processed']>0)
    assert set(result['sourceview'])==set(range(11))
