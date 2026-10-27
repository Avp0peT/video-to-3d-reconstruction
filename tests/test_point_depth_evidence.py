from copy import deepcopy
import numpy as np
import pytest
from integrations.mvs_surface.point_depth_evidence import assess_points


def scene(focal=100.):
    def view(i,c):
        return dict(view_id=i,image=np.full((128,128,3),100,np.uint8),mask=np.ones((128,128),bool),validity=np.ones((128,128),bool),depth=np.full((128,128),2.,np.float32),K=np.array([[focal,0,64],[0,focal,64],[0,0,1]]),R=np.eye(3),C=np.array([c,0,0]),depth_available=True)
    return [view(1,-.04),view(2,.04)]


def test_support_requires_two_depths_and_angle_without_modifying_inputs():
    views=scene();points=np.array([[0.,0,2.],[0,0,2.2],[0,0,1.8]])
    before=deepcopy((points,views))
    r=assess_points(points,views)
    assert r['classification'].tolist()==[1,0,2]
    assert r['state'][0].tolist()==[5,5]
    assert r['state'][1].tolist()==[6,6]
    assert r['state'][2].tolist()==[7,7]
    assert r['max_support_angle_degrees'][0]>=1
    assert r['origin_unknown'].all()
    np.testing.assert_array_equal(points,before[0])
    for a,b in zip(views,before[1]):
        for key in ('image','mask','validity','depth','K','R','C'): np.testing.assert_array_equal(a[key],b[key])


def test_axial_two_percent_not_secret_world_distance_gate():
    views=scene(focal=1000.)
    for v in views:v['depth'][:]=2.025
    result=assess_points(np.array([[0.,0.,2.]]),views)
    assert result['supported'][0]
    assert np.all(result['signed_relative_residual'][0]<0)
    assert np.all(result['world_distance_error'][0]>.004)
    np.testing.assert_allclose(result['sampled_depth'],2.025)


def test_origin_excluded_and_one_support_or_colocated_cameras_unverified():
    points=np.array([[0.,0,2.]])
    views=scene()
    r=assess_points(points,views,np.array([1]))
    assert r['state'][0].tolist()==[0,5]
    assert r['classification'][0]==0
    assert not r['origin_unknown'][0]
    views[1]['C']=views[0]['C'].copy()
    r=assess_points(points,views)
    assert r['support_count'][0]==2 and not r['supported'][0]


@pytest.mark.parametrize('mode,expected',[('missing',2),('unavailable',2),('invalid',4),('discontinuity',4),('background',3)])
def test_native_patch_admission_and_unknowns(mode,expected):
    views=scene()
    for v in views:
        if mode=='missing':v['depth'][:]=0
        if mode=='unavailable':v['depth_available']=False
        if mode=='invalid':v['validity'][63:66,60:68]=False
        if mode=='discontinuity':v['depth'][64,63:67]=3.
        if mode=='background':v['mask'][64:66,62:68]=False
    r=assess_points(np.array([[0.,0,2.]]),views)
    assert (r['state']==expected).all()
    assert not r['supported'].any()
    assert bool(r['conflict'][0])==(mode=='background')


def test_mask_background_diagnostic_available_even_without_depth_and_origin_excluded_first():
    views=scene()
    for v in views:v['mask'][:]=False;v['depth_available']=False;v['depth'][:]=0
    r=assess_points(np.array([[0.,0,2.]]),views,np.array([1]))
    assert r['state'].tolist()==[[0,3]]
    assert r['conflict'][0]
    assert r['mask_background_diagnostic'][0]
    assert not r['native_depth_conflict'][0]


def test_full_2x2_boundary_not_clamped_and_thin_mask_veto():
    views=scene()
    for v in views:v['C'][:]=0
    points=np.array([[1.26,0.,2.],[1.259,0.,2.]])
    for v in views:v['mask'][:,127]=False
    r=assess_points(points,views)
    assert r['state'][0].tolist()==[1,1]
    assert r['state'][1].tolist()==[4,4]


def test_inverse_depth_bilinear_not_axial_interpolation():
    views=scene()
    for v in views:
        v['C'][:]=0
        v['depth'][:,64]=2.
        v['depth'][:,65]=2.03
    r=assess_points(np.array([[.01,0,2.]]),views)
    expected=1/(.5/2.+.5/2.03)
    np.testing.assert_allclose(r['sampled_depth'],expected,rtol=1e-6)


def test_world_coordinate_rigid_transform_invariance():
    views=scene();points=np.array([[0.,0.,2.],[0,0,2.2],[0,0,1.8]])
    a=assess_points(points,views)
    theta=.4;q=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    translation=np.array([2.,3.,-4.])
    for view in views:
        view['C']=view['C']@q.T+translation
        view['R']=view['R']@q.T
    b=assess_points(points@q.T+translation,views)
    np.testing.assert_array_equal(a['state'],b['state'])
    np.testing.assert_allclose(a['max_support_angle_degrees'],b['max_support_angle_degrees'],atol=1e-6)


def test_duplicates_limits_bad_inputs_and_empty_output():
    views=scene();point=np.array([[0.,0,2.]])
    with pytest.raises(ValueError,match='duplicate'):assess_points(point,[views[0],views[0]])
    with pytest.raises(ValueError):assess_points(np.zeros((200001,3)),views)
    with pytest.raises(ValueError):assess_points(point,views,np.array([True]))
    with pytest.raises(ValueError):assess_points(np.array([[np.nan,0,2.]]),views)
    result=assess_points(np.empty((0,3)),views)
    assert result['state'].shape==(0,2) and result['counts']['points']==0


def test_bounded_chunks_and_source_count():
    views=scene();calls=[]
    points=np.tile(np.array([[0.,0.,2.]]),(8200,1))
    r=assess_points(points,views,check=lambda:calls.append(1))
    assert r['supported'].all() and len(calls)>=9
    assert all(sum(values.values())==len(points) for values in r['per_view'].values())


def test_one_background_neighbor_is_unknown_not_conflict_and_validity_required():
    views=scene()
    for v in views:
        v['C'][:]=0
        v['mask'][65,65]=False
    r=assess_points(np.array([[.01,.01,2.]]),views)
    assert r['state'].tolist()==[[4,4]]
    assert not r['conflict'][0]
    assert not r['mask_background_diagnostic'][0]
    assert r['unverified'][0]
    del views[0]['validity']
    with pytest.raises(ValueError,match='invalid validity'):
        assess_points(np.array([[0.,0.,2.]]),views)


@pytest.mark.parametrize('count',[17,64,67,128])
def test_extended_view_count_no_uint8_vote_overflow(count):
    base=scene();selected=[]
    for i in range(count):
        v=deepcopy(base[i%2]);v['view_id']=i;selected.append(v)
    r=assess_points(np.array([[0.,0.,2.]]),selected)
    assert r['state'].shape==(1,count)
    assert r['depth_support_count'].dtype==np.uint8
    assert int(r['depth_support_count'][0])==count
    assert r['supported'][0]
    assert r['policy']['max_views']==128
    excluded=assess_points(np.array([[0.,0.,2.]]),selected,np.array([count-1]))
    assert int(excluded['depth_support_count'][0])==count-1
    assert excluded['state'][0,-1]==0


def test_129_view_audit_rejected():
    with pytest.raises(ValueError,match='128'):assess_points(np.array([[0.,0.,2.]]),[scene()[0]]*129)
