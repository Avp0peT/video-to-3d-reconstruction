"""Pure-data registration predicate and ordered correspondence gates."""
import pytest
from integrations.demo_pipeline.registration_gate import bogus_camera, summarize_gate


def camera(k=0):
    return bogus_camera(100, 200, [200, 50, 100, k], .1, 10., 1.)


def link(source=0, target=1, point=7, camera_id=1, pose=True):
    return dict(source_row=source, target_image_id=target, target_point_row=3,
                point3D_id=point, camera_id=camera_id, has_pose=pose)


@pytest.mark.parametrize('f,k', [(20., -1.), (2000., 1.)])
def test_native_equal_thresholds_are_not_bogus(f, k):
    assert not bogus_camera(100, 200, [f, 0., 200., k], .1, 10., 1.)['bogus']


def test_negative_extra_uses_absolute_value_and_predicate_order():
    assert not camera(-1.)['bogus']
    assert camera(-1.00001)['extra_params_bogus']
    assert bogus_camera(100, 200, [1., -1., 100., -2.], .1, 10., 1.)['first_native_rejection'] == 'PRINCIPAL_POINT'


def test_bad_source_does_not_reserve_point_and_other_query_row_retained():
    rows = [link(camera_id=2), link(target=2), link(target=3), link(source=1, target=4)]
    result = summarize_gate({1: camera(), 2: camera(2.)}, rows, 2, visible_count=2)
    branch = result['structure_based']
    assert branch['potential_unique_2D3D'] == 2 and branch['retained_count'] == 2
    assert [v['structure_based'] for v in result['ledger']] == [
        'BOGUS_SOURCE_CAMERA', 'ACCEPTED', 'DUPLICATE_ACCEPTED_POINT3D_FOR_SOURCE_ROW', 'ACCEPTED']
    assert branch['gate_reachable'] is True
    assert result['status'] == 'PREDICTED_FROM_SAVED_SNAPSHOT' and not result['native_called']


def test_structureless_counts_no3d_and_duplicates_but_no_pose_is_rejected():
    rows = [link(point=None), link(target=2), link(target=3), link(target=4, pose=False)]
    result = summarize_gate({1: camera()}, rows, 1)
    assert result['structure_based']['retained_count'] == 1
    assert result['structure_based']['gate_reachable'] is None
    assert result['structureless']['retained_count'] == 3
    assert result['structureless']['min_inliers'] == 2
    assert result['structureless']['collection_gate_pass']
    assert result['structureless']['visible_correspondence_precheck'] == 'NOT_EVALUATED'


def test_visible_gate_precedes_target_reset_even_if_collected_gate_passes():
    result = summarize_gate({1: camera()}, [link()], 1, visible_count=0)
    assert result['structure_based']['collection_gate_pass']
    assert result['structure_based']['gate_reachable'] is False


@pytest.mark.parametrize('kwargs', [dict(width=True), dict(height=0), dict(params=[1, 2, 3, float('nan')]),
    dict(params=[True, 2, 3, 0]), dict(minratio=True), dict(maxratio=float('inf')), dict(maxextra=-1)])
def test_invalid_camera_input_distinct_from_bogus(kwargs):
    args=dict(width=100,height=200,params=[200,50,100,0],minratio=.1,maxratio=10.,maxextra=1.)
    args.update(kwargs)
    with pytest.raises(ValueError): bogus_camera(**args)


@pytest.mark.parametrize('field,value', [('source_row',True), ('has_pose',1), ('point3D_id',True),
                                       ('camera_id',False), ('target_point_row',float('nan'))])
def test_invalid_link_inputs_refused(field, value):
    row=link();row[field]=value
    with pytest.raises(ValueError): summarize_gate({1:camera()},[row],1)


@pytest.mark.parametrize('count', [True, 0, float('nan')])
def test_bad_minimum_rejected(count):
    with pytest.raises(ValueError): summarize_gate({1:camera()},[],count)
def test_native_zero_point_id_is_not_missing():
    from integrations.demo_pipeline.registration_gate import summarize_gate
    result=summarize_gate({1:{'bogus':False}},[dict(source_row=0,target_image_id=2,target_point_row=0,point3D_id=0,camera_id=1,has_pose=True)],1,1)
    assert result['structure_based']['retained_count']==1
    assert result['structure_based']['gate_reachable'] is True
