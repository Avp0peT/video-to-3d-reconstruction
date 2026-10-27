from copy import deepcopy
import numpy as np
import pytest
import integrations.demo_pipeline.area_tracking as module


class NoLK:
    def calcOpticalFlowPyrLK(self,*args,**kwargs):raise AssertionError('LK must never be called')


def fixture(targets=2,center=(100.5,100.5),size=256):
    domains={};decoded={}
    for i in range(targets+1):
        domains[str(i)]=dict(width=size,height=size,path=f'{i}.png',original_mask_status='OK' if i==0 else 'UNKNOWN_NOT_SEGMENTED',
            mask_rle=[[[0,size]] for _ in range(size)] if i==0 else None)
        decoded[i]=dict(width=size,height=size,path=f'{i}.png',gray=(np.indices((size,size)).sum(0)+i*31).astype(np.uint8))
    root=dict(region_id='root',depth=0,tracklet_id='track:1:segment:0:source:27:edge:0',
        source_input_index=27,source_decode_index=0,decode_path=list(range(targets+1)),center_raw_xy=list(center))
    return dict(regions=[root],frame_domains=domains),decoded


def alignment(matrix,status='ACCEPTED',reason=None):
    return dict(status=status,reasons=[reason] if reason else [],matrix_cv2x3=matrix,
                native_calls=dict(matchTemplate=1,findTransformECCWithMask=1),
                train=dict(count=700),holdout=dict(count=300),pixel_evidence={})


def test_nontranslation_crop_transform_has_exact_half_pixel_conjugation():
    matrix=module._global_raw([[1.1,-.1,10.25],[.2,.9,11.75]],
        dict(left=7,top=9),dict(left=31,top=43))
    # Raw (8.5,10.5) is global CV (8,10), hence template CV (1,1).
    # Crop-map output (11.25,12.85) is global raw (42.75,56.35).
    actual=module._map(matrix,np.array([[8.5,10.5]]))
    np.testing.assert_allclose(actual,[[42.75,56.35]],atol=1e-12)


def test_fixed_original_template_and_independent_reverse_global_raw_conversion(monkeypatch):
    plan,decoded=fixture();calls=[]
    def align(template,search,tm,sm,cv,*,budget):
        budget();calls.append((template.copy(),search.copy(),tm.copy(),sm.copy()))
        return alignment([[1.,0.,33.],[0.,1.,32.]]) if len(calls)%2 else alignment([[1.,0.,32.],[0.,1.,32.]])
    monkeypatch.setattr(module,'align_area',align)
    result=module.track_areas(plan,decoded,NoLK());area=result['regions'][0]
    assert result['status']=='AREA_TRACKING_COMPLETE' and area['status']=='REGION_TRACKLET_2D'
    assert area['verified_targets']==2 and len(calls)==4 and result['lk_calls']==0
    assert result['native_calls']==dict(matchTemplate=4,findTransformECCWithMask=4)
    np.testing.assert_array_equal(calls[0][0],calls[2][0])
    assert not np.array_equal(calls[1][0],calls[0][0])
    for i,leg in enumerate(area['region_legs'],1):
        np.testing.assert_allclose(leg['forward']['matrix_global_raw3x3'],[[1.,0.,i],[0.,1.,0.],[0.,0.,1.]])
        np.testing.assert_allclose(leg['backward']['matrix_global_raw3x3'],[[1.,0.,-i],[0.,1.,0.],[0.,0.,1.]])
        assert leg['cycle_error_px']<1e-10 and len(leg['cycle_points'])==5
        assert leg['forward']['search_mask_status']=='UNKNOWN_NOT_SEGMENTED'
        assert leg['forward']['unknown_search_sampling_mask'] is True
        assert np.all(calls[(i-1)*2][3]==255)
    assert area['source_polygon_raw4x2'][0]==[84.5,84.5]
    assert len(area['accepted_polygons'])==2 and result['independent_point_matches']==0
    assert result['geometry_accepted'] is False and result['new_model_points']==0


def test_no_point_steps_or_affine_motion_are_read(monkeypatch):
    plan,decoded=fixture()
    class Poison:
        def __iter__(self):raise AssertionError('old point measurements read')
        def __getitem__(self,key):raise AssertionError('old point measurements read')
    plan['regions'][0].update(points=Poison(),affine_diagnostics=Poison(),motion=Poison())
    monkeypatch.setattr(module,'align_area',lambda *args,**kwargs:alignment([[1.,0.,32.],[0.,1.,32.]]))
    assert module.track_areas(plan,decoded,NoLK())['regions'][0]['verified_targets']==2


def test_five_point_cycle_rejects_disagreeing_independent_reverse(monkeypatch):
    plan,decoded=fixture();calls=[]
    def align(*args,**kwargs):
        calls.append(1)
        return alignment([[1.,0.,32.],[0.,1.,32.]]) if len(calls)%2 else alignment([[1.,0.,35.],[0.,1.,32.]])
    monkeypatch.setattr(module,'align_area',align)
    result=module.track_areas(plan,decoded,NoLK());area=result['regions'][0];leg=area['region_legs'][0]
    assert len(calls)==2 and area['verified_targets']==0 and area['status']=='INCOMPLETE'
    assert leg['status']=='REJECTED' and leg['cycle_error_px']==3.
    assert area['stop_reason']=='AREA_FIVE_POINT_CYCLE_THRESHOLD'


def test_later_failure_preserves_accepted_fixed_source_prefix(monkeypatch):
    plan,decoded=fixture(3);calls=[]
    def align(*args,**kwargs):
        calls.append(1)
        return alignment([[1.,0.,32.],[0.,1.,32.]],'REJECTED','actual reverse failure') if len(calls)==6 else alignment([[1.,0.,32.],[0.,1.,32.]])
    monkeypatch.setattr(module,'align_area',align)
    area=module.track_areas(plan,decoded,NoLK())['regions'][0]
    assert area['verified_targets']==2 and len(area['accepted_polygons'])==2
    assert area['status']=='REGION_TRACKLET_2D' and area['stop_reason']=='BACKWARD_REJECTED'
    assert area['region_legs'][-1]['backward']['alignment']['reasons']==['actual reverse failure']


def test_known_target_corner_requires_all_four_bilinear_foreground_pixels(monkeypatch):
    plan,decoded=fixture();domain=plan['frame_domains']['1']
    domain.update(original_mask_status='OK',mask_rle=[[[0,256]] for _ in range(256)])
    domain['mask_rle'][84]=[[0,84],[85,256]]
    calls=[]
    def align(*args,**kwargs):calls.append(1);return alignment([[1.,0.,32.],[0.,1.,32.]])
    monkeypatch.setattr(module,'align_area',align)
    area=module.track_areas(plan,decoded,NoLK())['regions'][0]
    assert calls==[1] and area['stop_reason']=='FORWARD_POLYGON_OUTSIDE_ORIGINAL_FOREGROUND'
    assert area['region_legs'][0]['forward']['search_mask_status']=='OK'


def test_source_coverage_gate_and_real_boundary_crop(monkeypatch):
    plan,decoded=fixture(center=(20.5,20.5),size=80);search_shapes=[]
    def align(template,search,*args,**kwargs):
        search_shapes.append(search.shape);return alignment([[1.,0.,4.],[0.,1.,4.]])
    monkeypatch.setattr(module,'align_area',align)
    area=module.track_areas(plan,decoded,NoLK())['regions'][0]
    assert area['verified_targets']==2 and all(shape==(69,69) for shape in search_shapes)
    plan,decoded=fixture()
    plan['frame_domains']['0']['mask_rle']=[[] for _ in range(256)]
    area=module.track_areas(plan,decoded,NoLK())['regions'][0]
    assert area['source_mask_coverage']==0 and area['region_legs']==[]
    assert area['stop_reason']=='SOURCE_TEMPLATE_FOREGROUND_COVERAGE_BELOW_0_75'


def test_native_or_budget_exceptions_propagate_with_actual_partial_result(monkeypatch):
    plan,decoded=fixture()
    def align(*args,**kwargs):raise ValueError('shared deadline')
    monkeypatch.setattr(module,'align_area',align)
    with pytest.raises(ValueError,match='shared deadline') as error:module.track_areas(plan,decoded,NoLK())
    partial=error.value.partial_result
    assert partial['status']=='INTERRUPTED' and partial['native_calls_complete'] is False
    assert partial['counts']['targets_interrupted']==1 and partial['regions'][0]['verified_targets']==0


def test_empty_and_hard_root_or_path_bounds():
    assert module.track_areas(dict(regions=[]),{},NoLK())['counts']['targets_attempted']==0
    plan,decoded=fixture();plan['regions']*=9
    with pytest.raises(ValueError,match='ROOT_AREA_BOUND'):module.track_areas(plan,decoded,NoLK())
    plan,decoded=fixture();plan['regions'][0]['depth']=1
    with pytest.raises(ValueError,match='ONLY_ROOT'):module.track_areas(plan,decoded,NoLK())


def test_moving_search_follows_accepted_center_with_fixed_source_and_absolute_transforms(monkeypatch):
    plan,decoded=fixture(3);calls=[]
    def align(template,search,tm,sm,cv,*,budget):
        calls.append((template.copy(),search.copy()))
        # Each forward search starts at the preceding accepted position.
        # Reverse registration independently returns to the original source.
        return alignment([[1.,0.,52.],[0.,1.,32.]]) if len(calls)%2 else alignment([[1.,0.,32.],[0.,1.,32.]])
    monkeypatch.setattr(module,'align_area',align)
    area=module.track_areas(plan,decoded,NoLK())['regions'][0]
    assert area['verified_targets']==3 and len(calls)==6
    assert [leg['forward']['search_roi']['left'] for leg in area['region_legs']]==[52,72,92]
    for i,leg in enumerate(area['region_legs'],1):
        np.testing.assert_array_equal(calls[(i-1)*2][0],decoded[0]['gray'][84:117,84:117])
        np.testing.assert_allclose(leg['forward']['matrix_global_raw3x3'],[[1.,0.,20*i],[0.,1.,0.],[0.,0.,1.]])
        np.testing.assert_allclose(leg['backward']['matrix_global_raw3x3'],[[1.,0.,-20*i],[0.,1.,0.],[0.,0.,1.]])
        np.testing.assert_allclose(np.mean(leg['forward']['polygon_raw4x2'],axis=0),[100.5+20*i,100.5])
        assert leg['cycle_error_px']<1e-10


def test_rotated_area_moves_next_search_using_mapped_source_center_and_independent_reverse(monkeypatch):
    plan,decoded=fixture();calls=[]
    # A 90 degree rotation plus translation maps source center to (120.5,100.5).
    matrices=[[[0.,-1.,84.],[1.,0.,32.]],[[0.,1.,32.],[-1.,0.,64.]],
              [[0.,-1.,64.],[1.,0.,32.]],[[0.,1.,32.],[-1.,0.,64.]]]
    def align(template,search,tm,sm,cv,*,budget):
        calls.append((template.copy(),search.copy()))
        return alignment(matrices[len(calls)-1])
    monkeypatch.setattr(module,'align_area',align)
    area=module.track_areas(plan,decoded,NoLK())['regions'][0]
    assert area['verified_targets']==2 and len(calls)==4
    assert [leg['forward']['search_roi']['left'] for leg in area['region_legs']]==[52,72]
    for i,leg in enumerate(area['region_legs'],1):
        np.testing.assert_allclose(leg['forward']['matrix_global_raw3x3'],[[0.,-1.,221.],[1.,0.,0.],[0.,0.,1.]])
        np.testing.assert_allclose(leg['backward']['matrix_global_raw3x3'],[[0.,1.,0.],[-1.,0.,221.],[0.,0.,1.]])
        np.testing.assert_allclose(np.mean(leg['forward']['polygon_raw4x2'],axis=0),[120.5,100.5])
        np.testing.assert_array_equal(calls[(i-1)*2][0],decoded[0]['gray'][84:117,84:117])
        np.testing.assert_array_equal(calls[(i-1)*2+1][0],decoded[i]['gray'][84:117,104:137])
        np.testing.assert_array_equal(calls[(i-1)*2+1][1],decoded[0]['gray'][52:149,52:149])
        assert leg['cycle_error_px']<1e-10 and len(leg['cycle_points'])==5


@pytest.mark.parametrize('failed_stage',['forward','backward'])
def test_failed_registration_stops_route_without_later_leg_or_accepted_position(monkeypatch,failed_stage):
    plan,decoded=fixture(3);calls=[]
    fail_call=3 if failed_stage=='forward' else 4
    def align(*args,**kwargs):
        calls.append(1)
        if len(calls)==fail_call:return alignment([[1.,0.,52.],[0.,1.,32.]],'REJECTED','moving area failure')
        return alignment([[1.,0.,52.],[0.,1.,32.]]) if len(calls)%2 else alignment([[1.,0.,32.],[0.,1.,32.]])
    monkeypatch.setattr(module,'align_area',align)
    area=module.track_areas(plan,decoded,NoLK())['regions'][0]
    assert len(calls)==fail_call and area['verified_targets']==1
    assert [leg['target_decode_index'] for leg in area['region_legs']]==[1,2]
    assert [position['decode_index'] for position in area['accepted_polygons']]==[1]
    np.testing.assert_allclose(np.mean(area['accepted_polygons'][0]['polygon_raw4x2'],axis=0),[120.5,100.5])
    assert area['region_legs'][-1]['status']=='REJECTED'
    assert area['stop_reason']==failed_stage.upper()+'_REJECTED'
    if failed_stage=='forward':assert area['region_legs'][-1]['backward'] is None
