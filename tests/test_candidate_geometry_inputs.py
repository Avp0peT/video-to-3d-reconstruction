"""Synthetic no-I/O tests for camera/segment anchor input preparation."""
from copy import deepcopy
import pytest

from kozlov_lab.stereo.candidate_geometry_inputs import build_anchor_groups
from kozlov_lab.stereo.candidate_point_consumer import PROVISIONAL


def inputs():
    observations = [dict(frame_index=i, identity_segment=0 if i < 3 else 1,
        xy=[10.5+i, 20.5], label='unknown' if i == 1 else 'foreground',
        target_verified=i != 1, strong=True, fb_error_px=.1, lk_error=2.) for i in range(4)]
    tracking = dict(tracks=[dict(track_id=7, observations=observations)])
    ledger = [dict(input_index=78,frame_id='f78',status=PROVISIONAL,
        provisional_candidate_id='c1',scope='EXISTING_MEASURED_POINTS_ONLY',
        original_status='NEEDS_SELECTION',full_mask_accepted=False,
        new_point_seeding_allowed=False,publishable_for_geometry=False,
        supported_observations=[dict(track_id=7,query_point_xy=[11.,20.])])]
    frames = {77+i:dict(input_index=77+i,frame_id=f'f{77+i}',
        original_mask_status='NEEDS_SELECTION' if i == 1 else 'OK',
        mask_binding=None if i == 1 else dict(path='mask',sha256='hash'),
        measured_point_mask_checks=[] if i == 1 else [dict(track_id=7,frame_index=i,
            source_xy=deepcopy(observations[i]['xy']),inside_original_mask=True)]) for i in range(4)}
    camera = dict(image_id=1,camera_model='SIMPLE_RADIAL',params=[100.,50.,50.,0.],
                  R=[[1,0,0],[0,1,0],[0,0,1]],t=[0,0,0])
    models = [dict(model_key='a',binding=dict(path='saved'),
                   cameras={77:deepcopy(camera),79:dict(camera,image_id=3)}),
              dict(model_key='other-segment-only',binding={},cameras={80:camera})]
    return tracking, ledger, frames, models, 77


def test_segments_queries_and_camera_models_remain_separate():
    args=inputs(); before=deepcopy(args)
    result=build_anchor_groups(*args)
    assert args == before
    assert result['unique_query_count']==1 and len(result['groups'])==1
    group=result['groups'][0]
    assert group['model_key']=='a' and len(group['anchors'])==2
    assert group['ready_for_numeric_evaluation'] is True
    assert group['query_numeric_evaluable'] is False
    assert group['queries'][0]['status']=='NO_SAVED_CAMERA'
    assert len(group['assessments'])==3
    assert group['excluded_other_segment_frames']==[80]
    assert group['assessments'][1]['reasons']==[
        'NO_SAVED_CAMERA','UNKNOWN_SOURCE','ORIGINAL_MASK_PIXEL_NOT_CHECKED']
    assert result['numerical_geometry_evaluated'] is False
    assert result['geometry_accepted'] is False
    group['anchors'][0]['raw_pixels'][0]=99
    assert args == before


def test_no_camera_group_is_explicit():
    args=list(inputs()); args[3]=[]
    group=build_anchor_groups(*args)['groups'][0]
    assert group['model_key'] is None and group['status']=='NO_MODEL_WITH_ANCHORS'
    assert len(group['queries'])==1 and len(group['assessments'])==3
    assert not group['anchors'] and not group['ready_for_numeric_evaluation']


def test_multiple_models_do_not_duplicate_global_query_count():
    args=list(inputs()); args[3].append(dict(args[3][0],model_key='b'))
    result=build_anchor_groups(*args)
    assert len(result['groups'])==2 and result['unique_query_count']==1
    assert all(len(g['queries'])==1 for g in result['groups'])


@pytest.mark.parametrize('change,reason', [('weak','NON_STRONG'),
    ('fb','INVALID_FB_ERROR_PX'),('unknown','UNKNOWN_SOURCE'),
    ('outside','OUTSIDE_ORIGINAL_MASK'),('unchecked','ORIGINAL_MASK_PIXEL_NOT_CHECKED'),
    ('binding','NO_ORIGINAL_MASK_BINDING')])
def test_anchor_exclusions_are_retained(change,reason):
    args=list(inputs()); obs=args[0]['tracks'][0]['observations'][0]; frame=args[2][77]
    if change=='weak':obs['strong']=False
    if change=='fb':obs['fb_error_px']=float('nan')
    if change=='unknown':frame['original_mask_status']='NO_MATCH'
    if change=='outside':frame['measured_point_mask_checks'][0]['inside_original_mask']=False
    if change=='unchecked':frame['measured_point_mask_checks']=[]
    if change=='binding':frame['mask_binding']=None
    g=build_anchor_groups(*args)['groups'][0]
    assert reason in g['assessments'][0]['reasons']
    assert len(g['anchors'])==1 and not g['ready_for_numeric_evaluation']


@pytest.mark.parametrize('change', ['queryframe','missingframe','duplicatecheck',
    'checkcoord','boolcheck','duplicate_model','bad_camera','bad_decision','reorder'])
def test_invalid_identity_and_evidence_fail_closed(change):
    args=list(inputs()); frames=args[2]
    if change=='queryframe': frames[78]['frame_id']='wrong'
    if change=='missingframe': frames.pop(77)
    if change=='duplicatecheck':frames[77]['measured_point_mask_checks']*=2
    if change=='checkcoord':frames[77]['measured_point_mask_checks'][0]['source_xy'][0]+=1
    if change=='boolcheck':frames[77]['measured_point_mask_checks'][0]['inside_original_mask']=1
    if change=='duplicate_model':args[3].append(deepcopy(args[3][0]))
    if change=='bad_camera':args[3][0]['cameras'][77].pop('R')
    if change=='bad_decision':args[1][0]['publishable_for_geometry']=True
    if change=='reorder':args[0]['tracks'][0]['observations'].reverse()
    with pytest.raises(ValueError):build_anchor_groups(*args)


@pytest.mark.parametrize('key', [0, 1, True, None, '', [], {}])
def test_model_key_must_be_nonempty_string(key):
    args=list(inputs()); args[3][0]['model_key']=key
    with pytest.raises(ValueError):build_anchor_groups(*args)


def test_distinct_registry_indices_cannot_share_frame_identity():
    args=list(inputs())
    args[2][80]['frame_id']=args[2][77]['frame_id']
    with pytest.raises(ValueError,match='duplicate registry frame_id'):
        build_anchor_groups(*args)
