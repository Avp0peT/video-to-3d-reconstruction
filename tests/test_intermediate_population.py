from copy import deepcopy
import numpy as np
import pytest
from integrations.demo_pipeline.intermediate_population import build_population_plan, summarize_edge, _verified_leg, _cell


def fixture(count=2):
    frames = [dict(input_index=i, decode_index=d, frame_id=str(i), width=24, height=16,
                   original_mask_status='OK') for i, d in [(27, 9), (53, 7)]]
    mask = np.zeros((16, 24), np.uint8); mask[0, 0] = 255; mask[15, 23] = 255
    tracks = [dict(track_id=i, observations=[dict(frame_index=0, identity_segment=0,
        xy=[.5,.5] if i % 2 == 0 else [23.5,15.5], label='foreground', target_verified=True,
        tracking_status='seed' if i == 0 else 'flow_tracked', strong=i != 0, observed=True)]) for i in range(count)]
    edge = dict(model_key='m1', source_input_index=27, target_input_index=53,
                source_decode_index=9, target_decode_index=7, decode_path=[9,8,7], points=[])
    return frames, [mask,mask.copy()], dict(tracks=tracks), dict(edges=[edge,dict(edge,model_key='m2')])


def test_local_positions_reverse_intervals_model_dedup_seed_rights():
    plan = build_population_plan(*fixture()); edge = plan['edges'][0]
    assert len(plan['edges']) == 1 and edge['model_keys'] == ['m1','m2']
    assert edge['decode_path'] == [9,8,7]
    seed = next(p for p in edge['points'] if p['track_id'] == 0)
    assert seed['point_id'] == 'track:0:segment:0:source:27'
    assert seed['birth_kind'] == 'NEW_SEED' and seed['original_strong'] is False
    assert seed['original_tracking']['frame_index'] == 0 and seed['is_3d'] is False


def leg(error=0):
    record = dict(status=[[1]], error=[[0]], observed_raw_xy=[.5,.5])
    return dict(forward=deepcopy(record), backward=deepcopy(record), fb_error_px=error)


def test_sparse_mask_denominator_actual_endpoint_and_fb_failure_histogram():
    edge = build_population_plan(*fixture())['edges'][0]; rows=[]
    for i,p in enumerate(edge['points']):
        passed = i == 0
        rows.append(dict(group_id=p['point_id'],source_raw_xy=p['source_raw_xy'], outcome='RECOVERED' if passed else 'BOTH_FAIL',
            direct=dict(status='REJECTED'), dense=dict(status='PASSED' if passed else 'REJECTED',
            steps=[leg(),leg() if passed else leg(1.1)], observed_raw_coordinates=[dict(decode_index=7,raw_xy=[23.5,15.5])])) )
    summary=summarize_edge(edge,dict(rows=rows),{9:dict(time_seconds=9.),8:dict(time_seconds=8.),7:dict(time_seconds=7.)})
    assert [r['survived'] for r in summary['dense_per_leg_survival']] == [2,1]
    assert [r['count'] for r in summary['survival_histogram']] == [0,1,1]
    assert summary['coverage']['source_all']['total'] == 2
    assert summary['coverage']['target_dense_passed']['occupied_cells'] == [95]
    assert summary['durations'][1]['duration_seconds'] == 1
    assert summary['geometry_accepted'] is False
    assert edge['source_domain']['rows'] == 12 and edge['source_domain']['columns'] == 8
    assert _cell([12.5,8.5],[0,0,24,16]) == 52


def test_budget_is_frozen_without_refill():
    plan=build_population_plan(*fixture(771)); edge=plan['edges'][0]
    assert edge['source_candidate_count'] == 771 and len(edge['points']) == 768
    assert len(edge['batches']) == 12 and all(len(b['points'])==64 for b in edge['batches'])
    assert len(edge['omitted_points'])==3
    assert set(p['point_id'] for p in edge['points']).isdisjoint(p['point_id'] for p in edge['omitted_points'])
    assert build_population_plan(*fixture(771)) == plan


def test_no_interval_does_not_require_video_or_tracking_metadata():
    plan=build_population_plan([dict(input_index=17)], [None], dict(tracks=[]), dict(edges=[]))
    assert plan['edges'] == [] and plan['decode_indices'] == []
    assert plan['new_model_points'] == 0 and plan['geometry_accepted'] is False


def test_duplicate_conflict_and_explicit_skip_reasons():
    args=fixture(); obs=args[2]['tracks'][0]['observations'][0]
    args[2]['tracks'][0]['observations'].append(dict(obs,strong=True))
    with pytest.raises(ValueError,match='CONTRADICTORY'): build_population_plan(*args)
    args=fixture(3)
    args[2]['tracks'][0]['observations'][0]['label']='background'
    args[2]['tracks'][1]['observations'][0]['target_verified']=False
    args[2]['tracks'][2]['observations'][0]['xy']=[1.5,1.5]
    plan=build_population_plan(*args)
    assert {r['reason'] for r in plan['skipped_observations']} == {'NON_FOREGROUND','TARGET_UNVERIFIED','OUTSIDE_ORIGINAL_MASK'}


@pytest.mark.parametrize('part,value', [('fb',True), ('fb',-1), ('status',[[True]]), ('error',[[-.1]]), ('error',[[True]])])
def test_unaccepted_numeric_records_do_not_survive(part,value):
    step=leg()
    if part == 'fb': step['fb_error_px']=value
    else: step['backward'][part]=value
    assert _verified_leg(step) is False


def test_unknown_tracking_status_and_unselected_frame_are_not_cohorts():
    args=fixture()
    args[2]['tracks'][1]['observations'][0]['tracking_status']='invented'
    args[2]['tracks'][0]['observations'].append(dict(frame_index=1,identity_segment=0,xy=[float('nan'),0]))
    plan=build_population_plan(*args)
    assert len(plan['edges'][0]['points']) == 1
    assert plan['skipped_observations'][0]['reason'] == 'UNSUPPORTED_TRACKING_STATUS'


def test_summary_refuses_partial_or_contradictory_results():
    edge=build_population_plan(*fixture(1))['edges'][0]
    decoded={i:dict(time_seconds=float(i)) for i in [9,8,7]}
    with pytest.raises(ValueError,match='COMPARISON_COHORT'):
        summarize_edge(edge,dict(rows=[]),decoded)
    row=dict(group_id=edge['points'][0]['point_id'],source_raw_xy=[.5,.5],
             direct=dict(status='REJECTED'),dense=dict(status='REJECTED',steps=[]),outcome='RECOVERED')
    with pytest.raises(ValueError,match='OUTCOME_MISMATCH'):
        summarize_edge(edge,dict(rows=[row]),decoded)
