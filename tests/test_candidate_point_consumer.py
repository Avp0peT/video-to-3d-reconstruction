"""Synthetic CPU-only checks for the observation annotation boundary."""
from copy import deepcopy

import numpy as np
import pytest

from kozlov_lab.stereo.candidate_point_consumer import consume_point_decisions, PROVISIONAL


def inputs():
    obs = dict(frame_index=1, identity_segment=2, xy=[20.123456789, 30.5],
               label='unknown', target_verified=False, strong=True,
               fb_error_px=.2, lk_error=3, mask_status='unknown')
    tracking = dict(tracks=[dict(track_id=4, observations=[
        dict(obs, frame_index=0, label='foreground', target_verified=True,
             identity_segment=0), obs,
        dict(obs, frame_index=3, label='NO_MATCH', mask_status='NO_MATCH')])])
    decision = dict(input_index=78, frame_id='frame78', status=PROVISIONAL,
        provisional_candidate_id='candidate2', scope='EXISTING_MEASURED_POINTS_ONLY',
        full_mask_accepted=False, new_point_seeding_allowed=False,
        publishable_for_geometry=False, original_status='NEEDS_SELECTION',
        supported_observations=[dict(track_id=4,
            query_point_xy=np.asarray(np.asarray(obs['xy'])-.5, dtype=np.float32).tolist())])
    return tracking, [decision]


def test_exact_annotation_preserves_source_and_segments():
    tracking, decisions = inputs()
    before = deepcopy(tracking)
    result = consume_point_decisions(tracking, decisions, 77)
    assert tracking == before
    assert result['status'] == 'CANDIDATE_POINT_ANNOTATIONS_ONLY'
    assert result['counts'] == dict(observations=3, prior_labeled=2, prior_unknown=1, provisional=1)
    assert result['publishable_for_geometry'] is False
    assert [r['source_identity_segment'] for r in result['rows']] == [0, 2, 2]
    assert [r['source_declared_geometry_eligible'] for r in result['rows']] == [None]*3
    assert [r['source_original_foreground'] for r in result['rows']] == [True, False, False]
    assert result['original_geometry_filter_evaluated'] is False
    assert all(r['derived_geometry_eligible'] is False for r in result['rows'])
    assert all(r['geometry_eligible'] is False for r in result['rows'])
    row = result['rows'][1]
    assert row['source_label'] == 'unknown' and row['source_target_verified'] is False
    assert row['source_xy'] == before['tracks'][0]['observations'][1]['xy']
    assert row['derived_point_support'] == PROVISIONAL
    assert result['rows'][2]['source_label'] == 'NO_MATCH'
    assert result['rows'][2]['derived_point_support'] == 'UNCHANGED'
    row['source_xy'][0] = 99
    assert tracking == before


@pytest.mark.parametrize('change', ['coord', 'nan', 'boolcoord', 'frame', 'gap',
    'boolindex', 'booltrack', 'unknowntrack', 'strong', 'fb', 'lk', 'none',
    'negative', 'status', 'scope', 'original', 'flag', 'duplicate', 'duplicatepoint',
    'duplicatetrack', 'duplicateobs', 'boolsegment', 'empty', 'nomatch'])
def test_invalid_requests_fail_closed(change):
    tracking, decisions = inputs()
    d = decisions[0]
    obs = tracking['tracks'][0]['observations'][1]
    if change == 'coord': d['supported_observations'][0]['query_point_xy'][0] += .01
    if change == 'nan': d['supported_observations'][0]['query_point_xy'][0] = float('nan')
    if change == 'boolcoord': d['supported_observations'][0]['query_point_xy'][0] = True
    if change == 'frame': d['input_index'] = 76
    if change == 'gap': d['input_index'] = 79
    if change == 'boolindex': d['input_index'] = True
    if change == 'booltrack': d['supported_observations'][0]['track_id'] = True
    if change == 'unknowntrack': d['supported_observations'][0]['track_id'] = 99
    if change == 'strong': obs['strong'] = False
    if change == 'fb': obs['fb_error_px'] = 1.01
    if change == 'lk': obs['lk_error'] = 25.01
    if change == 'none': obs['fb_error_px'] = None
    if change == 'negative': obs['lk_error'] = -1
    if change == 'status': d['status'] = 'ACCEPTED'
    if change == 'scope': d['scope'] = 'WHOLE_MASK'
    if change == 'original': d['original_status'] = 'NO_MATCH'
    if change == 'flag': d['publishable_for_geometry'] = 0
    if change == 'duplicate': decisions.append(deepcopy(d))
    if change == 'duplicatepoint': d['supported_observations'] *= 2
    if change == 'duplicatetrack': tracking['tracks'] *= 2
    if change == 'duplicateobs': tracking['tracks'][0]['observations'].append(deepcopy(obs))
    if change == 'boolsegment': obs['identity_segment'] = True
    if change == 'empty': d['supported_observations'] = []
    if change == 'nomatch': obs['mask_status'] = 'NO_MATCH'
    before = deepcopy(tracking)
    with pytest.raises(ValueError): consume_point_decisions(tracking, decisions, 77)
    assert tracking == before


def test_original_states_ignored_and_clip_ids_stay_local():
    tracking, decisions = inputs()
    ignored = [dict(status=s, input_index=77+i, frame_id=str(i),
        provisional_candidate_id=None, supported_observations=[])
        for i,s in enumerate(('OK', 'NO_MATCH', 'NEEDS_SELECTION'))]
    result = consume_point_decisions(tracking, ignored, 77)
    assert all(r['derived_point_support'] == 'UNCHANGED' for r in result['rows'])
    first = consume_point_decisions(tracking, decisions, 77)
    second = consume_point_decisions(tracking, [], 49)
    assert first['counts']['provisional'] == 1 and second['counts']['provisional'] == 0


def test_distinct_frame_ids_cannot_repeat_source_index():
    tracking, decisions = inputs()
    duplicate = deepcopy(decisions[0]); duplicate['frame_id'] = 'other'
    with pytest.raises(ValueError): consume_point_decisions(tracking, decisions+[duplicate], 77)


@pytest.mark.parametrize('status', ['ORIGINAL_OK_UNCHANGED', 'NO_CANDIDATE',
    'INSUFFICIENT_EVIDENCE', 'AMBIGUOUS', 'CONFLICT', 'REFERENCE_CONFLICT_HOLD'])
def test_policy_nonprovisional_statuses(status):
    tracking, _ = inputs()
    decision = dict(status=status, input_index=78, frame_id='f78',
                    provisional_candidate_id=None, supported_observations=[])
    assert consume_point_decisions(tracking, [decision], 77)['counts']['provisional'] == 0
    decision['provisional_candidate_id'] = 'hidden-support'
    with pytest.raises(ValueError): consume_point_decisions(tracking, [decision], 77)


@pytest.mark.parametrize('change', ['unordered', 'segment', 'duplicateindex',
    'duplicatefid', 'boolindex', 'emptyfid', 'rights', 'support', 'invalid'])
def test_order_and_all_decision_rows_validated(change):
    tracking, decisions = inputs()
    other = dict(status='NO_CANDIDATE', input_index=79, frame_id='f79',
                 provisional_candidate_id=None, supported_observations=[])
    if change == 'unordered': tracking['tracks'][0]['observations'].reverse()
    if change == 'segment': tracking['tracks'][0]['observations'][2]['identity_segment'] = 1
    if change == 'duplicateindex': other['input_index'] = 78
    if change == 'duplicatefid': other['frame_id'] = 'frame78'
    if change == 'boolindex': other['input_index'] = True
    if change == 'emptyfid': other['frame_id'] = ''
    if change == 'rights': other['full_mask_accepted'] = True
    if change == 'support': other['supported_observations'] = [dict(track_id=4)]
    if change == 'invalid': other['status'] = 'INVALID_EVIDENCE'
    with pytest.raises(ValueError): consume_point_decisions(tracking, decisions+[other], 77)


def test_source_explicit_geometry_declaration_is_only_recorded():
    tracking, decisions = inputs()
    tracking['tracks'][0]['observations'][1]['geometry_eligible'] = True
    row = consume_point_decisions(tracking, decisions, 77)['rows'][1]
    assert row['source_declared_geometry_eligible'] is True
    assert row['derived_geometry_eligible'] is False
