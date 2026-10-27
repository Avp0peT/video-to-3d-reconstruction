import copy
import json

import numpy as np
import pytest

from integrations.demo_pipeline.coverage_triangulation import triangulate_candidate


def observations(centers=(-.6, 0., .6), noise=0.):
    xyz = np.array([.1, .2, 5.])
    rows = []
    for i, center in enumerate(centers):
        t = np.array([-center, 0., 0.]); local = xyz + t
        pixels = 800. * local[:2] / local[2] + [512., 384.]
        if i == 1: pixels[1] += noise
        rows.append(dict(image_id=i+1, feature_row=i, foreground=True,
            R=np.eye(3).tolist(), t=t.tolist(), params=[800., 512., 384., 0.],
            raw_pixels=pixels.tolist()))
    return rows


def check(rows):
    before = copy.deepcopy(rows)
    result = triangulate_candidate(rows)
    assert rows == before
    json.dumps(result, allow_nan=False)
    return result


def test_true_three_dimensional_point():
    result = check(observations())
    assert result['tier'] == 'STRICT_SUPPORTED'
    assert np.allclose(result['xyz'], [.1, .2, 5.])
    assert len(result['strict_decision']['loo']) == 3


def test_wrong_match_keeps_all_observations():
    result = check(observations(noise=70.))
    assert result['tier'] == 'REJECTED' and result['xyz'] is None
    assert result['reasons']
    assert len(result['relaxed_decision']['reprojection']) == 3


def test_baseline_too_small():
    result = check(observations(centers=(-.001, 0., .001)))
    assert result['tier'] == 'REJECTED'
    assert 'INSUFFICIENT_TRIANGULATION_ANGLE' in result['reasons']


def test_two_views_are_provisional():
    result = check(observations(centers=(-.6, .6)))
    assert result['tier'] == 'TWO_VIEW_PROVISIONAL'
    assert result['strict_decision'] is None and result['relaxed_decision'] is None
    assert result['metrics']['loo_evaluated'] is False


@pytest.mark.parametrize('change,reason', [
    ('foreground', 'NON_FOREGROUND_OBSERVATION'),
    ('identity', 'DUPLICATE_IMAGE_ID'),
    ('rotation', 'INVALID_ROTATION'),
    ('intrinsics', 'UNSUPPORTED_INTRINSICS'),
])
def test_invalid_observation(change, reason):
    rows = observations()
    if change == 'foreground': rows[0]['foreground'] = False
    if change == 'identity': rows[1]['image_id'] = rows[0]['image_id']
    if change == 'rotation': rows[0]['R'][0][0] = -1.
    if change == 'intrinsics': rows[0]['params'][3] = .01
    assert reason in check(rows)['reasons']


def test_relaxed_support_is_distinct():
    result = check(observations(centers=(-.07, 0., .07)))
    assert result['tier'] == 'RELAXED_SUPPORTED'
    assert result['strict_decision']['status'] == 'REJECTED'
    assert result['relaxed_decision']['status'] == 'CONDITIONAL_POINT_SUPPORTED'


def test_nonfinite_input_returns_finite_json():
    rows = observations(); rows[0]['raw_pixels'][0] = float('nan')
    result = triangulate_candidate(rows)
    assert result['tier'] == 'REJECTED' and result['reasons']
    json.dumps(result, allow_nan=False)


def test_two_view_wrong_match_rejected_without_loo():
    rows = observations(centers=(-.6, .6), noise=30.)
    result = check(rows)
    assert result['tier'] == 'REJECTED'
    assert 'REPROJECTION_ERROR_EXCEEDED' in result['reasons']
    assert result['metrics']['loo_evaluated'] is False


def test_negative_depth_rejected():
    rows = observations()
    for row in rows:
        row['raw_pixels'][0] = 1024. - row['raw_pixels'][0]
        row['raw_pixels'][1] = 768. - row['raw_pixels'][1]
    result = check(rows)
    assert result['tier'] == 'REJECTED'
    assert 'NON_POSITIVE_DEPTH' in result['reasons']


@pytest.mark.parametrize('count', [0, 1, 33])
def test_bounded_observation_count(count):
    result = triangulate_candidate(observations()[:1] * count)
    assert result['tier'] == 'REJECTED'
    assert 'OBSERVATION_COUNT_OUT_OF_BOUNDS' in result['reasons']
    json.dumps(result, allow_nan=False)
