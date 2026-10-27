import copy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.partial_point_gate import assess_point


def scene():
    xyz = np.array([0.2, 0.1, 5.0])
    observations = []
    for i, x in enumerate([-1.0, 0.0, 1.0, 2.0]):
        t = np.array([-x, 0.0, 0.0])
        q = xyz + t
        observations.append(dict(image_id=i + 1, feature_row=7, raw_pixels=(800 * q[:2] / q[2] + [320, 240]).tolist(), params=[800, 320, 240, 0], R=np.eye(3).tolist(), t=t.tolist(), foreground=True))
    return xyz.tolist(), observations


def test_supported_and_immutable():
    xyz, obs = scene()
    before = copy.deepcopy((xyz, obs))
    out = assess_point(xyz, obs)
    assert out['status'] == 'CONDITIONAL_POINT_SUPPORTED'
    assert len(out['loo']) == 4
    assert max(x['error_px'] for x in out['loo']) < 1e-8
    assert out['independent_quality_truth'] is False
    assert (xyz, obs) == before
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize('case', ['two', 'duplicate', 'bool_id', 'bad_r', 'reflection', 'nan', 'negative', 'foreground', 'wrong', 'k', 'f', 'too_many', 'zero_baseline'])
def test_rejections(case):
    xyz, obs = scene()
    if case == 'two': obs = obs[:2]
    elif case == 'duplicate': obs[1]['image_id'] = obs[0]['image_id']
    elif case == 'bool_id': obs[0]['image_id'] = True
    elif case == 'bad_r': obs[0]['R'][0][0] = 2
    elif case == 'reflection': obs[0]['R'][0][0] = -1
    elif case == 'nan': obs[0]['raw_pixels'][0] = float('nan')
    elif case == 'negative': xyz[2] = -5
    elif case == 'foreground': obs[0]['foreground'] = False
    elif case == 'wrong': obs[0]['raw_pixels'][0] += 100
    elif case == 'k': obs[0]['params'][3] = 1e-20
    elif case == 'f': obs[0]['params'][0] = 0
    elif case == 'too_many': obs = obs * 9
    elif case == 'zero_baseline':
        for o in obs: o['t'] = [0, 0, 0]; o['raw_pixels'] = [352, 256]
    out = assess_point(xyz, obs)
    assert out['status'] == 'REJECTED'
    assert out['reasons']
    json.dumps(out, allow_nan=False)


def test_loo_not_masked_by_xyz():
    xyz, obs = scene()
    obs = obs[:3]
    obs[0]['raw_pixels'][0] += 1.9
    out = assess_point(xyz, obs, max_loo_error_px=0.1)
    assert all(row['error_px'] <= 2 for row in out['reprojection'])
    assert 'LOO_ERROR_EXCEEDED' in out['reasons']


@pytest.mark.parametrize('xyz', [[float('inf'), 0, 1], [0, 0], None])
def test_bad_xyz(xyz):
    _, obs = scene()
    out = assess_point(xyz, obs)
    assert out['status'] == 'REJECTED'
    json.dumps(out, allow_nan=False)


def test_three_views_supported_and_condition_reported():
    xyz, obs = scene()
    out = assess_point(xyz, obs[:3])
    assert out['status'] == 'CONDITIONAL_POINT_SUPPORTED'
    assert out['jacobian_rank'] == 3
    assert out['jacobian_condition'] < out['max_condition']
    assert out['measurement_ray_angle_deg'] == out['max_angle_deg']
    assert abs(out['measurement_ray_angle_deg'] - out['candidate_ray_angle_deg']) < 1e-8


@pytest.mark.parametrize('kwargs', [{'max_error_px': -1}, {'min_angle_deg': float('nan')}, {'max_loo_error_px': float('inf')}])
def test_invalid_thresholds(kwargs):
    xyz, obs = scene()
    out = assess_point(xyz, obs, **kwargs)
    assert out['status'] == 'REJECTED'
    json.dumps(out, allow_nan=False)


def test_threshold_exceeded():
    xyz, obs = scene()
    obs[0]['raw_pixels'][1] += 2.01
    assert 'REPROJECTION_ERROR_EXCEEDED' in assess_point(xyz, obs)['reasons']


def test_measured_angle_cannot_be_faked_by_candidate():
    xyz, obs = scene()
    for o in obs: o['raw_pixels'] = [320, 240]
    out = assess_point(xyz, obs)
    assert out['candidate_ray_angle_deg'] > 2
    assert out['measurement_ray_angle_deg'] == 0
    assert 'INSUFFICIENT_TRIANGULATION_ANGLE' in out['reasons']
