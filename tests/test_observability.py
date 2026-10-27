"""Synthetic local diagnostics; no independent geometry acceptance."""
import copy
import json
import numpy as np
import pytest

from kozlov_lab.stereo.baseline.multiview_fixed import project
from kozlov_lab.stereo.baseline.observability import analyze


def fixture(centers=((0, 0, 0), (1, 0, 0), (0, 1, 0))):
    xyz = np.array([.2, -.15, 4.5])
    obs = []
    for i, center in enumerate(centers):
        o = dict(image_id=i, raw_pixels=[0., 0.], params=[800., 320., 240., -.05],
                 R=np.eye(3), t=-np.asarray(center, dtype=float), max_ideal_radius=1.)
        o['raw_pixels'] = project(xyz, o)['pixels']
        obs.append(o)
    return xyz, obs


def test_full_rank_and_positive_prediction_response_and_no_mutation():
    xyz, obs = fixture()
    before = copy.deepcopy(obs)
    result = analyze(xyz, obs, obs[0])
    assert result['status'] == 'DIAGNOSTIC_EVALUATED'
    assert result['rank'] == 3
    assert all(0 <= p['ray_separation_degrees'] <= 180 for p in result['pairs'])
    assert all(p['baseline_length'] > 0 for p in result['pairs'])
    assert np.linalg.norm(result['weak_world_direction']) == pytest.approx(1)
    s = result['singular_values']
    assert s[0] >= s[1] >= s[2] > 0
    response = result['prediction_response']
    assert response['label'] == 'linear_unit_pixel_noise_response'
    assert all(v > 0 for v in response['sqrt_eigenvalue_axes'])
    j = np.vstack([project(xyz, o)['jacobian'] for o in obs])
    mapping = project(xyz, obs[0])['jacobian'] @ np.linalg.pinv(j)
    np.testing.assert_allclose(response['covariance'], mapping @ mapping.T)
    json.dumps(result, allow_nan=False)
    for a, b in zip(obs, before):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_coincident_pure_rotation_is_rank_two():
    xyz, obs = fixture(((0, 0, 0), (0, 0, 0)))
    angle = .2
    obs[1]['R'] = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    obs[1]['raw_pixels'] = project(xyz, obs[1])['pixels']
    result = analyze(xyz, obs, obs[0])
    assert result['status'] == 'RANK_DEFICIENT'
    assert result['rank'] == 2
    assert result['prediction_response'] is None
    assert result['pairs'][0]['baseline_length'] == 0
    assert result['pairs'][0]['ray_separation_degrees'] < 1e-5


def test_similarity_scale_changes_jacobian_units_but_not_condition():
    xyz, obs = fixture()
    base = analyze(xyz, obs, obs[0])
    scaled = copy.deepcopy(obs)
    for o in scaled:
        o['t'] *= 7
    result = analyze(xyz*7, scaled, scaled[0])
    np.testing.assert_allclose(result['singular_values'], np.array(base['singular_values'])/7)
    assert result['singular_value_ratio'] == pytest.approx(base['singular_value_ratio'])
    np.testing.assert_allclose(result['prediction_response']['covariance'],
                               base['prediction_response']['covariance'])


def test_rays_use_observed_pixels_independent_of_candidate():
    xyz, obs = fixture()
    base = analyze(xyz, obs)
    shifted = analyze(xyz+np.array([.3, .1, .2]), obs)
    for a, b in zip(base['observations'], shifted['observations']):
        np.testing.assert_array_equal(a['normalized_world_ray'], b['normalized_world_ray'])
        assert a['candidate_raw_pixels'] != b['candidate_raw_pixels']


@pytest.mark.parametrize('bad', ['nan', 'bool', 'duplicate', 'domain', 'depth', 'single', 'prediction'])
def test_strict_refusal(bad):
    xyz, obs = fixture()
    prediction = None
    if bad == 'nan': xyz[0] = np.nan
    if bad == 'bool': obs[0]['raw_pixels'] = [True, 2]
    if bad == 'duplicate': obs[1]['image_id'] = 0
    if bad == 'domain': obs[0]['raw_pixels'] = [1e8, 1e8]
    if bad == 'depth': xyz[2] = -1
    if bad == 'single': obs = obs[:1]
    if bad == 'prediction':
        prediction = copy.deepcopy(obs[0])
        prediction['t'][2] = -10
    result = analyze(xyz, obs, prediction)
    assert result['status'] == 'DIAGNOSTIC_REFUSED'
    assert result['reason']
    json.dumps(result, allow_nan=False)
