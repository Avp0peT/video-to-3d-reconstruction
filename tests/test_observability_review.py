"""Independent degeneracy and coordinate-change checks for local diagnostics."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kozlov_lab.stereo.baseline import multiview_fixed as fixed
from kozlov_lab.stereo.baseline import observability


def observation(image_id, center, xyz, rotation=None):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    obs = dict(image_id=image_id, raw_pixels=[0., 0.],
               params=[700., 320., 240., 0.03], R=rotation,
               t=-rotation @ np.asarray(center), max_ideal_radius=2.)
    obs['raw_pixels'] = fixed.project(xyz, obs)['pixels'].tolist()
    return obs


def test_same_center_different_rotations_has_depth_nullspace():
    xyz = np.array([0.2, -0.1, 8.])
    angle = 0.2
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)],
                         [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    obs = [observation(1, [0., 0., 0.], xyz),
           observation(2, [0., 0., 0.], xyz, rotation)]
    result = observability.analyze(xyz, obs, obs[0])
    assert result['status'] == 'RANK_DEFICIENT'
    assert result['rank'] == 2
    assert result['prediction_response'] is None
    assert result['pairs'][0]['baseline_length'] == 0.
    assert abs(np.dot(result['weak_world_direction'], xyz / np.linalg.norm(xyz))) == pytest.approx(1.)


def test_small_baseline_is_full_rank_but_bad_for_distant_view_prediction():
    xyz = np.array([0.2, -0.1, 8.])
    prediction = observation(9, [2., 0., 0.], xyz)
    results = [observability.analyze(xyz,
               [observation(1, [0., 0., 0.], xyz),
                observation(2, [baseline, 0., 0.], xyz)], prediction)
               for baseline in (1.e-4, 1.)]
    assert all(r['rank'] == 3 for r in results)
    assert results[0]['singular_value_ratio'] < results[1]['singular_value_ratio'] / 1000.
    assert (results[0]['prediction_response']['sqrt_eigenvalue_axes'][0] >
            1000. * results[1]['prediction_response']['sqrt_eigenvalue_axes'][0])


def test_duplicate_poses_only_improve_response_under_stated_independent_noise_assumption():
    xyz = np.array([0.2, -0.1, 8.])
    obs = [observation(1, [0., 0., 0.], xyz), observation(2, [1., 0., 0.], xyz)]
    repeated = obs + [dict(o, image_id=o['image_id'] + 10) for o in obs]
    prediction = observation(9, [2., 0., 0.], xyz)
    single = observability.analyze(xyz, obs, prediction)
    double = observability.analyze(xyz, repeated, prediction)
    assert single['rank'] == double['rank'] == 3
    assert double['singular_value_ratio'] == pytest.approx(single['singular_value_ratio'])
    np.testing.assert_allclose(double['singular_values'], np.sqrt(2.) * np.array(single['singular_values']))
    np.testing.assert_allclose(double['prediction_response']['covariance'],
                               np.array(single['prediction_response']['covariance']) / 2.)
    assert 'independent' in double['prediction_response']['training_noise']


def test_world_scale_and_translation_preserve_pixel_response_and_angles():
    xyz = np.array([0.2, -0.1, 8.])
    centers = [np.array([0., 0., 0.]), np.array([1., 0., 0.])]
    prediction_center = np.array([2., 0., 0.])
    original = observability.analyze(xyz,
        [observation(i + 1, c, xyz) for i, c in enumerate(centers)],
        observation(9, prediction_center, xyz))
    scale, shift = 13., np.array([4., -3., 7.])
    changed_xyz = scale * xyz + shift
    changed = observability.analyze(changed_xyz,
        [observation(i + 1, scale * c + shift, changed_xyz) for i, c in enumerate(centers)],
        observation(9, scale * prediction_center + shift, changed_xyz))
    assert changed['rank'] == original['rank']
    assert changed['singular_value_ratio'] == pytest.approx(original['singular_value_ratio'])
    np.testing.assert_allclose(changed['singular_values'], np.array(original['singular_values']) / scale)
    np.testing.assert_allclose(changed['prediction_response']['covariance'], original['prediction_response']['covariance'])
    assert changed['pairs'][0]['ray_separation_degrees'] == pytest.approx(original['pairs'][0]['ray_separation_degrees'])
    assert changed['pairs'][0]['baseline_length'] == pytest.approx(scale * original['pairs'][0]['baseline_length'])
