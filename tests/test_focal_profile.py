import json

import numpy as np
import pytest

from integrations.demo_pipeline.focal_profile import profile_fundamental


def fundamental(f=1024., principal=(288., 512.), transpose=False):
    axis = np.array([.3, .7, -.2]); axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    R = np.eye(3) + np.sin(.3)*skew + (1-np.cos(.3))*(skew@skew)
    t = np.array([[0, -.4, -.2], [.4, 0, -.8], [.2, .8, 0]])
    K = np.array([[f, 0, principal[0]], [0, f, principal[1]], [0, 0, 1.]])
    F = np.linalg.inv(K).T @ t @ R @ np.linalg.inv(K)
    return F.T if transpose else F


def profile(F, **kwargs):
    return profile_fundamental(F, (288., 512.), (288., 512.), 1024., **kwargs)


def test_positive_and_json():
    result = profile(fundamental())
    assert result['status'] == 'FOCAL_PROFILE_EVALUATED'
    assert len(result['curve']) == 101
    assert result['minimum_grid_index'] == 50
    assert result['minimum_qess'] < 1e-25
    assert not result['flat'] and not result['boundary_minimum']
    assert result['focal_observability_status'] == 'CONDITIONAL_INTERIOR_MINIMUM'
    assert not any(result[k] for k in ('calibration_accepted', 'independent_truth', 'new_ba_authorized'))
    json.dumps(result, allow_nan=False)


def test_translation_flat():
    K = np.array([[1024., 0, 288], [0, 1024., 512], [0, 0, 1.]])
    t = np.array([[0., 0, 0], [0, 0, -1], [0, 1, 0]])
    result = profile(np.linalg.inv(K).T @ t @ np.linalg.inv(K))
    assert result['flat']
    assert result['focal_observability_status'] == 'FLAT_PROFILE'


@pytest.mark.parametrize('F,code', [(np.zeros((3, 3)), 'ZERO_F'),
                                  (np.eye(3), 'RANK_NOT_TWO'),
                                  (np.diag([1., 0, 0]), 'RANK_NOT_TWO'),
                                  (np.full((3, 3), np.nan), 'INVALID_INPUT'),
                                  (np.ones((2, 3)), 'INVALID_INPUT'),
                                  ([[True, 0., 0.], [0., 1., 0.], [0., 0., 0.]], 'INVALID_INPUT'),
                                  (np.ones((3, 3), dtype=bool), 'INVALID_INPUT')])
def test_invalid(F, code):
    result = profile(F)
    assert result['status'] == 'FOCAL_PROFILE_REFUSED'
    assert result['reason_code'] == code
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('config', [0, 1, 4, 5, 6, 7, True, 3.0, None])
def test_ambiguous_configuration_rejected_even_without_f(config):
    assert profile(None, config=config)['reason_code'] == 'CONFIG_AMBIGUOUS'


def test_homography_configuration_refuses_even_rank_two_f():
    # Any [e]_x H fits a planar correspondence set; configuration refuses it.
    assert profile(fundamental(), config=4)['status'] == 'FOCAL_PROFILE_REFUSED'


def test_truncated_domain():
    result = profile(fundamental(f=1024.*100))
    assert result['boundary_minimum']
    assert result['minimum_grid_index'] == 100


def test_scale_transpose_and_pixel_resize():
    F = fundamental()
    base = profile(F)
    expected = [v['qess'] for v in base['curve']]
    for changed in (profile(-17*F), profile(F.T)):
        np.testing.assert_allclose([v['qess'] for v in changed['curve']], expected, atol=1e-14)
    A = np.diag([.5, .5, 1.])
    scaled = profile_fundamental(np.linalg.inv(A).T @ F @ np.linalg.inv(A),
                                 (144., 256.), (144., 256.), 512.)
    np.testing.assert_allclose([v['qess'] for v in scaled['curve']], expected, atol=1e-14)
    np.testing.assert_allclose([v['focal'] for v in scaled['curve']],
                               np.array([v['focal'] for v in base['curve']])*.5)


@pytest.mark.parametrize('edge', [True, 0, -1, float('inf'), '1024'])
def test_bad_edge(edge):
    assert profile_fundamental(fundamental(), (0, 0), (0, 0), edge)['reason_code'] == 'INVALID_INPUT'


def test_bool_principal():
    assert profile_fundamental(fundamental(), (True, 0.), (0, 0), 1024)['reason_code'] == 'INVALID_INPUT'
