"""Synthetic fixed-camera numerical checks; no reconstruction acceptance."""
import copy
import numpy as np
import pytest
from kozlov_lab.stereo.baseline.multiview_fixed import evaluate, fit, project


def observations():
    point = np.array([0.2, -0.15, 4.5])
    obs = []
    for index, center in enumerate(([0, 0, 0], [1, 0, 0], [0, 0.8, 0], [-0.5, -0.3, 0])):
        o = dict(image_id=index, raw_pixels=[0., 0.], params=[800., 320., 240., -0.05],
                 R=np.eye(3), t=-np.array(center, dtype=float), max_ideal_radius=1.)
        o['raw_pixels'] = project(point, o)['pixels']
        obs.append(o)
    return point, obs


def test_exact_recovery():
    point, obs = observations()
    result = fit(obs)
    assert result['status'] == 'FITTED'
    np.testing.assert_allclose(result['xyz'], point, atol=1e-8, rtol=0)
    assert result['final_cost'] <= result['initial_cost']
    assert result['train_image_ids'] == [0, 1, 2, 3]


def test_noisy_objective_descends_and_preserves_inputs():
    _, obs = observations()
    for o, noise in zip(obs, ([1, -.5], [-.7, .3], [.4, .8], [-.2, -.6])):
        o['raw_pixels'] = o['raw_pixels']+noise
    before = copy.deepcopy(obs)
    result = fit(obs)
    assert result['status'] == 'FITTED'
    assert result['final_cost'] < result['initial_cost']
    assert result['iterations'] <= 20
    evaluated = evaluate(result['xyz'], obs)
    assert evaluated['status'] == 'EVALUATED'
    assert evaluated['rms']**2*len(obs) == pytest.approx(result['final_cost'])
    for a, b in zip(obs, before):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])


def test_analytic_jacobian():
    point, obs = observations()
    for o in obs:
        numeric = np.column_stack([(project(point+np.eye(3)[j]*1e-6, o)['pixels']-
                                    project(point-np.eye(3)[j]*1e-6, o)['pixels'])/2e-6
                                   for j in range(3)])
        np.testing.assert_allclose(project(point, o)['jacobian'], numeric, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('mutation', ['duplicate', 'bool', 'nan', 'rotation', 'focal', 'domain', 'coincident', 'degenerate'])
def test_invalid_fit_refused(mutation):
    _, obs = observations()
    if mutation == 'duplicate': obs[1]['image_id'] = obs[0]['image_id']
    if mutation == 'bool': obs[0]['raw_pixels'] = [True, 2]
    if mutation == 'nan': obs[0]['t'][0] = np.nan
    if mutation == 'rotation': obs[0]['R'][0, 0] = 2
    if mutation == 'focal': obs[0]['params'][0] = 0
    if mutation == 'domain': obs[0]['max_ideal_radius'] = 10.
    if mutation == 'coincident':
        for o in obs: o['t'] = np.zeros(3)
    if mutation == 'degenerate':
        for i, o in enumerate(obs):
            o['t'] = np.array([0., 0., -float(i)])
            o['raw_pixels'] = np.array([320., 240.])
    result = fit(obs)
    assert result['status'] == 'REFUSED'
    assert result['xyz'] is None
    assert result['reason']


def test_behind_camera_and_domain_refused():
    _, obs = observations()
    assert evaluate([0, 0, -1], obs)['status'] == 'PROJECTION_REFUSED'
    assert evaluate([20, 0, 1], obs)['status'] == 'PROJECTION_REFUSED'
    with pytest.raises(ValueError):
        project([0, 0, 0], obs[0])


def test_single_view_evaluation_and_fit_refusal():
    point, obs = observations()
    obs[0]['raw_pixels'] = obs[0]['raw_pixels']+np.array([3., 4.])
    result = evaluate(point, obs[:1])
    assert result['status'] == 'EVALUATED'
    assert result['residual_vectors'][0] == pytest.approx([-3., -4.])
    assert result['residual_norms'] == pytest.approx([5.])
    assert result['rms'] == pytest.approx(5.)
    assert result['max'] == pytest.approx(5.)
    assert fit(obs[:1])['status'] == 'REFUSED'
