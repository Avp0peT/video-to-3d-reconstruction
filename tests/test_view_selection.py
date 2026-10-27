"""Synthetic conditioning checks; not independent physical acceptance."""
import copy
import json

import numpy as np
import pytest

from kozlov_lab.stereo.baseline.view_selection import select_pair
from kozlov_lab.stereo.baseline.radial_diagnostics import distort_simple_radial


def observation(i, uv=(0., 0.), center=(0., 0., 0.), rotation=None):
    r = np.eye(3) if rotation is None else rotation
    return dict(image_id=i, raw_pixels=list(uv), params=[1., 0., 0., 0.],
                R=r, t=-r @ np.asarray(center), max_ideal_radius=None)


def test_ranking_ties_order_and_immutable_inputs():
    obs = [observation(9, (-1., 0.), (2., 0., 0.)),
           observation(2, (1., 0.), (1., 0., 0.)), observation(1)]
    before = copy.deepcopy(obs)
    result = select_pair(obs)
    assert result['selected_image_ids'] == [2, 9]
    assert result['selected_triangulation_angle_degrees'] == pytest.approx(90.)
    assert result['number_of_pairs'] == 3
    assert select_pair(obs[::-1]) == result
    json.dumps(result, allow_nan=False)
    for a, b in zip(obs, before):
        for key in a:
            np.testing.assert_equal(a[key], b[key])
    tied = [observation(3, (1., 0.), (2., 0., 0.)), observation(2, (1., 0.), (1., 0., 0.)), observation(1)]
    assert select_pair(tied)['selected_image_ids'] == [1, 2]


def test_similarity_rotation_and_tiny_world_units():
    obs = [observation(1), observation(2, (1., 0.), (1., 0., 0.)),
           observation(3, (-.5, .3), (0., 2., 0.))]
    base = select_pair(obs)
    q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    for scale in [1e-150, 7., 1e150]:
        changed = copy.deepcopy(obs)
        for o in changed:
            c = -o['R'].T @ o['t']
            o['R'] = o['R'] @ q.T
            o['t'] = -o['R'] @ (scale*(q @ c + np.array([3., 2., 1.])))
        result = select_pair(changed)
        assert result['selected_image_ids'] == base['selected_image_ids']
        assert result['selected_score'] == pytest.approx(base['selected_score'])
        assert [p['relative_center_separation'] for p in result['pairs']] == pytest.approx([p['relative_center_separation'] for p in base['pairs']])


def test_degenerate_centers_parallel_antiparallel_and_tiny_parallax():
    assert select_pair([observation(1), observation(2, (1., 0.))])['status'] == 'REFUSED'
    assert select_pair([observation(1), observation(2, center=(1., 0., 0.))])['pairs'][0]['score'] is None
    flipped = np.diag([-1., 1., -1.])
    assert select_pair([observation(1), observation(2, center=(1., 0., 0.), rotation=flipped)])['status'] == 'REFUSED'
    result = select_pair([observation(1), observation(2, (1e-8, 0.), (1., 0., 0.))])
    assert result['status'] == 'SELECTED'
    assert result['selected_score'] == pytest.approx(1e-8)
    obs = [observation(1), observation(2, (1., 0.), (1e-12, 0., 0.)), observation(3, (-1., 0.), (1., 0., 0.))]
    assert 'near-coincident' in select_pair(obs)['pairs'][0]['reason']


@pytest.mark.parametrize('bad', [[], [observation(1)], [observation(i) for i in range(65)],
                                [observation(1), observation(1)]])
def test_invalid_counts_and_duplicates(bad):
    assert select_pair(bad)['status'] == 'REFUSED'


@pytest.mark.parametrize('key,value', [('image_id', True), ('raw_pixels', [True, 0.]),
                                    ('raw_pixels', [float('nan'), 0.]),
                                    ('params', [0., 0., 0., 0.]), ('t', ['1', '2', '3']),
                                    ('R', np.zeros((3, 3))), ('max_ideal_radius', True)])
def test_strict_validation(key, value):
    obs = [observation(1), observation(2, (1., 0.), (1., 0., 0.))]
    obs[0][key] = value
    result = select_pair(obs)
    assert result['status'] == 'REFUSED'
    json.dumps(result, allow_nan=False)


def test_measured_outlier_can_win_and_extra_labels_are_ignored():
    obs = [observation(1), observation(2, (.1, 0.), (1., 0., 0.)), observation(3, (10., 0.), (2., 0., 0.))]
    base = select_pair(obs)
    assert 3 in base['selected_image_ids']
    for o in obs:
        o.update(xyz=[float('nan')]*3, groundtruth='not used', residual=-1., ideal_pixels=[0., 0.])
    assert select_pair(obs) == base


def test_radial_inverse_and_nearly_antiparallel_not_preferred():
    flipped = np.diag([-1., 1., -1.])
    obs = [observation(1), observation(2, (.01, 0.), (1., 0., 0.), flipped),
           observation(3, (1., 0.), (2., 0., 0.))]
    result = select_pair(obs)
    assert result['pairs'][0]['ray_separation_degrees'] > 179.
    assert result['selected_image_ids'] != [1, 2]
    base = [observation(1), observation(2, (.8, .2), (1., 0., 0.))]
    expected = select_pair(base)
    distorted = copy.deepcopy(base)
    for o in distorted:
        o['params'] = [1., 0., 0., -.05]
        o['max_ideal_radius'] = 1.
        o['raw_pixels'] = distort_simple_radial(np.array([o['raw_pixels']]), o['params'], max_ideal_radius=1.)[0]
    assert select_pair(distorted)['selected_score'] == pytest.approx(expected['selected_score'], abs=2e-8)
