"""Conditional radial branch tests; no cameras/models are loaded."""
import json
import math

import pytest

from kozlov_lab.stereo.baseline.radial_branch import propose_central_domain
from kozlov_lab.stereo.baseline import radial_diagnostics as radial


def test_actual_camera_central_branch_covers_raw_rectangle():
    p = [802.1466958831231, 288., 512., -.18581500167448858]
    result = propose_central_domain(p, 576, 1024, [[313.73779296875, 988.5667724609375]])
    assert result['full_image_coverage']['covered'] is True
    assert result['max_ideal_radius'] == pytest.approx(math.sqrt((1-1e-6)/(-3*p[3])))
    assert result['domain_radial_derivative'] == pytest.approx(1e-6)
    assert result['turning_ideal_radius'] == pytest.approx(1.3393651232977075)
    assert result['all_observations_invertible']
    assert result['per_point'][0]['forward_residual_pixels'] <= 1e-8
    assert not result['physical_inverse_unique'] and not result['geometry_acceptance']
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match='nonmonotone'):
        radial.undistort_simple_radial([[313., 988.]], p, max_ideal_radius=1.4646745166112096)


def test_outer_real_point_and_central_root_both_roundtrip():
    p = [100., 300., 300., -1/3]
    # r=1.5 lies beyond turn=1, yet has positive forward radius 0.375.
    outer = [[450., 300.]]
    raw = [[337.5, 300.]]  # Direct forward formula, without imposing a monotone domain.
    result = propose_central_domain(p, 600, 600, raw)
    ideal = result['per_point'][0]['ideal_pixels']
    assert result['all_observations_invertible']
    assert math.dist(ideal, outer[0]) > 100
    assert math.dist(radial.distort_simple_radial([ideal], p, max_ideal_radius=result['max_ideal_radius'])[0], raw[0]) < 1e-8
    assert 1.5*(1+p[3]*1.5**2) == pytest.approx(.375)
    assert result['physical_inverse_unique'] is False


def test_full_image_not_covered_but_central_observation_inverts():
    result = propose_central_domain([1., 0., 0., -1/3], 10, 10, [[.1, .1]])
    assert result['full_image_coverage']['covered'] is False
    assert result['all_observations_invertible'] is True


def test_fixed_domain_independent_of_observations_and_outliers():
    p = [100., 0., 0., -1/3]
    a = propose_central_domain(p, 200, 200, [[0., 0.]])
    b = propose_central_domain(p, 200, 200, [[10., 0.], [100., 0.]])
    assert a['max_ideal_radius'] == b['max_ideal_radius']
    assert b['per_point'][1]['reason'] == 'RAW_RADIUS_OUTSIDE_CENTRAL_BRANCH_IMAGE'
    assert not b['all_observations_invertible']


def test_turn_and_fixed_margin_boundary_refused():
    p = [100., 0., 0., -1/3]
    result = propose_central_domain(p, 100, 100, [[100*(2/3), 0.]])
    assert result['per_point'][0]['reason'] == 'NEAR_TURN_BOUNDARY'
    radius = result['max_ideal_radius']
    raw = 100*radius*(1+p[3]*radius*radius)
    boundary = propose_central_domain(p, 100, 100, [[raw, 0.]])
    assert boundary['per_point'][0]['status'] == 'INVERSE_REFUSED'


@pytest.mark.parametrize('k', [0., .1])
def test_nonnegative_monotonic_and_center(k):
    result = propose_central_domain([100., 50., 50., k], 100, 100, [[50., 50.], [90., 75.]])
    assert result['max_ideal_radius'] is None
    assert result['full_image_coverage']['covered']
    assert result['all_observations_invertible']
    assert result['per_point'][0]['ideal_pixels'] == [50., 50.]


@pytest.mark.parametrize('params,width,height,pixels', [
    ([True, 0, 0, -.1], 100, 100, [[0, 0]]),
    ([100, 0, 0, float('nan')], 100, 100, [[0, 0]]),
    ([0, 0, 0, -.1], 100, 100, [[0, 0]]),
    ([100, 0, 0, -.1], True, 100, [[0, 0]]),
    ([100, 0, 0, -.1], 100, 0, [[0, 0]]),
    ([100, 0, 0, -.1], 100, 100, [[False, 0]]),
    ([100, 0, 0, -.1], 100, 100, [[float('inf'), 0]]),
    ([100, 0, 0, -.1], 100, 100, [0, 0]),
    ([100, 0, 0, -.1], 100, 100, []),
])
def test_bad_inputs_refused(params, width, height, pixels):
    with pytest.raises(ValueError):
        propose_central_domain(params, width, height, pixels)
