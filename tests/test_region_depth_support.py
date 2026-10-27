import copy
import numpy as np
import pytest
from integrations.demo_pipeline.region_depth_support import backproject, project_pixels, classify, region_raster, separated_support, check_plan


def view():
    return dict(decode_index=7, camera=dict(K=np.eye(3), R=np.eye(3), C=np.zeros(3), width=5, height=5), mask=np.ones((5, 5), bool), valid=np.ones((5, 5), bool))


def test_camera_z_roundtrip_rotated_camera():
    c = view()['camera']; c['C'] = np.array([3., 2., 1.]); c['R'] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    uv = np.array([[1, 2], [3, 3]])
    p = backproject(uv, [4, 7], c)
    projected, z, inside = project_pixels(p, c)
    assert np.array_equal(projected, uv) and inside.all()
    assert np.allclose(z, [4, 7])


def test_occlusion_unknown_and_mask():
    v = view(); d = np.full((5, 5), 10.)
    p = np.array([[10, 10, 10], [5, 5, 5], [20, 20, 20], [0, 0, -1]])
    assert classify(p, v, d)[0].tolist() == [2, 3, 4, 0]
    assert classify(p, v, None)[0].tolist() == [1, 1, 1, 0]
    d[1, 1] = np.nan
    assert classify(p, v, d)[0][0] == 1
    v['valid'][1, 1] = False
    assert classify(p, v, d)[0][0] == 0


def test_region_identity_and_bounds():
    v = view(); s = dict(decode_index=7, cells=[dict(bounds=[1, 1, 3, 4])])
    assert region_raster(s, v).sum() == 6
    s['decode_index'] = 8
    with pytest.raises(ValueError, match='identity'):
        region_raster(s, v)
    s['decode_index'] = 7; s['cells'][0]['bounds'][2] = 10
    with pytest.raises(ValueError, match='outside'):
        region_raster(s, v)


def test_three_distinct_directions_required():
    points = np.array([[0., 0., 0.]])
    cameras = [dict(C=np.array(c)) for c in ([0, 0, 10], [0, 0, 10], [0, 0, 10])]
    assert not separated_support(points, cameras, np.ones((1, 3), bool))[0]
    cameras = [dict(C=np.array(c)) for c in ([0, 0, 10], [10, 0, 0], [0, 10, 0])]
    assert separated_support(points, cameras, np.ones((1, 3), bool))[0]
    assert not separated_support(points, cameras, np.array([[1, 1, 0]], bool))[0]


def test_plan_identity_drift():
    image = dict(name='a.png', global_id=1, width=5, height=5, K=np.eye(3).tolist(), R=np.eye(3).tolist(), C=[0, 0, 0])
    r = dict(sha256='same', bytes=3, path='same-input.png')
    p = dict(contract_version='mvs-surface-run-v1', scene=dict(sha256='scene'), frames=[dict(image=image, image_file=r, mask=r, validity=r)])
    check_plan(p, copy.deepcopy(p))
    q = copy.deepcopy(p); q['frames'][0]['image']['C'][0] = 1
    with pytest.raises(ValueError, match='camera'):
        check_plan(q, p)
    q = copy.deepcopy(p); q['scene']['sha256'] = 'different'
    with pytest.raises(ValueError, match='scene'):
        check_plan(q, p)


def test_two_percent_depth_gate_and_outside_image():
    v = view(); d = np.full((5, 5), 10.)
    p = np.array([[10.1, 10.1, 10.1], [10.3, 10.3, 10.3], [100, 100, 10]])
    assert classify(p, v, d)[0].tolist() == [2, 4, 0]


def test_plan_duplicate_and_input_hash_rejected():
    r = dict(sha256='same', bytes=3, path='a.png')
    image = dict(name='a.png', global_id=1, width=5, height=5, K=np.eye(3).tolist(), R=np.eye(3).tolist(), C=[0, 0, 0])
    p = dict(contract_version='mvs-surface-run-v1', scene=dict(sha256='scene'), frames=[dict(image=image, image_file=r, mask=r, validity=r)])
    q = copy.deepcopy(p); q['frames'].append(copy.deepcopy(q['frames'][0]))
    with pytest.raises(ValueError, match='duplicate'):
        check_plan(q, p)
    q = copy.deepcopy(p); q['frames'][0]['mask']['sha256'] = 'drift'
    with pytest.raises(ValueError, match='input identity'):
        check_plan(q, p)
