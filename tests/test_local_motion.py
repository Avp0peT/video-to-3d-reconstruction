import json

import cv2 as cv
import numpy as np
import pytest

from kozlov_lab.stereo.local_motion import evaluate_local_motion

cv.setNumThreads(2)


def fixture():
    mask = np.zeros((400, 400), np.uint8)
    mask[40:340, 40:340] = 255
    # 64 points per core, ID order interleaves quadrants in each heldout subset.
    p = np.array([(50+c*100+x*11, 50+r*100+y*11)
        for r in range(3) for c in range(3) for y in range(8) for x in range(8)], float)
    return mask, p, np.arange(len(p))


def run(mask, p, ids, q=None):
    return evaluate_local_motion(mask, mask, p, p.copy() if q is None else q, p, ids, cv)


def test_identity_measured_support_and_json():
    m, p, ids = fixture()
    result = run(m, p, ids)
    json.dumps(result, allow_nan=False)
    assert not result['publishable_for_geometry']
    assert result['physical_identity'] == 'UNVERIFIED'
    for profile in result['profiles'].values():
        assert len(profile['common_supported_track_ids']) > 90
        assert all(v['track_id'] % 5 == 0 for v in profile['measured_query_evidence'])
        for v in profile['measured_query_evidence']:
            assert v['query_point_xy'] == p[v['track_id']].tolist()


def test_piecewise_motion_supported_where_global_fails():
    m, p, ids = fixture()
    q = p.copy()
    q[:, 0] += np.where(p[:, 0] < 140, -9, np.where(p[:, 0] < 240, 0, 9))
    result = run(m, p, ids, q)
    fit = ids % 5 != 0
    cv.setRNGSeed(0)
    matrix, _ = cv.estimateAffinePartial2D(p[fit], q[fit], method=cv.RANSAC,
        ransacReprojThreshold=2.5, maxIters=1000, confidence=.99, refineIters=10)
    residual = np.linalg.norm(p @ matrix[:, :2].T+matrix[:, 2]-q, axis=1)
    assert np.percentile(residual[~fit], 90) > 4
    assert len(result['profiles']['local_similarity']['common_supported_track_ids']) >= 30


def test_shear_profiles_remain_separate():
    m, p, ids = fixture()
    q = p @ np.array([[1, 0], [.3, 1]]) + [-30, 0]
    result = run(m, p, ids, q)
    assert len(result['profiles']['local_affine']['common_supported_track_ids']) >= 80
    assert len(result['profiles']['local_similarity']['common_supported_track_ids']) < 30


def test_holdout_false_matches_not_used_in_any_fit():
    m, p, ids = fixture()
    baseline = run(m, p, ids)
    q = p.copy()
    q[ids % 5 == 0] += [15, 0]
    changed = run(m, p, ids, q)
    for key, profile in changed['profiles'].items():
        assert not profile['common_supported_track_ids']
        for side in ('before', 'after'):
            for a, b in zip(baseline['profiles'][key]['references'][side]['cells'], profile['references'][side]['cells']):
                assert a['matrix'] == b['matrix']
                assert set(b['fit_track_ids']).isdisjoint(changed['holdout_track_ids'])
                assert 'holdout_residual_high' in b['reasons']


def test_permutation_and_input_unchanged():
    m, p, ids = fixture()
    saved = (m.copy(), p.copy(), ids.copy())
    a = run(m, p, ids)
    order = np.random.default_rng(3).permutation(len(ids))
    b = run(m, p[order], ids[order])
    assert a == b
    assert all(np.array_equal(x, y) for x, y in zip((m, p, ids), saved))


@pytest.mark.parametrize('kind', ['empty', 'sparse', 'collinear', 'cluster'])
def test_insufficient_evidence_no_fit(kind):
    m, p, ids = fixture()
    if kind == 'empty': p = np.empty((0, 2))
    if kind == 'sparse': p = p[:10]
    if kind == 'collinear': p[:, 1] = 100
    if kind == 'cluster': p = 80 + (p-p.min(0))/100
    result = run(m, p, np.arange(len(p)))
    for profile in result['profiles'].values():
        assert profile['status'] == 'UNRESOLVED'
        assert all(c['matrix'] is None for c in profile['references']['before']['cells'])


@pytest.mark.parametrize('kind', ['nan', 'boolpoints', 'boolid', 'dupid', 'overflow', 'complex', 'many'])
def test_invalid_input(kind):
    m, p, ids = fixture()
    if kind == 'nan': p[0, 0] = np.nan
    if kind == 'boolpoints': p = p.astype(bool)
    if kind == 'boolid': ids = ids.tolist(); ids[0] = True
    if kind == 'dupid': ids[0] = ids[1]
    if kind == 'overflow': p[0, 0] = 400
    if kind == 'complex': p = p.astype(complex)
    if kind == 'many': p = np.tile(p, (2, 1)); ids = np.arange(len(p))
    with pytest.raises(ValueError): run(m, p, ids)


def test_reflection_rejected_by_affine():
    m, p, ids = fixture()
    q = p.copy(); q[:, 0] = 380-p[:, 0]
    result = run(m, p, ids, q)
    profile = result['profiles']['local_affine']
    assert not profile['common_supported_track_ids']
    assert all('nonpositive_determinant' in c['reasons'] for c in profile['references']['before']['cells'])


def test_extrapolated_holdouts_rejected_even_with_perfect_motion():
    m, p, ids = fixture()
    # Keep fit points occupying three quadrants and full 2-D spread, but pull
    # all heldouts beyond the fit convex hull while preserving core coverage.
    fit = ids % 5 != 0
    for r in range(3):
        for c in range(3):
            chosen = fit & (p[:, 0] >= 40+c*100) & (p[:, 0] < 140+c*100) & (p[:, 1] >= 40+r*100) & (p[:, 1] < 140+r*100)
            center = np.array([90+c*100, 90+r*100])
            p[chosen] = center+(p[chosen]-center)*.5
    result = run(m, p, ids)
    cells = result['profiles']['local_affine']['references']['before']['cells']
    assert any('holdout_extrapolation_unsupported' in c['reasons'] for c in cells)
    rejected = {tid for c in cells if 'holdout_extrapolation_unsupported' in c['reasons'] for tid in c['holdout_track_ids']}
    assert rejected.isdisjoint(result['profiles']['local_affine']['common_supported_track_ids'])


def test_query_hull_is_separate_measured_support_gate():
    m, p, ids = fixture()
    # Source holdouts sit 7px beyond a broad fitted square; scaling makes
    # their measured query extrapolation exceed8px despite zero residual.
    ids = np.arange(40)
    held = ids % 5 == 0
    p = np.empty((40, 2))
    p[~held] = [(x, y) for y in np.linspace(60, 114, 8) for x in [60, 78, 96, 114]]
    p[held] = [[53, 65], [121, 65], [53, 105], [121, 105], [65, 53], [105, 53], [65, 121], [105, 121]]
    q = p*1.3
    result = run(m, p, ids, q)
    cells = result['profiles']['local_affine']['references']['before']['cells']
    assert any('holdout_query_extrapolation_unsupported' in c['reasons'] for c in cells)
    cell = cells[0]
    assert max(cell['holdout_extrapolation_distances_px']) < 8
    assert max(cell['holdout_residuals_px']) < .01
    for c in cells:
        if c['matrix'] is not None:
            assert 'query_coverage' in c['groups']['holdout']


def test_shared_noninteger_edges_assign_heldouts_once_and_final_pixel():
    m = np.zeros((400, 400), np.uint8); m[40:341, 40:342] = 255
    edge_x = np.linspace(40, 342, 4)
    edge_y = np.linspace(40, 341, 4)
    points = np.array([[edge_x[1], 50], [edge_x[2], 50], [341, 340],
                       [50, edge_y[1]], [50, edge_y[2]]])
    # All five observations held out globally, while filler fit IDs remain unique.
    p = np.repeat(points, 5, axis=0)
    result = run(m, p, np.arange(len(p)))
    cells = result['profiles']['local_affine']['references']['before']['cells']
    assigned = [tid for cell in cells for tid in cell['holdout_track_ids']]
    assert sorted(assigned) == [0, 5, 10, 15, 20]
    assert 10 in cells[8]['holdout_track_ids']
