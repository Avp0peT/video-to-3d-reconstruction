"""Synthetic numerical contracts only; no registration/model execution."""
from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.neighbor_consensus import assess_motion


CENTER = [0., 0.]
CONTROLS = [[0., 0.], [-20., -20.], [20., -20.], [20., 20.], [-20., 20.]]
POINTS = [[-30., -30.], [30., -30.], [30., 30.], [-30., 30.], [0., 60.], [0., -60.]]


def records(matrix=None, points=POINTS, strict=True):
    matrix = np.eye(3) if matrix is None else np.asarray(matrix)
    return [dict(region_id=f'n{i}', source_raw_xy=p,
                 target_raw_xy=(matrix @ [*p, 1.])[:2].tolist(), strict_accepted=strict)
            for i, p in enumerate(points)]


def assess(rows, matrix=None, controls=CONTROLS):
    return assess_motion('candidate', CENTER, controls,
                         np.eye(3).tolist() if matrix is None else np.asarray(matrix).tolist(), rows)


@pytest.mark.parametrize('angle,scale', [(0., 1.), (.24, 1.15)])
def test_translation_rotation_scale(angle, scale):
    c, s = np.cos(angle)*scale, np.sin(angle)*scale
    matrix = np.array([[c, -s, 8.], [s, c, -5.], [0., 0., 1.]])
    result = assess(records(matrix), matrix)
    assert result['status'] == 'NEIGHBOR_CONSISTENT_CANDIDATE'
    np.testing.assert_allclose(result['neighbor_matrix'], matrix, atol=1e-12)
    assert result['matrix_global_raw3x3'] == result['neighbor_matrix']
    assert max(result['loo_errors_px']) < 1e-10
    assert result['geometry_accepted'] is result['identity_verified'] is False
    json.dumps(result, allow_nan=False)


def test_self_excluded_and_candidate_does_not_select_model():
    rows = records()
    baseline = assess(rows)
    rows.append(dict(region_id='candidate', source_raw_xy=[20., 20.],
                     target_raw_xy=[9999., -9999.], strict_accepted=True))
    conflict = np.eye(3)
    conflict[0, 2] = 20.
    changed = assess(rows, conflict)
    assert changed['neighbor_matrix'] == baseline['neighbor_matrix']
    assert changed['inlier_ids'] == baseline['inlier_ids']
    assert changed['status'] == 'NEIGHBOR_CONFLICT'
    assert changed['max_control_error_px'] == pytest.approx(20.)
    assert 'candidate' not in changed['selected_neighbor_ids']


def test_outlier_is_retained_and_input_is_unchanged():
    rows = records()
    rows[-1]['target_raw_xy'] = [200., 200.]
    before = deepcopy(rows)
    result = assess(rows)
    assert result['status'] == 'NEIGHBOR_CONSISTENT_CANDIDATE'
    assert result['outlier_ids'] == ['n5']
    assert len(result['actual_target_raw_xy']) == len(rows)
    assert len(result['neighbor_residuals_px']) == len(rows)
    assert result['max_selected_neighbor_error_px'] == result['max_neighbor_error_px']
    assert result['max_neighbor_error_px'] > 100.
    assert result['max_inlier_error_px'] < 1e-10
    assert rows == before


def test_degenerate_insufficient_and_extrapolation():
    insufficient = assess(records(points=POINTS[:3]))
    assert insufficient['status'] == 'INSUFFICIENT_SUPPORT'
    assert insufficient['matrix_global_raw3x3'] is None
    line = records(points=[[-60., 0.], [-30., 0.], [30., 0.], [60., 0.]])
    assert 'DEGENERATE_DESIGN' in assess(line)['reasons']
    outside = records(points=[[20., -20.], [40., -20.], [40., 20.], [20., 20.]])
    result = assess(outside)
    assert result['status'] == 'NO_CONSENSUS'
    assert result['matrix_global_raw3x3'] is None
    assert 'SOURCE_CENTER_OUTSIDE_INLIER_HULL' in result['reasons']


def test_unanchored_is_pending():
    assert assess(records(strict=False))['status'] == 'UNANCHORED_CONSENSUS'


def test_competing_affine_modes_are_ambiguous():
    points = [[-60., 0.], [-30., 0.], [30., 0.],
              [-30., -30.], [30., 30.], [-30., 30.], [30., -30.]]
    rows = records(points=points)
    for i, row in enumerate(rows):
        row['target_raw_xy'][1] *= .8 if i < 5 else 1.2
    result = assess(rows)
    assert result['status'] == 'AMBIGUOUS'
    alternative = np.eye(3)
    alternative[1, 1] = 1.2
    changed = assess(rows, alternative)
    assert changed['status'] == 'AMBIGUOUS'
    assert changed['neighbor_matrix'] == result['neighbor_matrix']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True])
def test_invalid_numeric_input(bad):
    rows = records()
    rows[0]['source_raw_xy'] = [bad, bad]
    with pytest.raises(ValueError):
        assess(rows)


def test_invalid_bool_and_duplicate_id():
    rows = records()
    rows[0]['strict_accepted'] = 1
    with pytest.raises(ValueError):
        assess(rows)
    rows = records()
    rows[1]['region_id'] = rows[0]['region_id']
    with pytest.raises(ValueError):
        assess(rows)


def test_budget_exception_propagates_during_fit():
    calls = 0
    def budget():
        nonlocal calls
        calls += 1
        if calls == 10:
            raise RuntimeError('budget exhausted')
    with pytest.raises(RuntimeError, match='budget exhausted'):
        assess_motion('candidate', CENTER, CONTROLS, np.eye(3).tolist(), records(), budget=budget)
