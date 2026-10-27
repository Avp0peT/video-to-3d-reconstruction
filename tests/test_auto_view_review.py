"""Independent adversarial checks: diagnostics never certify track identity."""
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline import view_geometry as audit
from kozlov_lab.stereo.baseline import multiview_fixed as fixed
from kozlov_lab.stereo.baseline.view_selection import select_pair


def scene():
    xyz = np.array([.2, -.1, 4.])
    observations = []
    for ident, center in enumerate(([0., 0., 0.], [1., 0., 0.], [0., 1., 0.]), 1):
        o = dict(image_id=ident, raw_pixels=[0., 0.], params=[800., 320., 240., 0.],
                 R=np.eye(3), t=-np.array(center), max_ideal_radius=2.)
        o['raw_pixels'] = fixed.project(xyz, o)['pixels'].tolist()
        observations.append(o)
    return xyz, observations


def test_no_third_view_is_retained_and_never_verified():
    _, obs = scene()
    result = audit.audit_track(70, obs[:2])
    assert result['point_id'] == 70
    assert result['status'] == 'PAIR_ONLY_NOT_VERIFIED'
    assert result['selected']['additional'] is None
    assert result['selected']['additional_image_ids'] == []
    assert result['geometry_acceptance'] is False


def test_one_bad_extra_retains_other_per_view_residuals():
    _, obs = scene()
    obs[2]['t'][2] = -10.  # selected pair still yields positive depth; third camera refuses
    result = audit.check_candidate(obs, [1, 2])
    assert result['status'] == 'PROJECTION_REFUSED'
    assert result['all_observations']['status'] == 'PROJECTION_REFUSED'
    views = {v['image_id']: v['diagnostic'] for v in result['per_view']}
    assert views[1]['status'] == views[2]['status'] == 'EVALUATED'
    assert len(views[1]['residual_vectors']) == 1
    assert views[3]['status'] == 'PROJECTION_REFUSED'
    assert result['geometry_acceptance'] is False


def test_ambiguous_native_view_cannot_be_full_check():
    _, obs = scene()
    result = audit.audit_track(71, obs, {'4': [2, 9]})
    assert result['status'] == 'INCOMPLETE_AMBIGUOUS_VIEWS'
    assert result['ambiguous_views_excluded'] == {'4': [2, 9]}
    assert result['all_native_views_checked'] is False
    assert result['geometry_acceptance'] is False


def test_selection_does_not_touch_xyz_labels_or_residuals():
    _, obs = scene()
    expected = select_pair(obs)
    poisoned = copy.deepcopy(obs)
    for o in poisoned:
        o.update(xyz=object(), source_track_id=object(), label=object(),
                 residual=object(), truth=object())
    assert select_pair(poisoned) == expected


def test_invalid_extra_cannot_be_silently_dropped_to_select_pair():
    _, obs = scene()
    obs[2]['raw_pixels'] = [float('nan'), 0.]
    result = audit.audit_track(72, obs)
    assert result['status'] == 'NO_USABLE_VIEW_PAIR'
    assert len(result['observations']) == 3
    assert result['selected'] is None


def test_over_budget_views_preserve_track_slot():
    _, obs = scene()
    many = [dict(copy.deepcopy(obs[0]), image_id=i) for i in range(65)]
    result = audit.audit_track(73, many)
    assert result['status'] == 'NOT_EVALUATED_VIEW_COUNT'
    assert len(result['observations']) == 65
    assert result['point_id'] == 73
