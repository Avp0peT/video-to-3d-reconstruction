import copy
import numpy as np
import pytest
from integrations.demo_pipeline.region_depth_tiers import validate_ledger, summarize_ledger


def fixture():
    p = np.array([[1., 1., 1.], [2., 2., 1.]])
    s = np.array([[2, 3, 4], [2, 4, 1]], np.uint8)
    o = np.array([1, 2]); chosen = np.ones(2, bool)
    views = [dict(decode_index=i, camera_name=str(i)) for i in range(3)]
    result = dict(sampled_candidates=2, selected_candidates=2, total_candidates=2)
    return p, s, o, chosen, views, views, result


def test_valid_and_two_tiers_use_raw_pixel_missing_domain():
    args = fixture(); validate_ledger(*args)
    p, s, o, take, views, _, _ = args
    c = dict(K=np.eye(3), R=np.eye(3), C=np.zeros(3), width=4, height=4)
    roi = np.ones((4, 4), bool); raw = np.zeros_like(roi); raw[1, 1] = True
    report, hits, no_front = summarize_ledger(p, s, o, take, views, c, roi, raw)
    assert no_front.tolist() == [False, True]
    assert report['raw_uncovered_pixels'] == 15
    assert report['tiers']['SUPPORT_ONLY']['target_unique_pixels'] == 2
    assert report['tiers']['NO_KNOWN_FRONT_CONFLICT']['raw_uncovered_unique_pixels'] == 1
    assert report['tiers']['NO_KNOWN_FRONT_CONFLICT']['samples_with_any_behind_state'] == 1


@pytest.mark.parametrize('change', ['nan', 'shape', 'state', 'floatstate', 'origin', 'bool', 'order', 'count'])
def test_invalid_ledgers_fail_closed(change):
    a = list(fixture()); a = copy.deepcopy(a)
    if change == 'nan': a[0][0, 0] = np.nan
    if change == 'shape': a[1] = a[1][:, :2]
    if change == 'state': a[1][0, 0] = 5
    if change == 'floatstate': a[1] = a[1].astype(float)
    if change == 'origin': a[2][0] = 3
    if change == 'bool': a[3] = a[3].astype(int)
    if change == 'order': a[5] = a[5][::-1]
    if change == 'count': a[6]['selected_candidates'] = 1
    with pytest.raises(ValueError): validate_ledger(*a)


def test_duplicate_target_projections_are_not_extra_pixels():
    p, s, o, chosen, views, _, _ = fixture(); p[1] = p[0]
    c = dict(K=np.eye(3), R=np.eye(3), C=np.zeros(3), width=4, height=4)
    roi = np.ones((4, 4), bool)
    r, _, _ = summarize_ledger(p, s, o, chosen, views, c, roi, np.zeros_like(roi))
    assert r['tiers']['SUPPORT_ONLY']['candidate_samples'] == 2
    assert r['tiers']['SUPPORT_ONLY']['target_unique_pixels'] == 1
