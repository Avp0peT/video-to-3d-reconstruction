from fractions import Fraction
import copy
import numpy as np
import pytest
from integrations.demo_pipeline.temporal_depth_consistency import analyze_window, analyze_windows


def fixture():
    centers = [[0, 0, 0], [0, 0, -10], [10, 0, 0], [0, 10, 0], [-10, 0, 0], [0, -10, 0]]
    cameras = [dict(K=np.eye(3), R=np.eye(3), C=np.array(c, float), width=10, height=10) for c in centers]
    return dict(points=np.array([[1., 1., 1.]]), states=np.array([[1, 2, 2, 2, 2, 3]], np.uint8),
                origins=np.array([1]), cameras=cameras, target_index=0,
                times=[Fraction(t) for t in (0, 1, 2, 3, 4, 10)],
                region=np.ones((10, 10), bool), raw_coverage=np.zeros((10, 10), bool),
                global_selected=np.array([True]))


def test_same_cohort_local_all_and_remote_conflict():
    a = fixture(); row, masks = analyze_window(**a, window=Fraction(4))
    assert row['support_cohort'] == 1
    assert masks['NO_LOCAL_FRONT_CONFLICT'].tolist() == [True]
    assert masks['NO_ALL_VIEW_FRONT_CONFLICT'].tolist() == [False]
    assert masks['REMOTE_ONLY_CONFLICT'].tolist() == [True]
    assert row['remote_only_conflict_count'] == 1
    assert row['exposures']['near']['same_support_cohort']['agree'] == 3


def test_self_votes_target_and_unknown_cannot_supply_three():
    a = fixture(); a['states'][0] = [2, 2, 2, 2, 1, 1]
    row, _ = analyze_window(**a, window=Fraction(4))
    assert row['support_cohort'] == 0  # target + generator do not vote
    assert row['exposures']['far']['same_support_cohort']['front_fraction_of_observable'] is None


def test_coincident_cameras_do_not_supply_angle_support():
    a = fixture()
    for c in a['cameras'][1:]: c['C'] = np.array([0., 0., -10.])
    row, _ = analyze_window(**a, window=Fraction(4))
    assert row['three_other_agree_before_angle'] == 1 and row['support_cohort'] == 0


def test_target_generated_candidate_rejected_even_with_three_other_votes():
    a = fixture(); a['origins'][0] = a['target_index']
    with pytest.raises(ValueError, match='target view cannot generate'):
        analyze_window(**a, window=Fraction(4))


def test_target_cannot_be_generator():
    a = fixture(); a['origins'][0] = a['target_index']
    with pytest.raises(ValueError, match='cannot generate'):
        analyze_window(**a, window=Fraction(4))


def test_fraction_closed_boundary_and_origin_requirement():
    a = fixture(); row, _ = analyze_window(**a, window=Fraction(4))
    assert row['near_view_ids'] == [1, 2, 3, 4]
    a['origins'][0] = 5
    row, _ = analyze_window(**a, window=Fraction(4))
    assert row['support_cohort'] == 0
    a = fixture(); a['times'][4] = Fraction(4000001, 1000000)
    row, _ = analyze_window(**a, window=Fraction(4))
    assert row['near_view_ids'] == [1, 2, 3]


def test_all_windows_same_denominator_and_behind_not_front():
    a = fixture(); a['states'][0, 5] = 4
    rows, masks = analyze_windows(**a)
    assert [r['window_seconds']['numerator'] for r in rows] == [1, 2, 4]
    assert all(r['frozen_candidate_denominator'] == 1 for r in rows)
    assert masks['4']['NO_ALL_VIEW_FRONT_CONFLICT'][0]
    assert rows[-1]['exposures']['far']['same_support_cohort']['behind'] == 1


@pytest.mark.parametrize('kind', ['nan', 'bool', 'states', 'origin', 'time', 'timeorder', 'duplicateids', 'raster'])
def test_invalid_inputs(kind):
    a = copy.deepcopy(fixture())
    if kind == 'nan': a['points'][0, 0] = np.nan
    if kind == 'bool': a['global_selected'] = np.array([1])
    if kind == 'states': a['states'][0, 0] = 5
    if kind == 'origin': a['origins'][0] = 6
    if kind == 'time': a['times'][1] = 1.0
    if kind == 'timeorder': a['times'][1] = Fraction(20)
    if kind == 'duplicateids': a['view_ids'] = [0]*6
    if kind == 'raster': a['region'] = np.ones((9, 9), bool)
    with pytest.raises(ValueError): analyze_window(**a, window=Fraction(4))
