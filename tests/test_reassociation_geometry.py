"""Pure-data frozen original-F link assessment; no OpenCV or native calls."""
from copy import deepcopy

import pytest

from integrations.demo_pipeline.reassociation_geometry import assess_reassociated_links


def inputs():
    # y_second == y_first is the epipolar constraint for this rank-two F.
    pair = dict(frames=[1, 4], status='CANDIDATE_SUPPORT_NOT_VERIFIED',
                F=[[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
                H=None, train_ids=['train'], holdout_ids=['hold'],
                candidates=[dict(stable_id='train', xy_first=[100., 100.], xy_second=[110., 100.])],
                planar_or_low_parallax_ambiguity=False)
    link = dict(from_track_id='f:1', to_track_id='f:2', direction='forward',
                from_frame=1, to_frame=4, xy_from=[10., 20.], xy_to=[30., 20.],
                status='CANDIDATE_LINK_NOT_VERIFIED', gap_frames=3)
    return dict(links=[link]), dict(pairs=[pair])


def assess(recovery, geometry):
    return assess_reassociated_links(recovery, geometry)['links'][0]


def test_good_candidate_keeps_source_data_and_no_acceptance():
    recovery, geometry = inputs()
    before = deepcopy((recovery, geometry))
    result = assess_reassociated_links(recovery, geometry)
    row = result['links'][0]
    assert row['status'] == 'EXISTING_F_SUPPORT_CANDIDATE'
    assert row['sampson_px'] == 0.
    assert row['original_pair_evidence']['train_ids'] == ['train']
    assert len(row['original_pair_evidence']['sha256']) == 64
    assert result['counts'] == {'EXISTING_F_SUPPORT_CANDIDATE': 1}
    assert not result['model_refit']
    assert not row['correspondence_verified'] and not row['geometry_acceptance']
    assert not row['eligible_for_triangulation']
    assert (recovery, geometry) == before
    row['xy_from'][0] = 999
    assert (recovery, geometry) == before


def test_backward_orientation_uses_original_pair_order():
    recovery, geometry = inputs()
    # Asymmetric epipolar relation: y_second = 2*y_first.
    geometry['pairs'][0]['F'] = [[0., 0., 0.], [0., 0., -1.], [0., 2., 0.]]
    link = recovery['links'][0]
    link.update(direction='backward', from_frame=4, to_frame=1,
                xy_from=[30., 40.], xy_to=[10., 20.])
    row = assess(recovery, geometry)
    assert row['status'] == 'EXISTING_F_SUPPORT_CANDIDATE'
    assert row['original_pair_frames'] == [1, 4]
    assert row['sampson_px'] == 0.


def test_bad_residual_disagrees():
    recovery, geometry = inputs()
    recovery['links'][0]['xy_to'][1] = 30.
    assert assess(recovery, geometry)['status'] == 'GEOMETRY_DISAGREES'


@pytest.mark.parametrize('endpoint,xy', [('xy_from', [101., 100.]), ('xy_to', [111., 100.])])
def test_either_training_endpoint_blocks_independence(endpoint, xy):
    recovery, geometry = inputs()
    recovery['links'][0][endpoint] = xy
    row = assess(recovery, geometry)
    assert row['status'] == 'TRAIN_ENDPOINT_DEPENDENT'
    assert row['dependent_train_ids'] == ['train']
    assert 'sampson_px' not in row


@pytest.mark.parametrize('fault', ['unsupported', 'missing', 'shape', 'nonfinite', 'zero',
                                  'zero_denominator', 'missing_training', 'missing_candidate'])
def test_unusable_original_evidence_never_grants_support(fault):
    recovery, geometry = inputs()
    pair = geometry['pairs'][0]
    if fault == 'unsupported': pair['status'] = 'HOLDOUT_RESIDUAL_LARGE'
    elif fault == 'missing': pair.pop('F')
    elif fault == 'shape': pair['F'] = [[1., 2.]]
    elif fault == 'nonfinite': pair['F'][0][0] = float('nan')
    elif fault == 'zero': pair['F'] = [[0.] * 3 for _ in range(3)]
    elif fault == 'zero_denominator': pair['F'] = [[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]
    elif fault == 'missing_training': pair.pop('train_ids')
    elif fault == 'missing_candidate': pair['candidates'] = []
    row = assess(recovery, geometry)
    assert row['status'] == 'NO_INDEPENDENT_GEOMETRY'
    assert row['reason']


def test_planar_ambiguity_preserved_without_veto_or_triangulation():
    recovery, geometry = inputs()
    geometry['pairs'][0]['planar_or_low_parallax_ambiguity'] = True
    row = assess(recovery, geometry)
    assert row['status'] == 'EXISTING_F_SUPPORT_CANDIDATE'
    assert row['planar_or_low_parallax_ambiguity'] is True
    assert row['eligible_for_triangulation'] is False
