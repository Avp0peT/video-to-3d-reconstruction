from copy import deepcopy
import json

import numpy as np
import pytest

from integrations.demo_pipeline import neighborhood_review as review


def photo(matrix=None):
    pixels = (np.indices((33, 33)).sum(0)+20).tolist()
    metrics = dict(coverage=1., ncc=.99, std_template=10., std_search=10., normalized_rmse=.1)
    return dict(status='ACCEPTED', reasons=[], matrix_cv2x3=matrix or [[1., 0., 52.], [0., 1., 32.]],
                train=deepcopy(metrics), holdout=deepcopy(metrics),
                pixel_evidence=dict(template_uint8=pixels, warped_search_gray=deepcopy(pixels),
                                    valid_mask_uint8=[[255]*33 for _ in range(33)]),
                native_calls=dict(matchTemplate=1, findTransformECCWithMask=1))


def fixture(centers=((100.5, 100.5),), targets=5):
    domains = {}
    decoded = {}
    for index in range(targets+1):
        domain = dict(width=256, height=256, path=f'not-on-disk-{index}.png',
                      original_mask_status='OK' if index == 0 else 'UNKNOWN_NOT_SEGMENTED',
                      mask_rle=[[[0, 256]] for _ in range(256)] if index == 0 else None)
        domains[str(index)] = domain
        decoded[index] = dict(width=256, height=256, path=domain['path'],
                              gray=(np.indices((256, 256)).sum(0)+index*17).astype(np.uint8))
    roots, areas = [], []
    for ordinal, (x, y) in enumerate(centers):
        roi = dict(left=round(x-.5)-16, top=round(y-.5)-16, width=33, height=33)
        search = dict(left=round(x-.5)-48, top=round(y-.5)-48, width=97, height=97)
        polygon = [[roi['left']+.5, roi['top']+.5], [roi['left']+32.5, roi['top']+.5],
                   [roi['left']+32.5, roi['top']+32.5], [roi['left']+.5, roi['top']+32.5]]
        root = dict(region_id=f'area-{ordinal}', source_input_index=27, source_decode_index=0,
                    decode_path=list(range(targets+1)), center_raw_xy=[x, y])
        forward = dict(template_roi=roi, search_roi=search, alignment=photo(),
                       matrix_global_raw3x3=[[1., 0., 20.], [0., 1., 0.], [0., 0., 1.]],
                       polygon_raw4x2=[[px+20, py] for px, py in polygon])
        leg = dict(source_decode_index=0, target_decode_index=1, status='REJECTED',
                   reasons=['AREA_FIVE_POINT_CYCLE_THRESHOLD'], cycle_error_px=3., forward=forward,
                   backward=dict(alignment=photo()))
        roots.append(root)
        areas.append(dict(region_id=root['region_id'], source_input_index=27, source_decode_index=0,
                          decode_path=root['decode_path'][:], template_roi=roi, region_legs=[leg]))
    return dict(regions=roots, frame_domains=domains), areas, decoded


@pytest.mark.parametrize('missing', ['template_uint8', 'warped_search_gray', 'valid_mask_uint8'])
def test_collect_requires_saved_forward_pixel_support_without_changing_strict_result(missing):
    plan, areas, _ = fixture(centers=((100.5, 100.5), (120.5, 100.5)))
    del areas[1]['region_legs'][0]['forward']['alignment']['pixel_evidence'][missing]
    before = deepcopy((plan, areas))
    records = review.collect_measurements(plan, areas)
    assert records[0]['strict_status'] == 'REJECTED'
    assert records[0]['strict_reasons'] == ['AREA_FIVE_POINT_CYCLE_THRESHOLD']
    assert records[0]['measurement']['target_center_raw_xy'] == [120.5, 100.5]
    assert records[0]['continuation_eligible'] is True
    assert records[1]['measurement'] is None
    assert records[1]['eligibility_reason'] == 'NO_ACCEPTED_FORWARD_PIXEL_EVIDENCE'
    assert records[1]['neighborhood'] is None
    assert (plan, areas) == before


def test_collect_rejects_changed_saved_matrix_and_forward_status_without_promoting_candidates():
    plan, areas, _ = fixture()
    areas[0]['region_legs'][0]['forward']['matrix_global_raw3x3'][0][2] = 21.
    with pytest.raises(ValueError, match='FORWARD_MATRIX_CHANGED'):
        review.collect_measurements(plan, areas)
    areas[0]['region_legs'][0]['forward']['alignment']['status'] = 'REJECTED'
    records = review.collect_measurements(plan, areas)
    assert records[0]['measurement'] is None and records[0]['strict_status'] == 'REJECTED'
    assert records[0]['geometry_accepted'] is False and records[0]['identity_verified'] is False


def test_classify_uses_frozen_same_source_target_pool_and_actual_consensus_excludes_self(monkeypatch):
    centers = ((100.5, 100.5), (80.5, 80.5), (120.5, 80.5), (80.5, 120.5), (120.5, 120.5))
    plan, areas, _ = fixture(centers=centers)
    areas[1]['region_legs'][0].update(status='ACCEPTED', reasons=[], cycle_error_px=0.)
    records = review.collect_measurements(plan, areas)
    for key, value in [('source_input_index', 99), ('source_decode_index', 9), ('target_decode_index', 9)]:
        foreign = deepcopy(records[1])
        foreign['region_id'] = 'foreign-'+key
        foreign[key] = value
        records.append(foreign)
    before = deepcopy(records)
    actual_assess = review.assess_motion
    seen = []
    def assess(candidate_id, center, controls, matrix, pool, *, budget):
        seen.append((candidate_id, deepcopy(pool)))
        assert len(controls) == 5
        return actual_assess(candidate_id, center, controls, matrix, pool, budget=budget)
    monkeypatch.setattr(review, 'assess_motion', assess)
    classified = review.classify_candidates(records)
    candidate = classified[0]
    assert candidate['neighborhood']['status'] == 'NEIGHBOR_CONSISTENT_CANDIDATE'
    assert set(candidate['neighborhood']['selected_neighbor_ids']) == {'area-1', 'area-2', 'area-3', 'area-4'}
    assert 'area-0' not in candidate['neighborhood']['selected_neighbor_ids']
    for _, pool in seen:
        assert [r['region_id'] for r in pool] == ['area-0', 'area-1', 'area-2', 'area-3', 'area-4']
        assert [r['strict_accepted'] for r in pool] == [False, True, False, False, False]
    assert records == before
    assert candidate['strict_status'] == 'REJECTED'
    assert candidate['geometry_accepted'] is False and candidate['identity_verified'] is False


def candidate_verdict(*args, **kwargs):
    return dict(status='NEIGHBOR_CONSISTENT_CANDIDATE', reasons=['TEST_NEIGHBOR_VERDICT'],
                matrix_global_raw3x3=[[1., 0., 80.], [0., 1., 0.], [0., 0., 1.]],
                geometry_accepted=False, identity_verified=False)


def test_continuation_only_next_two_frames_anchors_actual_seed_and_keeps_original_template(monkeypatch):
    plan, areas, decoded = fixture()
    seed = review.collect_measurements(plan, areas)[0]
    seed['neighborhood'] = candidate_verdict()
    calls = []
    def align(template, search, tm, sm, cv, *, budget):
        calls.append((template.copy(), search.copy()))
        return photo([[1., 0., 52. if len(calls)%2 else 32.], [0., 1., 32.]])
    monkeypatch.setattr(review, 'align_area', align)
    out = review._continue(plan, plan['regions'][0], seed, decoded, object(), budget=lambda: None)
    assert out['targets'] == [2, 3] and len(calls) == 4
    assert [leg['target_decode_index'] for leg in out['legs']] == [2, 3]
    assert [leg['search_anchor_raw_xy'] for leg in out['legs']] == [[120.5, 100.5], [140.5, 100.5]]
    assert [leg['forward']['search_roi']['left'] for leg in out['legs']] == [72, 92]
    for ordinal, dx in enumerate((40., 60.)):
        np.testing.assert_array_equal(calls[ordinal*2][0], decoded[0]['gray'][84:117, 84:117])
        np.testing.assert_allclose(out['legs'][ordinal]['forward']['matrix_global_raw3x3'],
                                   [[1., 0., dx], [0., 1., 0.], [0., 0., 1.]])
        assert out['legs'][ordinal]['cycle_error_px'] < 1e-10
    assert out['actual_extra_targets_accepted'] == 2 and out['seed_is_provisional'] is True
    assert out['geometry_accepted'] is False and out['identity_verified'] is False
    assert seed['strict_status'] == 'REJECTED'


def test_review_freezes_plan_and_pool_before_first_real_registration(tmp_path, monkeypatch):
    plan, areas, decoded = fixture()
    strict_before = deepcopy(areas)
    output = tmp_path/'review'
    monkeypatch.setattr(review, 'assess_motion', candidate_verdict)
    calls = []
    frozen = []
    def align(*args, **kwargs):
        calls.append(1)
        assert (output/'PLAN.json').is_file() and (output/'CANDIDATE_POOL.json').is_file()
        frozen.append((output/'PLAN.json').read_bytes())
        saved = json.loads((output/'CANDIDATE_POOL.json').read_text(encoding='utf-8'))
        assert saved[0]['strict_status'] == 'REJECTED' and saved[0]['neighborhood'] is None
        assert saved[0]['measurement']['target_center_raw_xy'] == [120.5, 100.5]
        return photo([[1., 0., 52. if len(calls)%2 else 32.], [0., 1., 32.]])
    monkeypatch.setattr(review, 'align_area', align)
    result, artifacts = review.review_neighborhood(plan, areas, decoded, output, object())
    assert len(calls) == 4 and len(set(frozen)) == 1
    assert result['status'] == 'NEIGHBORHOOD_REVIEW_COMPLETE'
    assert result['geometry_accepted'] is False and result['identity_verified'] is False
    assert result['new_model_points'] == 0 and areas == strict_before
    assert (output/'RESULT.json') in artifacts


def test_budget_failure_retains_interrupted_alignment_and_unknown_later_frames(tmp_path, monkeypatch):
    plan, areas, decoded = fixture()
    monkeypatch.setattr(review, 'assess_motion', candidate_verdict)
    partial = dict(status='BLOCKED', native_calls=dict(matchTemplate=1, findTransformECCWithMask=0),
                   matrix_cv2x3=None, reasons=['TEST_INTERRUPTED_BEFORE_ECC'])
    def align(*args, **kwargs):
        error = TimeoutError('shared budget')
        error.partial_result = deepcopy(partial)
        raise error
    monkeypatch.setattr(review, 'align_area', align)
    output = tmp_path/'interrupted'
    with pytest.raises(TimeoutError, match='shared budget'):
        review.review_neighborhood(plan, areas, decoded, output, object())
    result = json.loads((output/'FAILURE.json').read_text(encoding='utf-8'))
    assert result['status'] == 'RESOURCE_LIMIT' and not (output/'RESULT.json').exists()
    row = result['records'][0]
    assert row['strict_status'] == 'REJECTED'
    continuation = row['continuation']
    assert continuation['status'] == 'INTERRUPTED'
    assert continuation['interrupted_alignment'] == partial
    assert continuation['native_calls'] == partial['native_calls']
    assert result['native_calls'] == partial['native_calls']
    assert continuation['targets'] == [2, 3]
    assert [leg['target_decode_index'] for leg in continuation['legs']] == [2]
    assert continuation['legs'][0]['status'] == 'IN_PROGRESS'
    assert continuation['legs'][0]['forward'] is None
    assert continuation['actual_extra_targets_accepted'] == 0
    assert result['geometry_accepted'] is False and result['new_model_points'] == 0


def test_collection_budget_preserves_actual_prefix_without_claiming_missing_results(tmp_path):
    plan, areas, decoded = fixture(centers=((100.5, 100.5), (120.5, 100.5)))
    calls = []
    def budget():
        calls.append(1)
        # Initial review, first area and first leg have completed before area two.
        if len(calls) == 4:
            raise TimeoutError('collection deadline')
    output = tmp_path/'collection-interrupted'
    with pytest.raises(TimeoutError, match='collection deadline') as error:
        review.review_neighborhood(plan, areas, decoded, output, object(), budget=budget)
    saved = json.loads((output/'FAILURE.json').read_text(encoding='utf-8'))
    assert saved == error.value.neighborhood_result
    assert saved['status'] == 'RESOURCE_LIMIT'
    assert len(saved['records']) == 1
    row = saved['records'][0]
    assert row['region_id'] == 'area-0' and row['measurement'] is not None
    assert row['neighborhood'] is None and row['continuation'] is None
    assert not (output/'PLAN.json').exists() and not (output/'RESULT.json').exists()


def test_classification_budget_preserves_complete_frozen_pool_as_unknown(tmp_path, monkeypatch):
    plan, areas, decoded = fixture(centers=((100.5, 100.5), (120.5, 100.5)))
    def assess(*args, **kwargs):
        raise TimeoutError('classification deadline')
    monkeypatch.setattr(review, 'assess_motion', assess)
    output = tmp_path/'classification-interrupted'
    with pytest.raises(TimeoutError, match='classification deadline'):
        review.review_neighborhood(plan, areas, decoded, output, object())
    pool = json.loads((output/'CANDIDATE_POOL.json').read_text(encoding='utf-8'))
    failure = json.loads((output/'FAILURE.json').read_text(encoding='utf-8'))
    assert failure['records'] == pool and len(pool) == 2
    assert all(row['neighborhood'] is None and row['continuation'] is None for row in pool)
    assert failure['native_calls'] == dict(matchTemplate=0, findTransformECCWithMask=0)
    assert not (output/'PLAN.json').exists()
