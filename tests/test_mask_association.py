import json

import numpy as np
import pytest

from kozlov_lab.stereo.mask_association import associate_candidates

cv = pytest.importorskip("cv2")


def scene():
    mask = np.zeros((160, 160), np.uint8)
    mask[35:125, 35:125] = 255
    points = np.array([(x, y) for y in [40, 58, 79, 101, 120]
                       for x in [40, 58, 79, 101, 120]], dtype=float)
    before = cv.warpAffine(mask, np.float32([[1, 0, -2], [0, 1, 0]]), (160, 160))
    after = cv.warpAffine(mask, np.float32([[1, 0, 2], [0, 1, 0]]), (160, 160))
    # Freeze a spatially spread holdout by ID, including four corners and center.
    hold_positions = [0, 4, 12, 20, 24]
    ids = [None] * 25
    for position, ident in zip(hold_positions, range(0, 25, 5)):
        ids[position] = ident
    remaining = iter(i for i in range(25) if i % 5)
    ids = [next(remaining) if value is None else value for value in ids]
    return [before, after, points - [2, 0], points, points + [2, 0],
            ids, [dict(candidate_id="true", mask=mask)], cv]


def test_translated_true_beats_partial_allones_wrong_region_and_is_json():
    args = scene()
    partial = args[6][0]["mask"].copy()
    partial[:, 70:] = 0
    wrong = np.zeros_like(partial)
    wrong[5:30, 5:30] = 255
    args[6] += [dict(candidate_id="partial", mask=partial),
                dict(candidate_id="allones", mask=np.full_like(partial, 255)),
                dict(candidate_id="wrong", mask=wrong)]
    r = associate_candidates(*args)
    assert r["status"] == "TEMPORAL_CANDIDATE_SELECTED"
    assert r["selected_candidate_id"] == "true"
    assert r["holdout_track_ids"] == [0, 5, 10, 15, 20]
    assert r["references"]["before"]["holdout_max_px"] < 1e-8
    assert r["references"]["after"]["holdout_max_px"] < 1e-8
    assert all(not v["passed"] for v in r["candidates"][1:])
    json.dumps(r, allow_nan=False)
    assert not any(isinstance(v, np.ndarray) for v in r.values())


def test_equal_candidates_remain_ambiguous():
    args = scene()
    args[6].append(dict(candidate_id="duplicate_content", mask=args[6][0]["mask"].copy()))
    r = associate_candidates(*args)
    assert r["status"] == "UNRESOLVED"
    assert r["reasons"] == ["ambiguous_candidate_scores"]


def test_insufficient_points_and_clustered_anchors():
    args = scene()
    for i in [2, 3, 4, 5]: args[i] = args[i][:24]
    assert associate_candidates(*args)["reasons"] == ["insufficient_correspondences"]
    args = scene()
    p = np.array([(60+x, 60+y) for y in range(5) for x in range(5)], float)
    args[2:5] = [p-[2, 0], p, p+[2, 0]]
    r = associate_candidates(*args)
    assert r["status"] == "UNRESOLVED"
    assert "before:insufficient_reference_grid_coverage" in r["reasons"]


def test_heldout_error_is_not_fitted_away():
    args = scene()
    args[3][np.asarray(args[5]) % 5 == 0] += [12, 0]
    r = associate_candidates(*args)
    assert "before:holdout_residual_high" in r["reasons"]
    assert "after:holdout_residual_high" in r["reasons"]
    assert r["references"]["before"]["fit_inlier_fraction"] == 1
    assert r["references"]["before"]["holdout_max_px"] > 11


def test_holdout_frozen_by_sorted_ids_not_input_order():
    args = scene()
    baseline = associate_candidates(*args)
    order = np.random.default_rng(42).permutation(25)
    for i in (2, 3, 4): args[i] = args[i][order]
    args[5] = np.asarray(args[5])[order]
    shuffled = associate_candidates(*args)
    assert baseline == shuffled


@pytest.mark.parametrize("change", ["nan", "bool_points", "bool_ids", "float_ids", "duplicate_ids", "shape", "mask_dtype", "mask_values", "empty_mask", "null", "duplicate_candidates", "empty_points", "out_of_bounds"])
def test_bad_schema_rejected(change):
    args = scene()
    if change == "nan": args[2][0, 0] = np.nan
    if change == "bool_points": args[2] = args[2].astype(bool)
    if change == "bool_ids": args[5][0] = True
    if change == "float_ids": args[5][0] = 0.0
    if change == "duplicate_ids": args[5][1] = 0
    if change == "shape": args[1] = args[1][:100]
    if change == "mask_dtype": args[0] = args[0].astype(bool)
    if change == "mask_values": args[0][0, 0] = 1
    if change == "empty_mask": args[0][:] = 0
    if change == "null": args[2] = None
    if change == "duplicate_candidates": args[6] *= 2
    if change == "empty_points": args[2] = np.empty((0, 2))
    if change == "out_of_bounds": args[3][0, 0] = 200
    with pytest.raises(ValueError): associate_candidates(*args)


def test_no_candidate_preserves_no_candidate_status():
    args = scene()
    args[6] = []
    assert associate_candidates(*args)["status"] == "NO_CANDIDATE"


def test_warp_padding_is_unknown_and_can_block_selection():
    args = scene()
    # A large shift removes >10% of the image from the shared warp-valid domain.
    args[2] -= [18, 0]
    args[0] = cv.warpAffine(args[0], np.float32([[1, 0, -18], [0, 1, 0]]), (160, 160))
    r = associate_candidates(*args)
    assert r["status"] == "UNRESOLVED"
    assert r["common_valid_image_fraction"] < .90
    assert "insufficient_common_warp_valid_domain" in r["reasons"]


def test_original_mask_membership_required():
    args = scene()
    args[0][38:43, 36:41] = 0
    r = associate_candidates(*args)
    assert "before:reference_points_outside_original_mask" in r["reasons"]


def test_reference_prediction_shape_disagreement():
    args = scene()
    args[1][35:125, 35:80] = 0
    r = associate_candidates(*args)
    assert r["status"] == "UNRESOLVED"
    assert r["prediction_agreement_iou"] < .65


def test_shared_motion_hand_is_explicitly_unresolved_semantically():
    r = associate_candidates(*scene())
    # A pixel-identical co-moving hand is indistinguishable to this image test.
    assert r["status"] == "TEMPORAL_CANDIDATE_SELECTED"
    assert any("hand" in v and "cannot" in v for v in r["limitations"])


def test_no_input_mutation():
    args = scene()
    arrays = [args[i] for i in range(5)] + [args[6][0]["mask"]]
    snapshots = [v.copy() for v in arrays]
    associate_candidates(*args)
    for value, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(value, snapshot)


def test_scale_gate_is_independent_of_perfect_point_fit():
    args = scene()
    points = (args[3] - 80) / 1.5 + 80
    mask = np.zeros((160, 160), np.uint8)
    mask[50:111, 50:111] = 255
    args[0:3] = [mask, mask.copy(), points]
    args[4] = points.copy()
    r = associate_candidates(*args)
    assert r["references"]["before"]["holdout_max_px"] < .001
    assert r["references"]["before"]["scale"] > 1.4
    assert "before:scale_out_of_bounds" in r["reasons"]


def test_same_area_wrong_shape_does_not_pass_iou_gate():
    args = scene()
    wrong = np.zeros((160, 160), np.uint8)
    wrong[5:95, 5:95] = 255
    assert np.count_nonzero(wrong) == np.count_nonzero(args[6][0]["mask"])
    args[6] = [dict(candidate_id="equal_area_wrong", mask=wrong)]
    r = associate_candidates(*args)
    assert r["status"] == "UNRESOLVED"
    assert "candidate_reference_iou_low" in r["candidates"][0]["reasons"]


def test_collinear_holdout_is_rejected_despite_perfect_residuals():
    args = scene()
    args[5] = list(range(25))  # Every fifth observation has x=40.
    r = associate_candidates(*args)
    assert r['references']['before']['holdout_max_px'] < 1e-8
    assert 'before:holdout_insufficient_2d_spread' in r['reasons']
    assert r['status'] == 'UNRESOLVED'


def test_missing_one_holdout_cannot_hide_in_global_coverage():
    args = scene()
    # Only one of five holdout observations is absent: global coverage is 96%.
    x, y = args[3][0].astype(int)
    args[6][0]['mask'][y, x] = 0
    r = associate_candidates(*args)
    candidate = r['candidates'][0]
    assert candidate['query_point_coverage'] == .96
    assert candidate['fit_query_point_coverage'] == 1
    assert candidate['holdout_query_point_coverage'] == .8
    assert candidate['score'] > .99
    assert 'holdout_query_point_coverage_low' in candidate['reasons']
    assert r['status'] == 'UNRESOLVED'


def test_target_clipping_rejected_even_with_high_domain_and_perfect_iou():
    args = scene()
    mask = np.zeros((160, 160), np.uint8)
    mask[35:125, 70:160] = 255
    p = np.array([(x, y) for y in [40, 58, 79, 101, 120]
                  for x in [75, 93, 111, 129, 148]], float)
    args[0], args[1], args[2], args[3], args[4] = mask, mask.copy(), p, p+[8, 0], p.copy()
    args[6][0]['mask'] = cv.warpAffine(mask, np.float32([[1, 0, 8], [0, 1, 0]]), (160, 160))
    r = associate_candidates(*args)
    assert r['common_valid_image_fraction'] >= .90
    assert r['prediction_agreement_iou'] == 1
    assert r['candidates'][0]['score'] == 1
    assert 'before:source_foreground_clipped_by_warp' in r['reasons']
    assert r['references']['before']['source_foreground_retained_fraction'] < .95


def test_local_fitted_inlier_hull_cannot_support_distant_mask():
    args = scene()
    mask = np.zeros((640, 640), np.uint8)
    mask[20:620, 20:620] = 255
    p = np.array([(x, y) for y in [170, 240, 320, 400, 470]
                  for x in [170, 240, 320, 400, 470]], float)
    args[0], args[1] = mask, mask.copy()
    args[2], args[3], args[4] = p, p.copy(), p.copy()
    args[6][0]['mask'] = mask.copy()
    r = associate_candidates(*args)
    reference = r['references']['before']
    assert reference['fit_inlier_fraction'] == 1
    assert reference['groups']['fit_inliers']['occupied_grid_cell_count'] >= 6
    assert reference['source_foreground_beyond_inlier_hull_64px_fraction'] > .20
    assert 'before:source_foreground_extrapolation_unsupported' in r['reasons']
    assert r['status'] == 'UNRESOLVED'


def test_prediction_foreground_must_survive_other_reference_validity():
    args = scene()
    before, after = np.zeros((160, 160), np.uint8), np.zeros((160, 160), np.uint8)
    before[35:125, 120:156] = 255
    after[35:125, 128:160] = 255
    q = np.array([(x, y) for y in [40, 58, 79, 101, 120]
                  for x in [128, 134, 140, 146, 152]], float)
    args[0], args[1], args[2], args[3], args[4] = before, after, q-[4, 0], q, q+[4, 0]
    candidate = np.zeros((160, 160), np.uint8)
    candidate[35:125, 124:156] = 255
    args[6][0]['mask'] = candidate
    r = associate_candidates(*args)
    assert r['common_valid_image_fraction'] >= .90
    assert r['references']['before']['source_foreground_retained_fraction'] == 1
    assert r['references']['before']['prediction_common_valid_fraction'] < .95
    assert 'before:prediction_foreground_outside_common_valid_domain' in r['reasons']


def test_query_common_valid_fraction_gate_is_recorded_and_enforced():
    args = scene()
    mask = np.zeros((160, 160), np.uint8)
    mask[0:125, 0:125] = 255
    p = np.array([(x, y) for y in [4, 30, 60, 90, 120]
                  for x in [4, 30, 60, 90, 120]], float)
    q = p+[8, 0]
    q[0, 0] = q[4, 0] = 1
    args[0], args[1], args[2], args[3], args[4] = mask, mask.copy(), p, q, p.copy()
    args[6][0]['mask'] = mask.copy()
    r = associate_candidates(*args)
    assert r['query_common_valid_fraction'] < .95
    assert 'query_points_outside_common_valid_domain' in r['reasons']
