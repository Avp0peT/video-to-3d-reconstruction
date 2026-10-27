import numpy as np
import pytest

from integrations.video_masks import face_mask_gate as g


def scene():
    uv = np.array([[8., 8.], [56., 8.], [8., 56.]])
    vertices = np.column_stack((uv * 10, np.full(3, 10.)))
    return vertices, np.array([[0, 1, 2]])


def evaluate(mask=None, depth=None, validity=None, vertices=None, faces=None, **kwargs):
    original, original_faces = scene()
    mask = np.ones((64, 64), bool) if mask is None else mask
    validity = np.ones_like(mask) if validity is None else validity
    depth = np.where(mask, 10., 0.) if depth is None else depth
    return g.evaluate_view(original if vertices is None else vertices,
                           original_faces if faces is None else faces,
                           K=np.eye(3), R=np.eye(3), C=np.zeros(3), mask=mask,
                           validity=validity, depth=depth, view_id='a', **kwargs)


def reason(result, idx=0):
    return result['reason_names'][int(result['reason'][idx])]


def test_foreground_vertices_and_centroid_do_not_hide_crossed_edge_gap():
    mask = np.ones((64, 64), bool)
    mask[3:14, 24:40] = False  # A gap around the long top edge; centroid stays foreground.
    assert mask[24, 24]
    result = evaluate(mask)
    assert result['status'][0] == g.CONFLICT
    assert result['depth_supported_vertices'][0] == 3
    assert result['edge_background_pixels'][0] >= 4
    assert result['total_pixels'][0] >= result['interior_pixels'][0]


def test_complete_interior_raster_finds_hole_with_all_edges_foreground():
    mask = np.ones((64, 64), bool)
    mask[20:32, 20:32] = False
    result = evaluate(mask)
    assert result['status'][0] == g.CONFLICT
    assert result['background_pixels'][0] >= 4
    assert result['edge_background_pixels'][0] == 0


def test_boundary_margin_does_not_treat_thin_background_as_conflict():
    mask = np.ones((64, 64), bool)
    mask[:, 28:32] = False
    assert evaluate(mask)['status'][0] == g.NO_CONFLICT
    assert evaluate(mask, policy=g.Policy(boundary_px=0))['status'][0] == g.CONFLICT


def test_full_bbox_proof_is_explicit_and_never_invents_sample_count():
    result = evaluate()
    assert result['status'][0] == g.NO_CONFLICT
    assert reason(result) == 'BOUNDING_BOX_NO_DEEP_BACKGROUND'
    assert not result['raster_evaluated'][0]
    assert result['total_pixels'][0] == 0
    assert result['bounding_box_pixels'][0] == 49 * 49


def test_missing_depth_neighborhood_support_and_occlusion_are_separate():
    vertices, faces = scene()
    missing = g.evaluate_view(vertices, faces, K=np.eye(3), R=np.eye(3), C=np.zeros(3),
                             mask=np.ones((64, 64), bool), validity=np.ones((64, 64), bool),
                             depth=None, view_id=0)
    assert reason(missing) == 'MISSING_DEPTH'
    depth = np.zeros((64, 64))
    depth[8, 9] = depth[8, 55] = 10.  # Neighbor matches for two vertices; third remains unknown.
    result = evaluate(depth=depth)
    assert result['depth_supported_vertices'][0] == 2
    assert result['status'][0] == g.NO_CONFLICT
    depth[8, 8] = 5.  # Central foreground occluder overrides adjacent accidental match.
    result = evaluate(depth=depth)
    assert result['status'][0] == g.UNKNOWN
    assert result['occluded_vertices'][0] == 1
    assert reason(result) == 'OCCLUDED_VERTEX'


def test_local_unknown_preserves_other_face_status_and_original_indices():
    vertices, _ = scene()
    extra = np.array([[100, 100, -10], [120, 100, -10], [100, 120, -10.]])
    vertices = np.vstack((vertices, extra))
    faces = np.array([[3, 4, 5], [0, 1, 2]])
    result = evaluate(vertices=vertices, faces=faces)
    assert result['status'].tolist() == [g.UNKNOWN, g.NO_CONFLICT]
    assert result['face_indices'].tolist() == [0, 1]
    assert reason(result, 0) == 'CAMERA_PLANE'
    valid = np.ones((64, 64), bool)
    valid[24, 24] = False
    result = evaluate(validity=valid)
    assert result['status'][0] == g.UNKNOWN
    assert reason(result) == 'INVALID_SAMPLE_DOMAIN'


def test_raster_budget_subpixel_and_out_of_frame_are_unknown():
    assert reason(evaluate(policy=g.Policy(max_raster_pixels_per_face=20))) == 'RASTER_BUDGET'
    vertices, faces = scene()
    vertices[:, :2] = [[100., 100.], [101., 100.], [100., 101.]]
    assert reason(evaluate(vertices=vertices, faces=faces)) == 'SUBPIXEL_OR_DEGENERATE'
    vertices[1, 0] = 800.
    assert reason(evaluate(vertices=vertices, faces=faces)) == 'OUT_OF_FRAME'


def test_angular_separation_unique_views_and_witness_pair():
    vertices, faces = scene()
    center = vertices.mean(axis=0)
    statuses = [dict(view_id=x, status=np.array([g.CONFLICT])) for x in ['a', 'b', 'c']]
    centers = {key: center + np.array([10*np.sin(np.deg2rad(angle)), 0, -10*np.cos(np.deg2rad(angle))])
               for key, angle in [('a', 0), ('b', 3), ('c', 12)]}
    result = g.aggregate_views(vertices, faces, statuses[:2], centers)
    assert not result['quarantine'][0]
    result = g.aggregate_views(vertices, faces, statuses, centers)
    assert result['quarantine'][0]
    assert result['witness_view_ids'][0] == ['a', 'c']
    assert result['max_conflict_angle_degrees'][0] == pytest.approx(12.)
    assert result['status'] == 'CANDIDATE_NOT_ACCEPTED'
    with pytest.raises(ValueError, match='duplicate'):
        g.aggregate_views(vertices, faces, [statuses[0], statuses[0]], centers)


def test_aggregate_retains_indices_negative_and_unknown_votes_without_acceptance():
    vertices, _ = scene()
    faces = np.tile([[0, 1, 2]], (3, 1))
    rows = [dict(view_id=4, status=[g.UNKNOWN, g.CONFLICT, g.NO_CONFLICT]),
            dict(view_id=9, status=[g.UNKNOWN, g.CONFLICT, g.CONFLICT])]
    center = vertices.mean(axis=0)
    result = g.aggregate_views(vertices, faces, rows, [center+[0, 0, -10], center+[10, 0, 0]])
    assert result['quarantine_face_indices'].tolist() == [1]
    assert result['kept_face_indices'].tolist() == [0, 2]
    assert result['unknown_views'].tolist() == [2, 0, 0]
    assert result['no_conflict_views'].tolist() == [0, 0, 1]
    assert result['kept_status'] == 'KEPT_NOT_ACCEPTED'


def test_deadline_is_propagated_and_inputs_not_mutated():
    mask = np.ones((64, 64), bool)
    depth = np.full((64, 64), 10.)
    before_mask, before_depth = mask.copy(), depth.copy()
    calls = []
    def stop():
        calls.append(1)
        if len(calls) == 3:
            raise TimeoutError('fixture deadline')
    with pytest.raises(TimeoutError):
        evaluate(mask=mask, depth=depth, check=stop)
    assert np.array_equal(mask, before_mask)
    assert np.array_equal(depth, before_depth)


@pytest.mark.parametrize('kwargs', [dict(boundary_px=float('nan')), dict(min_depth_supported_vertices=1),
                                  dict(min_angle_degrees=0), dict(min_background_pixels=True)])
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        g.Policy(**kwargs)
