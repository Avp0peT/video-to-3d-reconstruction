import numpy as np
import pytest

from integrations.video_masks import boundary_geometry_evidence as kernel


def run(vertices=None, faces=None, *, depth=2., mask=None, valid=None, gap=None, check=lambda: None):
    v = np.array([[8., 8., 1.], [18., 8., 1.], [8., 18., 1.]]) if vertices is None else np.asarray(vertices, float)
    f = np.array([[0, 1, 2]]) if faces is None else np.asarray(faces)
    m = np.zeros((32, 32), bool) if mask is None else mask
    va = np.ones_like(m) if valid is None else valid
    g = np.ones_like(m) if gap is None else gap
    d = np.full(m.shape, depth, float) if depth is not None and np.isscalar(depth) else depth
    return kernel.evaluate_rgb_view(v, f, K=np.eye(3), R=np.eye(3), C=np.zeros(3),
        mask=m, validity=va, depth=d, supported_gap=g, view_id='native', check=check)


def reason(row):
    return row['reason_names'][int(row['reason'][0])]


def test_front_of_saved_surface_is_diagnostic_conflict_with_complete_raster():
    row = run()
    assert row['status'].dtype == np.uint8
    assert row['status'].tolist() == [2] and row['negative'].tolist() == [True]
    assert row['actual_pixel_count'].tolist() == [66]
    assert row['depth_checked_pixel_count'].tolist() == [66]
    assert row['front_pixel_count'].tolist() == [66]
    assert row['whole_raster_supported'][0] and row['all_depth_known'][0]
    assert not row['geometry_changed'] and row['quality_status'] == 'NOT_ACCEPTED'


@pytest.mark.parametrize('depth, expected', [
    (1., 'SURFACE_MATCH_OR_INSUFFICIENT_CLEARANCE'), (.5, 'OCCLUDED_SAMPLES')])
def test_surface_match_and_occluded_face_are_unknown(depth, expected):
    row = run(depth=depth)
    assert row['status'].tolist() == [0] and not row['negative'][0]
    assert reason(row) == expected


def test_exact_two_percent_boundary_is_not_strictly_in_front():
    for z, expected in ((.98, False), (.98000001, False), (.97999999, True)):
        uv = np.array([[8., 8.], [18., 8.], [8., 18.]])
        row = run(np.column_stack((uv * z, np.full(3, z))), depth=1.)
        assert bool(row['negative'][0]) is expected


def test_perspective_correct_reciprocal_depth_not_linear_vertex_depth():
    uv = np.array([[4., 4.], [16., 4.], [4., 16.]])
    z = np.array([1., 2., 4.])
    vertices = np.column_stack((uv * z[:, None], z))
    yy, xx = np.indices((32, 32))
    b = (xx - 4.) / 12.; c = (yy - 4.) / 12.; a = 1 - b - c
    reciprocal = a + b / 2 + c / 4
    depth = np.full((32, 32), 100.)
    inside = (a >= 0) & (b >= 0) & (c >= 0)
    depth[inside] = 1.03 / reciprocal[inside]
    row = run(vertices, depth=depth)
    assert row['negative'][0]
    assert row['minimum_relative_clearance'][0] == pytest.approx(1 - 1 / 1.03)
    assert 1 / reciprocal[8, 8] == pytest.approx(12 / 7)


@pytest.mark.parametrize('missing', [0., np.nan, np.inf, -1.])
def test_one_unknown_depth_sample_cannot_be_filtered_out(missing):
    depth = np.full((32, 32), 2.); depth[10, 10] = missing
    row = run(depth=depth)
    assert reason(row) == 'UNKNOWN_DEPTH' and not row['negative'][0]
    assert row['depth_checked_pixel_count'][0] == row['actual_pixel_count'][0] - 1


def test_absent_depth_is_unknown_not_free_space():
    row = run(depth=None)
    assert reason(row) == 'MISSING_DEPTH'
    assert not row['raster_evaluated'][0] and not row['negative'][0]


def test_partial_rgb_support_does_not_cover_entire_triangle():
    gap = np.ones((32, 32), bool); gap[10, 10] = False
    row = run(gap=gap)
    assert reason(row) == 'PARTIAL_RGB_SUPPORT' and not row['negative'][0]
    assert row['supported_pixel_count'][0] == row['actual_pixel_count'][0] - 1


def test_invalid_interior_is_unknown_even_when_all_vertices_valid():
    valid = np.ones((32, 32), bool); valid[10, 10] = False
    row = run(valid=valid, gap=valid.copy())
    assert reason(row) == 'INVALID_RASTER_DOMAIN' and not row['negative'][0]


def test_supported_rgb_may_overlap_original_union_foreground():
    row = run(mask=np.ones((32, 32), bool), depth=2.)
    assert row['negative'][0] and not row['core_protected'][0]


def test_original_foreground_vertex_core_is_independent_of_rgb_scope():
    mask = np.zeros((32, 32), bool); mask[8, 8] = True
    gap = np.ones_like(mask); gap[8, 8] = False
    depth = np.full(mask.shape, 2.); depth[8, 8] = 1.
    row = run([[8.2, 8.2, 1.], [18.2, 8.2, 1.], [8.2, 18.2, 1.]],
              mask=mask, gap=gap, depth=depth)
    assert row['core_protected'][0] and row['protected_sample_count'][0] == 1
    assert not row['negative'][0] and reason(row) == 'CORE_PROTECTED'


def test_three_dimensional_centroid_is_projected_for_core_support():
    uv = np.array([[4., 4.], [16., 4.], [4., 16.]])
    z = np.array([1., 2., 4.]); vertices = np.column_stack((uv * z[:, None], z))
    center = vertices.mean(axis=0); xy = np.floor(center[:2] / center[2] + .5).astype(int)
    assert xy.tolist() != [8, 8]  # the mean projected corners is a different point
    mask = np.zeros((32, 32), bool); mask[xy[1], xy[0]] = True
    depth = np.full(mask.shape, 100.); depth[xy[1], xy[0]] = center[2]
    row = run(vertices, mask=mask, depth=depth)
    assert row['core_protected'][0] and row['protected_sample_count'][0] == 1
    assert not row['negative'][0]


def test_pixel_work_budget_is_bounding_box_work(monkeypatch):
    monkeypatch.setattr(kernel, 'MAX_RASTER_PIXELS_PER_FACE', 100)
    row = run()
    assert reason(row) == 'RASTER_BUDGET'
    assert not row['raster_evaluated'][0] and not row['negative'][0]


def test_no_pixel_centers_and_fewer_than_four_are_not_vacuous_success():
    for triangle in (
        [[4.1, 4.1, 1.], [20.1, 4.2, 1.], [20.1, 4.3, 1.]],
        [[4.8, 4.8, 1.], [5.2, 4.8, 1.], [5., 5.2, 1.]],
    ):
        row = run(triangle)
        assert reason(row) == 'INSUFFICIENT_PIXEL_CENTERS'
        assert row['actual_pixel_count'][0] < 4 and not row['negative'][0]


@pytest.mark.parametrize('triangle, expected', [
    ([[-.6, 8., 1.], [18., 8., 1.], [8., 18., 1.]], 'OUT_OF_FRAME'),
    ([[8., 8., -1.], [18., 8., 1.], [8., 18., 1.]], 'CAMERA_PLANE'),
    ([[8., 8., 1.], [12., 12., 1.], [18., 18., 1.]], 'DEGENERATE_PROJECTION'),
])
def test_camera_domain_and_degenerate_faces_stay_unknown(triangle, expected):
    row = run(triangle)
    assert reason(row) == expected and not row['negative'][0]


def test_winding_and_face_indices_preserved():
    row = run(faces=[[0, 1, 2], [2, 1, 0]])
    assert row['negative'].tolist() == [True, True]
    assert row['actual_pixel_count'].tolist() == [66, 66]


def test_inputs_immutable_and_deadline_propagates():
    vertices = np.array([[8., 8., 1.], [18., 8., 1.], [8., 18., 1.]])
    faces = np.array([[0, 1, 2]])
    mask = np.zeros((32, 32), bool); valid = np.ones_like(mask); gap = valid.copy()
    depth = np.full(mask.shape, 2.)
    arrays = [vertices, faces, mask, valid, gap, depth]
    copies = [a.copy() for a in arrays]
    run(vertices, faces, mask=mask, valid=valid, gap=gap, depth=depth)
    assert all(np.array_equal(a, b) for a, b in zip(arrays, copies))
    def stop():
        raise TimeoutError('synthetic deadline')
    with pytest.raises(TimeoutError):
        run(check=stop)


def test_invalid_input_rejected_and_empty_mesh_bounded(monkeypatch):
    with pytest.raises(ValueError):
        run(gap=np.ones((32, 32), np.uint8))
    with pytest.raises(ValueError):
        run(depth=np.ones((31, 32)))
    with pytest.raises(ValueError):
        run(depth=np.ones((32, 32), complex))
    invalid = np.ones((32, 32), bool); invalid[0, 0] = False
    with pytest.raises(ValueError):
        run(valid=invalid)  # supported RGB outside validity is not accepted
    monkeypatch.setattr(kernel, 'MAX_FACES', 1)
    with pytest.raises(ValueError):
        run(faces=[[0, 1, 2], [2, 1, 0]])
    row = run(faces=np.empty((0, 3), int))
    assert row['status'].shape == (0,) and not row['geometry_changed']
