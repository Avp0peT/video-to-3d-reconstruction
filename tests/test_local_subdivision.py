"""Synthetic CPU fixtures for one-round conforming subdivision."""
import numpy as np
import pytest

from integrations.video_masks.local_subdivision import subdivide_conforming


def fixture():
    vertices = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0],
                         [2, -2, 0], [4, 4, 0], [-2, 2, 0]], float)
    vertices[:, 2] = .3 * vertices[:, 0] - .2 * vertices[:, 1] + 3
    faces = np.array([[0, 1, 2], [1, 0, 3], [2, 1, 4], [0, 2, 5]], np.int64)
    parents = np.array([90, 120, 240, 480], np.int64)
    eligible = np.array([True, False, False, False])
    return dict(vertices=vertices, faces=faces, parent_ids=parents, eligible=eligible,
                refine=np.zeros(4, bool))


def assert_geometry_partition(before, result):
    old_tri = before['vertices'][before['faces']]
    old_normals = np.cross(old_tri[:, 1] - old_tri[:, 0], old_tri[:, 2] - old_tri[:, 0])
    tri = result['vertices'][result['faces']]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    source = result['source_face_indices']
    assert np.all(np.einsum('ij,ij->i', normals, old_normals[source]) > 0)
    np.testing.assert_allclose(
        np.einsum('ijk,ik->ij', tri - old_tri[source, :1], old_normals[source]), 0, atol=1e-12)
    for i in range(len(old_tri)):
        np.testing.assert_allclose(normals[source == i].sum(axis=0), old_normals[i], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(normals[source == i], axis=1).sum(),
                                   np.linalg.norm(old_normals[i]), atol=1e-12)


def assert_no_hanging_vertices(result):
    v, f = result['vertices'], result['faces']
    edges = np.unique(np.sort(f[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0)
    used = np.unique(f)
    for a, b in edges:
        delta = v[b] - v[a]
        length2 = delta @ delta
        assert length2 > 0
        t = ((v[used] - v[a]) @ delta) / length2
        distance = np.linalg.norm(v[used] - (v[a] + t[:, None] * delta), axis=1)
        assert not np.any((t > 1e-10) & (t < 1 - 1e-10) & (distance < 1e-10))


@pytest.mark.parametrize('count', [0, 1, 2, 3])
@pytest.mark.parametrize('rotation', [0, 1, 2])
@pytest.mark.parametrize('reverse', [False, True])
def test_all_templates_keep_winding_plane_area_and_eligibility(count, rotation, reverse):
    args = fixture()
    args['faces'][0] = np.roll(args['faces'][0], rotation)
    if reverse:
        args['faces'] = args['faces'][:, ::-1].copy()
    args['refine'][1:count + 1] = True
    result = subdivide_conforming(**args)
    assert np.count_nonzero(result['source_face_indices'] == 0) == count + 1
    np.testing.assert_array_equal(result['parent_ids'], args['parent_ids'][result['source_face_indices']])
    np.testing.assert_array_equal(result['eligible'], args['eligible'][result['source_face_indices']])
    assert_geometry_partition(args, result)
    assert_no_hanging_vertices(result)


def test_single_selected_face_closes_shared_edge_without_inheriting_eligibility():
    args = fixture()
    args['refine'][0] = True
    result = subdivide_conforming(**args)
    assert np.bincount(result['source_face_indices']).tolist() == [4, 2, 2, 2]
    assert result['eligible'].sum() == 4
    lineage = result['midpoint_lineage']
    np.testing.assert_array_equal(lineage, [[6, 0, 1], [7, 0, 2], [8, 1, 2]])
    assert np.count_nonzero(result['faces'] == 6) > 2
    assert_no_hanging_vertices(result)


def test_rgb_midpoint_lineage_and_input_immutability():
    args = fixture()
    args['refine'][0] = True
    args['colors'] = np.array([[0, 255, 1], [1, 254, 2], [255, 255, 255],
                               [3, 4, 5], [6, 7, 8], [9, 10, 11]], np.uint8)
    copies = {k: a.copy() for k, a in args.items()}
    for a in args.values():
        a.setflags(write=False)
    result = subdivide_conforming(**args)
    for key, value in args.items():
        np.testing.assert_array_equal(value, copies[key])
    np.testing.assert_array_equal(result['vertices'][:6], args['vertices'])
    np.testing.assert_array_equal(result['colors'][:6], args['colors'])
    for new, a, b in result['midpoint_lineage']:
        np.testing.assert_array_equal(result['vertices'][new], (args['vertices'][a] + args['vertices'][b]) / 2)
        expected = (args['colors'][a].astype(np.uint16) + args['colors'][b] + 1) // 2
        np.testing.assert_array_equal(result['colors'][new], expected)
    assert result['midpoint_kind'] == 'INTERPOLATED_NOT_MEASURED'
    assert result['geometry_accepted'] is False


def test_second_round_can_refine_green_children_and_rebuild_current_edge_map():
    args = fixture()
    args['refine'][0] = True
    first = subdivide_conforming(**args)
    second_args = {key: first[key] for key in ('vertices', 'faces', 'parent_ids', 'eligible')}
    # Explicitly refine a previous green closure child; its eligibility remains false.
    second_args['refine'] = np.zeros(len(first['faces']), bool)
    second_args['refine'][np.flatnonzero(first['parent_ids'] == 120)[0]] = True
    second = subdivide_conforming(**second_args)
    assert_no_hanging_vertices(second)
    assert_geometry_partition(second_args, second)
    assert not second['eligible'][second['parent_ids'] == 120].any()
    assert set(second['parent_ids']) == set(args['parent_ids'])


def test_three_rounds_refining_eligible_descendants_stay_conforming():
    original = fixture()
    args = original
    for _ in range(3):
        args['refine'] = args['eligible'].copy()
        result = subdivide_conforming(**args)
        assert_geometry_partition(args, result)
        assert_no_hanging_vertices(result)
        args = {key: result[key] for key in ('vertices', 'faces', 'parent_ids', 'eligible')}
    assert np.count_nonzero(result['parent_ids'] == 90) == 64
    assert result['eligible'].sum() == 64


@pytest.mark.parametrize('budget', [{'max_faces': 9}, {'max_vertices': 8}, {'max_faces': 3}])
def test_budget_rejection_is_atomic_and_counts_closure(budget):
    args = fixture()
    args['refine'][0] = True  # Four children plus three two-child neighbors: 10, not 7.
    before = {key: value.copy() for key, value in args.items()}
    with pytest.raises(ValueError, match='budget'):
        subdivide_conforming(**args, **budget)
    for key in args:
        np.testing.assert_array_equal(args[key], before[key])


def test_exact_budget_allowed_and_results_deterministic():
    args = fixture()
    args['refine'][0] = True
    one = subdivide_conforming(**args, max_faces=10, max_vertices=9)
    two = subdivide_conforming(**args, max_faces=10, max_vertices=9)
    for key, value in one.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, two[key])
        else:
            assert value == two[key]


def test_noop_returns_independent_arrays_and_empty_lineage():
    args = fixture()
    result = subdivide_conforming(**args)
    for key in ('vertices', 'faces', 'parent_ids', 'eligible'):
        np.testing.assert_array_equal(result[key], args[key])
        assert not np.shares_memory(result[key], args[key])
    assert result['midpoint_lineage'].shape == (0, 3)
    assert result['colors'] is None


def test_empty_mesh_supported():
    result = subdivide_conforming(np.empty((0, 3)), np.empty((0, 3), np.int64),
                                 np.empty(0, np.int64), np.empty(0, bool), np.empty(0, bool))
    assert result['vertices'].shape == result['faces'].shape == (0, 3)
    assert result['midpoint_lineage'].shape == (0, 3)
    assert result['source_face_indices'].shape == (0,)


@pytest.mark.parametrize('key,value', [
    ('vertices', [[0, 0, np.nan]]), ('vertices', [[0, 0, np.inf]]),
    ('vertices', np.zeros((6, 2))), ('vertices', np.ones((6, 3), bool)),
    ('vertices', np.ones((6, 3), complex)),
    ('faces', np.array([[0., 1., 2.]])), ('faces', np.array([[0, 1, 6]])),
    ('faces', np.array([[0, 1, -1]])), ('faces', np.array([[0, 1, 1]])),
    ('faces', np.ones((4, 3), bool)), ('faces', np.zeros((4, 2), np.int64)),
    ('parent_ids', np.zeros(4, float)), ('parent_ids', np.array([1, 2, 3, -1])),
    ('parent_ids', np.ones(3, np.int64)),
    ('parent_ids', np.full(4, np.iinfo(np.uint64).max, np.uint64)),
    ('eligible', np.ones(4, int)), ('eligible', np.ones((4, 1), bool)),
    ('refine', np.ones(4, int)), ('refine', np.ones(3, bool)),
    ('colors', np.ones((6, 3), float)), ('colors', np.ones((6, 4), np.uint8)),
    ('max_faces', True), ('max_faces', 0), ('max_vertices', -1), ('max_vertices', 10.5),
])
def test_invalid_inputs_rejected(key, value):
    args = fixture()
    args[key] = value
    with pytest.raises(ValueError):
        subdivide_conforming(**args)


def test_midpoint_math_does_not_overflow_finite_large_coordinates():
    huge = np.finfo(float).max * .75
    result = subdivide_conforming(np.full((3, 3), huge), np.array([[0, 1, 2]]),
                                 np.array([1]), np.array([True]), np.array([True]))
    assert np.isfinite(result['vertices']).all()


def test_all_incident_faces_of_nonmanifold_edge_use_same_midpoint():
    v = np.array([[0, 0, 0], [4, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2]], float)
    f = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    result = subdivide_conforming(v, f, np.arange(3), np.array([True, False, False]),
                                 np.array([True, False, False]))
    assert np.bincount(result['source_face_indices']).tolist() == [4, 2, 2]
    shared = result['midpoint_lineage'][(result['midpoint_lineage'][:, 1:] == [0, 1]).all(axis=1)]
    assert len(shared) == 1
    for i in range(3):
        assert shared[0, 0] in result['faces'][result['source_face_indices'] == i]
