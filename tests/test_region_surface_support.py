"""Small, synthetic checks for regional surface diagnostics; no native launch."""
import hashlib
import json

import numpy as np
import pytest

from integrations.video_masks import surface_support as support


def _all_samples(result):
    return [sample for row in result['sampled_faces'] + result['sampled_edges']
            for sample in row['samples']]


def _assert_diagnostic(result):
    assert result['status'] == 'DIAGNOSTIC_ONLY'
    assert result['geometry_accepted'] is False
    json.dumps(result, allow_nan=False)


def test_regional_distance_and_local_six_neighbours_match_brute_force():
    dense = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.],
                      [0., 0., 3.], [4., 0., 0.], [0., 5., 0.],
                      [0., 0., 6.], [7., 2., 1.]])
    query = np.array([.13, .17, .19])
    # All four triangle samples and edge midpoints have the same known position.
    result = support.sampled_support(np.tile(query, (3, 1)), [[0, 1, 2]], dense)
    distances = np.linalg.norm(dense - query, axis=1)
    nearest = int(np.argmin(distances))
    neighbour_distances = np.linalg.norm(dense - dense[nearest], axis=1)
    spacing = np.median(np.sort(np.delete(neighbour_distances, nearest))[:6])
    samples = _all_samples(result)
    assert samples
    for sample in samples:
        assert sample['distance'] == pytest.approx(distances[nearest])
        assert sample['local_spacing'] == pytest.approx(spacing)
        assert sample['ratio'] == pytest.approx(distances[nearest] / spacing)
    _assert_diagnostic(result)


@pytest.mark.parametrize('dense,reason', [
    ([], 'NO_DENSE_POINTS'),
    ([[0., 0., 0.], [1., 0., 0.]], 'INSUFFICIENT_DENSE_POINTS'),
    ([[0., 0., 0.]] * 8, 'DUPLICATE_DENSE_POINTS'),
])
def test_regional_missing_spacing_remains_unknown(dense, reason):
    result = support.sampled_support([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]],
                                     [[0, 1, 2]], dense)
    samples = _all_samples(result)
    assert samples
    for sample in samples:
        assert sample['local_spacing'] is None and sample['ratio'] is None
        assert sample['reason'] == reason
        if reason == 'NO_DENSE_POINTS':
            assert sample['distance'] is None
        else:
            assert sample['distance'] is not None and sample['distance'] >= 0
    _assert_diagnostic(result)


def test_regional_zero_computed_spacing_is_not_divided():
    # Distinct float coordinates whose squared distances underflow to zero.
    dense = np.column_stack((np.arange(7) * 1e-200, np.zeros((7, 2))))
    result = support.sampled_support([[0., 0., 0.]] * 3, [[0, 1, 2]], dense)
    for sample in _all_samples(result):
        assert sample['local_spacing'] is None and sample['ratio'] is None
        assert sample['reason'] == 'ZERO_LOCAL_SPACING'
    _assert_diagnostic(result)


def test_regional_small_mesh_sampling_is_bounded_and_deterministic():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [2., 2., 0.]])
    faces = [[0, 1, 2], [1, 3, 2]]
    dense = np.array([[i, i * .17, i * .11] for i in range(8)])
    first = support.sampled_support(vertices, faces, dense)
    assert first == support.sampled_support(vertices, faces, dense)
    sampling = first['sampling']
    assert sampling['samples_per_face'] == 4
    for key in ('largest_faces', 'control_faces', 'longest_edges'):
        assert 0 <= sampling[key] <= 32
    for group in ('largest', 'control'):
        rows = [row for row in first['sampled_faces'] if row['group'] == group]
        ids = [row['face_index'] for row in rows]
        assert len(ids) == len(set(ids)) <= len(faces)
        assert all(0 <= index < len(faces) for index in ids)
    assert all(row['group'] in ('largest', 'control') for row in first['sampled_faces'])
    assert all(len(row['samples']) == 4 for row in first['sampled_faces'])
    assert len(first['sampled_edges']) <= 5
    assert all(len(row['samples']) == 1 for row in first['sampled_edges'])
    _assert_diagnostic(first)


@pytest.mark.parametrize('vertices,faces,dense', [
    ([[np.nan, 0., 0.]] * 3, [[0, 1, 2]], [[0., 0., 0.]]),
    ([[0., 0., 0.]] * 3, [[0, 1, 2]], [[np.inf, 0., 0.]]),
    ([[0., 0., 0.]] * 3, [[0, 1, 3]], [[0., 0., 0.]]),
    ([[0., 0., 0.]] * 3, [[-1, 1, 2]], [[0., 0., 0.]]),
    ([[0., 0., 0.]] * 3, [[0.5, 1., 2.]], [[0., 0., 0.]]),
])
def test_regional_invalid_arrays_rejected(vertices, faces, dense):
    with pytest.raises(ValueError):
        support.sampled_support(vertices, faces, dense)


def _write_ply(path, vertices, faces=()):
    lines = ['ply', 'format ascii 1.0', f'element vertex {len(vertices)}',
             'property double x', 'property double y', 'property double z']
    if faces:
        lines += [f'element face {len(faces)}', 'property list uchar int vertex_indices']
    lines += ['end_header']
    lines += [' '.join(map(str, point)) for point in vertices]
    lines += ['3 ' + ' '.join(map(str, face)) for face in faces]
    path.write_text('\n'.join(lines) + '\n', encoding='ascii')


def _binding(path, role, vertices, triangles):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'role': role, 'vertices': vertices, 'triangles': triangles}


@pytest.fixture
def regional_producer(tmp_path):
    source = tmp_path / 'regional'
    native = source / 'surface'
    native.mkdir(parents=True)
    dense = native / 'dense.ply'
    mesh = native / 'raw.ply'
    _write_ply(dense, [[i, i * .2, i * .1] for i in range(8)])
    _write_ply(mesh, [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    record = {'contract_version': 'mvs-surface-run-v1',
              'status': 'SURFACE_GENERATED_NOT_ACCEPTED', 'stages': [
                  {'name': 'dense', 'status': 'GEOMETRY_READBACK_PASSED',
                   'outputs': {'geometry': [_binding(dense, 'dense', 8, 0)]}},
                  {'name': 'mesh', 'status': 'GEOMETRY_READBACK_PASSED',
                   'outputs': {'geometry': [_binding(mesh, 'pre_cleanup_surface', 3, 1)]}}]}
    producer = native / 'result.json'
    producer.write_text(json.dumps(record), encoding='utf-8')
    return source, producer, dense, mesh, record


@pytest.mark.parametrize('direct_file', [False, True])
def test_regional_run_writes_only_new_report_and_keeps_sources(regional_producer, direct_file):
    source, producer, dense, mesh, _ = regional_producer
    originals = {path: path.read_bytes() for path in (producer, dense, mesh)}
    output = source / 'support_audit'
    result = support.run(producer if direct_file else source, output)
    _assert_diagnostic(result)
    assert result['source_unchanged'] is True
    for name in ('RESULT.json', 'index.html', 'candidates.png'):
        assert (output / name).is_file() and (output / name).stat().st_size > 0
    persisted = json.loads((output / 'RESULT.json').read_text(encoding='utf-8'))
    _assert_diagnostic(persisted)
    assert persisted['source_unchanged'] is True
    assert (output / 'candidates.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.parametrize('fault', [
    'producer_status', 'stage_status', 'missing_role', 'duplicate_role',
    'wrong_sha', 'wrong_bytes', 'wrong_vertex_count', 'wrong_triangle_count', 'source_drift',
])
def test_regional_producer_contract_and_bindings_fail_closed(regional_producer, fault):
    source, producer, dense, mesh, record = regional_producer
    geometry = record['stages'][0]['outputs']['geometry']
    if fault == 'producer_status':
        record['status'] = 'SUCCESS'
    elif fault == 'stage_status':
        record['stages'][0]['status'] = 'PROCESS_COMPLETED'
    elif fault == 'missing_role':
        geometry[0]['role'] = 'unknown'
    elif fault == 'duplicate_role':
        geometry.append(dict(geometry[0]))
    elif fault == 'wrong_sha':
        geometry[0]['sha256'] = '0' * 64
    elif fault == 'wrong_bytes':
        geometry[0]['bytes'] += 1
    elif fault == 'wrong_vertex_count':
        geometry[0]['vertices'] += 1
    elif fault == 'wrong_triangle_count':
        record['stages'][1]['outputs']['geometry'][0]['triangles'] += 1
    elif fault == 'source_drift':
        dense.write_bytes(dense.read_bytes() + b'\n')
    producer.write_text(json.dumps(record), encoding='utf-8')
    originals = {path: path.read_bytes() for path in (producer, dense, mesh)}
    with pytest.raises((ValueError, RuntimeError)):
        support.run(source, source / 'support_audit')
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_regional_existing_output_is_not_overwritten(regional_producer):
    source, producer, dense, mesh, _ = regional_producer
    output = source / 'support_audit'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('keep exactly', encoding='ascii')
    originals = {path: path.read_bytes() for path in (producer, dense, mesh, sentinel)}
    with pytest.raises((FileExistsError, ValueError)):
        support.run(source, output)
    assert set(output.iterdir()) == {sentinel}
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_regional_source_ancestor_output_refused(regional_producer):
    source, producer, dense, mesh, _ = regional_producer
    originals = {path: path.read_bytes() for path in (producer, dense, mesh)}
    with pytest.raises((FileExistsError, ValueError)):
        support.run(source, source.parent)
    assert all(path.read_bytes() == data for path, data in originals.items())
