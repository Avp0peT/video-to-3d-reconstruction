import hashlib
import json
import numpy as np
import pytest
from integrations.demo_pipeline import surface_support as s


def test_local_triangle_and_gap_face_sample_distances():
    dense = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [.05, .05, 0], [.03, .03, 0]])
    vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [10, 0, 0], [10, 10, 0]])
    result = s.diagnose_arrays(dense, vertices, [[0, 1, 2], [0, 3, 4]])
    values = result['samples']['ordinary']['nearest_dense_distances']
    assert max(values[0]) < .08 and min(values[1]) > 4
    assert result['top_one_percent']['face_ids'] == [1]
    assert result['top_one_percent']['area_fraction'] > .99
    assert result['geometry_acceptance'] is False
    json.dumps(result, allow_nan=False)


def test_zero_spacing_and_zero_area_not_divided():
    result = s.diagnose_arrays([[0, 0, 0], [0, 0, 0]], [[0, 0, 0]] * 3, [[0, 1, 2]])
    assert result['dense_spacing']['summary']['p50'] == 0
    assert result['samples']['ordinary']['distance_over_median_dense_spacing_summary'] is None
    assert result['top_one_percent']['area_fraction'] is None


def test_empty_dense_or_empty_faces_are_not_accepted_geometry():
    result = s.diagnose_arrays([], [[0, 0, 0]] * 3, [[0, 1, 2]])
    assert result['samples']['ordinary']['status'] == 'NO_DENSE_SUPPORT_REFERENCE'
    assert result['samples']['ordinary']['nearest_dense_distances'] is None
    empty = s.diagnose_arrays([], [], [])
    assert empty['face_count'] == 0 and empty['geometry_acceptance'] is False
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize('dense,vertices,faces', [
    ([[np.nan, 0, 0]], [[0, 0, 0]] * 3, [[0, 1, 2]]),
    ([[0, 0, 0]], [[np.inf, 0, 0]] * 3, [[0, 1, 2]]),
    ([[0, 0, 0]], [[0, 0, 0]] * 3, [[0, 1, 3]]),
    ([[0, 0, 0]], [[0, 0, 0]] * 3, [[0, -1, 2]]),
    ([[0, 0, 0]], [[0, 0, 0]] * 3, [[0., 1., 2.]]),
])
def test_invalid_input_refused(dense, vertices, faces):
    with pytest.raises(ValueError):
        s.diagnose_arrays(dense, vertices, faces)


def test_operation_budget_before_nearest(monkeypatch):
    monkeypatch.setattr(s, 'MAX_OPERATIONS', 100_000_000)
    monkeypatch.setattr(s, '_nearest', lambda *a, **k: pytest.fail('budget bypassed'))
    with pytest.raises(ValueError, match='operation budget'):
        s.diagnose_arrays(np.zeros((100000, 3)), np.zeros((3, 3)), np.tile([[0, 1, 2]], (20000, 1)))


def test_count_budget():
    with pytest.raises(ValueError, match='count budget'):
        s.diagnose_arrays(np.zeros((100001, 3)), [[0, 0, 0]] * 3, [[0, 1, 2]])


def test_deterministic_samples_and_top_exact_cardinality():
    result = s.diagnose_arrays([[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.tile([[0, 1, 2]], (300, 1)))
    assert len(result['samples']['ordinary']['face_ids']) == 256
    assert result['top_one_percent']['face_ids'] == [0, 1, 2]
    assert result['dense_spacing']['point_ids'] == [0, 1]


def test_sources_immutable_and_exclusive_report(tmp_path):
    sources = tmp_path / 'sources'
    sources.mkdir()
    mesh = sources / 'mesh.ply'
    dense = sources / 'dense.ply'
    mesh.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty double x\nproperty double y\nproperty double z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n')
    dense.write_text('ply\nformat ascii 1.0\nelement vertex 2\nproperty double x\nproperty double y\nproperty double z\nend_header\n0 0 0\n1 1 0\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = [digest(mesh), digest(dense)]
    output = tmp_path / 'diagnostic'
    record = s.review_surface_sources(mesh, dense, output)
    assert before == [digest(mesh), digest(dense)]
    assert record['geometry_acceptance'] is False
    assert record['result']['sha256'] == digest(output / 'RESULT.json')
    assert '不是可见性' in (output / 'index.html').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        s.review_surface_sources(mesh, dense, output)


def producer_fixture(tmp_path):
    mesh = tmp_path / 'arbitrary_mesh.bin'
    dense = tmp_path / 'arbitrary_cloud.bin'
    mesh.write_bytes(b'mesh')
    dense.write_bytes(b'dense')
    producer = tmp_path / 'producer.json'
    record = {'contract_version': 'mvs-surface-run-v1', 'stages': [
        {'name': 'dense', 'outputs': {'geometry': [dict(s._binding(dense), role='dense')]}},
        {'name': 'mesh', 'outputs': {'geometry': [dict(s._binding(mesh), role='pre_cleanup_surface')]}}]}
    producer.write_text(json.dumps(record))
    discovery = {'stages': {'surface': {'artifacts': [s._binding(mesh)], 'evidence_paths': [str(producer)]}}}
    return mesh, dense, producer, record, discovery


def test_discover_same_producer_and_append_verified_refs(tmp_path):
    mesh, dense, producer, record, discovery = producer_fixture(tmp_path)
    refs = []
    found = s.discover_surface_sources(discovery, refs)
    assert found['mesh_path'] == str(mesh.resolve()) and found['dense_path'] == str(dense.resolve())
    assert found['producer_record'] == str(producer.resolve()) and len(refs) == 3


def test_discover_missing_role_no_guess_or_partial_refs(tmp_path):
    _, _, producer, record, discovery = producer_fixture(tmp_path)
    record['stages'] = record['stages'][1:]
    producer.write_text(json.dumps(record))
    refs = []
    assert s.discover_surface_sources(discovery, refs) is None and refs == []
    assert s.discover_surface_sources({}, refs) is None


@pytest.mark.parametrize('fault', ['dense_drift', 'mesh_drift', 'foreign_mesh'])
def test_discover_drift_or_foreign_producer_refused(tmp_path, fault):
    mesh, dense, producer, record, discovery = producer_fixture(tmp_path)
    if fault == 'dense_drift':
        dense.write_bytes(b'changed')
    elif fault == 'mesh_drift':
        mesh.write_bytes(b'changed')
    else:
        other = tmp_path / 'other_mesh.bin'
        other.write_bytes(b'other')
        record['stages'][1]['outputs']['geometry'][0] = dict(s._binding(other), role='pre_cleanup_surface')
        producer.write_text(json.dumps(record))
    refs = []
    with pytest.raises(ValueError):
        s.discover_surface_sources(discovery, refs)
    assert refs == []


def test_binding_budget_before_read_and_bool_xyz(tmp_path, monkeypatch):
    big = tmp_path / 'large.bin'
    with big.open('wb') as stream:
        stream.truncate((64 << 20) + 1)
    monkeypatch.setattr(type(big), 'open', lambda *a, **k: pytest.fail('read before size budget'))
    with pytest.raises(ValueError, match='byte budget'):
        s._binding(big)
    with pytest.raises(ValueError, match='bool'):
        s.diagnose_arrays([[True, 0, 0]], [[0, 0, 0]] * 3, [[0, 1, 2]])
