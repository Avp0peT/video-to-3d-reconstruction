import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from integrations.demo_pipeline import silhouette_review as s


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def ref(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': s._hash(path)}


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    run = root / 'outputs/source'
    base = run / 'geometry/surface_pipeline'
    (base / 'surface/work').mkdir(parents=True)
    (base / 'masks').mkdir()
    monkeypatch.setattr(s, 'ROOT', root)
    mesh = base / 'surface/work/__mvs_surface_raw.ply'
    mesh.write_text('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n-1 -1 2\n1 -1 2\n0 1 2\n3 0 1 2\n')
    scene_path = base / 'scene.mvs'
    scene_path.write_bytes(b'fixture_scene_bytes')
    K, R, C = [[2, 0, 2], [0, 2, 2], [0, 0, 1]], np.eye(3).tolist(), [0, 0, 0]
    scene_images, mapping, frames, plans = [], [], [], []
    io_root = base / 'io'
    io_root.mkdir()
    io_rows = []
    for idx in (0, 17):
        photo, mask, validity = [base / ('%s_%s.png' % (idx, role)) for role in ('photo', 'mask', 'validity')]
        Image.new('RGB', (5, 5), (90, 80, 70)).save(photo)
        canonical = io_root / photo.name
        canonical.write_bytes(photo.read_bytes())
        io_rows.append({'input_index': idx, 'frame_id': 'frame%s' % idx, 'status': 'OK', 'image_output': photo.name})
        binary = np.zeros((5, 5), dtype=np.uint8)
        binary[1:4, 1:4] = 255
        Image.fromarray(binary).save(mask)
        Image.new('L', (5, 5), 255).save(validity)
        camera = {'global_id': idx + 1, 'name': photo.name, 'K': K, 'R': R, 'C': C, 'width': 5, 'height': 5}
        scene_images.append(camera)
        name = '%s.png' % idx
        mapping.append(dict(camera, model_image=name, scene_name=photo.name, scene_global_id=idx + 1, actual_image=str(photo)))
        frames.append({'input_index': idx, 'frame_id': 'frame%s' % idx, 'source_image': str(canonical),
            'source_image_sha256': s._hash(canonical), 'status': 'OK', 'image_name': name, 'target_size': [5, 5],
            'target_camera': {'model': 'PINHOLE', 'params': [2, 2, 2.5, 2.5]},
            'mask': dict(ref(mask), absolute_path=str(mask)), 'validity': dict(ref(validity), absolute_path=str(validity)),
            'core_counts': {'foreground': 9, 'valid': 25}})
        plans.append({'image': camera, 'image_file': ref(photo), 'mask': ref(mask), 'validity': ref(validity)})
    scene = {'path': str(scene_path), 'images': scene_images}
    monkeypatch.setattr(s, 'read_scene', lambda _: scene)
    write(run / 'pipeline.json', {'schema': 'photo-demo-pipeline-v1', 'stages': {'surface': {
        'status': 'SUCCESS', 'selected_indices': [0, 17], 'missing_indices': [8], 'artifacts': [ref(mesh)]}}})
    write(io_root / 'run.json', {'status': 'SUCCESS'})
    (io_root / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in io_rows), encoding='utf-8')
    write(base / 'masks/manifest.json', {'status': 'SUCCESS', 'coverage_status': 'COMPLETE', 'unverified_edges': [],
        'inputs': {'t01_run': str(io_root), 'source_io': {str(io_root): {'path': str(io_root),
            'run_json': ref(io_root / 'run.json'), 'frames_jsonl': ref(io_root / 'frames.jsonl')}}},
        'image_domain': 'colmap_undistorted', 'frames': frames, 'scene': dict(ref(scene_path), mapping=mapping)})
    write(base / 'surface/plan.json', {'scene': ref(scene_path), 'frames': plans})
    write(base / 'surface/result.json', {'status': 'SURFACE_GENERATED_NOT_ACCEPTED', 'exit_code': 0, 'input_drift': []})
    return run, root / 'outputs/review', base


def test_hand_computed_projected_triangle():
    points = [[0, 0, 2], [2, 0, 2], [0, 2, 2]]
    assert np.array_equal(s.project(points, np.eye(3), [0, 0, 0], [[2, 0, 1], [0, 2, 1], [0, 0, 1]]), [[1, 1], [3, 1], [1, 3]])


@pytest.mark.parametrize('vertices', [[[0, 0, -1]], [[0, 0, 0]], [[0, 0, float('nan')]]])
def test_camera_plane_and_nonfinite_rejected(vertices):
    with pytest.raises(ValueError):
        s.project(vertices, np.eye(3), [0, 0, 0], np.eye(3))


def test_known_raster_and_order_invariance():
    uv = np.array([[1., 1.], [3., 1.], [1., 3.]])
    actual = s.raster(uv, [[0, 1, 2]], 5, 5, 1)
    assert actual.sum() == 6
    assert actual[1, 1] and actual[1, 3] and actual[3, 1] and not actual[3, 3]
    assert np.array_equal(actual, s.raster(uv, [[2, 1, 0]], 5, 5, 1))


def test_mask_metrics_respect_unknowns_and_no_empty_perfect_score():
    pred = np.array([[True, True], [False, False]])
    sam = np.array([[True, False], [True, False]])
    v = np.ones((2, 2), dtype=bool)
    assert s.metrics(pred, sam, v)['IoU'] == 1 / 3
    v[0, 1] = False
    assert s.metrics(pred, sam, v)['IoU'] == .5
    assert s.metrics(pred, sam, np.zeros_like(v))['IoU'] is None


def test_complete_review_all_views_and_no_quality_promotion(source):
    run, output, _ = source
    before = {str(p): s._hash(p) for p in run.rglob('*') if p.is_file()}
    result = s.review(run, output)
    assert result['completed_views'] == result['expected_views'] == 2
    assert [v['input_index'] for v in result['views']] == [0, 17]
    assert result['excluded_indices'] == [8]
    assert result['source_unchanged'] and not result['full_object_accepted']
    assert result['physical_accuracy'] == 'NOT_EVALUATED'
    assert (output / 'index.html').is_file()
    assert before == {str(p): s._hash(p) for p in run.rglob('*') if p.is_file()}
    for b in result['output_bindings']:
        assert s._hash(output / b['path']) == b['sha256']


@pytest.mark.parametrize('kind', ['mesh_bytes', 'mask_bytes', 'photo_bytes', 'duplicate_view', 'missing_frame',
                                  'pixel_center', 'unknown_foreground', 'false_exit', 'scene_hash', 'swap_indices'])
def test_identity_and_pixel_failures_cannot_produce_review(source, kind):
    run, output, base = source
    path = base / 'masks/manifest.json'
    doc = s._read(path)
    if kind == 'mesh_bytes':
        (base / 'surface/work/__mvs_surface_raw.ply').write_text('changed')
    elif kind in ('mask_bytes', 'photo_bytes'):
        (base / ('0_mask.png' if kind == 'mask_bytes' else '0_photo.png')).write_bytes(b'changed')
    elif kind == 'duplicate_view':
        doc['frames'][1]['input_index'] = 0
    elif kind == 'swap_indices':
        doc['frames'][0]['input_index'], doc['frames'][1]['input_index'] = 17, 0
    elif kind == 'missing_frame':
        doc['frames'].pop()
    elif kind == 'pixel_center':
        doc['frames'][0]['target_camera']['params'][2] = 2
    elif kind == 'unknown_foreground':
        vpath = base / '0_validity.png'
        Image.new('L', (5, 5), 0).save(vpath)
        doc['frames'][0]['validity'].update(ref(vpath))
        plan_path = base / 'surface/plan.json'
        plan = s._read(plan_path)
        plan['frames'][0]['validity'] = ref(vpath)
        write(plan_path, plan)
    elif kind == 'false_exit':
        result_path = base / 'surface/result.json'
        result = s._read(result_path)
        result['exit_code'] = False
        write(result_path, result)
    else:
        doc['scene']['sha256'] = '0' * 64
    write(path, doc)
    with pytest.raises(ValueError):
        s.review(run, output)
    assert not output.exists()


def test_output_no_overwrite_or_source_overlap(source):
    run, output, _ = source
    output.mkdir()
    with pytest.raises(ValueError):
        s.review(run, output)
    with pytest.raises(ValueError):
        s.review(run, run / 'review')


def test_late_drift_keeps_partial_evidence_and_raises(source, monkeypatch):
    run, output, _ = source
    monkeypatch.setattr(s.Bindings, 'unchanged', lambda _: False)
    with pytest.raises(ValueError, match='source changed'):
        s.review(run, output)
    result = s._read(output / 'RESULT.json')
    assert result['status'] == 'INCOMPLETE' and not result['source_unchanged']
    assert result['completed_views'] == 2
    assert not (output / 'index.html').exists()


def test_finalisation_deadline_cannot_leave_success(source, monkeypatch):
    run, output, _ = source
    monkeypatch.setattr(s.time, 'monotonic', lambda: 10.0 if (output / 'index.html').exists() else 0.0)
    with pytest.raises(TimeoutError):
        s.review(run, output, seconds=1)
    result = s._read(output / 'RESULT.json')
    assert result['status'] == 'INCOMPLETE'
    assert result['completed_views'] == 2
