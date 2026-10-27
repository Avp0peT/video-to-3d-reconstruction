"""TEST_DATA: tiny native-camera/MVSI files; no reconstruction or MVS launch."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import pytest

pc = pytest.importorskip('pycolmap')
from integrations.video_masks import region_surface_masks as adapter

WIDTH, HEIGHT = 12, 10
IDS = (7, 19)


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf8')


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _record(path):
    return adapter.mio.file_record(path)


def _model(path, names, *, target=False, pose_delta=0.0):
    path.mkdir(parents=True, exist_ok=True)
    model = pc.Reconstruction()
    camera = pc.Camera(camera_id=3, model='PINHOLE' if target else 'SIMPLE_RADIAL',
                       width=WIDTH, height=HEIGHT,
                       params=[10, 10, 7, 5] if target else [10, 6, 5, 0])
    model.add_camera_with_trivial_rig(camera)
    for index, (iid, name) in enumerate(zip(IDS, names)):
        image = pc.Image(image_id=iid, camera_id=3, name=name)
        translation = np.array([index + 1.0 + (pose_delta if index else 0), 0, 0])
        model.add_image_with_trivial_frame(image, pc.Rigid3d(pc.Rotation3d(), translation))
    model.write(str(path))
    return model


def _scene(path, names, *, width=WIDTH, height=HEIGHT, ids=IDS, k_offset=0.0,
           center_delta=0.0, rotation_first=False, prefix='undistorted/images'):
    """Independent Interface.h v7 fixture writer, without native execution."""
    def text(value):
        data = value.encode('utf8')
        return struct.pack('<Q', len(data)) + data
    def doubles(values):
        return struct.pack('<' + 'd' * len(values), *values)
    eye = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
    data = b'MVSI' + struct.pack('<IIQ', 7, 0, 1) + text('TEST_DATA platform')
    data += struct.pack('<Q', 1) + text('camera') + text('RGB')
    data += struct.pack('<II', width, height)
    data += doubles([10, 0, 6.5 + k_offset, 0, 10, 4.5, 0, 0, 1])
    data += doubles(eye) + doubles([0, 0, 0]) + struct.pack('<Q', len(names))
    for index in range(len(names)):
        rotation = [0., -1., 0., 1., 0., 0., 0., 0., 1.] if rotation_first and index == 0 else eye
        data += doubles(rotation) + doubles([-index - 1.0 + center_delta, 0, 0])
    data += struct.pack('<Q', len(names))
    for index, (name, iid) in enumerate(zip(names, ids)):
        data += text(prefix + '/' + name) + text('') + struct.pack('<IIII', 0, 0, index, iid)
        data += struct.pack('<3fQ', 0, 1, 2, 0)
    data += struct.pack('<6Q', 0, 0, 0, 0, 0, 0)
    data += doubles([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    data += doubles(eye) + doubles([0, 0, 0]) + doubles([1, 1, 1])
    path.write_bytes(data)


def _seal(world, *, update_frame_hashes=False):
    export, bridge = world['export'], world['bridge']
    document = _load(export / 'EXPORT.json')
    if update_frame_hashes:
        for frame in document['frames']:
            for prefix in ('source_image', 'canonical_image', 'exported_png', 'source_mask', 'canonical_mask'):
                frame[prefix + '_sha256'] = _record(frame[prefix + '_path'])['sha256']
    _save(export / 'FRAME_MAP.json', dict(schema='region-surface-frame-map-v1', frames=document['frames']))
    document['frame_map_sha256'] = _record(export / 'FRAME_MAP.json')['sha256']
    document['source_bindings'] = [_record(row['path']) for row in document['source_bindings']]
    document['outputs'] = [_record(path) for path in sorted(export.rglob('*'))
                           if path.is_file() and path.name != 'EXPORT.json']
    if document.get('derivation_path'):
        document['derivation_sha256'] = _record(document['derivation_path'])['sha256']
    _save(export / 'EXPORT.json', document)
    bindings = [_record(path) for path in sorted((export / 'sparse').iterdir()) if path.is_file()]
    bindings += [_record(export / 'images' / frame['exported_name']) for frame in document['frames']]
    _save(bridge / 'INPUT_BINDINGS.json', bindings)


@pytest.fixture
def world(tmp_path):
    source, export, bridge = (tmp_path / name for name in ('source', 'export', 'bridge'))
    source.mkdir()
    (export / 'images').mkdir(parents=True)
    (bridge / 'undistorted' / 'images').mkdir(parents=True)
    names = [f'image_{iid:08d}.png' for iid in IDS]
    model = _model(export / 'sparse', names)
    _model(bridge / 'undistorted' / 'sparse', names, target=True)
    frames, sources = [], []
    for index, (iid, name) in enumerate(zip(IDS, names)):
        raw = source / f'original{index}.jpg'
        canonical = source / f'canonical{index}.jpg'
        mask = source / f'mask{index}.png'
        pattern = np.arange(WIDTH * HEIGHT * 3, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        Image.fromarray(pattern).save(raw)
        canonical.write_bytes(raw.read_bytes())
        with Image.open(raw) as image:
            pixels = np.array(image)
        Image.fromarray(pixels).save(export / 'images' / name)
        Image.fromarray(pixels).save(bridge / 'undistorted' / 'images' / name)
        labels = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        labels[2:8, 3:9] = 255
        Image.fromarray(labels).save(mask)
        sources.extend([_record(raw), _record(canonical), _record(mask)])
        camera = model.cameras[3]
        pose = model.images[iid].cam_from_world()
        pixel_digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        frames.append(dict(export_image_id=iid, image_id=iid, original_name=f'old/{index}.jpg',
            exported_name=name, input_index=10 + index * 32, decode_index=100 + index * 100,
            source_image_path=str(raw), source_image_sha256=_record(raw)['sha256'],
            canonical_image_path=str(canonical), canonical_image_sha256=_record(canonical)['sha256'],
            source_mask_path=str(mask), source_mask_sha256=_record(mask)['sha256'],
            canonical_mask_path=str(mask), canonical_mask_sha256=_record(mask)['sha256'],
            source_camera=dict(camera_id=3, model=camera.model_name, model_name=camera.model_name,
                               width=WIDTH, height=HEIGHT, params=camera.params.tolist()),
            source_pose=dict(quaternion_xyzw=pose.rotation.quat.tolist(),
                             rotation_matrix=pose.rotation.matrix().tolist(), translation=pose.translation.tolist()),
            exported_png_path=str(export / 'images' / name), exported_png_sha256=_record(export / 'images' / name)['sha256'],
            decoded_pixel_sha256=pixel_digest, exported_pixel_sha256=pixel_digest, pixel_equal=True))
    _save(export / 'EXPORT.json', dict(schema='region-surface-export-v1', status='EXPORTED_NOT_GEOMETRY_ACCEPTED',
        geometry_unchanged=True, source_geometry_unchanged=True, geometry_accepted=False,
        explicit_point_exclusion=False, excluded_point_ids=[], derivation_path=str(export / 'DERIVATION.json'),
        binary_path=str(export / 'sparse'), image_root=str(export / 'images'),
        frame_map_path=str(export / 'FRAME_MAP.json'), frames=frames, source_bindings=sources, outputs=[]))
    _save(export / 'DERIVATION.json', dict(schema='region-surface-derivation-v1', excluded_point_ids=[],
        source_modified=False, remaining_geometry_unchanged=True, all_2d_coordinates_unchanged=True,
        cameras_poses_rigs_frames_unchanged=True))
    _save(bridge / 'PLAN.json', dict(source_sparse=str(export / 'sparse'), source_images=str(export / 'images')))
    _save(bridge / 'RESULT.json', dict(status='PASS', input_bindings_unchanged=True))
    for name in ('UNDISTORT_COMPARISON.json', 'MVS_COMPARISON.json'):
        _save(bridge / name, dict(status='PASS'))
    _save(bridge / 'UNDISTORT_OPTIONS.json', dict(pycolmap='4.2.1', options={}))
    _scene(bridge / 'scene.mvs', names)
    result = dict(source=source, export=export, bridge=bridge, output=tmp_path / 'masks', names=names)
    _seal(result)
    return result


def _prepare(world):
    return adapter.prepare(world['export'], world['bridge'], world['output'])


def test_complete_masks_have_exact_pixels_and_separate_unknown_validity(world):
    originals = {str(path): _record(path) for path in world['output'].parent.rglob('*') if path.is_file()}
    result = _prepare(world)
    assert result['status'] == 'MASKS_PREPARED_NOT_CONSUMPTION_VERIFIED'
    assert result['available_views'] == result['registered_views'] == 2
    assert result['coverage_status'] == 'COMPLETE' and result['adapter_source_chain'] == 'VERIFIED'
    assert result['mask_consumption'] == 'NOT_VERIFIED' and result['native_launches'] == 0
    assert result['native_input_origin'] == 'unverified' and result['geometry_accepted'] is False
    assert result['surface_image_root'] == str(world['bridge'])
    assert result['scene_binding']['verified'] is True and result['scene_binding']['image_count'] == 2
    for row in result['frames']:
        with Image.open(row['source_mask_path']) as source:
            mask = np.array(source)
        expected = np.zeros_like(mask)
        expected[:, 1:] = mask[:, :-1]
        with Image.open(row['mask']['path']) as image:
            assert np.array_equal(np.array(image), expected)
        with Image.open(row['validity']['path']) as image:
            valid = np.array(image)
            assert np.all(valid[:, 0] == 0) and np.all(valid[:, 1:] == 255)
        assert row['counts']['unknown'] == HEIGHT
        assert Path(row['mask']['path']).name == Path(row['exported_name']).stem + '.mask.png'
        assert row['readback_verified'] is True
    assert originals == {path: _record(path) for path in originals}


@pytest.mark.parametrize('kind', ['missing', 'changed'])
def test_missing_or_changed_one_mask_refuses_before_output(world, kind):
    path = world['source'] / 'mask0.png'
    if kind == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'changed TEST_DATA')
    with pytest.raises(Exception, match='missing|binding differs'):
        _prepare(world)
    assert not world['output'].exists()


@pytest.mark.parametrize('label', [0, 127])
def test_empty_or_nonbinary_source_mask_is_not_filled(world, label):
    Image.fromarray(np.full((HEIGHT, WIDTH), label, dtype=np.uint8)).save(world['source'] / 'mask0.png')
    _seal(world, update_frame_hashes=True)
    with pytest.raises(Exception, match='empty|labels'):
        _prepare(world)
    assert not world['output'].exists()


@pytest.mark.parametrize('field', ['export_image_id', 'input_index', 'decode_index', 'exported_name'])
def test_duplicate_frame_identity_refuses(world, field):
    path = world['export'] / 'EXPORT.json'
    document = _load(path)
    document['frames'][1][field] = document['frames'][0][field]
    _save(path, document)
    _seal(world)
    with pytest.raises(ValueError, match='duplicate frame identity'):
        _prepare(world)
    assert not world['output'].exists()


@pytest.mark.parametrize('kind', ['camera', 'pose', 'pixel_digest', 'frame_mask_hash'])
def test_declared_frame_geometry_or_identity_drift_refuses(world, kind):
    path = world['export'] / 'EXPORT.json'
    document = _load(path)
    frame = document['frames'][0]
    if kind == 'camera':
        frame['source_camera']['params'][0] += 1
    elif kind == 'pose':
        frame['source_pose']['translation'][0] += 1
    elif kind == 'pixel_digest':
        frame['decoded_pixel_sha256'] = '0' * 64
    else:
        frame['source_mask_sha256'] = '0' * 64
    _save(path, document)
    _seal(world)
    with pytest.raises(ValueError, match='differs'):
        _prepare(world)
    assert not world['output'].exists()


@pytest.mark.parametrize('kwargs,match', [
    ({'prefix': 'wrong/images'}, 'scene images'),
    ({'width': WIDTH + 1}, 'target camera'),
    ({'ids': (8, 19)}, 'identity mismatch'),
    ({'k_offset': 0.25}, 'scene K mismatch'),
    ({'center_delta': 0.25}, 'scene C mismatch'),
    ({'rotation_first': True}, 'scene R mismatch'),
])
def test_scene_names_size_id_calibration_and_pose_are_all_checked(world, kwargs, match):
    _scene(world['bridge'] / 'scene.mvs', world['names'], **kwargs)
    with pytest.raises(Exception, match=match):
        _prepare(world)
    assert not world['output'].exists()


def test_undistorted_pose_mismatch_refuses_before_output(world):
    _model(world['bridge'] / 'undistorted' / 'sparse', world['names'], target=True, pose_delta=0.1)
    with pytest.raises(Exception, match='pose mismatch'):
        _prepare(world)
    assert not world['output'].exists()


def test_missing_bridge_image_binding_refuses(world):
    path = world['bridge'] / 'INPUT_BINDINGS.json'
    records = _load(path)
    _save(path, [row for row in records if Path(row['path']).name != world['names'][0]])
    with pytest.raises(ValueError, match='bridge source identity'):
        _prepare(world)
    assert not world['output'].exists()


def test_reencoded_png_must_still_match_source_pixels(world):
    path = world['export'] / 'images' / world['names'][0]
    with Image.open(path) as image:
        pixels = np.array(image)
    pixels[2, 2] ^= 255
    Image.fromarray(pixels).save(path)
    _seal(world, update_frame_hashes=True)
    with pytest.raises(ValueError, match='RGB pixels differ'):
        _prepare(world)
    assert not world['output'].exists()


def test_early_report_rejection_preserves_existing_output(world):
    world['output'].mkdir()
    sentinel = world['output'] / 'original.txt'
    sentinel.write_text('do not change', encoding='utf8')
    with pytest.raises(ValueError, match='fresh output'):
        _prepare(world)
    assert sentinel.read_text(encoding='utf8') == 'do not change'
    assert list(world['output'].iterdir()) == [sentinel]


def test_output_pixel_corruption_is_retained_as_failed_readback(world, monkeypatch):
    original = adapter.mio.write_label_png
    def corrupt(path, array):
        record = original(path, array)
        altered = array.copy()
        altered[2, 2] = 255 - altered[2, 2]
        Image.fromarray(altered).save(path)
        return record
    monkeypatch.setattr(adapter.mio, 'write_label_png', corrupt)
    with pytest.raises(Exception, match='does not match'):
        _prepare(world)
    result = _load(world['output'] / 'RESULT.json')
    assert result['status'] == 'FAILED' and result['coverage_status'] == 'INCOMPLETE'
    assert result['adapter_source_chain'] == 'NOT_VERIFIED' and result['mask_consumption'] == 'NOT_VERIFIED'
    assert result['available_views'] == 0


def test_explicit_geometry_derivation_is_bound_and_kept_distinct(world):
    path = world['export'] / 'EXPORT.json'
    document = _load(path)
    derivation = world['export'] / 'DERIVATION.json'
    _save(derivation, dict(schema='region-surface-derivation-v1', excluded_point_ids=[91],
        source_modified=False, remaining_geometry_unchanged=True, all_2d_coordinates_unchanged=True,
        cameras_poses_rigs_frames_unchanged=True))
    document.update(source_geometry_unchanged=False, explicit_point_exclusion=True,
                    excluded_point_ids=[91], derivation_path=str(derivation))
    _save(path, document)
    _seal(world)
    result = _prepare(world)
    assert result['adapter_source_chain'] == 'VERIFIED'
    assert result['source_geometry_unchanged'] is False and result['excluded_point_ids'] == [91]
    assert result['derivation_sha256'] == _record(derivation)['sha256']


def test_unbound_derivation_refuses(world):
    path = world['export'] / 'EXPORT.json'
    document = _load(path)
    document.update(explicit_point_exclusion=True, derivation_path=str(world['source'] / 'missing.json'),
                    derivation_sha256='0' * 64)
    _save(path, document)
    with pytest.raises(ValueError, match='derivation is not bound'):
        _prepare(world)
    assert not world['output'].exists()


def test_source_drift_during_warp_marks_failed_and_keeps_written_evidence(world, monkeypatch):
    original = adapter.warp_binary_mask
    changed = False
    def warp(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if not changed:
            (world['source'] / 'canonical0.jpg').write_bytes(b'TEST_DATA changed after preflight')
            changed = True
        return result
    monkeypatch.setattr(adapter, 'warp_binary_mask', warp)
    with pytest.raises(ValueError, match='drift after mask generation'):
        _prepare(world)
    result = _load(world['output'] / 'RESULT.json')
    assert result['status'] == 'FAILED' and result['adapter_source_chain'] == 'NOT_VERIFIED'
    assert result['input_drift'] == [str(world['source'] / 'canonical0.jpg')]
    assert len(result['outputs']) == 4


def test_derivation_mismatch_refuses_even_when_hashes_are_current(world):
    path = world['export'] / 'DERIVATION.json'
    document = _load(path)
    document['excluded_point_ids'] = [91]
    _save(path, document)
    _seal(world)
    with pytest.raises(ValueError, match='exclusion declarations disagree'):
        _prepare(world)
    assert not world['output'].exists()
