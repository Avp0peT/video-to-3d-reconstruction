"""Bind regional masks to a verified COLMAP/MVSI bridge, without running MVS.

Every exported view needs its original, nonempty binary mask. This adapter
verifies the regional export chain separately from the surface runner's
``unverified`` input-origin classification; neither is native mask consumption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from integrations.mvs_masks import manifest_io as mio
from integrations.mvs_masks.mask_geometry import ALGORITHM, warp_binary_mask

SCHEMA = 'region-surface-masks-v1'
MAX_VIEWS = 96
MAX_PIXELS = 16_777_216


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _save(path, document):
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


class _Bindings:
    def __init__(self):
        self.records = {}

    def bind(self, path, expected=None):
        current = mio.file_record(path)
        key = mio.normalised(current['path'])
        if expected is not None:
            mio.require_sha256(expected.get('sha256'), 'declared sha256')
            mio.require_int(expected.get('bytes'), 'declared bytes')
            _need(current['sha256'] == expected['sha256'] and current['bytes'] == expected['bytes'],
                  'declared byte binding differs: ' + str(path))
        _need(key not in self.records or self.records[key] == current, 'input changed during read: ' + str(path))
        self.records[key] = current
        return current

    def read(self, path, array=False):
        self.bind(path)
        loader = mio.load_json_array_strict if array else mio.load_json_strict
        document = loader(path)
        self.bind(path)
        return document

    def declared(self, records, label):
        _need(isinstance(records, list) and records, label + ' must be a nonempty binding list')
        result = {}
        for row in records:
            _need(isinstance(row, dict) and isinstance(row.get('path'), str)
                  and Path(row['path']).is_absolute(), label + ' requires absolute file paths')
            key = mio.normalised(row['path'])
            _need(key not in result, label + ' has duplicate path bindings')
            result[key] = self.bind(row['path'], row)
        return result

    def drift(self):
        changed = []
        for key, record in self.records.items():
            try:
                actual = mio.file_record(record['path'])
            except Exception:
                changed.append(record['path'])
            else:
                if actual != record:
                    changed.append(record['path'])
        return changed


def _rgb(path, size, *, png=False):
    with Image.open(path) as image:
        image.load()
        _need(image.format in (('PNG',) if png else ('JPEG', 'PNG'))
              and image.mode == 'RGB' and image.size == size
              and getattr(image, 'n_frames', 1) == 1
              and image.getexif().get(274, 1) in (None, 1)
              and 'transparency' not in image.info,
              'RGB image format/orientation/dimensions mismatch: ' + str(path))
        return np.asarray(image).copy()


def _array_equal(actual, expected, shape, label):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _need(a.shape == b.shape == shape and np.isfinite(a).all() and np.isfinite(b).all()
          and np.allclose(a, b, rtol=0, atol=1e-12), label + ' differs from exported native model')


def _frame_identity(frame, record, model):
    iid = mio.require_int(frame.get('export_image_id'), 'export_image_id')
    _need(frame.get('image_id') == iid and type(frame.get('image_id')) is int, 'export/native image ID mismatch')
    name = frame.get('exported_name')
    _need(name == f'image_{iid:08d}.png', 'explicit exported flat PNG name required')
    _need(record['source_image_id'] == record['target_image_id'] == iid, 'export/source/target image ID mismatch')
    image = model.images[iid]
    _need(image.name == name, 'exported model name differs from frame map')
    camera = model.cameras[image.camera_id]
    declared = frame.get('source_camera', {})
    _need(declared.get('camera_id') == record['source_camera_id']
          and declared.get('model') == declared.get('model_name') == camera.model_name
          and declared.get('width') == int(camera.width) and declared.get('height') == int(camera.height),
          'source camera identity/domain differs from frame map')
    _array_equal(camera.params, declared.get('params'), np.asarray(camera.params).shape, 'source camera params')
    _need(camera.verify_params() and int(camera.width) * int(camera.height) <= MAX_PIXELS,
          'invalid or oversized source camera')
    pose = image.cam_from_world()
    supplied = frame.get('source_pose', {})
    _array_equal(pose.rotation.quat, supplied.get('quaternion_xyzw'), (4,), 'source quaternion')
    _array_equal(pose.translation, supplied.get('translation'), (3,), 'source translation')
    _array_equal(pose.rotation.matrix(), supplied.get('rotation_matrix'), (3, 3), 'source rotation')
    for key in ('input_index', 'decode_index'):
        mio.require_int(frame.get(key), key)
    return camera


def _preflight(export_root, bridge_root, output):
    import pycolmap as pc
    import pycolmap._core as core
    from scripts import prepare_mvs_masks as bridge_tools
    from integrations.mvs_masks import mask_geometry
    from integrations.mvs_surface import scene as scene_reader

    _need(pc.__version__ == '4.2.1', 'pycolmap 4.2.1 reader required')
    export_root, bridge_root, output = (Path(p).resolve() for p in (export_root, bridge_root, output))
    _need(not output.exists(), 'fresh output directory required')
    for root in (export_root, bridge_root):
        _need(root.is_dir() and not output.is_relative_to(root) and not root.is_relative_to(output),
              'output must be disjoint from export and bridge directories')
    bound = _Bindings()
    for path in (__file__, mio.__file__, mask_geometry.__file__, scene_reader.__file__,
                 bridge_tools.__file__, core.__file__):
        bound.bind(path)
    export = bound.read(export_root / 'EXPORT.json')
    _need(export.get('schema') == 'region-surface-export-v1'
          and export.get('status') == 'EXPORTED_NOT_GEOMETRY_ACCEPTED'
          and export.get('geometry_unchanged') is True,
          'verified regional geometry-preserving export required')
    sources = bound.declared(export.get('source_bindings'), 'export source_bindings')
    products = bound.declared(export.get('outputs'), 'export outputs')
    for record in products.values():
        _need(Path(record['path']).is_relative_to(export_root), 'export output escapes export root')
    derivation_path = export.get('derivation_path')
    _need(isinstance(derivation_path, str) and Path(derivation_path).is_absolute(),
          'derivation must have a bound absolute path')
    key = mio.normalised(derivation_path)
    _need(key in products and export.get('derivation_sha256') == products[key]['sha256'],
          'derivation is not bound by export outputs')
    derivation = bound.read(derivation_path)
    excluded = derivation.get('excluded_point_ids')
    _need(isinstance(excluded, list) and all(type(pid) is int and pid >= 0 for pid in excluded)
          and excluded == sorted(set(excluded)) and excluded == export.get('excluded_point_ids')
          and export.get('explicit_point_exclusion') is bool(excluded)
          and export.get('source_geometry_unchanged') is (not excluded), 'derivation exclusion declarations disagree')
    _need(derivation.get('schema') == 'region-surface-derivation-v1'
          and derivation.get('source_modified') is False
          and all(derivation.get(key) is True for key in ('remaining_geometry_unchanged',
              'all_2d_coordinates_unchanged', 'cameras_poses_rigs_frames_unchanged')),
          'derivation does not preserve the declared camera and geometry domain')
    frame_map = Path(export['frame_map_path']).resolve(strict=True)
    _need(frame_map == export_root / 'FRAME_MAP.json' and mio.normalised(frame_map) in products
          and export.get('frame_map_sha256') == products[mio.normalised(frame_map)]['sha256'],
          'frame map is not bound by export outputs')
    mapping = bound.read(frame_map)
    frames = mapping.get('frames')
    _need(mapping.get('schema') == 'region-surface-frame-map-v1' and isinstance(frames, list)
          and 2 <= len(frames) <= MAX_VIEWS and export.get('frames') == frames,
          'complete regional frame map required')
    for key in ('export_image_id', 'exported_name', 'input_index', 'decode_index'):
        values = [frame.get(key) for frame in frames]
        _need(len(set(values)) == len(values), 'duplicate frame identity: ' + key)
    bridge = bridge_tools.bridge_artifacts(bridge_root)
    _need(bridge['result'].get('input_bindings_unchanged') is True, 'bridge inputs were not verified unchanged')
    for filename in ('PLAN.json', 'RESULT.json', 'INPUT_BINDINGS.json', 'UNDISTORT_COMPARISON.json',
                     'MVS_COMPARISON.json', 'UNDISTORT_OPTIONS.json'):
        document = bound.read(bridge_root / filename, array=filename == 'INPUT_BINDINGS.json')
        if filename.endswith('COMPARISON.json'):
            _need(document.get('status') == 'PASS', 'bridge numeric comparison did not pass: ' + filename)
    scene_record = bound.bind(bridge_root / 'scene.mvs')
    _need(Path(export['binary_path']).resolve(strict=True) == bridge['source_sparse']
          and Path(export['image_root']).resolve(strict=True) == bridge['source_images'],
          'bridge uses a different regional export')
    bridge_inputs = bound.declared(bound.read(bridge_root / 'INPUT_BINDINGS.json', array=True), 'bridge inputs')
    for path in sorted(bridge['source_sparse'].iterdir()):
        if path.is_file():
            key = mio.normalised(path)
            _need(key in products and key in bridge_inputs, 'source model file lacks export/bridge binding')
    for path in sorted(bridge['undistorted_sparse'].iterdir()):
        if path.is_file():
            bound.bind(path)
    models = bridge_tools.model_bindings(bridge)
    records, source, target = models['records'], models['source_model'], models['target_model']
    names = {frame['exported_name'] for frame in frames}
    _need(set(records) == names and len(source.reg_image_ids()) == len(target.reg_image_ids()) == len(frames),
          'export/source/undistorted all-view coverage differs')
    output_names = mio.output_plan(names)['by_image']
    scene = bridge_tools.scene_binding(bridge, records)
    _need(scene['sha256'] == scene_record['sha256'] and scene['bytes'] == scene_record['bytes'], 'scene changed during read')
    prepared = []
    for frame in frames:
        name = frame['exported_name']
        record = records[name]
        camera = _frame_identity(frame, record, source)
        size = (int(camera.width), int(camera.height))
        pixels = []
        for prefix in ('source_image', 'canonical_image', 'exported_png'):
            path = Path(frame[prefix + '_path']).resolve(strict=True)
            key = mio.normalised(path)
            declared = products if prefix == 'exported_png' else sources
            _need(key in declared, prefix + ' is not bound by the export')
            actual = bound.bind(path, declared[key])
            _need(frame.get(prefix + '_sha256') == actual['sha256'], prefix + ' frame hash differs')
            if prefix == 'exported_png':
                _need(path == bridge['source_images'] / name and key in bridge_inputs,
                      'exported PNG lacks exact bridge source identity')
                bridge_tools.verify_canonical_image(path, *size)
            pixels.append(_rgb(path, size, png=prefix == 'exported_png'))
        _need(all(np.array_equal(pixels[0], other) for other in pixels[1:])
              and frame.get('pixel_equal') is True, 'source/canonical/exported RGB pixels differ')
        pixel_sha = hashlib.sha256(pixels[0].tobytes()).hexdigest()
        _need(pixel_sha == frame.get('decoded_pixel_sha256') == frame.get('exported_pixel_sha256'),
              'exported pixel digest differs')
        mask_path = Path(frame['source_mask_path']).resolve(strict=True)
        canonical_mask = Path(frame['canonical_mask_path']).resolve(strict=True)
        _need(mask_path == canonical_mask and mio.normalised(mask_path) in sources,
              'original mask identity missing or inconsistent')
        mask_record = bound.bind(mask_path, sources[mio.normalised(mask_path)])
        _need(mask_record['sha256'] == frame.get('source_mask_sha256') == frame.get('canonical_mask_sha256'),
              'original mask hash differs')
        mask, mask_readback = mio.read_label_png(mask_path, *size)
        _need(mask_readback['foreground_pixels'] > 0 and mask_readback['sha256'] == mask_record['sha256'],
              'original mask is empty or changed during decoding')
        target_image = target.images[record['target_image_id']]
        target_camera = target.cameras[target_image.camera_id]
        _need(target_camera.verify_params() and int(target_camera.width) * int(target_camera.height) <= MAX_PIXELS,
              'invalid or oversized target camera')
        _rgb(record['undistorted_image']['path'], (int(target_camera.width), int(target_camera.height)), png=True)
        bound.bind(record['undistorted_image']['path'], record['undistorted_image'])
        prepared.append((frame, record, camera, target_camera, mask))
    _need(not bound.drift(), 'source drift during preflight')
    return dict(export=export, export_root=export_root, bridge=bridge, output=output,
                bindings=bound, scene=scene, prepared=prepared, output_names=output_names)


def prepare(export_root, bridge_root, output):
    """Write complete verified mask/validity rasters; never launch native MVS."""
    plan = _preflight(export_root, bridge_root, output)
    output, bound = plan['output'], plan['bindings']
    output.mkdir(parents=True, exist_ok=False)
    (output / 'masks').mkdir()
    (output / 'validity').mkdir()
    export = plan['export']
    result = dict(schema=SCHEMA, status='RUNNING', coverage_status='INCOMPLETE',
                  adapter_source_chain='NOT_VERIFIED', native_input_origin='unverified',
                  mask_consumption='NOT_VERIFIED', native_launches=0, geometry_accepted=False,
                  quality_status='NOT_EVALUATED', algorithm=ALGORITHM,
                  export_root=str(plan['export_root']), bridge_root=str(plan['bridge']['bridge']),
                  surface_image_root=str(plan['bridge']['bridge']),
                  mask_root=str(output / 'masks'), validity_root=str(output / 'validity'),
                  registered_views=len(plan['prepared']), available_views=0, frames=[], outputs=[],
                  scene_binding=plan['scene'], source_bindings=list(bound.records.values()),
                  source_geometry_unchanged=export.get('source_geometry_unchanged'),
                  explicit_point_exclusion=export.get('explicit_point_exclusion'),
                  excluded_point_ids=export.get('excluded_point_ids', []),
                  derivation_path=export.get('derivation_path'), derivation_sha256=export.get('derivation_sha256'),
                  export_geometry_reference='Export preserves the derived reference model; any explicit point exclusions '
                                            'from the original source remain separately declared.',
                  binding_scope='Export source/output declarations and bridge inputs checked against current bytes. '
                                'Bridge products are locally hashed and verified against parsed models/MVSI; '
                                'no historical output hash declaration is invented.',
                  limitations=['Prepared mask rasters do not prove OpenMVS consumed them.',
                               'Predicted source masks are not human ground truth.',
                               'Regional source-chain verification does not change native input_origin=unverified.'])
    try:
        for frame, record, source_camera, target_camera, mask in plan['prepared']:
            warped = warp_binary_mask(mask, source_camera, target_camera, max_pixels=MAX_PIXELS)
            _need(warped.counts['foreground'] > 0, 'warped foreground is empty')
            _need(not np.any((warped.mask != 0) & (warped.valid == 0)), 'foreground exists outside known raster support')
            mask_name = plan['output_names'][frame['exported_name']]
            validity_name = mask_name.removesuffix('.mask.png') + '.validity.png'
            output_records = []
            for folder, name, array in (('masks', mask_name, warped.mask), ('validity', validity_name, warped.valid)):
                path = output / folder / name
                written = mio.write_label_png(path, array)
                result['outputs'].append(written)
                checked = mio.verify_label_png(path, array, int(target_camera.width), int(target_camera.height))
                _need(all(written[key] == checked[key] for key in ('path', 'bytes', 'sha256')),
                      'output PNG bytes changed during readback')
                output_records.append(checked)
            result['frames'].append(dict(export_image_id=frame['export_image_id'],
                exported_name=frame['exported_name'], input_index=frame['input_index'], decode_index=frame['decode_index'],
                status='OK', readback_verified=True, mask=output_records[0], validity=output_records[1],
                counts=warped.counts, source_mask_path=frame['source_mask_path'],
                source_mask_sha256=frame['source_mask_sha256'], source_camera=record['source_camera'],
                target_camera=record['target_camera'], source_image_id=record['source_image_id'],
                target_image_id=record['target_image_id'], pose_max_abs=record['pose_max_abs']))
        result['input_drift'] = bound.drift()
        result['output_drift'] = [row['path'] for row in result['outputs'] if mio.file_record(row['path']) != row]
        _need(not result['input_drift'] and not result['output_drift'], 'input/output drift after mask generation')
        _need(len(result['frames']) == result['registered_views'], 'incomplete output coverage')
        result.update(status='MASKS_PREPARED_NOT_CONSUMPTION_VERIFIED', coverage_status='COMPLETE',
                      adapter_source_chain='VERIFIED', available_views=len(result['frames']))
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}',
                      available_views=len(result['frames']), input_drift=bound.drift())
        _save(output / 'RESULT.json', result)
        raise
    _save(output / 'RESULT.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('export-root', 'bridge-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.export_root, args.bridge_root, args.output)
    print(json.dumps(dict(status=result['status'], available_views=result['available_views'],
                          mask_consumption=result['mask_consumption']), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
