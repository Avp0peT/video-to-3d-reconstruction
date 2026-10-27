"""Bounded, read-only comparison of a delivered mesh with its SAM labels.

This measures internal image consistency, never independent object accuracy.
Every selected scene view is required; there is no best-view selection.
"""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

from integrations.demo_pipeline.quality import _read, _hash
from integrations.mvs_surface.scene import read_scene
from integrations.mvs_masks.manifest_io import read_jsonl_strict
from integrations.viewer.ply_data import read_ply

ROOT = Path(__file__).resolve().parents[2]
LIMITATIONS = [
    'SAM labels and geometry share source images: agreement is not independent accuracy.',
    'Only selected, segmented views are evaluated; excluded views and unseen surfaces remain unknown.',
    'Opaque triangle union; no background occlusion model or camera-plane clipping.',
    '2x polygon rasterization at threshold 128; 1x disagreement is reported separately.',
    'No physical scale, precision, full-object or CAD-quality acceptance is inferred.',
]


def project(vertices, rotation, center, intrinsic):
    vertices = np.asarray(vertices, dtype=float)
    rotation, center, intrinsic = (np.asarray(x, dtype=float) for x in (rotation, center, intrinsic))
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or rotation.shape != (3, 3)
            or center.shape != (3,) or intrinsic.shape != (3, 3)
            or not all(np.isfinite(a).all() for a in (vertices, rotation, center, intrinsic))):
        raise ValueError('finite Nx3 vertices and K/R/C required')
    camera = (vertices - center) @ rotation.T
    if len(camera) == 0 or (camera[:, 2] <= 0).any():
        raise ValueError('mesh touches/crosses camera plane; unverified clipping refused')
    homogeneous = camera @ intrinsic.T
    if (homogeneous[:, 2] == 0).any():
        raise ValueError('zero projective denominator')
    uv = homogeneous[:, :2] / homogeneous[:, 2, None]
    if not np.isfinite(uv).all():
        raise ValueError('nonfinite projection')
    return uv


def raster(uv, faces, width, height, scale=2, check=lambda: None):
    if any(type(x) is not int or not 1 <= x <= 2048 for x in (width, height)) or scale not in (1, 2):
        raise ValueError('bounded dimensions and scale 1/2 required')
    uv, faces = np.asarray(uv), np.asarray(faces)
    if (uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
            or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(uv)):
        raise ValueError('finite projections and nonempty indexed triangles required')
    canvas = Image.new('L', (width * scale, height * scale))
    draw = ImageDraw.Draw(canvas)
    triangles = uv[faces]
    lows, highs = triangles.min(axis=1), triangles.max(axis=1)
    visible = (highs[:, 0] >= -.5) & (highs[:, 1] >= -.5) & (lows[:, 0] <= width - .5) & (lows[:, 1] <= height - .5)
    for i, tri in enumerate(triangles[visible]):
        if i % 2048 == 0:
            check()
        draw.polygon((tri * scale + (scale - 1) / 2).ravel().tolist(), fill=255)
    if scale > 1:
        canvas = canvas.resize((width, height), Image.Resampling.BOX)
    return np.asarray(canvas) >= 128


def metrics(prediction, labels, valid):
    arrays = [np.asarray(a) for a in (prediction, labels, valid)]
    if any(a.dtype != np.bool_ or a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('same-shape binary arrays required')
    pred, sam, valid = arrays
    tp = int(np.count_nonzero(pred & sam & valid))
    fp = int(np.count_nonzero(pred & ~sam & valid))
    fn = int(np.count_nonzero(~pred & sam & valid))
    def ratio(a, b):
        return a / b if b else None
    return {'intersection_pixels': tp, 'mesh_only_pixels': fp, 'SAM_only_pixels': fn,
            'valid_pixels': int(valid.sum()), 'IoU': ratio(tp, tp + fp + fn),
            'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn),
            'undefined_reason': 'zero denominator' if not tp + fp or not tp + fn else None}


def _edge(mask):
    p = np.pad(mask, 1)
    return mask & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])


def _binary(path, size):
    with Image.open(path) as image:
        if image.mode != 'L' or image.size != tuple(size):
            raise ValueError('mask/validity must be L at actual target size')
        data = np.asarray(image).copy()
    if not np.isin(data, [0, 255]).all():
        raise ValueError('mask/validity labels outside {0,255}')
    return data == 255


class Bindings:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.records = {}

    def add(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('review input escapes completed run')
        if path.stat().st_size > 64 << 20:
            raise ValueError('review input exceeds 64 MiB')
        digest = _hash(path)
        if expected is not None and digest != expected:
            raise ValueError('input hash mismatch: ' + str(path))
        prior = self.records.get(str(path))
        record = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}
        if prior is not None and prior != record:
            raise ValueError('input drift while binding')
        self.records[str(path)] = record
        return path

    def unchanged(self):
        return all(Path(p).is_file() and Path(p).stat().st_size == r['bytes'] and _hash(p) == r['sha256']
                   for p, r in self.records.items())


def inspect(run):
    run = Path(run).resolve(strict=True)
    bindings = Bindings(run)
    def read(relative):
        return _read(bindings.add(run / relative))
    state = read('pipeline.json')
    base = 'geometry/surface_pipeline/'
    manifest = read(base + 'masks/manifest.json')
    surface = read(base + 'surface/result.json')
    plan = read(base + 'surface/plan.json')
    if (state.get('schema') != 'photo-demo-pipeline-v1'
            or state.get('stages', {}).get('surface', {}).get('status') != 'SUCCESS'
            or manifest.get('status') != 'SUCCESS' or manifest.get('coverage_status') != 'COMPLETE'
            or manifest.get('image_domain') != 'colmap_undistorted' or manifest.get('unverified_edges') != []
            or surface.get('status') != 'SURFACE_GENERATED_NOT_ACCEPTED'
            or type(surface.get('exit_code')) is not int or surface['exit_code'] != 0
            or surface.get('input_drift') != []):
        raise ValueError('successful bound surface and complete selected mask domain required')
    scene_ref = manifest['scene']
    scene_path = bindings.add(scene_ref['path'], scene_ref['sha256'])
    if plan['scene'] != {k: scene_ref[k] for k in ('path', 'bytes', 'sha256')}:
        raise ValueError('surface plan and mask scene differ')
    scene = read_scene(scene_path)
    artifacts = state['stages']['surface']['artifacts']
    mesh_path = (run / base / 'surface/work/__mvs_surface_raw.ply').resolve()
    declarations = [a for a in artifacts if Path(a['path']).resolve() == mesh_path]
    if len(declarations) != 1:
        raise ValueError('unique published raw mesh required')
    bindings.add(mesh_path, declarations[0]['sha256'])
    mesh = read_ply(mesh_path, max_bytes=64 << 20, max_vertices=500000, max_faces=1000000)
    if not mesh['faces']:
        raise ValueError('no surface triangles')
    frames = manifest['frames']
    indices = [f['input_index'] for f in frames]
    if (not 1 <= len(frames) <= 96 or any(type(i) is not int or i < 0 for i in indices)
            or len(set(indices)) != len(indices) or any(f['status'] != 'OK' for f in frames)
            or len(frames) != len(scene['images']) or len(frames) != len(plan['frames'])):
        raise ValueError('all selected scene views must have unique OK masks')
    if sorted(indices) != sorted(state['stages']['surface']['selected_indices']):
        raise ValueError('pipeline and mask selected views differ')
    io_root = Path(manifest['inputs']['t01_run']).resolve(strict=True)
    io_records = [r for r in manifest['inputs']['source_io'].values() if Path(r['path']).resolve() == io_root]
    if len(io_records) != 1:
        raise ValueError('unique upstream photo IO binding required')
    io_record = io_records[0]
    for key, name in (('run_json', 'run.json'), ('frames_jsonl', 'frames.jsonl')):
        ref = io_record[key]
        if Path(ref['path']).resolve() != io_root / name:
            raise ValueError('upstream IO path differs')
        bindings.add(ref['path'], ref['sha256'])
    source_rows = read_jsonl_strict(io_root / 'frames.jsonl')
    source_indices = [r['input_index'] for r in source_rows]
    if any(type(i) is not int for i in source_indices) or len(set(source_indices)) != len(source_indices):
        raise ValueError('upstream IO indices invalid')
    prepared = []
    seen_ids = set()
    for frame in sorted(frames, key=lambda f: f['input_index']):
        sources = [r for r in source_rows if r['input_index'] == frame['input_index']]
        if (len(sources) != 1 or sources[0]['status'] != 'OK' or sources[0]['frame_id'] != frame['frame_id']
                or (io_root / sources[0]['image_output']).resolve() != Path(frame['source_image']).resolve()):
            raise ValueError('upstream index/frame/image identity differs')
        bindings.add(frame['source_image'], frame['source_image_sha256'])
        matches = [m for m in scene_ref['mapping'] if m['model_image'] == frame['image_name']]
        if len(matches) != 1:
            raise ValueError('ambiguous image mapping')
        mapping = matches[0]
        cameras = [c for c in scene['images'] if c['global_id'] == mapping['scene_global_id']]
        planned = [p for p in plan['frames'] if p['image']['global_id'] == mapping['scene_global_id']]
        if len(cameras) != 1 or len(planned) != 1 or mapping['scene_global_id'] in seen_ids:
            raise ValueError('scene identity is not bijective')
        seen_ids.add(mapping['scene_global_id'])
        camera, planned = cameras[0], planned[0]
        if camera['name'] != mapping['scene_name'] or camera['name'] != planned['image']['name']:
            raise ValueError('scene photo name differs')
        for key in ('K', 'R', 'C', 'width', 'height'):
            if camera[key] != mapping[key] or camera[key] != planned['image'][key]:
                raise ValueError('camera mapping/plan drift')
        size = [camera['width'], camera['height']]
        if frame['target_camera']['model'] != 'PINHOLE' or frame['target_size'] != size:
            raise ValueError('target pinhole size mismatch')
        fx, fy, cx, cy = frame['target_camera']['params']
        if not np.allclose(camera['K'], [[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1]], atol=1e-12, rtol=0):
            raise ValueError('OpenMVS/COLMAP pixel-center convention mismatch')
        masks = []
        for role in ('mask', 'validity'):
            ref = frame[role]
            path = bindings.add(ref['absolute_path'], ref['sha256'])
            if Path(planned[role]['path']).resolve() != path or planned[role]['sha256'] != ref['sha256']:
                raise ValueError('plan mask identity differs')
            masks.append(_binary(path, size))
        sam, valid = masks
        if (sam & ~valid).any() or int(sam.sum()) != frame['core_counts']['foreground'] or int(valid.sum()) != frame['core_counts']['valid']:
            raise ValueError('mask/validity counts or unknown-domain foreground differ')
        image = bindings.add(planned['image_file']['path'], planned['image_file']['sha256'])
        if (image != Path(mapping['actual_image']).resolve()
                or image != (scene_path.parent / camera['name']).resolve()):
            raise ValueError('photo identity differs')
        with Image.open(image) as photo:
            if photo.size != tuple(size):
                raise ValueError('photo dimensions differ')
        prepared.append((frame, camera, sam, valid, image))
    return state, mesh, prepared, bindings


def review(run, output, *, seconds=180):
    return render_review(run, output, inspect, seconds=seconds)


def render_review(run, output, inspector, *, seconds=180, source_kind=None):
    """Render a verified source supplied by an explicit source-specific reader."""
    if type(seconds) is not int or not 1 <= seconds <= 300:
        raise ValueError('seconds must be 1..300')
    run, output = Path(run).resolve(strict=True), Path(output).resolve()
    if (output.exists() or not output.is_relative_to(ROOT / 'outputs')
            or output.is_relative_to(run) or run.is_relative_to(output)):
        raise ValueError('new disjoint project outputs directory required')
    start = time.monotonic()
    def check():
        if time.monotonic() - start > seconds:
            raise TimeoutError('shared review deadline reached')
        if output.exists() and sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > 256 << 20:
            raise ValueError('256 MiB review output budget reached')
    state, mesh, prepared, bindings = inspector(run)
    check()
    output.mkdir(parents=True, exist_ok=False)
    result = {'schema': 'surface-silhouette-review-v1', 'status': 'RUNNING', 'source_run': str(run),
              'scope': 'ALL_SELECTED_VIEWS_SAM_CONSISTENCY_ONLY', 'views': [],
              'selected_indices': [x[0]['input_index'] for x in prepared],
              'excluded_indices': state['stages']['surface'].get('missing_indices'),
              'full_object_accepted': False, 'physical_accuracy': 'NOT_EVALUATED',
              'limitations': LIMITATIONS, 'mesh_vertices': len(mesh['vertices']), 'mesh_faces': len(mesh['faces']),
              'seconds_limit': seconds, 'output_limit_bytes': 256 << 20,
              'runtime': {'python': sys.version, 'numpy': np.__version__, 'pillow': Image.__version__},
              'implementation': [{'path': str(p), 'sha256': _hash(p)} for p in (
                  Path(__file__), ROOT / 'integrations/demo_pipeline/quality.py',
                  ROOT / 'integrations/mvs_surface/scene.py', ROOT / 'integrations/viewer/ply_data.py') if p.is_file()]}
    if source_kind is not None:
        result.update(source_kind=source_kind, source_scope=state['source_scope'],
                      pipeline_acceptance=False)
        extra = ROOT / 'integrations/demo_pipeline/experimental_silhouette.py'
        result['implementation'].append({'path': str(extra), 'sha256': _hash(extra)})
    def save():
        (output / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    try:
        vertices, faces = np.asarray(mesh['vertices']), np.asarray(mesh['faces'])
        for frame, camera, sam, valid, photo in prepared:
            check()
            uv = project(vertices, camera['R'], camera['C'], camera['K'])
            w, h = camera['width'], camera['height']
            pred = raster(uv, faces, w, h, 2, check)
            coarse = raster(uv, faces, w, h, 1, check)
            with Image.open(photo) as image:
                overlay = np.array(image.convert('RGB'))
            overlay[_edge(sam)] = [0, 255, 0]
            overlay[_edge(pred)] = [255, 0, 255]
            name = 'view_%06d' % frame['input_index']
            Image.fromarray(overlay).save(output / (name + '_overlay.png'))
            Image.fromarray(pred.astype('uint8') * 255).save(output / (name + '_silhouette.png'))
            result['views'].append({'input_index': frame['input_index'], 'image_name': frame['image_name'],
                'scene_global_id': camera['global_id'], 'dimensions': [w, h],
                'metrics_2x': metrics(pred, sam, valid), 'metrics_1x': metrics(coarse, sam, valid),
                'raster_disagreement_pixels': int(np.count_nonzero(pred != coarse)),
                'overlay': name + '_overlay.png', 'silhouette': name + '_silhouette.png'})
            save()
        check()
        if not bindings.unchanged():
            raise ValueError('source changed during review')
        if any(_hash(Path(b['path'])) != b['sha256'] for b in result['implementation']):
            raise ValueError('review implementation changed during execution')
        values = [r['metrics_2x']['IoU'] for r in result['views'] if r['metrics_2x']['IoU'] is not None]
        result.update(status='COMPLETED_SAM_CONSISTENCY_ONLY', source_unchanged=True,
                      IoU_min=min(values) if values else None, IoU_median=float(np.median(values)) if values else None,
                      IoU_max=max(values) if values else None)
        cards = ''.join('<article><h2>View %s · IoU %s</h2><img src="%s"></article>' %
                        (v['input_index'], v['metrics_2x']['IoU'], html.escape(v['overlay'])) for v in result['views'])
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>SAM silhouette consistency</title>'
            '<style>body{font:16px sans-serif;margin:2em;background:#eee}article{display:inline-block;width:46%;padding:1%}img{width:100%}</style>'
            '<h1>全选定视图：网格与 SAM 掩膜一致性</h1><p>绿色：SAM；紫色：网格。不是独立真值或物理精度。未选视图和不可见表面仍未知。</p>'
            + cards, encoding='utf-8')
    except Exception as exc:
        result.update(status='INCOMPLETE', error=type(exc).__name__ + ': ' + str(exc), source_unchanged=bindings.unchanged())
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - start, source_bindings=list(bindings.records.values()),
                      completed_views=len(result['views']), expected_views=len(prepared))
        result['output_bindings'] = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': _hash(p)}
                                     for p in sorted(output.iterdir()) if p.is_file() and p.name != 'RESULT.json']
        save()
        if result['status'] == 'COMPLETED_SAM_CONSISTENCY_ONLY':
            try:
                check()
            except Exception as exc:
                result.update(status='INCOMPLETE', error=type(exc).__name__ + ': ' + str(exc),
                              elapsed_seconds=time.monotonic() - start)
                save()
                raise
    return result
