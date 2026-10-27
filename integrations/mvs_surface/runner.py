"""Plan and execute a fixed CPU OpenMVS surface recipe in a new directory.

Masks are explicitly supplied undistorted labels. This engine binds their bytes
but does not authenticate SAM3/B provenance or certify segmentation quality.
Planning never starts a process; execution is a separate explicit CLI action.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
from pathlib import Path
import time
import unicodedata
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from .scene import read_scene
from .photo_inputs import validate as validate_photo_source
from integrations.viewer.ply_data import read_ply

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = 'mvs-surface-run-v1'
PROFILE = 'openmvs-2.4.0-cpu-preserve-raw-v1'
LOW_TEXTURE_PROFILE = 'openmvs-2.4.0-cpu-low-texture-v1'
PROFILES = (PROFILE, LOW_TEXTURE_PROFILE)
LOW_TEXTURE_CONFIG = 'Descriptor Min Magnitude Threshold = 0.01\n'
TOOL_HASHES = {
    'DensifyPointCloud.exe': 'baa34151aa66ca5b909645330fdb79c271087f08759872dd2d474717dc901d77',
    'ReconstructMesh.exe': 'bb84317e4881effed587a560a80c574c1e22fc3dd9b1d9655127944f66b8abcc',
}
CODE_FILES = [
    'scripts/build_mvs_surface.py', 'integrations/mvs_surface/runner.py',
    'integrations/mvs_surface/scene.py', 'integrations/mvs_surface/supervision.py',
    'integrations/mvs_surface/__init__.py',
    'integrations/mvs_surface/photo_inputs.py',
    'integrations/mvs_surface/experimental_inputs.py', 'scripts/prepare_experimental_surface_inputs.py',
    'scripts/prepare_mvs.py', 'scripts/prepare_mvs_masks.py',
    'integrations/mvs_masks/manifest_io.py', 'integrations/mvs_masks/mask_geometry.py',
    'integrations/mvs_surface/general_subset.py', 'integrations/mvs_surface/subset.py',
    'scripts/derive_general_subset.py',
    'integrations/cad_delivery/supervise.py', 'integrations/viewer/ply_data.py',
]
DENSE_OPTIONS = {
    'resolution-level': 1, 'max-resolution': 1024, 'min-resolution': 640,
    'sub-resolution-levels': 2, 'number-views': 5, 'number-views-fuse': 2,
    'iters': 3, 'geometric-iters': 2, 'estimate-colors': 2, 'estimate-normals': 2,
    'estimate-scale': 0, 'estimate-segmentation': 0, 'sub-scene-area': 0,
    'sample-mesh': 0, 'fusion-mode': 0, 'fusion-filter': 2,
    'fusion-depth-diff-threshold': 0.01, 'fusion-reprojection-threshold': 1.2,
    'postprocess-dmaps': 0, 'filter-point-cloud': 0, 'export-number-views': 0,
    'roi-border': 0, 'estimate-roi': 0, 'crop-to-roi': 0, 'up-axis': -1,
    'remove-dmaps': 0, 'tower-mode': 0, 'normalize-coordinates': 0,
    'force-neighbors-from-images': 0, 'sample-mesh-for-neighbors': 0,
}
MESH_OPTIONS = {
    'min-point-distance': 1.5, 'integrate-only-roi': 0, 'constant-weight': 1,
    'free-space-support': 0, 'thickness-factor': 1, 'quality-factor': 1,
    'decimate': 1, 'target-face-num': 0, 'remove-spurious': 0,
    'remove-spikes': 0, 'close-holes': 0, 'smooth': 0, 'edge-length': 0,
    'roi-border': 0, 'crop-to-roi': 0, 'split-max-area': 0,
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def binding(path: str | Path, limit: int = 128 * 1024**2) -> dict:
    p = Path(path).resolve(strict=True)
    if not p.is_file() or p.stat().st_size > limit:
        raise ValueError(f'invalid or oversized file: {p}')
    digest = hashlib.sha256(); total = 0
    with p.open('rb') as stream:
        before = os.fstat(stream.fileno())
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > limit:
                raise ValueError('file exceeded byte bound during reading')
            digest.update(block)
        after = os.fstat(stream.fileno())
    now = p.stat()
    if (total != before.st_size or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError('input changed during hash read')
    return {'path': str(p), 'bytes': total, 'sha256': digest.hexdigest()}


def _under(root: Path, relative: str) -> Path:
    p = root.joinpath(*relative.split('/')).resolve(strict=True)
    p.relative_to(root)
    if not p.is_file():
        raise ValueError('expected regular input file')
    return p


def _image(path: Path, size: tuple[int, int], labels: bool = False) -> tuple[dict, np.ndarray | None]:
    record = binding(path, 64 * 1024**2)
    with path.open('rb') as stream:
        data = stream.read(64 * 1024**2 + 1)
    if len(data) > 64 * 1024**2:
        raise ValueError('image grew beyond decode bound')
    if hashlib.sha256(data).hexdigest() != record['sha256']:
        raise ValueError('image changed before decode')
    with Image.open(io.BytesIO(data)) as image:
        if (image.size != size or math.prod(size) > 16_777_216
                or getattr(image, 'n_frames', 1) != 1 or image.getexif().get(274, 1) != 1):
            raise ValueError('image dimensions, frame count or orientation mismatch')
        if labels:
            if (image.format != 'PNG' or image.mode != 'L' or len(data) < 26
                    or data[24:26] != b'\x08\x00' or 'transparency' in image.info):
                raise ValueError('label must be original 8-bit L PNG without transparency')
            pixels = np.array(image)
            if not np.isin(pixels, (0, 255)).all():
                raise ValueError('non-binary label values')
        else:
            if image.format != 'PNG' or image.mode not in ('L', 'RGB') or 'transparency' in image.info:
                raise ValueError('prepared image must be opaque L/RGB PNG')
            image.load(); pixels = None
    return record, pixels


def _budgets(total_seconds: int, stage_seconds: int, memory_bytes: int, output_bytes: int) -> dict:
    for key, value, low, high in [
        ('total_seconds', total_seconds, 10, 1800), ('stage_seconds', stage_seconds, 5, 1800),
        ('memory_bytes', memory_bytes, 128 * 1024**2, 12 * 1024**3),
        ('output_bytes', output_bytes, 16 * 1024**2, 8 * 1024**3),
    ]:
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'invalid budget: {key}')
    if stage_seconds > total_seconds:
        raise ValueError('stage budget exceeds total budget')
    return dict(total_seconds=total_seconds, stage_seconds=stage_seconds,
                memory_bytes=memory_bytes, output_bytes=output_bytes, cleanup_seconds_per_stage=5,
                threads=2, poll_seconds=0.1, native_launch_cap=2, retries=0)


def make_plan(scene_path: str | Path, image_root: str | Path, mask_root: str | Path,
              validity_root: str | Path, tool_root: str | Path, *, total_seconds: int,
              stage_seconds: int, memory_bytes: int, output_bytes: int,
              input_origin: str = 'unverified', ignore_label: int = 0,
              photo_source: str | Path | None = None, profile: str = PROFILE) -> dict:
    """Validate complete inputs and bind a deterministic execution recipe; no native call."""
    if profile not in PROFILES:
        raise ValueError('unsupported fixed surface profile')
    budget = _budgets(total_seconds, stage_seconds, memory_bytes, output_bytes)
    if input_origin not in ('unverified', 'synthetic', 'real', 'experimental-real'):
        raise ValueError('input_origin must be explicitly declared')
    if (input_origin in ('real', 'experimental-real')) != (photo_source is not None):
        raise ValueError('real inputs require --photo-source; other input origins must not supply it')
    if input_origin == 'experimental-real':
        from .experimental_inputs import validate as validate_experimental_source
        photo_link = validate_experimental_source(photo_source,scene_path,image_root,mask_root,validity_root)
    else:
        photo_link = validate_photo_source(photo_source,scene_path,image_root) if photo_source is not None else None
    if type(ignore_label) is not int or ignore_label not in (0, -1):
        raise ValueError('ignore_label must be 0 (masked) or -1 (separate control)')
    roots = [Path(p).resolve(strict=True) for p in (image_root, mask_root, validity_root, tool_root)]
    if not all(p.is_dir() for p in roots):
        raise ValueError('input roots must be directories')
    image_root, mask_root, validity_root, tool_root = roots
    scene = read_scene(scene_path)
    if (not 2 <= len(scene['images']) <= 256 or scene['point_count'] == 0
            or scene['min_point_views'] < 2):
        raise ValueError('surface profile requires 2..256 images and sparse points')
    scene_binding = binding(scene_path)
    if (scene_binding['sha256'] != scene['sha256'] or scene_binding['bytes'] != scene['bytes']):
        raise ValueError('scene bytes changed after parsing')
    if scene['transform'] != np.eye(4).tolist():
        raise ValueError('nonidentity scene transform requires a separate reviewed profile')
    tools = []
    for name, digest in TOOL_HASHES.items():
        rec = binding(tool_root / name)
        if rec['sha256'] != digest:
            raise ValueError(f'fixed CPU tool identity mismatch: {name}')
        tools.append(rec)
    dlls = sorted(tool_root.glob('*.dll'), key=lambda p: p.name.casefold())
    if not dlls or len(dlls) > 128:
        raise ValueError('missing or excessive private DLL set')
    tools += [binding(p) for p in dlls]
    frames = []; targets = set()
    for image in scene['images']:
        name = image['name']
        if image['mask_name'] or name.split('/')[0].casefold().startswith('__mvs_'):
            raise ValueError('scene maskName or reserved staging prefix is unsupported')
        if Path(name).suffix.casefold() != '.png':
            raise ValueError('prepared scene images must be PNG')
        stem = Path(name).stem
        key = unicodedata.normalize('NFC', stem).casefold()
        if key in targets:
            raise ValueError('mask basename collision')
        targets.add(key)
        size = (image['width'], image['height'])
        img, _ = _image(_under(image_root, name), size)
        mask, mask_pixels = _image(_under(mask_root, stem + '.mask.png'), size, True)
        validity, valid_pixels = _image(_under(validity_root, stem + '.validity.png'), size, True)
        if not np.count_nonzero(mask_pixels) or np.any((mask_pixels == 255) & (valid_pixels == 0)):
            raise ValueError('empty foreground or foreground in unknown mask domain')
        frames.append({'image': image, 'image_file': img, 'mask': mask, 'validity': validity,
                       'mask_name': stem + '.mask.png', 'validity_name': stem + '.validity.png',
                       'foreground_pixels': int(np.count_nonzero(mask_pixels)),
                       'unknown_pixels': int(np.count_nonzero(valid_pixels == 0))})
    staging_bytes = scene['bytes'] + sum(f[k]['bytes'] for f in frames for k in ('image_file', 'mask', 'validity'))
    if staging_bytes + 1024**2 >= output_bytes:
        raise ValueError('output budget does not cover immutable staging plus metadata reserve')
    recipe = {
        'contract_version': CONTRACT, 'profile': profile,
        'arguments': {'scene_path': str(Path(scene_path).resolve()), 'image_root': str(image_root),
                      'mask_root': str(mask_root), 'validity_root': str(validity_root), 'tool_root': str(tool_root),
                      'total_seconds': total_seconds, 'stage_seconds': stage_seconds,
                      'memory_bytes': memory_bytes, 'output_bytes': output_bytes,
                      'input_origin': input_origin, 'ignore_label': ignore_label,
                      'photo_source': str(Path(photo_source).resolve()) if photo_source is not None else None},
        'scene': scene_binding, 'point_count': scene['point_count'], 'frames': frames,
        'tools': tools, 'implementation': [binding(ROOT / p) for p in CODE_FILES],
        'budgets': budget, 'staging_bytes': staging_bytes,
        'options': {'dense': DENSE_OPTIONS.copy(), 'mesh': MESH_OPTIONS.copy()},
        'input_origin': 'caller_declared_' + input_origin,
        'mask_source_chain': 'NOT_VERIFIED', 'mask_quality': 'NOT_EVALUATED',
        'cross_view': 'NOT_EVALUATED', 'mask_consumption': 'NOT_VERIFIED',
        'scale_status': 'unknown', 'length_unit': None, 'native_execution': 'NOT_RUN',
        'photo_inputs': photo_link,
    }
    if profile == LOW_TEXTURE_PROFILE:
        recipe['arguments']['profile'] = profile
        data = LOW_TEXTURE_CONFIG.encode('ascii')
        recipe['config_initial'] = dict(filename='__mvs_dense.cfg', text=LOW_TEXTURE_CONFIG,
                                       bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return recipe


def plan_files(plan: dict) -> list[dict]:
    return [plan['scene'], *plan['tools'], *plan['implementation'],
            *((plan.get('photo_inputs') or {}).get('bindings',[])),
            *(frame[key] for frame in plan['frames'] for key in ('image_file', 'mask', 'validity'))]


def save_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        stream.write(text)
        stream.flush()


def _copy(record: dict, target: Path, deadline: float = math.inf) -> dict:
    if binding(record['path']) != record:
        raise ValueError('staging source identity changed')
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(record['path']).open('rb') as source, target.open('xb') as out:
        remaining = record['bytes']
        while remaining:
            if time.monotonic() >= deadline:
                raise TimeoutError('budget exhausted during byte copy')
            block = source.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError('staging input was truncated')
            out.write(block); remaining -= len(block)
        if source.read(1):
            raise ValueError('staging input grew beyond frozen length')
    actual = binding(target)
    if actual['sha256'] != record['sha256'] or actual['bytes'] != record['bytes']:
        raise ValueError('staged bytes differ from frozen source')
    return actual


def stage_inputs(plan: dict, work: Path, deadline: float) -> list[dict]:
    staged = [_copy(plan['scene'], work / '__mvs_input.mvs', deadline)]
    for frame in plan['frames']:
        if time.monotonic() >= deadline:
            raise TimeoutError('budget exhausted during input staging')
        for record, relative in [(frame['image_file'], frame['image']['name']),
                (frame['mask'], '__mvs_masks/' + frame['mask_name']),
                (frame['validity'], '__mvs_validity/' + frame['validity_name'])]:
            staged.append(_copy(record, work / relative, deadline))
    config = work / '__mvs_empty.cfg'
    config.write_bytes(b'')
    staged.append(binding(config))
    if plan['profile'] == LOW_TEXTURE_PROFILE:
        (work / '__mvs_dense.cfg').write_bytes(plan['config_initial']['text'].encode('ascii'))
        _verify_initial_dense_config(plan, work)
    return staged


def _verify_initial_dense_config(plan: dict, work: Path) -> None:
    if plan['profile'] != LOW_TEXTURE_PROFILE:
        return
    rec = binding(work / '__mvs_dense.cfg', 1 << 20)
    expected = plan['config_initial']
    if rec['bytes'] != expected['bytes'] or rec['sha256'] != expected['sha256']:
        raise ValueError('initial dense configuration changed')


def _effective_dense_config(plan: dict, work: Path, evidence: Path) -> dict | None:
    if plan['profile'] != LOW_TEXTURE_PROFILE:
        return None
    path = work / '__mvs_dense.cfg'
    before = binding(path, 1 << 20)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != before['sha256']:
        raise ValueError('effective dense configuration changed while reading')
    values = []
    for line in data.decode('utf-8-sig').splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            if key.strip() == 'Descriptor Min Magnitude Threshold':
                values.append(value.strip())
    if len(values) != 1:
        raise ValueError('effective descriptor threshold missing or duplicate')
    try:
        value = float(values[0])
    except ValueError as exc:
        raise ValueError('invalid effective descriptor threshold') from exc
    if not math.isfinite(value) or value != .01:
        raise ValueError('effective descriptor threshold differs from fixed profile')
    snapshot = evidence / 'dense_config_effective.cfg'
    with snapshot.open('xb') as stream:
        stream.write(data)
    return dict(descriptor_min_magnitude_threshold=value, native_config=before,
                effective_snapshot=binding(snapshot, 1 << 20),
                note='Native rewrites the dense configuration; initial bytes are not immutable staged input.')


def commands(plan: dict, work: Path) -> list[tuple[str, list[str]]]:
    tool = Path(plan['arguments']['tool_root'])
    common = ['-w', str(work) + os.sep, '--max-threads', '2', '--archive-type', '1',
              '--verbosity', '3', '--config-file', str(work / '__mvs_empty.cfg')]
    dense = [str(tool / 'DensifyPointCloud.exe'), '-i', str(work / '__mvs_input.mvs'),
             '-o', str(work / '__mvs_dense.mvs'), '--mask-path', str(work / '__mvs_masks'),
             '--ignore-mask-label', str(plan['arguments']['ignore_label']),
             '--dense-config-file', str(work / '__mvs_dense.cfg'), *common]
    mesh = [str(tool / 'ReconstructMesh.exe'), '-i', str(work / '__mvs_dense.mvs'),
            '-o', str(work / '__mvs_surface.mvs'), '--export-type', 'ply', *common]
    for argv, options in [(dense, DENSE_OPTIONS), (mesh, MESH_OPTIONS)]:
        for key, value in options.items():
            argv.extend(['--' + key, str(value)])
    return [('dense', dense), ('mesh', mesh)]


def _surface_outputs(work: Path, stage: str) -> dict:
    stem = '__mvs_dense' if stage == 'dense' else '__mvs_surface'
    native = binding(work / (stem + '.mvs'), 512 * 1024**2)
    if native['bytes'] == 0:
        raise ValueError('empty native MVS output')
    names = [stem + '.ply'] if stage == 'dense' else [stem + '_raw.ply', stem + '.ply']
    out = {'native_scene': native, 'geometry': []}
    for name in names:
        path = work / name
        ply = read_ply(path, max_bytes=128 * 1024**2, max_vertices=500000, max_faces=1000000)
        if not ply['vertices'] or (stage == 'mesh' and not ply['faces']):
            raise ValueError('empty dense cloud or triangle surface')
        points = np.asarray(ply['vertices'], dtype=np.float64)
        if stage == 'mesh':
            faces = np.asarray(ply['faces'], dtype=np.int64)
            cross = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
            if not np.isfinite(cross).all() or np.any(np.max(np.abs(cross), axis=1) == 0):
                raise ValueError('degenerate or numerically invalid surface triangles')
        rec = binding(path)
        if rec['sha256'] != ply['sha256']:
            raise ValueError('geometry changed after readback')
        out['geometry'].append({**rec, 'vertices': len(ply['vertices']), 'triangles': len(ply['faces']),
                                'bbox': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                                'role': 'pre_cleanup_surface' if name.endswith('_raw.ply') else stage})
    return out


def _dense_mask_log_guard(work: Path, evidence: Path, native: dict) -> dict:
    """Fail closed on missing logs or frozen v2.4.0 mask-load diagnostics.

    No diagnostic found is only a negative log check, not mask consumption proof.
    Scan bytes because native logs may use Windows ANSI rather than UTF-8.
    """
    adapter = native.get('mvs_surface_adapter')
    if (not isinstance(adapter, dict) or adapter.get('schema') != 'mvs-surface-supervision-v1'
            or adapter.get('output') != str(evidence)):
        raise ValueError('dense mask log guard: missing/mismatched supervision log directory')
    logs = sorted(work.glob('DensifyPointCloud-*.log'))
    if not logs:
        raise ValueError('dense mask log guard: required native DensifyPointCloud log missing')
    patterns = [
        rb'warning:\s*can\s+not\s+load\s+the\s+segmentation\s+mask\b',
        rb'error:\s*cannot\s+load\s+mask\s+image\b',
    ]
    files, matches = [], []
    for path in [evidence / 'stdout.txt', evidence / 'stderr.txt', *logs]:
        record = binding(path, 16 * 1024**2)
        data = path.read_bytes()
        if (len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']
                or binding(path, 16 * 1024**2) != record):
            raise ValueError('dense mask log guard: log changed during read')
        if b'\x00' in data or (path in logs and not data.strip()):
            raise ValueError('dense mask log guard: empty/binary native log')
        files.append(record)
        for pattern in patterns:
            for match in re.finditer(pattern, data, re.IGNORECASE):
                matches.append({'path': str(path), 'byte_offset': match.start(),
                                'diagnostic': match.group().decode('ascii')})
    return {'status': 'MASK_LOAD_DIAGNOSTIC_FOUND' if matches else 'NO_MASK_LOAD_DIAGNOSTIC_FOUND',
            'files': files, 'matches': matches, 'mask_consumption': 'NOT_VERIFIED',
            'scope': 'negative diagnostic guard only; no positive native mask loading proof'}


def execute_plan(plan: dict, output: str | Path) -> dict:
    """Execute at most two native launches; caller must explicitly request this action."""
    from .supervision import run_supervised, strict_directory_size
    start = time.monotonic()
    target = Path(output).resolve()
    if target.exists():
        raise ValueError('output must not exist; no resume, overwrite or retry')
    for name in ('image_root', 'mask_root', 'validity_root', 'tool_root'):
        if target.is_relative_to(Path(plan['arguments'][name]).resolve()):
            raise ValueError('output must not be inside an input/tool directory')
    if target.is_relative_to(Path(plan['arguments']['scene_path']).resolve().parent):
        raise ValueError('output must not be inside the source scene directory')
    # Reject overlap before any directory creation/copy, including ancestor aliases.
    for row in plan_files(plan):
        p = Path(row['path']).resolve()
        if target == p or target in p.parents or p in target.parents:
            raise ValueError('output overlaps a frozen input')
    current = make_plan(**plan['arguments'])
    if current != plan:
        raise ValueError('plan does not match current code/tools/inputs/recipe')
    deadline = start + plan['budgets']['total_seconds']
    target.mkdir(parents=False)
    work = target / 'work'; work.mkdir()
    save_json(target / 'plan.json', plan, exclusive=True)
    result = {'contract_version': CONTRACT, 'profile': plan['profile'], 'status': 'RUNNING', 'exit_code': None,
              'started_utc': utc(), 'stages': [], 'native_launches': 0,
              'known_native_launches': 0, 'launch_attempts': 0, 'launch_count_complete': True, 'staged': [],
              'dense_status': 'NOT_RUN', 'raw_mesh_status': 'NOT_RUN',
              'mask_consumption': 'NOT_VERIFIED', 'mask_source_chain': 'NOT_VERIFIED',
              'quality_status': 'NOT_EVALUATED', 'CAD_status': 'NOT_RUN',
              'scale_status': 'unknown', 'length_unit': None, 'input_origin': plan['input_origin'],
              'photo_input_identity_status': (plan.get('photo_inputs') or {}).get('status','NOT_VERIFIED'),
              'photo_geometry_lineage': 'NOT_VERIFIED'}
    try:
        save_json(target / 'result.json', result)
        result['staged'] = stage_inputs(plan, work, deadline)
        for name, argv in commands(plan, work):
            if name == 'dense':
                _verify_initial_dense_config(plan, work)
            # An unexpected config edit must not change the second stage's
            # effective options while retaining the same recorded argv.
            if binding(work / '__mvs_empty.cfg')['sha256'] != hashlib.sha256(b'').hexdigest():
                raise ValueError('empty command configuration changed')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('shared execution budget exhausted')
            evidence = work / '__mvs_runtime' / name; evidence.mkdir(parents=True)
            step = {'name': name, 'argv': argv, 'cwd': str(work), 'started_utc': utc(), 'status': 'STARTING'}
            result['launch_attempts'] += 1
            result['launch_count_complete'] = False
            result['native_launches'] = None
            result['stages'].append(step); save_json(target / 'result.json', result)
            native = run_supervised(argv, cwd=work, output=evidence,
                wall_seconds=min(remaining, plan['budgets']['stage_seconds']),
                memory_bytes=plan['budgets']['memory_bytes'], output_bytes=plan['budgets']['output_bytes'], label=name)
            # Capture actual creation evidence before disk writes can fail. A
            # thrown supervisor call leaves the count unknown, never zero.
            if not isinstance(native, dict) or 'pid' not in native:
                raise RuntimeError('missing supervisor creation outcome')
            if native['pid'] is not None and (type(native['pid']) is not int or native['pid'] <= 0):
                raise RuntimeError('invalid supervisor creation identity')
            if native.get('pid') is not None:
                result['known_native_launches'] += 1
                if name == 'dense':
                    # A PID proves creation; the process may still be suspended.
                    result['dense_status'] = 'PROCESS_CREATED_OUTPUT_UNVERIFIED'
            result['launch_count_complete'] = True
            result['native_launches'] = result['known_native_launches']
            step.update(ended_utc=utc(), returncode=native.get('root_exit_code'),
                        pid=native.get('pid'), status='STOPPED')
            save_json(evidence / 'supervisor_result.json', native, exclusive=True)
            step['supervisor'] = binding(evidence / 'supervisor_result.json')
            if (native.get('status') != 'completed' or type(native.get('root_exit_code')) is not int
                    or native['root_exit_code'] != 0 or native.get('reclaim_confirmed') is not True
                    or type(native.get('pid')) is not int or native['pid'] <= 0
                    or native.get('argv') != argv or native.get('cwd') != str(work)
                    or native.get('errors') or native.get('log_errors')):
                raise RuntimeError(f'{name} failed supervision/exit/cleanup; no retry')
            if name == 'dense':
                effective = _effective_dense_config(plan, work, evidence)
                if effective is not None:
                    step['dense_config_effective'] = effective
                try:
                    step['mask_log_guard'] = _dense_mask_log_guard(work, evidence, native)
                except Exception as exc:
                    step['mask_log_guard'] = {'status': 'LOG_EVIDENCE_UNAVAILABLE',
                                              'error': f'{type(exc).__name__}: {exc}',
                                              'mask_consumption': 'NOT_VERIFIED'}
                    raise
                if step['mask_log_guard']['matches']:
                    raise RuntimeError('dense mask loading/decoding diagnostic detected; mesh not started')
            try:
                step['outputs'] = _surface_outputs(work, name)
            except Exception:
                step['status'] = 'OUTPUT_READBACK_FAILED'
                if name == 'dense':
                    result['dense_status'] = 'NO_VERIFIED_GEOMETRY'
                raise
            if time.monotonic() >= deadline or strict_directory_size(work) > plan['budgets']['output_bytes']:
                raise TimeoutError('shared budget/output threshold exceeded after native output')
            step['status'] = 'GEOMETRY_READBACK_PASSED'
            result['dense_status' if name == 'dense' else 'raw_mesh_status'] = 'NONEMPTY_FINITE_GEOMETRY'
            save_json(target / 'result.json', result)
        result.update(status='SURFACE_GENERATED_NOT_ACCEPTED', exit_code=0)
    except TimeoutError as exc:
        result.update(status='BUDGET_EXCEEDED', exit_code=5, error=str(exc))
    except Exception as exc:
        result.update(status='FAILED', exit_code=4, error=f'{type(exc).__name__}: {exc}')
    finally:
        drift = []
        effective_files = [record for step in result['stages']
                           for record in (step.get('dense_config_effective') or {}).values()
                           if isinstance(record, dict) and {'path', 'bytes', 'sha256'} <= record.keys()]
        for row in plan_files(plan) + result['staged'] + effective_files:
            try:
                if binding(row['path']) != row:
                    drift.append(row['path'])
            except Exception:
                drift.append(row['path'])
        result['input_drift'] = drift
        if drift:
            result.update(status='INPUT_DRIFT', exit_code=4)
        try:
            result['final_work_bytes'] = strict_directory_size(work)
            if result['exit_code'] == 0 and (time.monotonic() >= deadline
                    or result['final_work_bytes'] > plan['budgets']['output_bytes']):
                result.update(status='BUDGET_EXCEEDED', exit_code=5)
        except Exception as exc:
            result['final_observation_error'] = f'{type(exc).__name__}: {exc}'
            if result['exit_code'] == 0:
                result.update(status='OBSERVATION_ERROR', exit_code=4)
        result.update(ended_utc=utc(), wall_seconds=time.monotonic() - start)
        save_json(target / 'result.json', result)
    return result
