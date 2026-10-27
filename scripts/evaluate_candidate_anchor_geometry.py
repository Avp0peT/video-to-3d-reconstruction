"""One saved-camera diagnostic; no pose fitting, source mutation or publication."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src'), str(ROOT)]
import numpy as np
from PIL import Image
from review_mask_association import read, save, bind, verify, index_frames, verified_tracks, CLIPS
from review_local_motion import fresh_output
from integrations.demo_pipeline.coverage_expansion import camera_table
from kozlov_lab.stereo.candidate_geometry_inputs import build_anchor_groups
from kozlov_lab.stereo.candidate_anchor_geometry import evaluate_anchor_group

VERSION = 'candidate-anchor-geometry-v1'


def sealed_source(source, digest):
    source = Path(source).resolve(strict=True)
    task = source.parent.parent
    if source.name != 'point_support' or source.parent.name != 'real01':
        raise ValueError('sealed point-support layout required')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('explicit lowercase seal SHA256 required')
    refs = [verify(dict(path=str(task / 'SHA256SUMS.txt'), sha256=digest))]
    entries = {}
    for line in (task / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected, relative = line.split('  ', 1)
        path = (task / relative).resolve(strict=True)
        key = str(path).casefold()
        if not path.is_relative_to(task) or key in entries:
            raise ValueError('unsafe or duplicate sealed path')
        entries[key] = expected
        refs.append(verify(dict(path=str(path), sha256=expected)))
    required = [task / 'DELIVERY_CHECK.json', task / 'INDEPENDENT_AUDIT.json']
    required += [source / (name + '.json') for name in
                 ('PLAN', 'RESULT', 'POINT_SUPPORT_LEDGER', 'MASK_SELECTION_LEDGER')]
    if any(str(p.resolve()).casefold() not in entries for p in required):
        raise ValueError('incomplete source seal')
    for p in required[:2]:
        data = read(p)
        if data.get('status') != 'PASS' or data.get('failures') != []:
            raise ValueError('source delivery audit failed')
    plan, result, ledger = [read(source / (n + '.json')) for n in
                            ('PLAN', 'RESULT', 'POINT_SUPPORT_LEDGER')]
    if (result['status'] != 'POINT_SUPPORT_CONSUMED_MASK_UNACCEPTED'
            or result['source_unchanged'] is not True
            or read(source / 'MASK_SELECTION_LEDGER.json') != []):
        raise ValueError('source status or mask permissions differ')
    for group in ('source_bindings', 'implementation'):
        refs.extend(verify(r) for r in plan[group])
    return plan, ledger, refs


def require_bound(path, refs):
    path = Path(path).resolve(strict=True)
    found = [r for r in refs if Path(r['path']).resolve() == path]
    if not found or len({r['sha256'] for r in found}) != 1:
        raise ValueError('missing or contradictory authority binding: ' + str(path))
    return verify(found[0])


def verify_camera(c, image, camera, frame, mapping):
    """Read actual native camera identity and extrinsics, not requested options."""
    if (c['image_id'] != mapping['database_image_id'] or c['input_index'] != frame['input_index']
            or mapping['input_index'] != frame['input_index']
            or mapping['frame_id'] != frame['frame_id']
            or mapping['canonical_image_sha256'] != frame['sha256']
            or image.name != mapping['staged_image_name']):
        raise ValueError('camera/frame/image mapping differs')
    pose = image.cam_from_world()
    if (camera.model_name != 'SIMPLE_RADIAL'
            or (camera.width, camera.height) != (frame['width'], frame['height'])
            or not np.allclose(c['params'], camera.params, rtol=0, atol=1e-12)
            or not np.allclose(c['R'], pose.rotation.matrix(), rtol=0, atol=1e-12)
            or not np.allclose(c['t'], pose.translation, rtol=0, atol=1e-12)):
        raise ValueError('ledger/native calibration or pose differs')
    R = np.asarray(c['R'], float)
    if (not np.isfinite(R).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-10, rtol=0)
            or abs(np.linalg.det(R) - 1) > 1e-10 or c['params'][3] != 0):
        raise ValueError('unsupported rotation or radial distortion')
    return dict(c, camera_model=camera.model_name, width=camera.width, height=camera.height,
                frame_id=frame['frame_id'], canonical_image_sha256=frame['sha256'])


def load_cameras(local, registry, refs):
    result_path = local / 'RESULT.json'
    refs.append(require_bound(result_path, refs))
    data = read(result_path)
    if data['status'] != 'PARTIAL_CONDITIONAL_MODEL' or data['source_unchanged'] is not True:
        raise ValueError('saved local model lacks conditional status')
    refs.extend(verify(r) for r in data['source_bindings'])
    sfm_refs = [r for r in data['source_bindings'] if Path(r['path']).name == 'run.json'
                and Path(r['path']).parent.parent.name == 'WHOLE_VIDEO_NATIVE_20261016']
    if len(sfm_refs) != 1:
        raise ValueError('unique bound SfM run required')
    sfm_path = Path(sfm_refs[0]['path'])
    sfm = read(sfm_path)
    map_path = sfm_path.parent / 'frames_map.jsonl'
    refs.append(require_bound(map_path, refs))
    mappings = index_frames([read_json_line(s) for s in map_path.read_text(encoding='utf-8').splitlines() if s])
    old_masks = {}
    for mask in sfm['effective_options']['foreground']['mask_bindings']:
        index = mask['input_index']
        if type(index) is not int or index < 0 or index in old_masks:
            raise ValueError('invalid or duplicate native mask input index')
        old_masks[index] = mask
    import pycolmap as pc
    if pc.__version__ != '4.2.1':
        raise ValueError('pinned pycolmap 4.2.1 required')
    runtime = require_bound(pc.__file__, refs)
    models = []
    for group in data['groups']:
        for model in group.get('models', []):
            if model['status'] != 'PARTIAL_CONDITIONAL_MODEL':
                continue
            if not model['ba_converged'] or model['roundtrip_differences']:
                raise ValueError('saved model lacks convergence or native roundtrip')
            folder = local / ('group_%02d' % group['group_id']) / ('model_%d' % model['model_id'])
            bound = [require_bound(folder / name, refs) for name in
                     ('POINT_LEDGER.json', 'cameras.bin', 'images.bin', 'points3D.bin', 'rigs.bin', 'frames.bin')]
            refs.extend(bound)
            cameras = camera_table(read(folder / 'POINT_LEDGER.json'))
            if set(cameras) != set(model['registered_input_indices']):
                raise ValueError('ledger registered camera scope differs')
            native = pc.Reconstruction(folder)
            if set(native.reg_image_ids()) != {c['image_id'] for c in cameras.values()}:
                raise ValueError('native registered camera scope differs')
            verified = {}
            for index, c in cameras.items():
                frame = registry[index]
                if (frame['original_mask_status'] != 'OK' or frame['mask_binding'] is None
                        or old_masks[index]['sha256'] != frame['mask_binding']['sha256']):
                    raise ValueError('native camera original-mask binding differs')
                image = native.image(c['image_id'])
                verified[index] = verify_camera(c, image, native.cameras[image.camera_id], frame, mappings[index])
            models.append(dict(model_key=f"group_{group['group_id']:02d}/model_{model['model_id']}",
                               cameras=verified, binding=dict(files=bound, native_readback=True,
                               source_model_status=model['status'], geometry_accepted=False)))
    if not 1 <= len(models) <= 8:
        raise ValueError('bounded nonempty local camera models required')
    return models, runtime


def read_json_line(line):
    # Same strict decoder as the source module, without creating temporary files.
    from integrations.demo_pipeline import parallel_photo_audit as io
    return io.loads(line)


def mask_check(mask, xy):
    raw = np.asarray(xy, float)
    if raw.shape != (2,) or not np.isfinite(raw).all():
        raise ValueError('invalid saved COLMAP coordinates')
    x, y = np.floor(raw).astype(int)
    return bool(0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] == 255)


def add_mask_checks(tracking, ledger, registry, clip_start, refs):
    selected = {o['track_id'] for d in ledger for o in d['supported_observations']}
    masks = {}
    for track in tracking['tracks']:
        if track['track_id'] not in selected:
            continue
        for obs in track['observations']:
            index = obs['frame_index'] + clip_start
            frame = registry[index]
            if frame['original_mask_status'] != 'OK' or frame['mask_binding'] is None:
                continue
            if index not in masks:
                refs.extend([verify(dict(path=frame['path'], sha256=frame['sha256'])), verify(frame['mask_binding'])])
                with Image.open(frame['mask_binding']['path']) as image:
                    if image.mode != 'L' or image.size != (frame['width'], frame['height']):
                        raise ValueError('original mask format differs')
                    mask = np.asarray(image).copy()
                if not np.isin(mask, [0, 255]).all():
                    raise ValueError('original mask labels differ')
                masks[index] = mask
            frame.setdefault('measured_point_mask_checks', []).append(dict(track_id=track['track_id'],
                frame_index=obs['frame_index'], source_xy=deepcopy(obs['xy']),
                inside_original_mask=mask_check(masks[index], obs['xy'])))


def run(source, seal, coverage, local, tracks_review, output):
    start = time.monotonic()
    output = fresh_output(output, (source, coverage, local, tracks_review))
    plan, ledger, refs = sealed_source(source, seal)
    refs += [require_bound(coverage / (n + '.json'), refs) for n in ('INPUTS', 'RESULT')]
    inputs = read(coverage / 'INPUTS.json')
    if (read(coverage / 'RESULT.json')['status'] != 'PARTIAL_CONDITIONAL_EXPANSION'
            or inputs['track_profile']['profile_id'] != 'whole-video-incremental-v1'):
        raise ValueError('wrong original coverage inputs')
    refs.extend(verify(r) for r in inputs['source_bindings'])
    registry = index_frames(deepcopy(inputs['frames']))
    tracking = {}
    for lo, hi in CLIPS:
        name = f'{lo}_{hi}'
        data, bound = verified_tracks(tracks_review, name)
        for ref in bound:
            require_bound(ref['path'], refs)
        refs.extend(bound)
        tracking[name] = data
    implementation = [bind(ROOT / p) for p in (
        'scripts/evaluate_candidate_anchor_geometry.py',
        'src/kozlov_lab/stereo/candidate_geometry_inputs.py',
        'src/kozlov_lab/stereo/candidate_anchor_geometry.py',
        'src/kozlov_lab/stereo/candidate_point_consumer.py',
        'integrations/demo_pipeline/coverage_expansion.py',
        'integrations/demo_pipeline/coverage_triangulation.py',
        'integrations/demo_pipeline/partial_point_gate.py')]
    output.mkdir(parents=True)
    save(output / 'PLAN.json', dict(version=VERSION, status='FROZEN_BEFORE_NATIVE_READ_AND_GEOMETRY',
        source=bind(source / 'POINT_SUPPORT_LEDGER.json'), source_bindings=refs,
        implementation=implementation, source_seal_sha256=seal,
        camera_policy='saved native cameras only, no fitting or interpolation',
        anchor_policy='same segment, original foreground, actual mask pixel, strong FB<=1 LK<=25',
        thresholds=dict(two_view=dict(reprojection_px=4, angle_deg=1),
                        strict=dict(reprojection_px=2, angle_deg=2, loo_px=3)),
        geometry_accepted=False, publishable_for_geometry=False, scale_status='unknown'))
    try:
        models, runtime = load_cameras(local, registry, refs)
        save(output / 'CAMERAS.json', models)
        groups = []
        for lo, hi in CLIPS:
            name = f'{lo}_{hi}'
            subset = [d for d in ledger if lo <= d['input_index'] <= hi]
            add_mask_checks(tracking[name], subset, registry, lo, refs)
            prepared = build_anchor_groups(tracking[name], subset, registry, models, lo)
            for group in prepared['groups']:
                group['clip'] = name
                group['geometry'] = evaluate_anchor_group(group['anchors'])
                groups.append(group)
        save(output / 'GROUPS.json', groups)
        unique_queries = {(g['clip'], g['track_id'], g['identity_segment'], q['input_index']) for g in groups for q in g['queries']}
        available = {(g['clip'], g['track_id'], g['identity_segment'], q['input_index']) for g in groups for q in g['queries']
                     if q['status'] == 'SAVED_CAMERA_PRESENT'}
        missing = unique_queries - available
        # Query geometry remains untested even where an anchor-only point is computable.
        result = dict(version=VERSION, status='ANCHOR_DIAGNOSTIC_ONLY', groups=len(groups),
            unique_queries=len(unique_queries), queries_without_saved_camera=len(missing),
            native_saved_model_reads=len(models), native_camera_count=sum(len(m['cameras']) for m in models),
            native_runtime=runtime, numeric_attempts=sum(len(g['anchors']) >= 2 for g in groups),
            anchor_status_counts=dict(Counter(g['geometry']['status'] for g in groups)),
            candidate_query_geometry='NOT_EVALUATED', publishable_for_geometry=False, geometry_accepted=False,
            independent_quality_truth=False, physical_identity='UNVERIFIED', scale_status='unknown',
            new_model_points=0, new_masks=0, new_observations=0, pose_fits=0, tracking_calls=0,
            gpu_calls=0, mvs_calls=0, cad_calls=0, source_unchanged=True,
            duration_seconds=time.monotonic()-start)
        for ref in refs + implementation + [runtime]:
            verify(ref)
        save(output / 'FINAL_INPUT_BINDINGS.json', refs + implementation + [runtime])
        save(output / 'RESULT.json', result)
        return result
    except Exception as exc:
        save(output / 'FAILURE.json', dict(type=type(exc).__name__, message=str(exc),
                                         geometry_accepted=False, publishable_for_geometry=False))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('source', 'coverage', 'local', 'tracks-review', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--seal-sha256', required=True)
    args = p.parse_args()
    result = run(args.source, args.seal_sha256, args.coverage, args.local, args.tracks_review, args.output)
    print(result['status'])


if __name__ == '__main__':
    main()
