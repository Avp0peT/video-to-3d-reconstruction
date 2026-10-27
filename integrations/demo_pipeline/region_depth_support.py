"""Bounded conditional depth evidence; never a surface acceptance gate."""
from pathlib import Path
import json
import time
import numpy as np
from PIL import Image
from integrations.video_masks.boundary_geometry_binding import _bind, _read, _need
from integrations.video_masks.face_mask_adapter import verify_depth
from integrations.openmvs_depth.dmap import read_dmap
from integrations.demo_pipeline.shape_support_audit import context


def project_pixels(points, camera):
    xyz = (np.asarray(points) - camera['C']) @ np.asarray(camera['R']).T
    q = xyz @ np.asarray(camera['K']).T
    uv = np.zeros((len(xyz), 2), dtype=np.int64)
    front = np.isfinite(q).all(axis=1) & (xyz[:, 2] > 0)
    uv[front] = np.floor(q[front, :2] / q[front, 2:] + .5).astype(np.int64)
    inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < camera['width']) & (uv[:, 1] >= 0) & (uv[:, 1] < camera['height'])
    return uv, xyz[:, 2], inside


def backproject(uv, depth, camera):
    rays = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(camera['K']).T
    return (rays * np.asarray(depth)[:, None]) @ np.asarray(camera['R']) + camera['C']


def classify(points, view, depth):
    """0 outside mask/valid, 1 unknown, 2 agree, 3 in front, 4 behind."""
    uv, z, inside = project_pixels(points, view['camera'])
    state = np.zeros(len(points), np.uint8)
    ids = np.flatnonzero(inside)
    ids = ids[view['mask'][uv[ids, 1], uv[ids, 0]] & view['valid'][uv[ids, 1], uv[ids, 0]]]
    state[ids] = 1
    if depth is not None:
        d = depth[uv[ids, 1], uv[ids, 0]]
        known = np.isfinite(d) & (d > 0)
        ii = ids[known]; dd = d[known]
        state[ii] = np.where(abs(z[ii] - dd) <= .02 * dd, 2, np.where(z[ii] < dd, 3, 4))
    return state, uv


def separated_support(points, cameras, agreeing, degrees=8, check=None):
    """Conservative greedy pairwise separated triple; no coincident-center votes."""
    result = np.zeros(len(points), bool)
    for n, point in enumerate(points):
        if check is not None and n % 256 == 0:
            check()
        directions = []
        for j in np.flatnonzero(agreeing[n]):
            d = np.asarray(cameras[j]['C']) - point
            norm = np.linalg.norm(d)
            if norm <= 1e-12:
                continue
            d = d / norm
            if all(float(d @ old) <= np.cos(np.deg2rad(degrees)) for old in directions):
                directions.append(d)
            if len(directions) >= 3:
                result[n] = True
                break
    return result


def region_raster(spec, view):
    _need(spec['decode_index'] == view['decode_index'], 'region frame identity mismatch')
    h, w = view['mask'].shape
    region = np.zeros((h, w), bool)
    _need(bool(spec['cells']), 'empty region cells')
    for cell in spec['cells']:
        b = cell['bounds']
        _need(len(b) == 4 and all(type(x) is int for x in b), 'integer region bounds required')
        x0, y0, x1, y1 = b
        _need(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, 'region outside native domain')
        region[y0:y1, x0:x1] = True
    return region & view['mask'] & view['valid']


def check_plan(plan, oldplan):
    _need(plan.get('contract_version') == 'mvs-surface-run-v1', 'depth plan contract mismatch')
    _need(plan['scene']['sha256'] == oldplan['scene']['sha256'], 'depth scene identity mismatch')
    def indexed(p):
        rows = {Path(x['image']['name']).name: x for x in p['frames']}
        _need(len(rows) == len(p['frames']), 'duplicate plan frame')
        return rows
    a, b = indexed(plan), indexed(oldplan)
    _need(a.keys() == b.keys(), 'depth plan frame set mismatch')
    for name in a:
        for role in ('image_file', 'mask', 'validity'):
            _need(all(a[name][role][k] == b[name][role][k] for k in ('sha256', 'bytes')), 'depth plan input identity mismatch')
            _need(Path(a[name][role]['path']).resolve() == Path(b[name][role]['path']).resolve(), 'depth plan input path mismatch')
        for k in ('global_id', 'width', 'height', 'K', 'R', 'C'):
            _need(np.array_equal(a[name]['image'][k], b[name]['image'][k]), 'depth plan camera mismatch')


def run(source, depth_run, region_json, output):
    started = time.monotonic(); refs = []
    source, depth_run, region_json, output = map(lambda p: Path(p).resolve(), (source, depth_run, region_json, output))
    _need(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, depth_run, region_json.parent)), 'fresh disjoint output required')
    output.mkdir(parents=True)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    def budget():
        if time.monotonic() - started > 600:
            raise TimeoutError('600 second CPU audit deadline')
    try:
        _bind(__file__, refs)
        _bind(Path(__file__).resolve().parents[2] / 'scripts/audit_region_depth_support.py', refs)
        _, _, _, _, views = context(source, None, refs)
        budget()
        plan = _read(depth_run / 'plan.json', refs)
        check_plan(plan, _read(source / 'surface/plan.json', refs))
        _bind(plan['scene']['path'], refs, plan['scene'])
        result = _read(depth_run / 'result.json', refs)
        _need(result.get('status') == 'SURFACE_GENERATED_NOT_ACCEPTED' and not result.get('input_drift'), 'depth run incomplete or drifted')
        spec = _read(region_json, refs)
        target = [v for v in views if v['decode_index'] == spec['decode_index']]
        _need(len(target) == 1, 'region target identity absent')
        target = target[0]; region = region_raster(spec, target)
        names = {Path(v['camera']['name']).name: v for v in views}
        depths = {}
        for path in sorted((depth_run / 'work').glob('depth*.dmap')):
            budget(); d = read_dmap(path)
            name = Path(d.image_name_utf8.replace('\\', '/')).name
            _need(name in names and name not in depths, 'unknown or duplicate depth identity')
            verify_depth(d, names[name]['camera'])
            _bind(path, refs, dict(bytes=d.file_bytes, sha256=d.sha256))
            depths[name] = d.depth.copy()
        save('BINDINGS_PRE.json', refs)
        rng = np.random.default_rng(0)
        points = np.empty((0, 3)); origins = np.empty(0, int); priorities = np.empty(0)
        matrix = []; total = 0
        for j, v in enumerate(views):
            budget(); name = Path(v['camera']['name']).name; d = depths.get(name)
            row = dict(decode_index=v['decode_index'], depth_available=d is not None)
            if d is None or v is target:
                row['candidate_count_before_sampling'] = 0; matrix.append(row); continue
            yy, xx = np.indices(d.shape)
            good = v['mask'] & v['valid'] & np.isfinite(d) & (d > 0) & (xx % 2 == 0) & (yy % 2 == 0)
            y, x = np.where(good)
            p = backproject(np.column_stack((x, y)), d[y, x], v['camera'])
            uv, _, inside = project_pixels(p, target['camera'])
            ids = np.flatnonzero(inside); ids = ids[region[uv[ids, 1], uv[ids, 0]]]
            p = p[ids]; total += len(p); row['candidate_count_before_sampling'] = len(p)
            state, _ = classify(p, target, depths.get(Path(target['camera']['name']).name))
            row['target_state_counts'] = {label: int((state == code).sum()) for code, label in enumerate(('outside', 'unknown', 'agree', 'front', 'behind'))}
            matrix.append(row)
            points = np.concatenate((points, p)); origins = np.concatenate((origins, np.full(len(p), j)))
            priorities = np.concatenate((priorities, rng.random(len(p))))
            if len(points) > 50000:
                keep = np.argpartition(priorities, 49999)[:50000]
                points, origins, priorities = points[keep], origins[keep], priorities[keep]
        # Fixed-seed uniform priority reservoir across all source pixels; no hand selection.
        states = np.zeros((len(points), len(views)), np.uint8)
        for j, v in enumerate(views):
            budget(); states[:, j], _ = classify(points, v, depths.get(Path(v['camera']['name']).name))
        target_index = next(j for j, v in enumerate(views) if v is target)
        agreeing = states == 2
        agreeing[:, target_index] = False
        # Exclude the generating depth map: its agreement is tautological.
        agreeing[np.arange(len(points)), origins] = False
        supported = (agreeing.sum(axis=1) >= 3) & separated_support(points, [v['camera'] for v in views], agreeing, check=budget)
        accepted = supported & np.isin(states[:, target_index], [1, 2])
        p = points[accepted]
        with (output / 'candidates.ply').open('w', encoding='ascii') as f:
            f.write('ply\nformat ascii 1.0\nelement vertex %d\nproperty double x\nproperty double y\nproperty double z\nend_header\n' % len(p))
            np.savetxt(f, p, fmt='%.17g')
        np.savez_compressed(output / 'CANDIDATE_LEDGER.npz', points=points, source_view=origins, states=states, other_agreeing=agreeing, selected=accepted)
        with Image.open(target['image']) as im:
            rgb = np.array(im.convert('RGB'))
        uv, _, inside = project_pixels(p, target['camera'])
        rgb[uv[inside, 1], uv[inside, 0]] = [0, 255, 255]
        Image.fromarray(rgb).save(output / 'target_overlay.png')
        for j, row in enumerate(matrix):
            row['sampled_origin_candidates'] = int((origins == j).sum())
            row['selected_origin_candidates'] = int(((origins == j) & accepted).sum())
            row['validation_state_counts'] = {label: int((states[:, j] == code).sum()) for code, label in enumerate(('outside', 'unknown', 'agree', 'front', 'behind'))}
        budget()
        post = []
        for r in refs:
            _bind(r['path'], post, r)
        save('BINDINGS_POST.json', post)
        save('RESULT.json', dict(status='DEPTH_SUPPORTED_CANDIDATES_NOT_ACCEPTED', source_unchanged=True, roi_pixels=int(region.sum()), total_candidates=total, sampled_candidates=len(points), selected_candidates=len(p), target_unknown_selected=int((accepted & (states[:, target_index] == 1)).sum()), matrix=matrix, views=[dict(decode_index=v['decode_index'], camera_name=v['camera']['name']) for v in views], policy=dict(stride=2, max_candidates=50000, seed=0, relative_depth_tolerance=.02, min_other_views=3, pairwise_direction_degrees=8, seconds=600, threads=2), elapsed_seconds=time.monotonic()-started, limitations=['Shared depth estimates are correlated, not independent truth.', 'No target depth means unknown; candidates do not establish physical surface in that region.', 'Generating and target depth views are excluded from the three validating views.', 'Greedy pairwise direction separation is conservative and may reject valid triples.', 'No geometry adopted, no surface interpolated, no native or GPU execution.']))
    except Exception as exc:
        save('FAILURE.json', dict(status='FAILED_NOT_ACCEPTED', error=repr(exc), elapsed_seconds=time.monotonic()-started, bindings=refs))
        raise
