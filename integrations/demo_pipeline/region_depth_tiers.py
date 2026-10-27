"""Read-only separation of conditional depth support and front-conflict tiers."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from PIL import Image
from integrations.demo_pipeline.region_depth_support import project_pixels, region_raster
from integrations.demo_pipeline.shape_support_audit import context
from integrations.demo_pipeline.silhouette_review import project, raster
from integrations.video_masks.boundary_geometry_binding import _bind, _read, _need


def validate_ledger(points, states, source_view, selected, views, expected_views, result):
    points, states, source_view, selected = map(np.asarray, (points, states, source_view, selected))
    n = len(points)
    _need(points.ndim == 2 and points.shape[1] == 3 and points.dtype.kind in 'fiu' and np.isfinite(points).all(), 'finite Nx3 points required')
    _need(n <= 50000 and result['sampled_candidates'] == n and type(result['sampled_candidates']) is int, 'candidate count mismatch')
    _need(views == expected_views and len(views) > 0, 'view order or source identity mismatch')
    _need(states.shape == (n, len(views)) and states.dtype.kind in 'iu' and np.isin(states, [0, 1, 2, 3, 4]).all(), 'state domain mismatch')
    _need(source_view.shape == (n,) and source_view.dtype.kind in 'iu' and (not n or (source_view.min() >= 0 and source_view.max() < len(views))), 'source view index mismatch')
    _need(selected.shape == (n,) and selected.dtype == np.bool_, 'bool selected required')
    _need(type(result['selected_candidates']) is int and int(selected.sum()) == result['selected_candidates'], 'selected count mismatch')
    _need(type(result['total_candidates']) is int and result['total_candidates'] >= n, 'total count mismatch')


def summarize_ledger(points, states, source_view, selected, views, target_camera, region, baseline_coverage):
    """No depth recomputation, candidate adoption or threshold tuning."""
    _need(region.dtype == np.bool_ and baseline_coverage.dtype == np.bool_ and region.ndim == 2 and region.shape == baseline_coverage.shape, 'same-domain bool region and baseline required')
    _need(region.shape == (target_camera['height'], target_camera['width']), 'target raster domain mismatch')
    front_conflict = (states == 3).any(axis=1)
    no_front = selected & ~front_conflict
    uv, _, inside = project_pixels(points, target_camera)
    ix = np.flatnonzero(selected)
    _need(inside[ix].all() and region[uv[ix, 1], uv[ix, 0]].all(), 'selected projection escapes frozen region')
    missing = region & ~baseline_coverage
    tiers = {}; hits = {}
    for name, take in [('SUPPORT_ONLY', selected), ('NO_KNOWN_FRONT_CONFLICT', no_front)]:
        image = np.zeros_like(region); ids = np.flatnonzero(take & inside)
        image[uv[ids, 1], uv[ids, 0]] = True
        image &= region; hits[name] = image
        tiers[name] = dict(candidate_samples=int(take.sum()), target_unique_pixels=int(image.sum()), raw_uncovered_unique_pixels=int((image & missing).sum()), samples_with_any_front_conflict=int((take & front_conflict).sum()), samples_with_any_behind_state=int((take & (states == 4).any(axis=1)).sum()))
    matrix = [dict(**v, support_front_conflicts=int((selected & (states[:, j] == 3)).sum()), support_behind=int((selected & (states[:, j] == 4)).sum()), support_unknown=int((selected & (states[:, j] == 1)).sum()), support_outside=int((selected & (states[:, j] == 0)).sum()), support_agree=int((selected & (states[:, j] == 2)).sum()), generating_samples=int((selected & (source_view == j)).sum())) for j, v in enumerate(views)]
    return dict(tiers=tiers, matrix=matrix, roi_pixels=int(region.sum()), raw_uncovered_pixels=int(missing.sum())), hits, no_front


def run(source, ledger_run, region_json, output):
    source, ledger_run, region_json, output = [Path(x).resolve() for x in (source, ledger_run, region_json, output)]
    _need(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source, ledger_run, region_json.parent)), 'fresh disjoint output required')
    output.mkdir(parents=True); refs = []; started = time.monotonic()
    def save(name, obj):
        (output / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    def check():
        if time.monotonic() - started > 600:
            raise TimeoutError('600 second ledger analysis budget')
    save('COMMAND.json', dict(argv=sys.argv, executable=sys.executable, threads=2, native_execution=False))
    try:
        for p in (__file__, Path(__file__).resolve().parents[2] / 'scripts/analyze_region_depth_ledger.py'):
            _bind(p, refs)
        old = _read(ledger_run / 'RESULT.json', refs)
        _need(old['status'] == 'DEPTH_SUPPORTED_CANDIDATES_NOT_ACCEPTED' and old['source_unchanged'] is True, 'unrecognized depth diagnostic')
        pre = _read(ledger_run / 'BINDINGS_PRE.json', refs)
        oldpost = _read(ledger_run / 'BINDINGS_POST.json', refs)
        _need(pre == oldpost, 'ledger input binding snapshots differ')
        for r in pre:
            check(); _bind(r['path'], refs, r)
        _, raw, _, _, vv = context(source, None, refs)
        views = [dict(decode_index=v['decode_index'], camera_name=v['camera']['name']) for v in vv]
        spec = _read(region_json, refs)
        target_index = [j for j, v in enumerate(vv) if v['decode_index'] == spec['decode_index']]
        _need(len(target_index) == 1, 'target view identity mismatch')
        j = target_index[0]; target = vv[j]; region = region_raster(spec, target)
        _need(any(Path(r['path']).resolve() == region_json for r in pre), 'region not in original ledger bindings')
        ledger = _bind(ledger_run / 'CANDIDATE_LEDGER.npz', refs)
        with np.load(ledger, allow_pickle=False) as a:
            points, states, origins, selected, other = [a[k].copy() for k in ('points', 'states', 'source_view', 'selected', 'other_agreeing')]
        validate_ledger(points, states, origins, selected, old['views'], views, old)
        expected = states == 2; expected[:, j] = False; expected[np.arange(len(points)), origins] = False
        _need(other.dtype == np.bool_ and other.shape == states.shape and np.array_equal(other, expected), 'validating-view ledger mismatch')
        _need((other[selected].sum(axis=1) >= 3).all() and np.isin(states[selected, j], [1, 2]).all(), 'selected support contract violated')
        check(); c = target['camera']
        baseline = raster(project(raw['vertices'], c['R'], c['C'], c['K']), raw['faces'], c['width'], c['height'], check=check)
        summary, hits, no_front = summarize_ledger(points, states, origins, selected, views, c, region, baseline)
        save('BINDINGS_PRE.json', refs)
        for name, take in [('SUPPORT_ONLY', selected), ('NO_KNOWN_FRONT_CONFLICT', no_front)]:
            with (output / (name + '.ply')).open('w', encoding='ascii') as f:
                f.write('ply\nformat ascii 1.0\nelement vertex %d\nproperty double x\nproperty double y\nproperty double z\nend_header\n' % int(take.sum()))
                np.savetxt(f, points[take], fmt='%.17g')
            with Image.open(target['image']) as im:
                rgb = np.array(im.convert('RGB'))
            rgb[region & ~baseline] = (.55 * rgb[region & ~baseline] + .45 * np.array([255, 150, 0])).astype(np.uint8)
            rgb[hits[name]] = [0, 255, 255]
            Image.fromarray(rgb).save(output / (name + '.png'))
        Image.fromarray((baseline.astype(np.uint8) * 255)).save(output / 'RAW_TRIANGLE_RASTER.png')
        Image.fromarray(((region & ~baseline).astype(np.uint8) * 255)).save(output / 'RAW_UNCOVERED_ROI.png')
        post = []
        for r in refs:
            check(); _bind(r['path'], post, r)
        save('BINDINGS_POST.json', post)
        save('RESULT.json', dict(status='LEDGER_DIAGNOSTIC_COMPLETE_NOT_ADOPTED', geometry_accepted=False, source_unchanged=True, target_decode_index=spec['decode_index'], **summary, raster_policy='bound raw triangles; silhouette_review.raster scale=2 majority threshold; projected union, not visible physical surface', elapsed_seconds=time.monotonic()-started, limitations=['Depth estimates share source images and are not independent ground truth.', 'SUPPORT_ONLY is conditional multi-view depth support, not new usable or adopted 3D points.', 'NO_KNOWN_FRONT_CONFLICT excludes any state3 only; state4 can be ordinary occlusion and remains reported.', 'Any-view front-conflict veto is conservative; pose error, nonrigidity or bad depth may cause rejection, not proof of physical error.', 'Unknown target depth and no known conflict do not establish physical surface.', 'Samples from several depth maps can represent the same surface; unique target pixels are reported separately.']))
    except Exception as exc:
        save('FAILURE.json', dict(status='FAILED_NOT_ADOPTED', error=repr(exc), bindings=refs))
        raise
