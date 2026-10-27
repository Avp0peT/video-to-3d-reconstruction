"""Prepare an exact calibrated temporal subset; no native process or SfM solve."""
from pathlib import Path
from fractions import Fraction
import json
import shutil
import numpy as np
from .general_subset import compare
from .subset import views
from .photo_inputs import MODEL_BIN
from integrations.demo_pipeline.shape_support_audit import context
from integrations.video_masks.boundary_geometry_binding import _read, _bind, _need


def scene_intrinsic_from_native(params):
    """InterfaceCOLMAP pixel-centre comparison only; no model mutation."""
    p = np.asarray(params, float)
    _need(p.shape == (4,) and np.isfinite(p).all() and (p[:2] > 0).all(), 'finite native PINHOLE parameters required')
    fx, fy, cx, cy = p
    return np.array([[fx, 0, cx-.5], [0, fy, cy-.5], [0, 0, 1.]])


def copied_relative_path(exported_name, role):
    _need(isinstance(exported_name, str) and Path(exported_name).name == exported_name and Path(exported_name).suffix.lower() == '.png', 'bare exported PNG name required')
    stem = Path(exported_name).stem
    if role == 'image':
        return Path('undistorted/images') / exported_name
    _need(role in ('mask', 'validity'), 'unknown copy role')
    return Path('masks' if role == 'mask' else 'validity') / (stem + ('.mask.png' if role == 'mask' else '.validity.png'))


def select_views(frame_map, timeline_join, analysis, window_seconds):
    _need(type(window_seconds) is int and window_seconds == 2, 'predeclared middle-scale window is exactly 2 seconds, not an optimum')
    _need(analysis.get('status') == 'TEMPORAL_DEPTH_DIAGNOSTIC_COMPLETE_NOT_ADOPTED' and analysis.get('source_unchanged') is True, 'unverified temporal diagnostic')
    target = analysis['target_decode_index']
    fm = {x['decode_index']: x for x in frame_map['frames']}
    tr = {x['decode_index']: x for x in timeline_join['rows']}
    _need(len(fm) == len(frame_map['frames']) and len(tr) == len(timeline_join['rows']) and fm.keys() == tr.keys() and target in fm, 'timeline/frame identity sets differ')
    _need(all(type(i) is int and i >= 0 for i in fm), 'integer decode identities required')
    times = {}
    ids = []
    for i in sorted(fm):
        f, t = fm[i], tr[i]
        _need(type(t['pts']) is int and len(t['time_base']) == 2 and all(type(v) is int and v > 0 for v in t['time_base']), 'exact integer PTS/timebase required')
        timestamp = Fraction(t['pts'] * t['time_base'][0], t['time_base'][1])
        _need(list((timestamp.numerator, timestamp.denominator)) == t['seconds_fraction'], 'PTS fraction differs')
        _need(Path(t['canonical_mask_path']).resolve() == Path(f['canonical_mask_path']).resolve() and t['canonical_mask_sha256'] == f['canonical_mask_sha256'], 'canonical mask identity differs')
        _need(type(f['export_image_id']) is int and f['export_image_id'] > 0, 'export image identity required')
        ids.append(f['export_image_id']); times[i] = timestamp
    _need(len(set(ids)) == len(ids) and all(times[a] < times[b] for a, b in zip(sorted(times), sorted(times)[1:])), 'duplicate native ID or nonmonotonic time')
    selected = [i for i in sorted(fm) if abs(times[i] - times[target]) <= Fraction(2)]
    rows = [r for r in analysis['rows'] if r['window_seconds'] == dict(numerator=2, denominator=1)]
    _need(len(rows) == 1 and rows[0]['target_view_id'] == target and rows[0]['near_view_ids'] == [i for i in selected if i != target], 'analysis window selection differs')
    _need(len(selected) >= 4, 'insufficient local views for fixed comparison')
    return [dict(decode_index=i, export_image_id=fm[i]['export_image_id'], exported_name=fm[i]['exported_name'], seconds_fraction=[times[i].numerator, times[i].denominator], is_target=i == target) for i in selected]


def prepare(source_root, temporal_analysis, window_seconds, output):
    """Write fresh subset+byte copies and a future command description only."""
    source_root, temporal_analysis = [Path(p).resolve(strict=True) for p in (source_root, temporal_analysis)]
    output = Path(output).resolve()
    _need(not output.exists() and not any(output.is_relative_to(p) or p.is_relative_to(output) for p in (source_root, temporal_analysis)), 'fresh disjoint subset output required')
    _need(type(window_seconds) is int and window_seconds == 2, 'only predeclared 2-second preparation allowed')
    refs = []; generated = []
    analysis = _read(temporal_analysis / 'RESULT.json', refs)
    timeline = _read(temporal_analysis / 'TIMELINE_JOIN.json', refs)
    pre = _read(temporal_analysis / 'BINDINGS_PRE.json', refs)
    post = _read(temporal_analysis / 'BINDINGS_POST.json', refs)
    _need(pre == post, 'temporal analysis input snapshots differ')
    # Historical code may have newer revisions. All bound data must still match;
    # preserve historical code hashes in the derivation instead of relabelling it.
    historical_code = [r for r in pre if Path(r['path']).suffix == '.py']
    for r in pre:
        if Path(r['path']).suffix != '.py':
            _bind(r['path'], refs, r)
    fmap_path = source_root / 'export/FRAME_MAP.json'
    _need(any(Path(r['path']).resolve() == fmap_path for r in pre), 'analysis belongs to another source')
    fmap = _read(fmap_path, refs)
    chosen = select_views(fmap, timeline, analysis, window_seconds)
    _, _, _, _, bound_views = context(source_root, None, refs)
    by_decode = {v['decode_index']: v for v in bound_views}
    source_path = source_root / 'bridge/undistorted/sparse'
    for name in MODEL_BIN:
        _bind(source_path / name, refs)
    for name in (__file__, Path(__file__).with_name('general_subset.py'), Path(__file__).with_name('subset.py')):
        _bind(name, refs)
    import pycolmap
    _need(pycolmap.__version__ == '4.2.1', 'pycolmap 4.2.1 required')
    parent = pycolmap.Reconstruction(str(source_path)); original_views = views(parent)
    retained = {r['export_image_id'] for r in chosen}
    _need(retained <= original_views.keys(), 'selected native IDs absent')
    for row in chosen:
        view = original_views[row['export_image_id']]; bound = by_decode[row['decode_index']]; c = bound['camera']
        _need(view['name'] == row['exported_name'] == Path(c['name']).name and view['camera_model'] == 'PINHOLE' and len(view['params']) == 4, 'native PINHOLE image mapping differs')
        k = scene_intrinsic_from_native(view['params'])
        rotation = np.asarray(view['R']); center = -rotation.T @ np.asarray(view['t'])
        _need((view['width'], view['height']) == (c['width'], c['height']) and np.allclose(k, c['K'], atol=1e-9, rtol=0) and np.allclose(rotation, c['R'], atol=1e-9, rtol=0) and np.allclose(center, c['C'], atol=1e-9, rtol=0), 'scene/native calibration differs')
    child = pycolmap.Reconstruction(parent)
    for pid, point in list(child.points3D.items()):
        track = list(point.track.elements)
        if len({e.image_id for e in track}) != len(track):
            child.delete_point3D(pid)
    for iid, record in original_views.items():
        if iid not in retained:
            _need(len(record['frame_data_ids']) == 1, 'cannot deregister shared frame')
            child.deregister_frame(record['frame_id'])
    relation = compare(parent, child, retained)
    output.mkdir(parents=True)
    def save(name, obj):
        (output / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    try:
        sparse = output / 'undistorted/sparse'; sparse.mkdir(parents=True)
        child.write_binary(str(sparse))
        relation_after = compare(parent, pycolmap.Reconstruction(str(sparse)), retained)
        _need(relation == relation_after, 'serialized subset relation differs')
        for name in MODEL_BIN:
            _bind(sparse / name, generated)
        copied = []
        for row in chosen:
            v = by_decode[row['decode_index']]
            # Context already strictly bound the actual MVS pixels and binary labels.
            mf = next(x for x in _read(source_root / 'masks/RESULT.json', refs)['frames'] if x['decode_index'] == row['decode_index'])
            paths = dict(image=v['image'], mask=mf['mask']['path'], validity=mf['validity']['path'])
            entry = dict(row)
            for role, src in paths.items():
                src = _bind(src, refs)
                dst = output / copied_relative_path(row['exported_name'], role)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                original = next(r for r in refs if r['path'] == str(src))
                _bind(dst, generated, original); entry[role] = str(dst)
            copied.append(entry)
        save('SELECTED_VIEWS.json', dict(target_decode_index=analysis['target_decode_index'], window_seconds=2, selection='predeclared middle scale; not optimized', rows=copied))
        save('INPUT_BINDINGS.json', refs)
        checked = []
        for binding in refs:
            _bind(binding['path'], checked, binding)
        save('INPUT_BINDINGS_POST.json', checked)
        save('DERIVATION.json', dict(status='LOCAL_MVS_SUBSET_PREPARED_NOT_EXECUTED', geometry_accepted=False, source_unchanged=True, relation=relation_after, historical_analysis_code_bindings=historical_code, generated=generated, retained_native_ids=sorted(retained), intrinsic_comparison='InterfaceCOLMAP principal point minus 0.5; comparison only, native model unmodified', future_interface_colmap=dict(executable='InterfaceCOLMAP', argv=['--input-file', str(output / 'undistorted'), '--image-folder', str(output / 'undistorted/images'), '--output-file', str(output / 'scene.mvs'), '--normalize', '0', '--max-threads', '2'], executed=False), limitations=['No bundle adjustment, pose estimation, scale change or depth reconstruction performed.', 'Sparse coordinates and retained observations are exact restrictions of globally estimated SfM, not independently locally estimated cameras.', 'Future local MVS is an experiment, not guaranteed thumb recovery.']))
        return dict(output=str(output), selected_views=len(chosen), status='LOCAL_MVS_SUBSET_PREPARED_NOT_EXECUTED')
    except Exception as exc:
        save('FAILURE.json', dict(error=repr(exc), status='PREPARATION_FAILED_NOT_EXECUTED'))
        raise
