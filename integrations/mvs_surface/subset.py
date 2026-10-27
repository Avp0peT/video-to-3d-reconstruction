"""Independently recompute the explicit clean44 -> subset43 relation.

Certificates bind current files and a declared view exclusion, not photo truth.
No mapping, bundle adjustment, triangulation or model repair is performed here.
"""
from pathlib import Path
import hashlib, math
from .photo_inputs import _binding, _json, MODEL_BIN, _under

PROFILE = 'clean44-exclude59-native43-v1'


def need(value, message):
    if not value:
        raise ValueError('native subset: ' + message)


def read(path, lines=False):
    record = _binding(path, 4 * 1024**2)
    raw = Path(record['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == record['sha256'], 'record read drift')
    value = [_json(s) for s in raw.decode('utf-8-sig').splitlines() if s.strip()] if lines else _json(raw.decode('utf-8-sig'))
    need(isinstance(value, list if lines else dict), 'record root type')
    return value, record


def _native(path):
    import pycolmap
    need(pycolmap.__version__ == '4.2.1', 'requires pycolmap 4.2.1')
    return pycolmap.Reconstruction(str(path))


def views(model):
    result = {}
    for iid in model.reg_image_ids():
        image = model.image(iid)
        camera = model.camera(image.camera_id)
        pose = image.cam_from_world()
        frame = model.frame(image.frame_id)
        result[int(iid)] = {'name': image.name, 'camera_id': int(image.camera_id),
            'camera_model': camera.model_name, 'width': int(camera.width), 'height': int(camera.height),
            'params': camera.params.tolist(), 'R': pose.rotation.matrix().tolist(), 't': pose.translation.tolist(),
            'frame_id': int(image.frame_id),
            'frame_data_ids': sorted((str(x.sensor_id.type), int(x.sensor_id.id), int(x.id)) for x in frame.data_ids)}
    return result


def points(model):
    return {int(pid): {'xyz': p.xyz.tolist(), 'rgb': p.color.tolist(),
            'track': sorted((int(e.image_id), int(e.point2D_idx)) for e in p.track.elements)}
            for pid, p in model.points3D.items()}


def _tracks_valid(model, registered):
    records = points(model)
    tracksets = {pid: set(p['track']) for pid, p in records.items()}
    image_points = {iid: model.image(iid).points2D for iid in registered}
    for pid, p in records.items():
        need(all(math.isfinite(v) for v in p['xyz']), 'nonfinite point geometry')
        track = p['track']
        need(len(track) >= 2 and len({i for i, _ in track}) == len(track), 'short or repeated-view track')
        for iid, idx in track:
            need(iid in registered, 'track points to unregistered image')
            observations = image_points[iid]
            need(0 <= idx < len(observations) and int(observations[idx].point3D_id) == pid,
                 'track/image reciprocal mismatch')
    for iid in registered:
        for idx, point in enumerate(image_points[iid]):
            if point.has_point3D():
                pid = int(point.point3D_id)
                need(pid in records and (iid, idx) in tracksets[pid],
                     'image/track reciprocal mismatch')


def compare_models(original, clean, derived, excluded_id):
    """Generic relation core, also tested with small synthetic native models."""
    ov, cv, dv = views(original), views(clean), views(derived)
    need(ov == cv, 'clean parent differs from original registered cameras/poses/IDs')
    need(excluded_id in cv, 'excluded native image not registered')
    data = cv[excluded_id]['frame_data_ids']
    need(len(data) == 1 and data[0][2] == excluded_id, 'excluded frame is not single-image')
    need(dv == {i: v for i, v in cv.items() if i != excluded_id}, 'derived view set/camera/pose/frame identity differs')
    op, cp, dp = points(original), points(clean), points(derived)
    # The existing clean44 removes entire points having duplicate image tracks.
    expected_clean = {pid: p for pid, p in op.items() if len({i for i, _ in p['track']}) == len(p['track'])}
    need(cp == expected_clean, 'clean44 is not exact whole-point duplicate-track restriction of original')
    for iid in cv:
        left, right = original.image(iid).points2D, clean.image(iid).points2D
        need(len(left) == len(right), 'clean parent points2D count differs')
        for a, b in zip(left, right):
            need(a.xy.tolist() == b.xy.tolist(), 'clean parent measured points2D changed')
            wanted = int(a.point3D_id) if int(a.point3D_id) in cp else None
            actual = int(b.point3D_id) if b.has_point3D() else None
            need(wanted == actual, 'clean parent point2D association differs')
    expected = {}
    removed = []
    for pid, p in cp.items():
        track = [pair for pair in p['track'] if pair[0] != excluded_id]
        removed.extend({'point3D_id': pid, 'image_id': iid, 'point2D_idx': idx}
                       for iid, idx in p['track'] if iid == excluded_id)
        if len(track) >= 2:
            expected[pid] = dict(p, track=track)
    need(dp == expected, 'derived XYZ/RGB/point IDs/retained tracks differ')
    # Check unchanged measured 2D coordinates and point associations in kept views.
    for iid in dv:
        left, right = clean.image(iid).points2D, derived.image(iid).points2D
        need(len(left) == len(right), 'retained points2D count differs')
        for a, b in zip(left, right):
            need(a.xy.tolist() == b.xy.tolist(), 'retained measured points2D changed')
            expected_pid = int(a.point3D_id) if int(a.point3D_id) in expected else None
            actual_pid = int(b.point3D_id) if b.has_point3D() else None
            need(expected_pid == actual_pid, 'retained point2D association changed')
    need(clean.is_valid() and derived.is_valid(), 'native consistency gate failed')
    _tracks_valid(clean, set(cv)); _tracks_valid(derived, set(dv))
    return {'original_registered_ids': sorted(ov), 'parent_registered_ids': sorted(cv),
            'original_points':len(op),'clean_removed_duplicate_points':sorted(set(op)-set(cp)),
            'derived_registered_ids': sorted(dv), 'parent_points': len(cp), 'derived_points': len(dp),
            'removed_points': sorted(set(cp) - set(dp)), 'removed_observations': removed}


def evaluate(clean_parent, derived_model, sfm_run, io_run, model_id='0'):
    """Fixed production profile: exact44, exclude canonical59/image60/frame60."""
    roots = {k: Path(v).resolve(strict=True) for k, v in dict(clean_parent=clean_parent,
             derived_model=derived_model, sfm_run=sfm_run, io_run=io_run).items()}
    need(all(p.is_dir() for p in roots.values()), 'model/run directories required')
    sfm, sfm_binding = read(roots['sfm_run'] / 'run.json')
    io, io_binding = read(roots['io_run'] / 'run.json')
    maps, maps_binding = read(roots['sfm_run'] / 'frames_map.jsonl', True)
    frames, frames_binding = read(roots['io_run'] / 'frames.jsonl', True)
    need(model_id == '0' and sfm.get('upstream_io_run_id') == io.get('run_id') and io.get('status') == 'SUCCESS'
         and io.get('contract_version')=='photo-io-v1' and sfm.get('contract_version') in
         ('sfm-native-v1','sfm-native-profile-v1','sfm-native-object-v1') and sfm.get('status') in ('SUCCESS','PARTIAL'),
         'model0/T01/SfM run identity differs')
    need(0<len(frames)==len(maps)<=512 and
         all(type(v.get('input_index')) is int for v in frames+maps) and
         [v['input_index'] for v in frames]==[v['input_index'] for v in maps]==list(range(len(frames))),
         'T01/SfM mapping rows must be complete unique ordered plain indices')
    need(type(io.get('input_count')) is int and io['input_count']==len(frames) and
         type(io.get('ok_count')) is int and io['ok_count']==len(frames) and
         type(io.get('error_count')) is int and io['error_count']==0 and
         all(f.get('status')=='OK' for f in frames) and
         type(sfm.get('counts',{}).get('input_rows')) is int and sfm['counts']['input_rows']==len(maps),
         'T01/SfM complete row counts differ')
    models = [m for m in sfm.get('models', []) if m.get('model_id') == model_id]
    need(len(models) == 1 and models[0].get('usable') is True, 'model0 missing or not usable')
    original_path = _under(roots['sfm_run'], models[0]['binary_path'])
    bindings = [sfm_binding, io_binding, maps_binding, frames_binding]
    for path in (original_path, roots['clean_parent'], roots['derived_model']):
        bindings.extend(_binding(path / f) for f in MODEL_BIN)
    original, parent, child = (_native(p) for p in (original_path, roots['clean_parent'], roots['derived_model']))
    need(original.num_points3D()==2413 and parent.num_points3D()==child.num_points3D()==2393,
         'profile native point counts differ before relation readback')
    need(len(views(original)) == len(views(parent)) == 44 and len(views(child)) == 43, 'profile requires original/clean44 and derived43')
    need(all(type(i) is int for i in models[0]['registered_image_ids']) and
         sorted(models[0]['registered_image_ids']) == sorted(views(original)), 'SfM declared registered IDs differ from original44')
    by_index = {x['input_index']: x for x in frames}; mapping = {x['input_index']: x for x in maps}
    need(len(by_index) == len(frames) and len(mapping) == len(maps), 'duplicate input mapping index')
    joined = []
    for iid, view in sorted(views(parent).items()):
        rows = [m for m in maps if m.get('staged_image_name') == view['name']]
        need(len(rows) == 1, 'profile requires unique canonical input per view')
        m = rows[0]; f = by_index.get(m['input_index'])
        regs = [v for v in m.get('registrations', []) if v.get('model_id') == model_id]
        need(f is not None and f.get('status') == 'OK' and len(regs) == 1 and
             type(regs[0].get('image_id')) is int and regs[0]['image_id'] == iid and
             regs[0].get('camera_id') == view['camera_id'] and m.get('frame_id') == f.get('frame_id') and
             m.get('upstream_io_run_id') == io['run_id'] and m.get('upstream_image_output') == f.get('image_output'),
             'native/SfM/T01 view identity differs')
        canonical = _binding(_under(roots['io_run'], f['image_output']))
        staged = _binding(_under(roots['sfm_run'], view['name']))
        need(canonical['sha256'] == staged['sha256'] == m.get('canonical_image_sha256'), 'canonical/staged image hash differs')
        need((f.get('canonical_width'), f.get('canonical_height')) == (view['width'], view['height']), 'canonical/native image dimensions differ')
        bindings.extend((canonical, staged))
        joined.append({'input_index': m['input_index'], 'frame_id': f['frame_id'], 'image_id': iid,
                       'frame_id_native': view['frame_id'], 'image_name': view['name'],
                       'width': view['width'], 'height': view['height'], 'canonical': canonical})
    excluded = [x for x in joined if x['input_index'] == 59]
    need(len(excluded) == 1 and excluded[0]['image_id'] == 60 and excluded[0]['frame_id_native'] == 60, 'profile must exclude exact input59/native image60/frame60')
    relation = compare_models(original, parent, child, 60)
    need(relation['original_points']==2413 and len(relation['clean_removed_duplicate_points'])==20 and
         relation['parent_points'] == relation['derived_points'] == 2393 and
         len(relation['removed_observations']) == 185 and not relation['removed_points'],
         'profile expected2393 points and185 removed observations; stop for review')
    bindings.extend((_binding(Path(__file__)), _binding(Path(__file__).parents[2] / 'scripts/derive_native_subset.py')))
    for b in bindings:
        need(_binding(b['path']) == b, 'inputs changed during subset verification')
    return {'schema_version': 'native-subset43-v1', 'profile': PROFILE, 'model_id': model_id,
            'roots': {k: str(v) for k, v in roots.items()}, 'original_model': str(original_path),
            'excluded': excluded[0], 'selected_inputs': [x for x in joined if x['input_index'] != 59],
            'relation': relation, 'bindings': bindings,
            'scope': 'Current explicit view restriction only; not geometry/photo authenticity or complete object evidence'}


def validate_reference(reference, *, clean_parent=None, derived_model=None, sfm_run=None, io_run=None, original_model=None):
    need(isinstance(reference, dict) and set(reference) == {'path', 'sha256'} and
         isinstance(reference['path'], str) and Path(reference['path']).is_absolute(), 'certificate needs absolute path and sha256')
    cert, binding = read(reference['path'])
    need(reference['sha256'] == binding['sha256'], 'certificate hash differs')
    need(cert.get('schema_version') == 'native-subset43-v1' and cert.get('profile') == PROFILE, 'unsupported subset certificate')
    roots = cert.get('roots'); need(isinstance(roots, dict) and set(roots) == {'clean_parent','derived_model','sfm_run','io_run'}, 'certificate roots invalid')
    for name, value in roots.items():
        need(isinstance(value, str) and Path(value).is_absolute(), 'certificate roots must be absolute')
    for name, value in dict(clean_parent=clean_parent, derived_model=derived_model, sfm_run=sfm_run, io_run=io_run).items():
        if value is not None:
            need(Path(roots[name]).resolve() == Path(value).resolve(), 'certificate belongs to another ' + name)
    if original_model is not None:
        need(Path(cert['original_model']).resolve() == Path(original_model).resolve(), 'certificate original model differs')
    # Compare the entire producer record with independent recomputation; PASS
    # text/counts in a caller-supplied certificate never authorize a subset.
    actual = evaluate(**roots, model_id=cert.get('model_id'))
    need(cert == actual, 'certificate content differs from independently recomputed relation')
    need(_binding(binding['path']) == binding, 'certificate changed during verification')
    return dict(actual, certificate=binding, bindings=[binding, *actual['bindings']])
