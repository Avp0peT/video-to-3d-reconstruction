"""Explicit real experimental native inputs; never a substitute for a T01/SfM run.

The closed producer profile authenticates recorded execution and current bytes,
not capture accuracy, object completeness, or the historical intent of a mask.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image

from integrations.mvs_masks import manifest_io as mio

SCHEMA = 'experimental-native-photo-source-v1'
SURFACE_SCHEMA = 'experimental-native-surface-input-v1'
PROFILE = 'paired-cpu-pycolmap421-v1'
DERIVED_SCHEMA = 'experimental-native-derived-photo-source-v1'
DERIVED_POLICY = 'delete-whole-repeated-or-short-view-points-v1'
PROFILE_WORKER_SHA256 = 'ca8c761ac39a73d1c8471476b308bdab0b1aad3dc4b7c78f126b5fd15bee39d4'


def need(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    # Reuse the bounded, race-detecting byte reader without importing runner.
    from .photo_inputs import _binding
    return _binding(path)


def checked(item, bindings):
    need(isinstance(item, dict) and set(item) == {'path', 'bytes', 'sha256'}, 'incomplete byte binding')
    actual = record(item['path'])
    need(actual == item, 'conflicting byte binding: ' + str(item['path']))
    bindings[actual['path']] = actual
    return Path(actual['path'])


def load(path):
    return mio.load_json_strict(path, 'experimental input')


def safe(root, name):
    need(isinstance(name, str) and '\\' not in name, 'invalid native relative name')
    path = (Path(root) / name).resolve(strict=True)
    need(path.is_relative_to(Path(root).resolve()) and path.is_file(), 'unsafe native relative name')
    return path


def immutable_database(path):
    """Require a closed database before opening; never create WAL/SHM in source."""
    path = Path(path).resolve(strict=True)
    need(not Path(str(path) + '-wal').exists() and not Path(str(path) + '-shm').exists(),
         'database sidecars exist; a coherent task-local snapshot is required')
    return sqlite3.connect(path.as_uri() + '?immutable=1', uri=True)


def scene_point_records(path):
    """Retain the bounded MVSI vertex identity fields for the existing matcher.

    The shared parser validates the whole file first. Its Reader and camera/image
    routines are reused; only XYZ/RGB/view-name records are retained here.
    """
    from . import scene as mvsi
    import hashlib
    metadata=mvsi.read_scene(path)
    data=Path(path).read_bytes()
    need(hashlib.sha256(data).hexdigest()==metadata['sha256'],'scene changed before vertex identity read')
    reader=mvsi._Reader(data);reader.unpack('4sII','header')
    images=mvsi._read_images(reader,mvsi._read_platforms(reader))
    count=reader.count('vertices',mvsi.MAX_VERTICES,20)
    records=[]
    for i in range(count):
        xyz=reader.floats(3,'vertex.position','f')
        views=[]
        for _ in range(reader.count('vertex.views',len(images),8)):
            image_id,confidence=reader.unpack('If','vertex.view')
            name=images[image_id]['name']
            need(name.startswith('undistorted/images/'),'scene vertex image path conflict')
            views.append(name[len('undistorted/images/'):])
        records.append({'id':i,'xyz':xyz,'views':sorted(views),'rgb':None})
    normals=reader.count('verticesNormal',count,12);reader.offset+=normals*12
    colors=reader.count('verticesColor',count,3)
    need(colors==count and count>0,'scene vertex colors missing/empty')
    # MVSI Interface.h Col3 serializes x=B,y=G,z=R; native COLMAP is RGB.
    for item in records:item['rgb']=list(reversed(reader.unpack('BBB','vertex.color')))
    return records


def validate_identity(spec_path, *, _derivation_parent=False):
    """Reopen producer records, copied inputs, masks, DB and selected native model."""
    import pycolmap
    from scripts.prepare_mvs import structure, image_files
    bindings = {}
    spec_path = Path(spec_path).resolve(strict=True)
    bindings[str(spec_path)] = record(spec_path)
    spec = load(spec_path)
    if spec.get('schema_version') == DERIVED_SCHEMA:
        return validate_derived_identity(spec_path)
    need(spec.get('schema_version') == SCHEMA and spec.get('producer_profile') == PROFILE,
         'unsupported experimental source schema/profile')
    need(spec.get('source_kind') == 'real-experimental-native-reconstruction', 'source kind must be explicit')
    required = {'producer_plan', 'producer_config', 'producer_entry', 'producer_result', 'database'}
    docs = {key: checked(spec.get(key), bindings) for key in required}
    plan, config, entry, result = (load(docs[key]) for key in
                                  ('producer_plan', 'producer_config', 'producer_entry', 'producer_result'))
    arm = config.get('arm')
    need(arm in ('masked28', 'unmasked28') and result.get('arm') == arm, 'producer arm conflict')
    need(config in plan.get('configs', []), 'configuration not frozen by producer PLAN')
    need(spec['producer_config'] in plan.get('config_hashes', []), 'configuration byte binding absent')
    need(entry.get('argv') in plan.get('argv', []) and
         entry['argv'][-1] == str(docs['producer_config']), 'actual argv differs from frozen producer')
    need(Path(entry.get('cwd', '')).resolve() == Path(config['output']).resolve(), 'producer cwd conflict')
    need(entry.get('status') == 'completed' and type(entry.get('root_exit_code')) is int and entry.get('root_exit_code') == 0 and
         type(entry.get('pid')) is int and entry['pid'] > 0 and
         entry.get('reclaim_confirmed') is True and not entry.get('errors') and not entry.get('log_errors'),
         'producer execution/reclamation incomplete')
    need(result.get('status') == 'COMPLETED' and type(result.get('root_exit_code')) is int and result.get('root_exit_code') == 0 and
         result.get('native_calls') == ['extract_features', 'match_exhaustive', 'incremental_mapping'] and
         result.get('actual_options') == config.get('frozen_options'), 'native producer result conflict')
    source_bindings = plan.get('source_bindings')
    need(isinstance(source_bindings, list) and source_bindings, 'producer source bindings absent')
    archives = spec.get('historical_code_copies', [])
    need(isinstance(archives,list), 'invalid historical code copies')
    archived = {}
    permitted = {str(Path(__file__).resolve().parents[2]/name) for name in
                 ('integrations/mvs_surface/supervision.py','integrations/cad_delivery/supervise.py')}
    for item in archives:
        need(isinstance(item,dict) and set(item)=={'original_path','archive'} and
             item['original_path'] in permitted and item['original_path'] not in archived,
             'invalid/duplicate historical supervisor copy')
        checked(item['archive'],bindings)
        originals=[b for b in source_bindings if b['path']==item['original_path']]
        need(len(originals)==1 and all(originals[0][k]==item['archive'][k] for k in ('bytes','sha256')),
             'historical copy does not match frozen original bytes')
        archived[item['original_path']]=item['archive']
    for item in source_bindings:
        if item['path'] not in archived: checked(item, bindings)
    need(any(b['path'] == entry['argv'][-2] for b in source_bindings), 'worker code binding absent')
    need(record(entry['argv'][-2])['sha256'] == PROFILE_WORKER_SHA256, 'unsupported experimental producer code')
    need(pycolmap.__version__ == '4.2.1', 'native reader version mismatch')
    import pycolmap._core as core
    need(any(b == record(core.__file__) for b in source_bindings), 'producer/reader native binary mismatch')
    need(Path(config['database_path']).resolve() == docs['database'], 'database path conflict')
    model_id = spec.get('model_id')
    need(type(model_id) is int and model_id >= 0, 'explicit integer model_id required')
    choices = [m for m in result.get('models', []) if m.get('model_id') == model_id]
    need(len(choices) == 1, 'selected native model missing/ambiguous')
    model_root = Path(config['models_root']).resolve() / str(model_id)
    need(Path(choices[0]['path']).resolve() == model_root, 'selected model root conflict')
    files = spec.get('model_files')
    need(isinstance(files, list) and files and len({b.get('path') for b in files}) == len(files),
         'model byte bindings missing/duplicate')
    actual_files = {str(p.resolve()) for p in model_root.iterdir() if p.is_file()}
    need({str(checked(b, bindings)) for b in files} == actual_files, 'selected model file coverage conflict')
    need({'cameras.bin', 'images.bin', 'points3D.bin'} <= {Path(p).name for p in actual_files}, 'native binaries absent')
    model = pycolmap.Reconstruction(str(model_root))
    if _derivation_parent:
        # Authenticate the original producer without admitting dirty geometry
        # for a bridge. A disposable in-memory clean model lets the shared
        # relation validator check all source reciprocity and exact exclusions.
        clean = pycolmap.Reconstruction(str(model_root))
        for pid, point in list(clean.points3D.items()):
            ids = [e.image_id for e in point.track.elements]
            if len(set(ids)) != len(ids) or len(set(ids)) < 2: clean.delete_point3D(pid)
        derived_relation(model, clean)
        counts = {'images':len(model.reg_image_ids()),'points':len(model.points3D),
                  'observations':sum(len(p.track.elements) for p in model.points3D.values())}
    else:
        counts = structure(model)
    need(sorted(model.reg_image_ids()) == choices[0]['registered_native_ids'] and
         counts['points'] == choices[0]['points3D'], 'producer/model counts conflict')
    for point in model.points3D.values():
        need(np.isfinite(point.xyz).all(), 'nonfinite native point')
    rows = [row for row in plan.get('inputs', []) if row.get('arm') == arm]
    maps = {'image': {}, 'mask': {}}
    for row in rows:
        kind, idx = row.get('kind'), row.get('input_index')
        need(kind in maps and type(idx) is int and idx >= 0 and idx not in maps[kind], 'duplicate/invalid input mapping')
        source, copy = checked(row.get('source'), bindings), checked(row.get('copy'), bindings)
        need(row['source']['sha256'] == row['copy']['sha256'] and
             row['source']['bytes'] == row['copy']['bytes'], 'copy/source byte mismatch')
        root = config['image_root'] if kind == 'image' else config['mask_root']
        need(safe(root, row['relative']) == copy, 'input copy/name conflict')
        maps[kind][idx] = row
    names = config.get('image_names')
    need(isinstance(names, list) and 2 <= len(names) <= 256 and len(names) == len(set(names)) and
         set(names) == {r['relative'] for r in maps['image'].values()} and
         set(maps['image']) == set(maps['mask']), 'requested image/mask coverage conflict')
    need(config['frozen_options']['extract_features']['reader']['mask_path'] ==
         (config['mask_root'] if arm == 'masked28' else '.'), 'feature mask semantic conflict')
    manifest_refs = spec.get('mask_manifests')
    need(isinstance(manifest_refs, list) and manifest_refs, 'original accepted-mask provenance absent')
    manifests = [mio.parse_object_manifest(checked(b, bindings)) for b in manifest_refs]
    need(len({m['object_id'] for m in manifests}) == 1, 'different mask objects')
    source_ios = [mio.resolve_source_io(m['source_io'], m['manifest_path']) for m in manifests]
    need(len({r['path'] for r in source_ios}) == 1, 'different canonical IO sources')
    io = source_ios[0]
    for key in ('run_json', 'frames_jsonl'):
        checked(io[key], bindings)
    io_frames = [json.loads(line) for line in Path(io['frames_jsonl']['path']).read_text(encoding='utf-8').splitlines()]
    by_index = {}
    for frame in io_frames:
        idx = frame.get('input_index')
        need(type(idx) is int and idx not in by_index, 'duplicate canonical frame index')
        by_index[idx] = frame
    selections = spec.get('mask_selections')
    need(isinstance(selections, list) and len(selections) == len(maps['image']), 'explicit mask selection coverage required')
    selected_masks = {}
    for selection in selections:
        idx, manifest_index = selection.get('input_index'), selection.get('manifest_index')
        need(type(idx) is int and idx in maps['image'] and idx not in selected_masks and
             type(manifest_index) is int and 0 <= manifest_index < len(manifests), 'invalid mask selection')
        frames = [f for f in manifests[manifest_index]['frames'] if f['input_index'] == idx]
        need(len(frames) == 1 and frames[0]['status'] == 'OK', 'selected mask is not accepted OK')
        f, image_row, mask_row = frames[0], maps['image'][idx], maps['mask'][idx]
        canonical = by_index.get(idx)
        need(canonical and canonical['status'] == 'OK' and f['frame_id'] == canonical['frame_id'] and
             f['image_domain'] == 'photo_io_canonical', 'canonical frame identity conflict')
        canonical_path = safe(io['path'], canonical['image_output'])
        canonical_record = record(canonical_path); bindings[str(canonical_path)] = canonical_record
        need(canonical_record['sha256'] == image_row['copy']['sha256'] == f['source_image_sha256'] and
             Path(f['source_image']).resolve() == canonical_path and
             f['photo_io_image_sha256'] == canonical['image_sha256'], 'canonical RGB provenance conflict')
        chosen = f['chosen']; mask_path = Path(chosen['resolved_mask_path']).resolve()
        need(mask_path == Path(mask_row['source']['path']).resolve() and
             chosen['mask_sha256'] == mask_row['copy']['sha256'] and
             mask_row['relative'] == image_row['relative'] + '.png', 'mask provenance/name conflict')
        with Image.open(canonical_path) as handle:
            need(handle.size == (f['width'], f['height']) ==
                 (canonical['canonical_width'], canonical['canonical_height']), 'canonical dimensions conflict')
        pixels = mio.read_label_png(mask_path, f['width'], f['height'])
        # read_label_png returns the labels and their immutable byte record.
        array = pixels['array'] if isinstance(pixels, dict) else pixels[0]
        need(int(np.count_nonzero(array)) == chosen['foreground_pixels'] > 0, 'foreground count conflict')
        selected_masks[idx] = {'path': str(mask_path), 'frame_id': f['frame_id'], 'manifest_index': manifest_index}
    db_before = record(docs['database'])
    frames = []
    with immutable_database(docs['database']) as db:
        db_rows = db.execute('SELECT image_id,name,camera_id FROM images').fetchall()
        need(len(db_rows) == len(names) and {r[1] for r in db_rows} == set(names), 'database requested image coverage conflict')
        by_name = {row[1]: row for row in db_rows}
        need(len(by_name) == len(db_rows), 'database native name alias')
        input_by_name = {r['relative']: idx for idx, r in maps['image'].items()}
        for iid in sorted(model.reg_image_ids()):
            image = model.images[iid]; camera = model.cameras[image.camera_id]
            row = by_name.get(image.name)
            need(row == (int(iid), image.name, int(image.camera_id)), 'native/DB identity conflict')
            need(np.isfinite(camera.params).all() and camera.focal_length > 0, 'invalid optimized camera')
            db_camera = db.execute('SELECT model,width,height FROM cameras WHERE camera_id=?', (int(image.camera_id),)).fetchone()
            need(db_camera and pycolmap.CameraModelId(db_camera[0]).name == camera.model_name and
                 db_camera[1:] == (camera.width,camera.height), 'native/DB camera model or dimensions conflict')
            keypoint = db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=?', (int(iid),)).fetchone()
            need(keypoint and keypoint[1] >= 2, 'native DB keypoints absent')
            xy = np.frombuffer(keypoint[2], dtype='<f4').reshape(keypoint[0], keypoint[1])[:, :2]
            need(np.isfinite(xy).all() and np.array_equal(xy.astype(float), np.array([p.xy for p in image.points2D])), 'native/DB keypoint coordinates conflict')
            idx = input_by_name[image.name]
            with Image.open(maps['image'][idx]['copy']['path']) as handle:
                need(handle.size == (camera.width, camera.height), 'native/RGB camera dimensions conflict')
            frames.append({'input_index': idx, 'image_id': int(iid), 'camera_id': int(image.camera_id),
                           'native_name': image.name, 'mask': selected_masks[idx]})
    need(record(docs['database']) == db_before, 'database changed during validation')
    for item in image_files(model, Path(config['image_root'])):
        checked(item, bindings)
    for item in list(bindings.values()):
        need(record(item['path']) == item, 'source changed during validation')
    return {'schema_version': 'experimental-native-identity-link-v1',
            'status': 'DERIVATION_PARENT_AUTHENTICATED_NOT_BRIDGE_ADMITTED' if _derivation_parent else 'EXPERIMENTAL_IDENTITIES_BOUND',
            'source_kind': spec['source_kind'], 'model_id': model_id, 'source_sparse': str(model_root),
            'source_images': config['image_root'], 'counts': counts, 'frames': frames,
            'requested_indices': sorted(maps['image']),
            'unregistered_in_selected_model': sorted(set(maps['image']) - {f['input_index'] for f in frames}),
            'bindings': sorted(bindings.values(), key=lambda r: r['path']),
            'historical_code_copies': archives,
            'not_original_t01_sfm_run': True, 'object_shape_status': 'NOT_EVALUATED', 'full_object_accepted': False}


def derived_code_bindings():
    root = Path(__file__).resolve().parents[2]
    return [record(root/name) for name in (
        'integrations/mvs_surface/experimental_inputs.py',
        'integrations/mvs_surface/general_subset.py',
        'integrations/mvs_surface/subset.py',
        'integrations/mvs_surface/photo_inputs.py',
        'scripts/prepare_experimental_surface_inputs.py')]


def derived_relation(parent, child):
    from .general_subset import compare
    # Retain every registered view; compare independently computes the exact
    # whole-point deletion set and every reciprocal observation association.
    relation = compare(parent, child, sorted(parent.reg_image_ids()))
    need(set(parent.images) == set(child.images) and set(parent.cameras) == set(child.cameras) and
         set(parent.rigs) == set(child.rigs) and set(parent.frames) == set(child.frames),
         'derived camera/rig/frame identity differs')
    for kind in ('cameras', 'rigs', 'frames'):
        left, right = getattr(parent, kind), getattr(child, kind)
        for key in left:
            def canonical(value):
                if isinstance(value, dict): return {str(k):canonical(v) for k,v in value.items()}
                if isinstance(value, (list,tuple)): return [canonical(v) for v in value]
                if isinstance(value, np.ndarray): return value.tolist()
                if hasattr(value, 'todict'): return canonical(value.todict())
                if isinstance(value, (str,int,float,bool)) or value is None: return value
                return str(value)
            need(canonical(left[key].todict()) == canonical(right[key].todict()), 'derived '+kind+' changed')
    return relation


def validate_derived_identity(spec_path):
    import pycolmap
    spec_path = Path(spec_path).resolve(strict=True)
    spec = load(spec_path); bindings = {str(spec_path): record(spec_path)}
    need(set(spec) == {'schema_version','policy','parent_identity_spec','derived_sparse',
                      'model_files','certificate','code_bindings'}, 'derived identity fields differ')
    need(spec['schema_version'] == DERIVED_SCHEMA and spec['policy'] == DERIVED_POLICY,
         'derived identity policy differs')
    parent_path = checked(spec['parent_identity_spec'], bindings)
    need(load(parent_path).get('schema_version') == SCHEMA, 'original experimental parent required')
    parent = validate_identity(parent_path, _derivation_parent=True)
    bindings.update({b['path']:b for b in parent['bindings']})
    need(spec['code_bindings'] == derived_code_bindings(), 'current derived code drift')
    for item in spec['code_bindings']: checked(item, bindings)
    root = Path(spec['derived_sparse'])
    need(root.is_absolute() and root.resolve(strict=True) != Path(parent['source_sparse']).resolve(),
         'distinct absolute derived sparse required')
    files = spec['model_files']
    need(isinstance(files,list) and len(files) == len({b['path'] for b in files}) and
         {str(checked(b,bindings)) for b in files} == {str(p.resolve()) for p in root.iterdir() if p.is_file()},
         'derived model file coverage differs')
    source, child = (pycolmap.Reconstruction(str(p)) for p in (parent['source_sparse'], root))
    relation = derived_relation(source, child)
    cert_path = checked(spec['certificate'],bindings); cert = load(cert_path)
    expected = {'policy':DERIVED_POLICY,'parent_identity_spec':spec['parent_identity_spec'],
                'source_sparse':parent['source_sparse'],'derived_sparse':str(root),
                'relation':relation,'code_bindings':spec['code_bindings']}
    need(cert == expected, 'derived certificate differs from independently recomputed relation')
    for item in bindings.values(): need(record(item['path']) == item, 'derived input drift')
    from scripts.prepare_mvs import structure
    return {**parent, 'schema_version':'experimental-native-derived-identity-link-v1',
            'status':'EXPERIMENTAL_DERIVED_IDENTITIES_BOUND',
            'source_sparse':str(root), 'counts':structure(child), 'parent_source_sparse':parent['source_sparse'],
            'derivation':relation,'bindings':sorted(bindings.values(),key=lambda b:b['path'])}


def validate(spec_path, scene_path, image_root, mask_root, validity_root):
    """Surface admission reruns identity, native bridge and all mask pixel checks."""
    from scripts.prepare_mvs_masks import bridge_artifacts, model_bindings, scene_binding
    from scripts.prepare_mvs import structure, numeric_comparison, point_records, correspondence
    from integrations.mvs_masks.mask_geometry import warp_binary_mask
    import pycolmap
    spec = load(spec_path); bindings = {str(Path(spec_path).resolve()): record(spec_path)}
    need(spec.get('schema_version') == SURFACE_SCHEMA, 'experimental surface schema required')
    identity_path = checked(spec.get('identity_spec'), bindings)
    identity = validate_identity(identity_path)
    bindings.update({b['path']: b for b in identity['bindings']})
    bridge = bridge_artifacts(spec.get('bridge_run'))
    bridge_files = spec.get('bridge_bindings')
    need(isinstance(bridge_files,list) and bridge_files, 'bridge byte bindings absent')
    actual_bridge_files = {str(p.resolve()) for p in bridge['bridge'].rglob('*') if p.is_file()}
    need({str(checked(item,bindings)) for item in bridge_files} == actual_bridge_files and
         len(bridge_files) == len(actual_bridge_files), 'bridge byte coverage conflict')
    from scripts.prepare_mvs import sha
    need(bridge['plan'].get('script_sha256') == sha(Path(__file__).resolve().parents[2]/'scripts'/'prepare_mvs.py'),
         'bridge producer code drift')
    stages=bridge['plan'].get('stages')
    steps=json.loads((bridge['bridge']/'STEPS.json').read_text(encoding='utf-8'))
    need(isinstance(stages,list) and [s.get('stage') for s in stages] ==
         ['undistort','mvs_import','mvs_export','verify'] and len(steps)==4, 'bridge execution incomplete')
    for frozen,actual in zip(stages,steps):
        need(actual.get('stage')==frozen['stage'] and actual.get('argv')==frozen['argv'] and
             actual.get('cwd')==frozen['cwd'] and type(actual.get('returncode')) is int and actual.get('returncode')==0 and
             type(actual.get('pid')) is int and actual['pid']>0, 'bridge actual stage conflict')
    tool=Path(stages[1]['argv'][0]).resolve(strict=True)
    need(sha(tool)==bridge['plan'].get('tool_sha256'), 'bridge tool byte drift')
    bindings[str(tool)]=record(tool)
    need(str(bridge['source_sparse']) == identity['source_sparse'] and
         str(bridge['source_images']) == str(Path(identity['source_images']).resolve()) and
         Path(scene_path).resolve() == Path(bridge['scene']['path']) and
         Path(image_root).resolve() == bridge['bridge'], 'experimental bridge path conflict')
    models = model_bindings(bridge); scene_binding(bridge, models['records'])
    a, b = models['source_model'], models['target_model']
    need(structure(a) == structure(b), 'undistortion structure conflict')
    numeric_comparison(a, b, same_ids=True)
    need(set(a.points3D) == set(b.points3D), 'undistortion point IDs changed')
    for pid, point in a.points3D.items():
        other = b.points3D[pid]
        need(np.allclose(point.xyz, other.xyz, rtol=0, atol=1e-12) and np.array_equal(point.color, other.color) and
             [(e.image_id,e.point2D_idx) for e in point.track.elements] ==
             [(e.image_id,e.point2D_idx) for e in other.track.elements], 'undistortion changed native point/track')
    for iid in a.reg_image_ids():
        left,right = a.images[iid], b.images[iid]
        need(len(left.points2D) == len(right.points2D), 'undistortion keypoint indexing changed')
        need([p.point3D_id for p in left.points2D] == [p.point3D_id for p in right.points2D], 'undistortion keypoint ownership changed')
        rays_a = a.cameras[left.camera_id].cam_from_img(np.array([p.xy for p in left.points2D]))
        rays_b = b.cameras[right.camera_id].cam_from_img(np.array([p.xy for p in right.points2D]))
        need(np.isfinite(rays_a).all() and np.isfinite(rays_b).all() and
             np.max(np.abs(rays_a-rays_b)) <= 1e-5, 'undistortion normalized rays conflict')
    roundtrip = pycolmap.Reconstruction(str(bridge['bridge'] / 'roundtrip' / 'sparse'))
    need(structure(b) == structure(roundtrip), 'bridge roundtrip structure conflict')
    left,right,_=numeric_comparison(b, roundtrip, same_ids=False)
    for name,image in left.items():
        ca,cb=b.cameras[image.camera_id],roundtrip.cameras[right[name].camera_id]
        need((ca.width,ca.height)==(cb.width,cb.height) and
             np.isfinite(cb.params).all() and
             np.max(np.abs(ca.calibration_matrix()-cb.calibration_matrix())) <= 1e-7,
             'bridge roundtrip calibration conflict')
    extent = max(1., float(np.max(np.abs([p.xyz for p in b.points3D.values()]))))
    correspondence(point_records(b), point_records(roundtrip), extent * 1e-6)
    correspondence(point_records(b),scene_point_records(bridge['scene']['path']),extent*1e-6)
    for point in roundtrip.points3D.values():
        for track in point.track.elements:
            image=roundtrip.images[track.image_id];camera=roundtrip.cameras[image.camera_id]
            xy=camera.img_from_cam(image.cam_from_world()*point.xyz)
            need(np.isfinite(xy).all() and
                 np.max(np.abs(xy-image.points2D[track.point2D_idx].xy)) <= 1e-4,
                 'bridge roundtrip reprojection convention conflict')
    for path in (bridge['source_sparse'], bridge['undistorted_sparse'], bridge['bridge'] / 'roundtrip' / 'sparse'):
        for p in path.iterdir():
            if p.is_file(): bindings[str(p.resolve())] = record(p)
    for name in ('PLAN.json','RESULT.json','INPUT_BINDINGS.json','UNDISTORT_COMPARISON.json','scene.mvs'):
        p = bridge['bridge']/name; bindings[str(p)] = record(p)
    outputs = spec.get('outputs')
    need(isinstance(outputs, list) and len(outputs) == len(identity['frames']), 'warped output coverage conflict')
    by_name = {f['native_name']: f for f in identity['frames']}
    need(len({o.get('native_name') for o in outputs}) == len(outputs) and
         {o.get('native_name') for o in outputs} == set(by_name), 'warped output name conflict')
    for output in outputs:
        name = output['native_name']; f = by_name[name]; rec = models['records'][name]
        source_camera = a.cameras[a.images[f['image_id']].camera_id]
        target_camera = b.cameras[b.images[f['image_id']].camera_id]
        source = mio.read_label_png(f['mask']['path'], source_camera.width, source_camera.height)
        array = source['array'] if isinstance(source, dict) else source[0]
        expected = warp_binary_mask(array, source_camera, target_camera)
        need(expected.counts['foreground'] > 0, 'empty warped foreground')
        for kind, root, pixels in (('mask',mask_root,expected.mask),('validity',validity_root,expected.valid)):
            p = checked(output.get(kind), bindings)
            suffix = '.mask.png' if kind == 'mask' else '.validity.png'
            need(p == safe(root, Path(name).stem + suffix), 'warped output path conflict')
            mio.verify_label_png(p, pixels, target_camera.width, target_camera.height)
        bindings[rec['undistorted_image']['path']] = record(rec['undistorted_image']['path'])
    for item in list(bindings.values()): need(record(item['path']) == item, 'experimental surface input drift')
    return {**identity, 'schema_version': 'experimental-native-surface-link-v1',
            'status': 'EXPERIMENTAL_SOURCE_AND_WARP_BOUND', 'bindings': sorted(bindings.values(),key=lambda r:r['path']),
            'geometry_lineage': 'CURRENT_NATIVE_BRIDGE_NUMERIC_CHECKS', 'quality_status': 'NOT_EVALUATED'}
