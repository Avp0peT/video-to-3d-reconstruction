"""Read registered world-to-camera poses from this pipeline's saved SfM run.

No reconstruction, fitting, camera conversion, or historical ledger lookup.
"""
import importlib
from pathlib import Path

import numpy as np

from . import parallel_photo_audit as io
from .temporal_review import bind, read_bound, integer, unique


def _native_module():
    pc = importlib.import_module('pycolmap')
    if pc.__version__ != '4.2.1':
        raise ValueError('pycolmap 4.2.1 required for saved camera readback')
    return pc


def load_registered_cameras(discovery: dict, frames: list[dict], refs: list) -> dict:
    """Return identity-bound cameras; malformed or changed evidence raises.

    Only SIMPLE_RADIAL with exactly zero k enters computable models. Other
    lenses retain exact parameters and poses in unsupported camera records.
    """
    path = discovery.get('stages', {}).get('sfm', {}).get('run_json')
    empty = dict(status='NO_MODEL', models=[], unsupported_models=[], native_reads=0,
                 reason='this pipeline has no saved SfM model')
    if not path:
        return empty
    path = Path(path).resolve(strict=True)
    # Append each successful read immediately so exceptions retain its binding.
    start_ref = len(refs)
    local = refs
    run = read_bound(path, local)
    run_ref = local[-1]
    if run.get('contract_version') not in ('sfm-native-object-v1', 'sfm-native-object-video-v1', 'sfm-native-object-pinhole-v1'):
        raise ValueError('saved object SfM run required')
    models = run.get('models')
    if type(models) is not list:
        raise ValueError('SfM models must be a list')
    if not models:
        if run.get('status') not in ('NO_MODEL', 'SUCCESS', 'PARTIAL'):
            raise ValueError('SfM run did not produce a valid no-model outcome')
        return empty
    if run.get('status') not in ('SUCCESS', 'PARTIAL'):
        raise ValueError('saved SfM model has unsuccessful run status')
    if run.get('backend', {}).get('name') != 'pycolmap' or run['backend'].get('version') != '4.2.1':
        raise ValueError('saved pycolmap 4.2.1 backend required')
    upstream = run.get('upstream_io_run_id')
    if type(upstream) is not str or not upstream:
        raise ValueError('SfM upstream identity required')
    frame_by_index = unique(frames, 'input_index')
    map_path = path.parent/'frames_map.jsonl'
    map_ref = bind(map_path, local)
    rows = [io.loads(line) for line in map_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    mapped = unique(rows, 'input_index')
    if set(mapped) != set(frame_by_index):
        raise ValueError('SfM frame scope differs from prepared frames')
    mask_rows = run.get('effective_options', {}).get('foreground', {}).get('mask_bindings', [])
    masks = unique(mask_rows, 'input_index')
    keys = [str(m['model_id']) for m in models]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate SfM model identity')
    registrations = {key: {} for key in keys}
    for index, row in mapped.items():
        integer(index, 'frame input index')
        frame = frame_by_index[index]
        if (row['upstream_io_run_id'] != upstream or row['frame_id'] != frame['frame_id']
                or row['canonical_image_sha256'] != frame['sha256']):
            raise ValueError('SfM upstream/frame/image identity differs')
        mask = frame.get('mask_binding')
        # frames_map canonical_mask describes optional photo-io masks. SAM
        # object masks are independently bound by effective_options.foreground.
        if mask is not None and (row.get('canonical_mask_sha256') not in (None, mask['sha256'])
                                 or masks.get(index, {}).get('sha256') != mask['sha256']):
            raise ValueError('SfM mask identity differs')
        seen = set()
        for reg in row['registrations']:
            key = str(reg['model_id'])
            iid = integer(reg['image_id'], 'registered image ID')
            integer(reg['camera_id'], 'registered camera ID')
            if key not in registrations or key in seen or iid in registrations[key]:
                raise ValueError('ambiguous or unknown SfM registration')
            if row['status'] != 'REGISTERED':
                raise ValueError('registration attached to unregistered frame')
            seen.add(key)
            registrations[key][iid] = (frame, row, reg)
    pc = _native_module()
    bind(pc.__file__, local)
    output = []; excluded = []; reads = 0; unsupported = False
    for model, key in zip(models, keys):
        ids = model.get('registered_image_ids')
        if (type(ids) is not list or any(type(i) is not int or i <= 0 for i in ids)
                or len(set(ids)) != len(ids) or set(ids) != set(registrations[key])):
            raise ValueError('SfM registered image IDs differ from frame map')
        if model.get('roundtrip_verified') is not True:
            raise ValueError('roundtrip-verified saved SfM model required')
        binary = io.under(path.parent, model['binary_path'])
        binaries = [bind(binary/name, local) for name in ('cameras.bin', 'images.bin', 'points3D.bin')]
        for name in ('rigs.bin', 'frames.bin'):
            if (binary/name).exists():
                binaries.append(bind(binary/name, local))
        rec = pc.Reconstruction(str(binary)); reads += 1
        if set(rec.reg_image_ids()) != set(ids):
            raise ValueError('native/run registered image IDs differ')
        cameras = {}; unsupported_cameras = {}
        for iid in sorted(ids):
            frame, row, reg = registrations[key][iid]
            image = rec.image(iid)
            camera = rec.cameras[image.camera_id]
            if image.name != row['staged_image_name'] or image.camera_id != reg['camera_id']:
                raise ValueError('native image name/camera differs from frame binding')
            if (camera.width, camera.height) != (frame['width'], frame['height']):
                raise ValueError('native camera dimensions differ')
            params = np.asarray(camera.params, dtype=float)
            pose = image.cam_from_world()
            R = np.asarray(pose.rotation.matrix(), dtype=float)
            t = np.asarray(pose.translation, dtype=float)
            if (params.ndim != 1 or R.shape != (3, 3) or t.shape != (3,)
                    or not all(np.isfinite(v).all() for v in (params, R, t))
                    or not np.allclose(R @ R.T, np.eye(3), atol=1e-8, rtol=0)
                    or not np.isclose(np.linalg.det(R), 1., atol=1e-8, rtol=0)):
                raise ValueError('invalid native camera parameters or rigid pose')
            lens = camera.model_name
            count = {'SIMPLE_PINHOLE': 3, 'PINHOLE': 4, 'SIMPLE_RADIAL': 4}.get(lens)
            if count is not None and (len(params) != count or params[0] <= 0 or (lens == 'PINHOLE' and params[1] <= 0)):
                raise ValueError('invalid native lens parameters')
            supported = lens == 'SIMPLE_RADIAL' and params[3] == 0
            unsupported |= not supported
            target = cameras if supported else unsupported_cameras
            target[frame['input_index']] = dict(input_index=frame['input_index'], image_id=iid,
                camera_id=int(image.camera_id), camera_model=lens, params=params.tolist(),
                R=R.tolist(), t=t.tolist(), width=int(camera.width), height=int(camera.height),
                status='SUPPORTED' if supported else 'UNSUPPORTED',
                pose_convention='world_to_camera', frame_id=frame['frame_id'])
        record = dict(model_key=key, cameras=cameras, unsupported_cameras=unsupported_cameras,
                      registered_image_ids=sorted(ids),
                      binding=dict(run=run_ref, frames_map=map_ref, binaries=binaries))
        if cameras:
            output.append(record)
        else:
            record['status'] = 'UNSUPPORTED'
            excluded.append(record)
    # Detect replacement or mutation during native reading before publishing.
    for ref in local[start_ref:]:
        bind(ref['path'], [], ref)
    return dict(status='UNSUPPORTED' if unsupported else 'LOADED', models=output, unsupported_models=excluded,
                native_reads=reads, reason='unsupported native camera model present' if unsupported else None)
