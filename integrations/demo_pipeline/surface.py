"""Bounded post-SfM orchestration; all native work uses the caller's Job stage."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def read(path):
    path = Path(path)
    if path.stat().st_size > 8 << 20:
        raise ValueError('record exceeds 8 MiB')
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        return value
    def bad(text):
        raise ValueError('nonfinite JSON constant')
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=pairs,
                      parse_float=number, parse_constant=bad)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def choose_model(models, model_id=None):
    usable = [m for m in models if m.get('usable') is True and m.get('roundtrip_verified') is True
              and m.get('num_registered_images', 0) >= 2]
    if model_id is not None:
        usable = [m for m in usable if str(m['model_id']) == str(model_id)]
    if not usable:
        raise ValueError('requested usable SfM model is unavailable')
    ranked = sorted(usable, key=lambda m: (-m.get('num_registered_images', 0),
                                          -m.get('num_points3D', 0), str(m['model_id'])))
    selected = ranked[0]
    return selected, {'policy': 'explicit_model_id' if model_id is not None else
                     'most_registered_then_most_points_then_lexical_model_id',
                     'selected_model_id': str(selected['model_id']),
                     'other_model_ids': [str(m['model_id']) for m in models
                                         if m['model_id'] != selected['model_id']],
                     'components_merged': False}


def mask_indices(manifests, io_run):
    """Candidate selection only; the existing consumer authenticates all bytes."""
    found = {}; missing = {}
    for path in manifests:
        manifest = read(path)
        if manifest.get('contract_version') != 'object-mask-v1':
            raise ValueError('unsupported SAM3 manifest')
        if Path(manifest['source_io']['path']).resolve() != Path(io_run).resolve():
            raise ValueError('SAM3 manifest belongs to another T01 run')
        for frame in manifest['frames']:
            index = frame['input_index']
            if type(index) is not int or index < 0:
                raise ValueError('invalid mask input index')
            if frame.get('status') == 'OK' and frame.get('selected_mask'):
                if index in found:
                    raise ValueError('duplicate successful mask requires an explicit selection ledger')
                found[index] = frame
            else:
                missing[index] = frame.get('status', 'MISSING')
    return sorted(found), missing


def selection_ledger(manifests, selected_indices, subset_certificate):
    """Choose the unique OK source; keep non-OK rows for consumer verification."""
    selected = set(selected_indices)
    ledger = {'schema_version': 'object-mask-selection-v1',
              'subset_certificate': subset_certificate, 'manifests': [],
              'selections': [], 'excluded_source_rows': []}
    identities = {}; chosen = set()
    for path in manifests:
        path = Path(path).resolve(); digest = sha(path)
        ledger['manifests'].append({'path': str(path), 'sha256': digest})
        for frame in read(path)['frames']:
            index = frame['input_index']; identity = frame['frame_id']
            if index in identities and identities[index] != identity:
                raise ValueError('conflicting frame identity across mask manifests')
            identities[index] = identity
            if index in selected:
                if frame.get('status') != 'OK':
                    continue
                if index in chosen:
                    raise ValueError('duplicate successful mask requires an explicit selection ledger')
                mask = frame.get('selected_mask')
                if not isinstance(mask, dict) or not frame.get('selected_candidate_id'):
                    raise ValueError('OK source lacks a selected candidate/mask')
                chosen.add(index)
                ledger['selections'].append({'input_index': index, 'frame_id': identity,
                    'manifest_path': str(path), 'selected_candidate_id': frame['selected_candidate_id'],
                    'selected_mask_sha256': mask['sha256']})
            else:
                ledger['excluded_source_rows'].append({'manifest_path': str(path),
                    'manifest_sha256': digest, 'input_index': index,
                    'frame_id': identity, 'reason': 'CERTIFIED_VIEW_EXCLUDED'})
    if chosen != selected:
        raise ValueError('selected subset has no unique OK source for every view')
    return ledger


def complete_surface(sfm_run, io_run, dataset_manifest, mask_manifests, run, root,
                     python, tool_root, blender_exe, model_id=None, stage=None):
    """Generate an unaccepted real surface and independently reopened .blend.

    ``stage(name, argv, seconds, memory, extra=None)`` must supervise and reclaim
    its process tree. No fallback subprocess, retry, merging or mask filling.
    """
    if stage is None:
        raise ValueError('a bounded caller stage supervisor is required')
    root, run, sfm_run, io_run = map(lambda p: Path(p).resolve(), (root, run, sfm_run, io_run))
    if not run.is_relative_to(root / 'outputs') or run == root / 'outputs':
        raise ValueError('run must be a child of project outputs')
    work = run / 'surface_pipeline'
    work.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'exit_code': 4, 'stages': {},
              'surface': {'status': 'NOT_RUN', 'artifacts': [],
                          'quality_status': 'NOT_EVALUATED', 'mask_consumption': 'NOT_VERIFIED',
                          'limitations': ['Physical scale is unknown.',
                                          'SAM3 labels and photo shape fidelity are not accepted.',
                                          'Native mask consumption is not positively verified.']},
              'blender': {'status': 'NOT_RUN'}}
    def launch(name, argv, seconds=180, memory=2 << 30):
        outcome = stage(name, [str(x) for x in argv], seconds, memory, extra=[work])
        if type(outcome) is int:
            code = outcome
        elif (isinstance(outcome, dict) and outcome.get('reclaim_confirmed') is True
              and outcome.get('status') == 'completed' and not outcome.get('errors')
              and not outcome.get('log_errors')):
            code = outcome.get('root_exit_code', 4)
        else:
            code = 4
        if type(code) is not int:
            code = 4
        result['stages'][name] = {'exit_code': code, 'status': 'SUCCESS' if code == 0 else 'FAILED'}
        save(work / 'RESULT.json', result)
        if code != 0:
            result['exit_code'] = code or 4
            raise RuntimeError(f'{name} failed; see bounded stage logs')
    base = [python, '-B', '-X', 'utf8']
    try:
        selected, selection = choose_model(read(sfm_run / 'run.json')['models'], model_id)
        result['surface'].update(selection)
        indices, missing = mask_indices(mask_manifests, io_run)
        # Aliases share a native image. Retain a view only when every declared
        # alias has a successful mask, avoiding partial native identity sets.
        maps = [json.loads(line) for line in (sfm_run / 'frames_map.jsonl').read_text(
            encoding='utf-8-sig').splitlines() if line.strip()]
        groups = {}
        registered = set()
        for row in maps:
            groups.setdefault(row['staged_image_name'], []).append(row['input_index'])
            if any(str(r.get('model_id')) == str(selected['model_id'])
                   for r in row.get('registrations', [])):
                registered.add(row['input_index'])
            else:
                missing.setdefault(row['input_index'], 'UNREGISTERED_IN_SELECTED_MODEL')
        available = set(indices) & registered
        for aliases in groups.values():
            if not set(aliases) <= available:
                for index in aliases:
                    available.discard(index)
                    missing.setdefault(index, 'INCOMPLETE_MASK_ALIAS_GROUP')
        indices = sorted(available)
        selection_path = work / 'SUBSET_INDICES.json'
        reasons_path = work / 'SUBSET_REASONS.json'
        save(selection_path, indices); save(reasons_path, missing)
        launch('native_subset', base + [root / 'scripts/derive_general_subset.py',
               '--sfm-run', sfm_run, '--io-run', io_run, '--model-id', str(selected['model_id']),
               '--input-indices-json', selection_path, '--exclusion-reasons-json', reasons_path,
               '--output', work / 'subset'], 180)
        derived = read(work / 'subset' / 'RESULT.json')
        cert_path = Path(derived['subset_certificate']['path'])
        sparse = Path(derived['selected_sparse'])
        result['surface'].update(selected_indices=derived['selected_indices'],
                                 missing_indices=derived['missing_indices'],
                                 subset_certificate=str(cert_path))
        bridge = work / 'bridge'
        launch('mvs_bridge', base + [root / 'scripts/prepare_mvs.py', '--sparse', sparse,
               '--image-root', sfm_run, '--interface-colmap', Path(tool_root) / 'InterfaceCOLMAP.exe',
               '--output', bridge, '--wall-seconds', '180', '--max-output-mib', '512'], 210)
        if read(bridge / 'RESULT.json').get('status') != 'PASS':
            raise ValueError('bridge acceptance failed')
        masks = work / 'masks'
        argv = base + [root / 'scripts/prepare_mvs_masks.py', '--bridge', bridge,
                        '--sfm-run', sfm_run, '--output', masks]
        ledger = selection_ledger(mask_manifests, derived['selected_indices'], derived['subset_certificate'])
        ledger_path = work / 'MASK_SELECTION.json'; save(ledger_path, ledger)
        argv += ['--selection-ledger', ledger_path]
        for manifest in mask_manifests:
            argv += ['--mask-manifest', Path(manifest).resolve()]
        launch('mvs_masks', argv, 240)
        consumer = read(masks / 'manifest.json')
        if consumer.get('coverage_status') != 'COMPLETE' or consumer.get('publishable_arguments') is not True:
            raise ValueError('selected views lack complete verified mask coverage')
        photo = work / 'PHOTO_SOURCE.json'
        save(photo, {'schema_version': 'surface-photo-input-v1', 'bridge_run': str(bridge),
                     'sfm_run': str(sfm_run), 'io_run': str(io_run),
                     'dataset_manifest': str(Path(dataset_manifest).resolve()),
                     'model_id': str(selected['model_id']),
                     'subset_certificate': {'path': str(cert_path), 'sha256': sha(cert_path)}})
        plan = work / 'SURFACE_PLAN.json'
        launch('surface_plan', base + [root / 'scripts/build_mvs_surface.py', 'plan',
               '--scene-path', bridge / 'scene.mvs', '--image-root', bridge,
               '--mask-root', masks / 'masks', '--validity-root', masks / 'validity',
               '--tool-root', tool_root, '--total-seconds', '1200', '--stage-seconds', '900',
               '--memory-bytes', str(8 << 30), '--output-bytes', str(2 << 30),
               '--input-origin', 'real', '--photo-source', photo, '--plan-output', plan], 180)
        surface = work / 'surface'
        launch('surface_reconstruction', base + [root / 'scripts/build_mvs_surface.py', 'run',
               '--plan', plan, '--plan-sha256', sha(plan), '--output', surface], 1230, 9 << 30)
        actual = read(surface / 'result.json')
        if actual.get('status') != 'SURFACE_GENERATED_NOT_ACCEPTED' or actual.get('exit_code') != 0:
            raise ValueError('surface finite geometry readback failed')
        ply = surface / 'work' / '__mvs_surface_raw.ply'
        result['surface'].update(status='SUCCESS', artifacts=[{'path': str(ply), 'sha256': sha(ply),
                                'bytes': ply.stat().st_size}], evidence_paths=[str(surface / 'result.json')])
        spec = work / 'SCENES.json'
        save(spec, {'schema': 'ply-viewer-scenes-v1', 'scenes': [{'ply': str(ply), 'sha256': sha(ply),
                    'title': 'Current photo reconstruction surface', 'data_kind': 'real_photos',
                    'declared_unit': None, 'note': 'Raw reconstructed surface; quality not evaluated.',
                    'evidence': [str(surface / 'result.json')]}]})
        argv = base + [root / 'scripts/export_blender.py', '--spec', spec, '--output', work / 'blender',
                       '--profile', 'surface400k']
        if blender_exe:
            argv += ['--blender-exe', blender_exe]
        launch('blender_export', argv, 660, 8 << 30)
        exported = read(work / 'blender' / 'RESULT.json')
        if exported.get('status') != 'SUCCESS':
            raise ValueError('Blender independent reopen did not pass')
        stages = exported.get('stages', [])
        if ([s.get('mode') for s in stages] != ['create', 'verify']
                or any(s.get('data', {}).get('status') != 'PASS'
                       or s.get('supervision', {}).get('reclaim_confirmed') is not True
                       or s.get('supervision', {}).get('status') != 'completed'
                       or s.get('supervision', {}).get('root_exit_code') != 0
                       or s.get('supervision', {}).get('errors')
                       or s.get('supervision', {}).get('log_errors')
                       or s.get('data', {}).get('payload_sha256') != exported.get('payload_sha256')
                       for s in stages)):
            raise ValueError('Blender create/verify evidence incomplete')
        if stages[0]['data'].get('scenes') != stages[1]['data'].get('scenes'):
            raise ValueError('Blender independently reopened geometry differs')
        blend = exported['blend']
        if (Path(blend['path']).resolve() != work / 'blender' / 'model.blend'
                or Path(blend['path']).stat().st_size != blend['bytes']
                or sha(blend['path']) != blend['sha256']):
            raise ValueError('Blender output hash mismatch')
        if exported.get('blend_sha256_before_readback') != blend['sha256']:
            raise ValueError('Blender hash changed across readback')
        result['blender'] = {'status': 'SUCCESS', **blend, 'reopen_verified': True,
                             'result_path': str(work / 'blender' / 'RESULT.json')}
        result.update(status='SUCCESS', exit_code=0)
    except Exception as exc:
        result.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        if result['surface']['status'] != 'SUCCESS':
            result['surface'].update(status='FAILED', error=result['error'])
    save(work / 'RESULT.json', result)
    return result
