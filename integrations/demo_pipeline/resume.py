"""Strict, explicit surface continuation; old inputs/results are read-only.

Legacy unified producers without a committed resume checkpoint are refused.
The module never manufactures historical commit evidence from current bytes.
"""
from pathlib import Path
import json
import time

from .surface import read, save, choose_model, complete_surface
from .pipeline import ROOT, _invoke, entry_code, source_info, verify_geometry_outputs
from integrations.mvs_surface.photo_inputs import _binding, _under, _rows_by_index, _json, MODEL_BIN
from integrations.mvs_masks import manifest_io as mio

SCHEMA = 'photo-surface-continuation-v1'
MAX_FILES = 2048
MAX_TOTAL_BYTES = 3 << 30


def need(ok, message):
    if not ok:
        raise ValueError('surface continuation: ' + message)


def plan(run, output, *, python, tool_root, blender_exe, model_id=None, root=ROOT, _checkpoint_write=False):
    """Read-only strict plan; missing producer checkpoint means refusal."""
    root = Path(root).resolve(strict=True)
    run = Path(run).resolve(strict=True)
    output = Path(output).resolve()
    need(run.is_relative_to(root / 'outputs'), 'source run outside project outputs')
    need(output.is_relative_to(root / 'outputs') and output != root / 'outputs', 'output outside project outputs')
    need(not output.exists(), 'output already exists')
    need(not output.is_relative_to(run) and not run.is_relative_to(output), 'output overlaps source run')
    frozen = {}; total = 0
    def bind(path, declared=None, limit=512 << 20):
        nonlocal total
        actual = _binding(path, limit)
        if declared is not None:
            need(type(declared.get('bytes')) is int and
                 all(declared.get(k) == actual[k] for k in ('path', 'bytes', 'sha256')), 'declared file binding differs')
        if actual['path'] not in frozen:
            total += actual['bytes']
        need(len(frozen) < MAX_FILES and total <= MAX_TOTAL_BYTES, 'input file/byte budget exceeded')
        need(frozen.get(actual['path'], actual) == actual, 'input drift during plan')
        frozen[actual['path']] = actual
        return actual
    def doc(path, limit=8 << 20):
        before = bind(path, limit=limit)
        with Path(path).open('rb') as stream:
            payload = stream.read(limit + 1)
        need(len(payload) <= limit, 'record exceeds its payload limit')
        value = _json(payload)
        need(bind(path, limit=limit) == before and type(value) is dict, 'record changed or invalid')
        return value
    meta = doc(run / 'pipeline.json'); old_plan = doc(run / 'PLAN.json')
    need(meta.get('schema') == 'photo-demo-pipeline-v1' and old_plan.get('schema') == 'photo-demo-plan-v1', 'unsupported source schemas')
    if _checkpoint_write:
        need(meta.get('status') == 'RUNNING' and meta.get('exit_code') is None, 'checkpoint must be committed by the running producer')
    else:
        need(meta.get('status') in ('SUCCESS', 'PARTIAL', 'FAILED', 'CANCELLED')
             and type(meta.get('exit_code')) is int and meta.get('updated_utc'), 'source pipeline has no committed terminal state')
    stages = meta.get('stages')
    need(type(stages) is dict and all(type(s) is dict and s.get('status') != 'RUNNING' for s in stages.values()), 'source stage still RUNNING')
    need(meta.get('source') == old_plan.get('source', {}).get('path') and
         meta.get('prompt') == old_plan.get('prompt') and meta.get('object_id') == old_plan.get('object_id'), 'pipeline/plan identity differs')
    inventory = doc(run / 'INPUT_INVENTORY.json')
    current_source = source_info(Path(meta['source']))
    need(current_source == old_plan['source'] and inventory.get('files') == current_source['files']
         and inventory.get('source') == meta['source'], 'original input inventory drift')
    need(not output.is_relative_to(Path(meta['source'])) and not Path(meta['source']).is_relative_to(output), 'output overlaps originals')
    for row in current_source['files']:
        bind(row['path'], {k: row[k] for k in ('path', 'bytes', 'sha256')})
    # Every launched old native stage must be reclaimed, including downstream
    # failures; otherwise the old process could still mutate reused files.
    terminal_entry_paths = set()
    for name, value in stages.items():
        if 'argv' not in value and 'entry_path' not in value:
            continue
        need(value.get('reclaim_confirmed') is True and isinstance(value.get('entry_path'), str), 'source native reclamation unconfirmed')
        entry_path = Path(value['entry_path']).resolve(strict=True)
        need(entry_path.is_relative_to(run), 'source supervisor entry escapes run')
        entry = doc(entry_path)
        if name not in ('input', 'sfm', 'segmentation'):
            # These records are written AFTER the producer's upstream commit.
            # Require current terminal/reclaim evidence and freeze it for this
            # continuation, without pretending it existed in that earlier seal.
            terminal_entry_paths.add(str(entry_path))
        need(entry.get('reclaim_confirmed') is True and not entry.get('errors') and not entry.get('log_errors')
             and entry.get('argv') == value.get('argv') and entry.get('cwd') == str(root)
             and type(entry.get('pid')) is int and entry['pid'] > 0, 'source supervisor identity/reclamation differs')
        need(entry_code(entry) == value.get('exit_code'), 'source stage exit declaration differs')
    need(not stages.get('segmentation', {}).get('execution_stages'),
         'aggregation segmentation resume is unsupported; retain original and recovery execution evidence')
    for name in ('input', 'sfm', 'segmentation'):
        value = stages.get(name, {})
        need(value.get('status') in (('SUCCESS',) if name == 'input' else ('SUCCESS', 'PARTIAL'))
             and value.get('reclaim_confirmed') is True and value.get('entry_path'), 'required committed upstream stage unavailable')
        need(value.get('exit_code') == (1 if value['status'] == 'PARTIAL' else 0), 'upstream status/exit differs')
    prepared = doc(run / 'input/RESULT.json')
    need(prepared.get('contract_version') == 'object-input-v1' and prepared.get('status') == 'INPUTS_PREPARED'
         and prepared.get('exit_code') == 0 and prepared.get('data_kind') == 'real', 'prepared input incomplete')
    # Reuse the full existing readonly source->prepared->T01 verifier. Its
    # pinned runtime/implementation rules apply; mismatches are not bypassed.
    from integrations.object_input.session import verify_input
    checked = verify_input(run / 'input', project_root=root)
    need(checked.get('status') == 'INPUT_BINDINGS_VALID', 'prepared input independent verification failed')
    for row in prepared.get('source_bindings', []) + prepared.get('implementation', []):
        bind(row['path'], row)
    for frame in prepared.get('frames', []):
        if type(frame.get('prepared')) is dict:
            bind(frame['prepared']['path'], frame['prepared'])
    io_ref = prepared['photo_io']['run']; manifest_ref = prepared['photo_io']['manifest']
    io_file = Path(io_ref['path']).resolve(strict=True)
    dataset = Path(manifest_ref['path']).resolve(strict=True)
    need(io_file.is_relative_to(run / 'input') and dataset.is_relative_to(run / 'input'), 'prepared receipt escapes pipeline input')
    bind(io_file, io_ref); bind(dataset, manifest_ref)
    io = doc(io_file); io_root = io_file.parent
    frame_path = io_root / 'frames.jsonl'; bind(frame_path, limit=8 << 20)
    frames = _rows_by_index(mio.read_jsonl_strict(frame_path), 'IO')
    need(len(frames) <= 96 and io.get('status') == 'SUCCESS', 'IO incomplete or outside profile')
    sfm_file = Path(stages['sfm']['run_json']).resolve(strict=True)
    sfm_root = sfm_file.parent
    need(sfm_root.is_relative_to(root / 'outputs') and not output.is_relative_to(sfm_root)
         and not sfm_root.is_relative_to(output), 'SfM/output roots overlap or escape outputs')
    sfm = doc(sfm_file)
    need(sfm.get('status') == stages['sfm']['status'] and sfm.get('counts') == stages['sfm'].get('counts')
         and sfm.get('contract_version') in ('sfm-native-v1', 'sfm-native-profile-v1', 'sfm-native-object-v1', 'sfm-native-object-pinhole-v1')
         and sfm.get('data_kind') == 'real' and sfm.get('upstream_io_run_id') == io.get('run_id'), 'SfM source identity differs')
    selected, selection = choose_model(sfm.get('models', []), model_id)
    model_dir = _under(sfm_root, selected['binary_path']).parent
    # New producers must retain a passing native-domain audit for the exact
    # selected component. Legacy producers keep their existing checkpoint rules.
    camera_required = (old_plan.get('automatic_recovery') is True
                       or sfm.get('contract_version') == 'sfm-native-object-pinhole-v1'
                       or 'automatic_camera_recovery' in meta)
    if camera_required:
        recovery_path = run / 'camera_recovery/RESULT.json'
        recovery = doc(recovery_path)
        need(recovery == meta.get('automatic_camera_recovery'), 'camera recovery metadata differs')
        need(recovery.get('surface_ready') is True
             and recovery.get('status') in ('ORIGINAL_DOMAIN_VALID', 'CONSTRAINED_DOMAIN_VALID')
             and recovery.get('selected_sfm') == str(sfm_file)
             and recovery.get('selected_sfm_sha256') == bind(sfm_file)['sha256'],
             'selected camera domain is invalid or unbound')
        original = Path(recovery.get('original_sfm', '')).resolve(strict=True)
        need(original.is_relative_to(root / 'outputs')
             and bind(original)['sha256'] == recovery.get('original_sfm_sha256'),
             'original camera SfM binding differs')
        if recovery['status'] == 'ORIGINAL_DOMAIN_VALID':
            need(original == sfm_file, 'original camera selection differs')
        else:
            need(recovery.get('candidate_sfm') == str(sfm_file)
                 and recovery.get('candidate_sfm_sha256') == bind(sfm_file)['sha256']
                 and stages.get('sfm', {}).get('selected_from') == 'sfm_constrained',
                 'constrained camera selection execution differs')
        expected_binary = _under(sfm_root, selected['binary_path'])
        chosen_audit = False
        audits = recovery.get('audits')
        need(type(audits) is list and 1 <= len(audits) <= 2, 'camera audit ledger missing')
        labels = set()
        for item in audits:
            label = item.get('label')
            need(label in ('original', 'candidate') and label not in labels, 'camera audit label invalid')
            labels.add(label)
            audit_path = Path(item.get('path', '')).resolve(strict=True)
            need(audit_path == (run / 'camera_recovery' / (label + '.json')).resolve(),
                 'camera audit path differs')
            audit = doc(audit_path)
            need(audit == item.get('result') and bind(audit_path)['sha256'] == item.get('sha256'),
                 'camera audit receipt differs')
            stage = stages.get('camera_domain_' + label, {})
            need(isinstance(stage.get('entry_path'), str) and isinstance(stage.get('argv'), list)
                 and stage.get('reclaim_confirmed') is True and stage.get('exit_code') == item.get('exit_code')
                 and stage.get('exit_code') in (0, 1), 'camera audit execution unavailable')
            source_model = Path(audit.get('source_model', '')).resolve(strict=True)
            need(source_model.is_relative_to(root / 'outputs') and source_model.is_dir(), 'camera model escapes outputs')
            before, after = audit.get('source_bindings_before'), audit.get('source_bindings_after')
            need(audit.get('source_unchanged') is True and type(before) is dict and before and before == after,
                 'camera audit source drift or bindings absent')
            actual = {p.relative_to(source_model).as_posix(): p for p in source_model.rglob('*') if p.is_file()}
            need(set(actual) == set(before), 'camera audit file inventory differs')
            for relative, path in actual.items():
                receipt = bind(path)
                need(not path.is_symlink() and path.resolve().is_relative_to(source_model) and receipt['bytes'] == before[relative].get('bytes')
                     and receipt['sha256'] == before[relative].get('sha256'), 'camera audit source binding differs')
            if source_model == expected_binary:
                need(audit.get('status') == 'CAMERA_DOMAIN_VALID' and stage['exit_code'] == 0,
                     'selected camera domain audit rejected')
                chosen_audit = True
        need(chosen_audit, 'selected camera component has no passing native audit')
    checkpoint = None
    if not _checkpoint_write:
        checkpoint = doc(run / 'SURFACE_RESUME_SOURCE.json')
        need(checkpoint.get('schema') == 'photo-surface-resume-source-v1'
             and checkpoint.get('source_run') == str(run)
             and checkpoint.get('producer_status') == 'RUNNING'
             and checkpoint.get('upstream_stages') == {k: stages[k] for k in ('input', 'sfm', 'segmentation')}, 'committed upstream checkpoint differs')
        for row in checkpoint.get('bindings', []):
            bind(row['path'], row)
        need(type(checkpoint.get('bindings')) is list and checkpoint['bindings'], 'source checkpoint binding ledger missing')
    actual_files = sorted(p.resolve() for p in model_dir.rglob('*') if p.is_file())
    need(len(actual_files) <= 128, 'component file budget exceeded')
    committed_paths = {r['path'] for r in checkpoint['bindings']} if checkpoint else set()
    for path in actual_files:
        need(_checkpoint_write or str(path) in committed_paths, 'SOURCE_BINDING_REQUIRED: model file has no producer checkpoint binding')
        bind(path)
    need(all((_under(sfm_root, selected['binary_path']) / f).is_file() for f in MODEL_BIN), 'native component missing')
    # Native descriptions contain three full copies (source/binary/text) of
    # points and tracks: the real 33-view model is 45,323,944 bytes. Keep small
    # control records bounded separately from this explicit geometry payload.
    descriptions = doc(_under(sfm_root, selected['descriptions_path']), limit=128 << 20)
    need(selected.get('source_vs_binary_differences') == selected.get('source_vs_text_differences') == []
         and all(k in descriptions for k in ('source', 'binary', 'text')), 'roundtrip evidence incomplete')
    map_path = sfm_root / 'frames_map.jsonl'; bind(map_path, limit=8 << 20)
    maps = _rows_by_index(mio.read_jsonl_strict(map_path), 'SfM')
    need(len(maps) == len(frames) == sfm.get('counts', {}).get('input_rows'), 'SfM input rows differ')
    for frame, mapped in zip(frames, maps):
        need(frame.get('status') == 'OK' and mapped.get('frame_id') == frame.get('frame_id')
             and mapped.get('upstream_io_run_id') == io.get('run_id')
             and mapped.get('upstream_image_output') == frame.get('image_output'), 'SfM frame identity differs')
        canonical = bind(_under(io_root, frame['image_output']))
        staged = bind(_under(sfm_root, mapped['staged_image_name']))
        need(canonical['sha256'] == staged['sha256'] == mapped.get('canonical_image_sha256'), 'staged image differs')
    product = doc(run / 'segmentation/PRODUCT_OUTCOME.json')
    need(product.get('status') == stages['segmentation']['status'] and
         product.get('exit_code') == stages['segmentation']['exit_code'] and
         product.get('counts') == stages['segmentation'].get('counts') and
         product.get('run_dir') == stages['segmentation'].get('run_dir'), 'SAM product differs from pipeline')
    need(product.get('stop_state', {}).get('latched') is False, 'SAM stop state unresolved')
    mask_root = Path(product['run_dir']).resolve(strict=True)
    need(mask_root.is_relative_to(run / 'segmentation'), 'SAM product escapes source run')
    mask_path = mask_root / 'manifest.json'; bind(mask_path, limit=8 << 20)
    manifest = mio.parse_object_manifest(mask_path)
    upstream = mio.resolve_source_io(manifest['source_io'], 'continuation')
    need(Path(upstream['path']) == io_root and manifest['status'] == product['status']
         and manifest['counts'] == product['counts'] and manifest['object_id'] == meta['object_id'], 'SAM manifest identity differs')
    mask_frames = manifest['frames']
    need(len(mask_frames) == len(frames) and {f['input_index'] for f in mask_frames} == set(range(len(frames))), 'SAM complete input ledger missing')
    for row in mask_frames:
        frame = frames[row['input_index']]
        canonical = frozen[str(_under(io_root, frame['image_output']))]
        need(row.get('frame_id') == frame.get('frame_id') and row.get('source_image_sha256') == canonical['sha256']
             and row.get('photo_io_image_sha256') == frame.get('image_sha256')
             and Path(row.get('source_image', '')).resolve() == Path(canonical['path'])
             and (row.get('width'), row.get('height')) == (frame.get('canonical_width'), frame.get('canonical_height')), 'SAM frame source identity differs')
        for candidate in row['candidates']:
            actual = bind(candidate['resolved_mask_path'], limit=64 << 20)
            need(actual['sha256'] == candidate['mask_sha256'], 'committed candidate mask hash differs')
        # Only manifest-selected candidate masks are eligible; orphan files do
        # not participate. Existing surface consumer performs raster validation.
    config = doc(run / 'segmentation/CONFIG.json')
    need(config.get('io_run') == str(io_root) and config.get('text_prompt') == meta['prompt']
         and config.get('object_id') == meta['object_id'] and config.get('indices') == list(range(len(frames))), 'SAM request identity differs')
    if checkpoint is not None:
        need(all(p in committed_paths or p in terminal_entry_paths or p in (str(run / 'pipeline.json'), str(run / 'SURFACE_RESUME_SOURCE.json'))
                 for p in frozen), 'SOURCE_BINDING_REQUIRED: dependency lacks producer checkpoint hash')
    tools = [Path(python).resolve(), Path(blender_exe).resolve()]
    tool_root = Path(tool_root).resolve()
    if not _checkpoint_write:
        tools += [tool_root / f for f in ('InterfaceCOLMAP.exe', 'DensifyPointCloud.exe', 'ReconstructMesh.exe')]
        for tool in tools:
            bind(tool)
        code = [Path(__file__), Path(__file__).with_name('surface.py'), Path(__file__).with_name('pipeline.py')]
        code += sorted((root / 'integrations/demo_pipeline').glob('*.py'))
        code += [root / 'scripts' / n for n in ('derive_general_subset.py', 'prepare_mvs.py', 'prepare_mvs_masks.py', 'build_mvs_surface.py', 'export_blender.py')]
        for path in code:
            bind(path, limit=8 << 20)
    for row in list(frozen.values()):
        need(_binding(row['path'], 512 << 20) == row, 'source changed during plan')
    if _checkpoint_write:
        frozen.pop(str(run / 'pipeline.json'), None)
    return {'schema': SCHEMA, 'source_run': str(run), 'output': str(output), 'root': str(root),
            'sfm_run': str(sfm_root), 'io_run': str(io_root), 'dataset_manifest': str(dataset),
            'mask_manifests': [str(mask_path)], 'model_id': str(selected['model_id']),
            'requested_model_id': model_id, 'selection': selection,
            'python': str(tools[0]), 'blender_exe': str(tools[1]), 'tool_root': str(tool_root),
            'upstream_stages': {k: stages[k] for k in ('input', 'sfm', 'segmentation')},
            'source_status': meta['status'], 'source_prompt': meta['prompt'], 'source_object_id': meta['object_id'],
            'bindings': sorted(frozen.values(), key=lambda r: r['path']), 'native_attempts_per_stage': 1,
            'automatic_retry': False, 'cad_status': 'NOT_RUN', 'research_status': 'NOT_RUN',
            'upstream_reuse': ['input', 'sfm', 'segmentation'], 'status': 'READY'}


def execute(frozen, *, executor=None, surface_builder=None):
    """Revalidate the exact plan; execute only bounded downstream work once.

    Injection seams are for offline tests. The default uses the existing Job
    supervisor and complete_surface. Old run files are never rewritten.
    """
    need(type(frozen) is dict and frozen.get('schema') == SCHEMA, 'invalid continuation plan')
    repeated = plan(frozen['source_run'], frozen['output'], python=frozen['python'],
                    tool_root=frozen['tool_root'], blender_exe=frozen['blender_exe'],
                    model_id=frozen['requested_model_id'], root=frozen['root'])
    need(repeated == frozen, 'plan changed since approval')
    root, output = Path(frozen['root']), Path(frozen['output'])
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'PLAN.json', frozen)
    state = {'schema': SCHEMA, 'status': 'RUNNING', 'source_run': frozen['source_run'],
             'source_bindings': frozen['bindings'], 'prompt': frozen['source_prompt'],
             'object_id': frozen['source_object_id'], 'stages': {}, 'exit_code': None,
             'surface': {'status': 'NOT_RUN'}, 'blender': {'status': 'NOT_RUN'},
             'cad_status': 'NOT_RUN', 'research_status': 'NOT_RUN', 'deliverable_ready': False,
             'source_pipeline_status': frozen['source_status'],
             'reused_stage_statuses': {k: v['status'] for k, v in frozen['upstream_stages'].items()},
             'scope': 'surface and Blender continuation; CAD/research/quality not accepted'}
    save(output / 'CONTINUATION.json', state)
    invoke = executor or _invoke
    attempted = set(); stopped = False; first_stop_code = None
    def unchanged():
        for row in frozen['bindings']:
            need(_binding(row['path'], 512 << 20) == row, 'frozen input drift')
    def stage(name, argv, seconds, memory, extra=None):
        nonlocal stopped, first_stop_code
        need(not stopped and name not in attempted, 'stage retry or launch after stop refused')
        allowed = {'native_subset': (180, 2 << 30), 'mvs_bridge': (210, 2 << 30),
                   'mvs_masks': (240, 2 << 30), 'surface_plan': (180, 2 << 30),
                   'surface_reconstruction': (1230, 9 << 30), 'blender_export': (660, 8 << 30)}
        need(name in allowed and type(seconds) is int and 0 < seconds <= allowed[name][0]
             and type(memory) is int and 0 < memory <= allowed[name][1], 'stage outside fixed budget')
        unchanged(); attempted.add(name)
        entry = invoke(root, output, name, argv, seconds=seconds, memory=memory,
                       additional_roots=extra or [])
        save(output / (name + '_ENTRY.json'), entry)
        code = entry_code(entry)
        state['stages'][name] = {'status': 'SUCCESS' if code == 0 else 'FAILED', 'exit_code': code,
                                 'entry_path': str(output / (name + '_ENTRY.json')),
                                 'reclaim_confirmed': entry.get('reclaim_confirmed')}
        if code != 0:
            stopped = True
            if first_stop_code is None:
                first_stop_code = code
        save(output / 'CONTINUATION.json', state)
        unchanged()
        # Match the unified producer's stage API: the adjudicated code includes
        # cancellation, deadlines and reclaim, unlike the raw child returncode.
        return code
    try:
        builder = surface_builder or complete_surface
        outcome = builder(sfm_run=Path(frozen['sfm_run']), io_run=Path(frozen['io_run']),
                          dataset_manifest=Path(frozen['dataset_manifest']),
                          mask_manifests=[Path(p) for p in frozen['mask_manifests']],
                          run=output / 'geometry', root=root, python=Path(frozen['python']),
                          tool_root=Path(frozen['tool_root']), blender_exe=Path(frozen['blender_exe']),
                          model_id=frozen['model_id'], stage=stage)
        need(type(outcome) is dict and type(outcome.get('exit_code')) is int, 'surface builder result missing')
        verify_geometry_outputs(outcome, output / 'geometry')
        unchanged()
        state.update(surface=outcome.get('surface', {'status': 'NOT_RUN'}),
                     blender=outcome.get('blender', {'status': 'NOT_RUN'}), geometry_steps=outcome.get('stages', {}))
        code = first_stop_code if first_stop_code is not None else outcome['exit_code']
        ready = code == 0 and not stopped and state['surface'].get('status') == state['blender'].get('status') == 'SUCCESS'
        need(code != 0 or ready, 'success result lacks completed surface/Blender')
        partial = any(v['status'] == 'PARTIAL' for v in frozen['upstream_stages'].values())
        state.update(status=('PARTIAL' if partial else 'SUCCESS') if ready else ('CANCELLED' if code == 130 else 'FAILED'), exit_code=code if code != 0 else 4,
                     deliverable_ready=ready)
        if ready:
            state['exit_code'] = 1 if partial else 0
        if outcome.get('error'):
            state['error'] = outcome['error']
    except Exception as exc:
        code = first_stop_code if first_stop_code is not None else 4
        state.update(status='CANCELLED' if code == 130 else 'FAILED', exit_code=code,
                     error=f'{type(exc).__name__}: {exc}', deliverable_ready=False)
    state['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    save(output / 'CONTINUATION.json', state)
    return state


def write_resume_checkpoint(run, *, root=ROOT):
    """Producer-only first commit after input/SfM/SAM completion and reclaim.

    Call after persisting all three upstream stage records, before surface
    starts, while pipeline.status remains RUNNING and exit_code is None.
    Never call against a historical terminal run to retrofit commit evidence.
    """
    run = Path(run).resolve(strict=True)
    checkpoint = run / 'SURFACE_RESUME_SOURCE.json'
    need(not checkpoint.exists(), 'producer checkpoint already exists')
    old_plan = read(run / 'PLAN.json')
    # Planning remains read-only; this mode freezes the currently executing
    # producer's upstream files and does not require downstream tool availability.
    draft = plan(run, run.parent / ('__checkpoint_probe_' + run.name),
                 python=old_plan['python'], tool_root=old_plan['mvs_tool_root'],
                 blender_exe=old_plan.get('blender_exe') or run / '__unused_blender',
                 model_id=old_plan.get('model_id'), root=root, _checkpoint_write=True)
    record = {'schema': 'photo-surface-resume-source-v1', 'source_run': str(run),
              'producer_status': 'RUNNING', 'committed_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'upstream_stages': draft['upstream_stages'], 'bindings': draft['bindings'],
              'source_prompt': draft['source_prompt'], 'source_object_id': draft['source_object_id']}
    with checkpoint.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    return _binding(checkpoint, 8 << 20)
