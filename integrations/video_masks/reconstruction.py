"""Reconstruct from an explicitly bound full-video mask fusion delivery.

Reuse the source sampled photographs for a controlled before/after comparison.
Do not relabel historical SAM3 outcomes or rerun image/video segmentation.
"""
from __future__ import annotations

import html
from pathlib import Path
import uuid

from integrations.demo_pipeline.pipeline import (
    ROOT, LIGHT_REL, MVS_REL, _invoke, entry_code, make_viewer, read, save, sha, utc,
)


def _drift(bindings):
    changed = []
    for path, expected in bindings.items():
        try:
            if sha(Path(path)) != expected:
                changed.append(path)
        except OSError:
            changed.append(path)
    return changed


def _view_surface(run, summary):
    from integrations.viewer.export import export_html
    work = run / 'geometry' / 'surface_pipeline' / 'surface' / 'work'
    links = summary.setdefault('viewer_links', {})
    for label, filename in [('dense', '__mvs_dense.ply'), ('mesh', '__mvs_surface_raw.ply')]:
        source = work / filename
        if not source.is_file():
            continue
        try:
            directory = run / (label + '_viewer')
            directory.mkdir(exist_ok=False)
            spec = directory / 'scenes.json'
            save(spec, {'schema': 'ply-viewer-scenes-v1', 'scenes': [{
                'ply': str(source), 'sha256': sha(source), 'title': label,
                'data_kind': 'real_photos', 'declared_unit': None,
                'note': 'Unchanged native geometry; no hole filling; quality not evaluated.',
                'evidence': [str(work.parent / 'result.json')],
            }]})
            result = export_html(spec, directory / 'viewer.html', profile='surface400k')
            save(directory / 'RESULT.json', result)
            links[label] = str(directory / 'viewer.html')
            summary[label + '_path'] = str(source)
        except Exception as exc:
            summary.setdefault('diagnostic_errors', []).append(f'{label} viewer: {type(exc).__name__}: {exc}')


def render(run):
    """Report-only: never invoke a reconstruction tool."""
    from integrations.video_masks.reconstruction_report import build_report
    run = Path(run).resolve(strict=True)
    summary = read(run / 'SUMMARY.json')
    result = build_report(run, summary)
    summary['report'] = result
    save(run / 'SUMMARY.json', summary)
    target = Path(result.get('report_html', run / 'diagnostics' / 'index.html'))
    if not target.is_absolute():
        target = run / target
    relative = target.relative_to(run).as_posix()
    (run / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='
        + html.escape(relative, quote=True) + '"><a href="'
        + html.escape(relative, quote=True) + '">Open reconstruction stages</a>', encoding='utf-8')
    return result


def execute(source, output):
    from integrations.video_masks.reconstruction_input import prepare, validate_derivation
    from integrations.demo_pipeline.surface import complete_surface, choose_model
    from integrations.blender_export.export import find_blender

    source = Path(source).resolve(strict=True)
    run = Path(output).resolve()
    if not run.is_relative_to(ROOT / 'outputs') or run == ROOT / 'outputs' or run.exists():
        raise ValueError('output must be a fresh child of repository outputs')
    python = (ROOT / LIGHT_REL).resolve(strict=True)
    tools = (ROOT / MVS_REL).resolve(strict=True)
    blender = find_blender()
    for name in ('InterfaceCOLMAP.exe', 'DensifyPointCloud.exe', 'ReconstructMesh.exe'):
        if not (tools / name).is_file():
            raise ValueError('missing existing MVS executable: ' + name)
    plan_source = read(source / 'PLAN.json')
    baseline = Path(plan_source['pipeline_plan_path']).parent
    sfm_output = ROOT / 'outputs' / 'delegation' / 'T02A' / ('FUSED_' + run.name + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    plan = {
        'schema': 'video-mask-reconstruction-plan-v1', 'source_run': str(source),
        'baseline_run': str(baseline), 'python': str(python), 'mvs_tool_root': str(tools),
        'blender_exe': str(blender), 'sfm_output': str(sfm_output),
        'sfm_profile': 'cad-object-v1', 'point_selection': 'native',
        'input_selection': 'Reuse all original sampled T01 rows; no newly sampled frames.',
        'native_attempts_per_stage': 1, 'automatic_retry': False,
        'budgets': {'sfm_seconds': 720, 'sfm_commit_bytes': 2 << 30,
                    'mvs_seconds': 1200, 'mvs_commit_bytes': 8 << 30,
                    'threads': 2, 'output_bytes': 3 << 30},
        'camera_failure_policy': 'Retain sparse results and stop surface; no implicit model substitution.',
        'quality_status': 'NOT_EVALUATED', 'scale_status': 'unknown',
    }
    save(run / 'PLAN.json', plan)
    code_paths = [Path(__file__), ROOT / 'scripts/run_photo_demo.py',
                  ROOT / 'integrations/video_masks/reconstruction_input.py',
                  ROOT / 'src/kozlov_lab/reconstruction/foreground.py',
                  ROOT / 'integrations/mvs_masks/manifest_io.py',
                  ROOT / 'integrations/mask_review/review.py',
                  ROOT / 'integrations/demo_pipeline/surface.py',
                  ROOT / 'integrations/demo_pipeline/supervision.py']
    bindings = {str(p): sha(p) for p in code_paths}
    save(run / 'CODE_BINDINGS.json', bindings)
    state = {'schema': 'video-mask-reconstruction-v1', 'status': 'RUNNING',
             'started_utc': utc(), 'source_run': str(source), 'baseline_run': str(baseline),
             'stages': {}, 'errors': [], 'viewer_links': {}, 'quality_status': 'NOT_EVALUATED',
             'scale_status': 'unknown', 'new_sam3_executions': 0, 'whole_image_fallback': False}
    halted = False

    def persist():
        save(run / 'SUMMARY.json', state)

    def stage(name, argv, seconds, memory, extra=None):
        nonlocal halted
        if halted:
            raise RuntimeError('native stages stopped after cancellation/resource/reclaim failure')
        if name in state['stages']:
            raise RuntimeError('refusing a second execution of stage ' + name)
        print('STAGE_BEGIN ' + name, flush=True)
        state['stages'][name] = {'status': 'RUNNING', 'started_utc': utc(), 'argv': [str(a) for a in argv]}
        persist()
        entry = _invoke(ROOT, run, name, [str(a) for a in argv], seconds=seconds,
                        memory=memory, additional_roots=extra or [])
        code = entry_code(entry)
        state['stages'][name].update(status='COMPLETED' if code in (0, 1) else 'FAILED',
                                    exit_code=code, finished_utc=utc(),
                                    reclaim_confirmed=entry.get('reclaim_confirmed'),
                                    entry=str(run / 'logs' / name / 'ENTRY.json'))
        if (entry.get('reclaim_confirmed') is not True or entry.get('status') != 'completed'
                or entry.get('cancellation_requested') or code in (5, 130)
                or entry.get('termination_reason') in ('job_memory_limit', 'memory_limit_exceeded')):
            halted = True
        persist()
        print(f'STAGE_END {name} rc={code}', flush=True)
        return code

    manifest = None
    try:
        prepared = prepare(source, run / 'input_masks')
        state['input_preparation'] = prepared
        manifest = Path(prepared['manifest_path']).resolve(strict=True)
        io_run = Path(prepared['io_run']).resolve(strict=True)
        dataset = Path(prepared['dataset_manifest']).resolve(strict=True)
        doc = read(manifest)
        validate_derivation(doc, manifest.parent)
        if not 2 <= len(doc['frames']) <= 96:
            raise ValueError('source sampled input count exceeds cad-object-v1 profile')
        state.update(input_manifest=str(manifest), io_run=str(io_run), input_count=len(doc['frames']))
        persist()
        code = stage('sfm', [python, '-B', '-X', 'utf8', ROOT / 'scripts/reconstruct_object.py',
            '--io-run', io_run, '--output', sfm_output, '--profile', 'cad-object-v1',
            '--object-mask-manifest', manifest, '--point-selection', 'native'], 720, 2 << 30, [sfm_output])
        candidates = list(sfm_output.glob('*/run.json'))
        if len(candidates) != 1:
            raise RuntimeError('SfM did not produce exactly one run record')
        sfm_file = candidates[0]
        sfm = read(sfm_file)
        state.update(sfm_run=str(sfm_file.parent), sfm_status=sfm.get('status'),
                     sfm_counts=sfm.get('counts'), foreground=sfm.get('foreground'))
        persist()
        if code not in (0, 1) or halted or sfm.get('status') not in ('SUCCESS', 'PARTIAL'):
            raise RuntimeError('no usable SfM model; retained its actual failure and diagnostics')
        selected, selection = choose_model(sfm['models'])
        state['model_selection'] = selection
        sparse = make_viewer(sfm_file.parent, run)
        state['sparse_viewer'] = sparse
        if sparse.get('status') == 'SUCCESS':
            state['viewer_links']['sparse'] = str(run / sparse['viewer_path'])
        model = (sfm_file.parent / selected['binary_path']).resolve(strict=True)
        domain_file = run / 'CAMERA_DOMAIN.json'
        code = stage('camera_domain_original', [python, '-B', '-X', 'utf8',
            ROOT / 'scripts/audit_camera_domain.py', '--model', model, '--output', domain_file], 90, 1 << 30)
        domain = read(domain_file) if domain_file.is_file() else {}
        state['camera_domain'] = domain
        code_geometry = stage('view_geometry', [python, '-B', '-X', 'utf8',
            ROOT / 'scripts/audit_view_geometry.py', '--sfm-run', sfm_file.parent,
            '--model-id', str(selected['model_id']), '--output', run / 'view_geometry'], 90, 1 << 30)
        if code_geometry == 0 and (run / 'view_geometry' / 'report.html').is_file():
            state['viewer_links']['camera_geometry'] = str(run / 'view_geometry' / 'report.html')
        if code != 0 or domain.get('status') != 'CAMERA_DOMAIN_VALID' or domain.get('source_unchanged') is not True:
            raise RuntimeError('camera domain rejected; surface not started')
        if halted:
            raise RuntimeError('stopped after geometric diagnostic resource/cancellation failure')
        geometry = run / 'geometry'
        result = complete_surface(sfm_run=sfm_file.parent, io_run=io_run, dataset_manifest=dataset,
            mask_manifests=[manifest], run=geometry, root=ROOT, python=python, tool_root=tools,
            blender_exe=blender, model_id=str(selected['model_id']), stage=stage)
        state.update(surface_run=str(geometry / 'surface_pipeline'), geometry=result)
        _view_surface(run, state)
        state['status'] = ('SURFACE_GENERATED_NOT_ACCEPTED' if result.get('status') == 'SUCCESS'
                           else 'PARTIAL')
        if result.get('error'):
            state['errors'].append(result['error'])
    except Exception as exc:
        state.update(status='PARTIAL' if state.get('sfm_run') else 'FAILED')
        state['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        state['finished_utc'] = utc()
        state['code_drift'] = _drift(bindings)
        if manifest is not None:
            try:
                validate_derivation(read(manifest), manifest.parent)
                state['input_recheck'] = 'PASS'
            except Exception as exc:
                state['input_recheck'] = 'FAILED'
                state['errors'].append('input recheck: ' + str(exc))
        if state['code_drift'] or state.get('input_recheck') == 'FAILED':
            state['status'] = 'INPUT_DRIFT_REFUSED'
        persist()
        try:
            render(run)
        except Exception as exc:
            state.setdefault('diagnostic_errors', []).append('report: ' + str(exc))
            persist()
    return read(run / 'SUMMARY.json')


def continue_surface(source_run, output):
    """Reuse bound SfM/camera inputs; retry no native phase in the old run.

    A new surface attempt has its own preparation, budgets, logs and output.
    Segmentation and SfM are not executed again. Old failed attempts stay intact.
    """
    import copy
    from integrations.video_masks.reconstruction_input import validate_derivation
    from integrations.demo_pipeline.surface import complete_surface, choose_model
    from integrations.blender_export.export import find_blender
    source_run = Path(source_run).resolve(strict=True)
    run = Path(output).resolve()
    if not run.is_relative_to(ROOT / 'outputs') or run == ROOT / 'outputs' or run.exists():
        raise ValueError('continuation output must be a new child of repository outputs')
    source_summary = source_run / 'SUMMARY.json'
    parent = read(source_summary)
    manifest = Path(parent['input_manifest']).resolve(strict=True)
    validate_derivation(read(manifest), manifest.parent)
    sfm = Path(parent['sfm_run']).resolve(strict=True)
    sfm_doc = read(sfm / 'run.json')
    chosen, _ = choose_model(sfm_doc['models'], parent['model_selection']['selected_model_id'])
    domain_path = source_run / 'CAMERA_DOMAIN.json'
    domain = read(domain_path)
    binary = (sfm / chosen['binary_path']).resolve(strict=True)
    if (domain.get('status') != 'CAMERA_DOMAIN_VALID' or domain.get('source_unchanged') is not True
            or Path(domain['source_model']).resolve() != binary):
        raise ValueError('no accepted numerical camera-domain check for this component')
    if sfm_doc.get('foreground', {}).get('mask_consumption_verified') is not True:
        raise ValueError('foreground feature consumption was not verified')
    refs = {str(source_summary): sha(source_summary), str(manifest): sha(manifest),
            str(sfm / 'run.json'): sha(sfm / 'run.json'), str(domain_path): sha(domain_path)}
    for name, ref in domain['source_bindings_after'].items():
        path = binary / name
        if path.stat().st_size != ref['bytes'] or sha(path) != ref['sha256']:
            raise ValueError('native component has changed since camera audit')
        refs[str(path)] = ref['sha256']
    prepared = read(manifest.parent / 'RESULT.json')
    io_run = Path(prepared['io_run']).resolve(strict=True)
    dataset = Path(prepared['dataset_manifest']).resolve(strict=True)
    python = (ROOT / LIGHT_REL).resolve(strict=True)
    tools = (ROOT / MVS_REL).resolve(strict=True)
    blender = find_blender()
    run.mkdir(parents=True, exist_ok=False)
    save(run / 'SOURCE_BINDINGS.json', refs)
    codes = [Path(__file__), ROOT / 'scripts/run_photo_demo.py', ROOT / 'scripts/prepare_mvs_masks.py',
             ROOT / 'integrations/video_masks/reconstruction_input.py',
             ROOT / 'integrations/demo_pipeline/surface.py', ROOT / 'integrations/mvs_masks/manifest_io.py',
             ROOT / 'integrations/mask_review/review.py']
    code_hashes = {str(p): sha(p) for p in codes}
    save(run / 'CODE_BINDINGS.json', code_hashes)
    save(run / 'PLAN.json', {'schema': 'video-mask-surface-continuation-v1',
        'source_run': str(source_run), 'source_bindings': refs,
        'segmentation_executions': 0, 'sfm_executions': 0,
        'surface_attempts': 1, 'automatic_retry': False, 'threads': 2,
        'mvs_total_seconds': 1200, 'mvs_commit_bytes': 8 << 30,
        'quality_status': 'NOT_EVALUATED', 'scale_status': 'unknown'})
    state = {k: copy.deepcopy(parent[k]) for k in (
        'source_run', 'baseline_run', 'input_manifest', 'io_run', 'input_count',
        'sfm_run', 'sfm_status', 'sfm_counts', 'foreground', 'model_selection',
        'camera_domain', 'viewer_links') if k in parent}
    state.update(schema='video-mask-reconstruction-v1', status='RUNNING', stages={}, errors=[],
        started_utc=utc(), previous_attempt=str(source_run), previous_errors=parent.get('errors', []),
        reused_stages={k: v for k, v in parent.get('stages', {}).items()
                       if k in ('sfm', 'camera_domain_original', 'view_geometry')},
        quality_status='NOT_EVALUATED', scale_status='unknown', new_sam3_executions=0,
        new_sfm_executions=0, input_recheck='PASS')
    halted = False
    def persist():
        save(run / 'SUMMARY.json', state)
    def stage(name, argv, seconds, memory, extra=None):
        nonlocal halted
        if halted or name in state['stages']:
            raise RuntimeError('stage is stopped or already attempted: ' + name)
        state['stages'][name] = dict(status='RUNNING', started_utc=utc(), argv=[str(x) for x in argv])
        persist();print('STAGE_BEGIN ' + name, flush=True)
        entry = _invoke(ROOT, run, name, [str(x) for x in argv], seconds=seconds,
                        memory=memory, additional_roots=extra or [])
        code = entry_code(entry)
        state['stages'][name].update(status='COMPLETED' if code == 0 else 'FAILED', exit_code=code,
            finished_utc=utc(), reclaim_confirmed=entry.get('reclaim_confirmed'),
            entry=str(run / 'logs' / name / 'ENTRY.json'))
        halted = (entry.get('status') != 'completed' or entry.get('reclaim_confirmed') is not True
                  or entry.get('cancellation_requested') or code in (5, 130))
        persist();print(f'STAGE_END {name} rc={code}', flush=True)
        return code
    persist()
    try:
        result = complete_surface(sfm_run=sfm, io_run=io_run, dataset_manifest=dataset,
            mask_manifests=[manifest], run=run / 'geometry', root=ROOT, python=python,
            tool_root=tools, blender_exe=blender, model_id=str(chosen['model_id']), stage=stage)
        state.update(geometry=result, surface_run=str(run / 'geometry' / 'surface_pipeline'))
        state['status'] = 'SURFACE_GENERATED_NOT_ACCEPTED' if result['status'] == 'SUCCESS' else 'PARTIAL'
        if result.get('error'):
            state['errors'].append(result['error'])
        _view_surface(run, state)
    except Exception as exc:
        state.update(status='PARTIAL')
        state['errors'].append(f'{type(exc).__name__}: {exc}')
    finally:
        state['source_drift'] = _drift(refs)
        state['code_drift'] = _drift(code_hashes)
        try:
            validate_derivation(read(manifest), manifest.parent)
        except Exception as exc:
            state['input_recheck'] = 'FAILED'
            state['errors'].append('input recheck: ' + str(exc))
        if state['source_drift'] or state['code_drift'] or state['input_recheck'] != 'PASS':
            state['status'] = 'INPUT_DRIFT_REFUSED'
        state['finished_utc'] = utc();persist()
        try:
            render(run)
        except Exception as exc:
            state.setdefault('diagnostic_errors', []).append('report: ' + str(exc));persist()
    return read(run / 'SUMMARY.json')
