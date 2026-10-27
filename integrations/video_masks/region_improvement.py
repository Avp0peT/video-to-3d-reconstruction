"""Bounded reverse-region supplementation; old evidence remains immutable."""
from pathlib import Path
import argparse

from integrations.demo_pipeline.pipeline import ROOT, LIGHT_REL, _invoke, entry_code, read, save, sha, utc
from .region_experiment import drift, resolve_source_geometry


CODE = (
    'scripts/improve_mask_regions.py',
    'integrations/video_masks/region_improvement.py',
    'integrations/video_masks/region_experiment.py',
    'integrations/video_masks/region_reverse.py',
    'integrations/video_masks/region_anchor.py',
    'integrations/video_masks/region_tracks.py',
    'integrations/video_masks/point_density.py',
    'integrations/video_masks/density_sampling.py',
    'integrations/video_masks/region_geometry.py',
    'integrations/video_masks/region_improvement_assessment.py',
    'integrations/video_masks/region_identity_report.py',
    'integrations/demo_pipeline/camera_domain_gate.py',
    'integrations/demo_pipeline/photometric_tracks.py',
    'integrations/demo_pipeline/temporal_review.py',
    'integrations/demo_pipeline/pipeline.py',
    'integrations/demo_pipeline/supervision.py',
    'integrations/viewer/export.py',
)


def prepare(source_run, output, source_geometry=None):
    source = Path(source_run).resolve(strict=True)
    out = Path(output).resolve()
    if out.exists() or out == source or source.is_relative_to(out):
        raise FileExistsError('fresh disjoint output required')
    source_geometry = resolve_source_geometry(source, source_geometry)
    original = read(source / 'PLAN.json')
    geometry = read(source / source_geometry / 'RESULT.json')
    if not geometry.get('models'):
        raise ValueError('source needs a real reconstruction for comparison')
    tracks = read(source / 'tracking' / 'REGION_TRACKS.json')
    if tracks.get('schema') != 'glove-region-tracks-v1':
        raise ValueError('unsupported source tracks')
    refs = dict(original['source_bindings'])
    for ref in tracks['source_bindings']:
        refs[ref['path']] = ref['sha256']
    for name in ('PLAN.json', 'tracking/REGION_TRACKS.json', 'COMPARISON.json',
                 source_geometry+'/PLAN.json', source_geometry+'/RESULT.json',
                 source_geometry+'/database.db'):
        path = source / name
        refs[str(path)] = sha(path)
    for model in geometry['models']:
        for key in ('binary_path', 'text_path', 'ply_path'):
            path = Path(model[key]).resolve(strict=True)
            for item in sorted(path.rglob('*')) if path.is_dir() else [path]:
                if item.is_file():
                    refs[str(item)] = sha(item)
    if drift(refs):
        raise ValueError('source input binding drift')
    code = {str(ROOT / p): sha(ROOT / p) for p in CODE}
    plan = dict(schema='glove-region-improvement-v1', created_utc=utc(),
                object_scope='object-agnostic mask-bound region candidates; legacy glove schema retained for compatibility',
                source_run=str(source), source_geometry=source_geometry,
                source_identity_model=max(geometry['models'], key=lambda m:(m['registered_views'],m['points']))['model_id'],
                source_plan=str(source / 'PLAN.json'),
                forward_tracks=str(source / 'tracking' / 'REGION_TRACKS.json'),
                baseline_sfm=original['baseline_sfm'], source_bindings=refs,
                code_bindings=code,
                changed_factor='additional independently seeded reverse tracks with direct bidirectional endpoint checks',
                preserved=['all original forward IDs and coordinates', 'all original SIFT raw matches',
                           'same original endpoint images', 'native verification and mapper options',
                           'mask/texture/FB thresholds'],
                interpretation=['Reverse IDs are separate candidates, not new unique physical points.',
                                'Anchor UNKNOWN is not a proved mismatch.',
                                'Original forward identities are not upgraded by this experiment.',
                                'Single-run mapper comparison is not a statistical efficacy proof.',
                                'No automatic component merge or hole fill.'],
                budgets=dict(identity_seconds=180, tracking_seconds=600,
                             geometry_seconds=600, assessment_seconds=180, commit_bytes=4 << 30))
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'PLAN.json', plan)
    save(out / 'SUMMARY.json', dict(status='PREPARED', stages={}, geometry_accepted=False,
                                    quality_status='NOT_EVALUATED', scale_status='unknown'))
    return plan


def execute(run):
    run = Path(run).resolve(strict=True)
    plan = read(run / 'PLAN.json')
    state = read(run / 'SUMMARY.json')
    if state['status'] != 'PREPARED' or (run / 'logs').exists():
        raise ValueError('already attempted; fresh planned output required')
    if drift(plan['source_bindings']) or drift(plan['code_bindings']):
        raise ValueError('source or code changed after preparation')
    python = str(ROOT / LIGHT_REL)
    driver = str(ROOT / 'scripts/improve_mask_regions.py')
    state.update(status='RUNNING', started_utc=utc())
    save(run / 'SUMMARY.json', state)
    try:
        stages = [('identity', 180), ('tracking', 600), ('geometry', 600), ('assessment', 180)]
        for label, seconds in stages:
            argv = [python, '-B', '-X', 'utf8', driver, label+'-worker', '--run', str(run)]
            state['stages'][label] = dict(status='RUNNING', argv=argv, started_utc=utc())
            save(run / 'SUMMARY.json', state)
            print('STAGE_BEGIN', label, flush=True)
            entry = _invoke(ROOT, run, label, argv, seconds=seconds, memory=4 << 30)
            code = entry_code(entry)
            state['stages'][label].update(status='COMPLETED' if code == 0 else 'FAILED',
                                          exit_code=code, reclaimed=entry.get('reclaim_confirmed'),
                                          finished_utc=utc())
            save(run / 'SUMMARY.json', state)
            print('STAGE_END', label, code, flush=True)
            if code != 0:
                raise RuntimeError(label+' did not complete; no automatic retry')
        read(run / 'assessment' / 'ASSESSMENT.json')
        state.update(status='EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED',
                     assessment_path=str(run / 'assessment' / 'index.html'))
    except Exception as exc:
        state.update(status='PARTIAL_DIAGNOSTIC', error=f'{type(exc).__name__}: {exc}')
    finally:
        state.update(finished_utc=utc(), input_drift=drift(plan['source_bindings']),
                     code_drift=drift(plan['code_bindings']))
        if state['input_drift'] or state['code_drift']:
            state['status'] = 'DRIFT_REFUSED'
        save(run / 'SUMMARY.json', state)
    return state


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run', 'execute', 'identity-worker',
                                    'tracking-worker', 'geometry-worker', 'assessment-worker'])
    p.add_argument('--source-run', type=Path)
    p.add_argument('--source-geometry', default=None)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args(argv)
    if a.action in ('prepare', 'execute'):
        if a.source_run is None:
            p.error('--source-run required for preparation')
        prepare(a.source_run, a.run, a.source_geometry)
        if a.action == 'prepare':
            return 0
    if a.action in ('run', 'execute'):
        result = execute(a.run)
        return 0 if result['status'] == 'EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED' else 1
    plan = read(a.run / 'PLAN.json')
    if a.action == 'identity-worker':
        from .region_identity_report import render
        render(plan['source_run'], a.run / 'identity_audit', geometry=plan['source_geometry'],
               model_id=plan['source_identity_model'])
    elif a.action == 'tracking-worker':
        from .region_reverse import run
        run(plan['source_plan'], plan['forward_tracks'], a.run / 'tracking')
    elif a.action == 'geometry-worker':
        from .region_geometry import run
        run(plan['baseline_sfm'], a.run / 'tracking' / 'REGION_TRACKS.json', a.run / 'geometry')
    else:
        from .region_improvement_assessment import assess
        assess(plan['source_run'], a.run)
    return 0
