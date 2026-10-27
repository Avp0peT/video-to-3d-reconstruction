"""Frozen whole-video patch tracks and independent SIFT supplementation.

No segmentation inference, no mutation of the baseline, no automatic parameter
retry. All substantive stages are supervised; report-only never runs geometry.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from integrations.demo_pipeline.pipeline import ROOT, LIGHT_REL, _invoke, entry_code, read, save, sha, utc


def resolve_source_geometry(source_run, source_geometry=None, *, required=True):
    """Select one contained geometry directory without chronological guessing."""
    source = Path(source_run).resolve(strict=True)
    if source_geometry is None:
        candidates = [name for name in ('geometry', 'geometry02')
                      if (source / name / 'RESULT.json').is_file()]
        if len(candidates) > 1:
            raise ValueError('ambiguous source geometry; explicitly select a geometry directory')
        if not candidates:
            if required:
                raise ValueError('no source geometry RESULT.json found in geometry or geometry02')
            return None
        source_geometry = candidates[0]
    selected = Path(source_geometry)
    if '..' in selected.parts:
        raise ValueError('source geometry path must not contain ..')
    resolved = (source / selected).resolve(strict=True)
    if resolved == source or not resolved.is_relative_to(source) or not resolved.is_dir():
        raise ValueError('source geometry must be a directory inside source run')
    if not (resolved / 'RESULT.json').is_file():
        raise ValueError('source geometry directory must contain RESULT.json')
    return resolved.relative_to(source).as_posix()


def prepare(source_mask_run, baseline_sfm, baseline_report, output):
    from .reconstruction_input import _collect
    source, io, io_doc, old, selected, refs = _collect(source_mask_run)
    if not 2 <= len(selected) <= 64:
        raise ValueError('region experiment supports 2..64 canonical endpoint images')
    sfm = Path(baseline_sfm).resolve(strict=True)
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError('fresh experiment directory required')
    manifest = read(source / 'MANIFEST.json')
    decode_file = Path(read(source / 'START_BINDINGS.json')['source_run']) / 'DECODE.json'
    decoded = {f['decode_index']: f for f in read(decode_file)['frames']}
    endpoints = {f['decode_index']: f for f in selected}
    timeline = []
    for row in manifest['rows']:
        n = row['decode_index']
        if row.get('placeholder') is not False or n not in decoded:
            raise ValueError('incomplete timeline or mask placeholder')
        frame = source / 'frames' / f'{n:06d}.jpg'
        expected = decoded[n]['sha256']
        kind = 'decoded_jpeg'
        if n in endpoints:
            frame = Path(endpoints[n]['source_image'])
            expected = endpoints[n]['source_image_sha256']
            kind = 'original_canonical_png'
        mask = Path(row['path'])
        if not mask.resolve().is_relative_to((source / 'final_masks').resolve()):
            raise ValueError('mask outside producer final directory')
        if sha(frame) != expected or sha(mask) != row['sha256']:
            raise ValueError('timeline input drift')
        refs[str(frame)] = expected
        refs[str(mask)] = row['sha256']
        item = dict(decode_index=n, image_path=str(frame), image_sha256=expected,
                    mask_path=str(mask), mask_sha256=row['sha256'], width=selected[0]['width'],
                    height=selected[0]['height'], image_encoding_source=kind)
        if n in endpoints:
            item.update(input_index=endpoints[n]['input_index'], frame_id=endpoints[n]['frame_id'])
        timeline.append(item)
    if [r['decode_index'] for r in timeline] != list(range(len(timeline))):
        raise ValueError('this version requires complete consecutive decoded timeline')
    mapped = [__import__('json').loads(s) for s in (sfm / 'frames_map.jsonl').read_text(encoding='utf8').splitlines() if s]
    by_index = {r['input_index']: r for r in mapped}
    if len(mapped) != len(selected) or set(by_index) != {r['input_index'] for r in selected}:
        raise ValueError('SIFT and region endpoint identity sets differ')
    for row in selected:
        if by_index[row['input_index']]['canonical_image_sha256'] != row['source_image_sha256']:
            raise ValueError('SIFT/region canonical input bytes differ')
    for name in ('run.json', 'frames_map.jsonl', 'database.db'):
        refs[str(sfm / name)] = sha(sfm / name)
    refs[str(decode_file)] = sha(decode_file)
    from .region_tracks import PROFILE
    from .density_sampling import default_policy
    plan = dict(schema='glove-region-plan-v1', created_utc=utc(), source_mask_run=str(source),
                object_scope='object-agnostic mask-bound region candidates; legacy glove schema retained for compatibility',
                baseline_sfm=str(sfm), baseline_report=str(Path(baseline_report).resolve(strict=True)),
                io_run=str(io), timeline=timeline, endpoint_count=len(selected),
                profile=PROFILE, density_policy=default_policy(), source_bindings=refs,
                experiment='SIFT raw correspondences plus continuous mask-bound patch tracks',
                changed_factors=['region correspondence producer', 'intermediate video observations'],
                same_factors=[f'{len(selected)} original canonical keyframes', 'union masks', 'native camera and mapper settings'],
                notes=['Intermediate JPEGs are not pixel-identical to canonical PNGs. Endpoint tracking uses original PNG bytes.',
                       'No track gap interpolation or geometric identity ground truth.',
                       'Regional tracking is a classical patch method, not LoFTR or TAPNext inference.'],
                budgets=dict(tracking_seconds=600, geometry_seconds=600, job_commit_bytes=4 << 30),
                quality_status='NOT_EVALUATED', scale_status='unknown')
    out.mkdir(parents=True)
    save(out / 'PLAN.json', plan)
    save(out / 'SUMMARY.json', dict(status='PREPARED', stages={}, quality_status='NOT_EVALUATED', scale_status='unknown'))
    return plan


def drift(refs):
    errors = []
    for path, expected in refs.items():
        if not Path(path).is_file() or sha(Path(path)) != expected:
            errors.append(path)
    return errors


def execute(run):
    run = Path(run).resolve(strict=True)
    plan = read(run / 'PLAN.json')
    state = read(run / 'SUMMARY.json')
    if state['status'] != 'PREPARED' or (run / 'logs').exists():
        raise ValueError('experiment already attempted; use report or a separately planned experiment')
    if drift(plan['source_bindings']):
        raise ValueError('input drift before execution')
    code_paths = ['scripts/run_mask_region_experiment.py',
                  'integrations/video_masks/region_experiment.py',
                  'integrations/video_masks/region_tracks.py',
                  'integrations/video_masks/point_density.py',
                  'integrations/video_masks/density_sampling.py',
                  'integrations/video_masks/region_geometry.py',
                  'integrations/video_masks/region_comparison.py',
                  'integrations/video_masks/region_report.py']
    bindings = {str(ROOT / p): sha(ROOT / p) for p in code_paths}
    save(run / 'CODE_BINDINGS.json', bindings)
    python = str(ROOT / LIGHT_REL)
    driver = str(ROOT / 'scripts/run_mask_region_experiment.py')
    def stage(label, argv, seconds):
        state['stages'][label] = dict(status='RUNNING', started_utc=utc(), argv=argv)
        save(run / 'SUMMARY.json', state)
        print('STAGE_BEGIN', label, flush=True)
        entry = _invoke(ROOT, run, label, argv, seconds=seconds, memory=4 << 30)
        code = entry_code(entry)
        state['stages'][label].update(status='COMPLETED' if code == 0 else 'FAILED',
                                      exit_code=code, finished_utc=utc(),
                                      reclaim_confirmed=entry.get('reclaim_confirmed'))
        save(run / 'SUMMARY.json', state)
        print('STAGE_END', label, code, flush=True)
        return code
    try:
        state['status'] = 'RUNNING'
        code = stage('tracking', [python, '-B', '-X', 'utf8', driver, 'tracking-worker', '--run', str(run)], 600)
        if code != 0:
            raise RuntimeError('region tracking stage did not complete; no geometry launch')
        code = stage('geometry', [python, '-B', '-X', 'utf8', driver, 'geometry-worker', '--run', str(run)], 600)
        if code != 0 or not (run / 'geometry' / 'RESULT.json').is_file():
            raise RuntimeError('geometry stage did not complete')
        from .region_comparison import run_control, analyse
        from .region_report import render
        run_control(run)
        state['stages']['sift_control'] = dict(status='COMPLETED', finished_utc=utc())
        analyse(run, geometry_name='geometry')
        render(run)
        state['status'] = 'EXPERIMENT_COMPLETE_NOT_ACCEPTED'
    except Exception as exc:
        state.update(status='PARTIAL_DIAGNOSTIC', error=f'{type(exc).__name__}: {exc}')
    finally:
        state.update(input_drift=drift(plan['source_bindings']), code_drift=drift(bindings), finished_utc=utc())
        if state['input_drift'] or state['code_drift']:
            state['status'] = 'DRIFT_REFUSED'
        save(run / 'SUMMARY.json', state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'execute', 'report', 'tracking-worker', 'geometry-worker'])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source-mask-run', type=Path)
    parser.add_argument('--baseline-sfm', type=Path)
    parser.add_argument('--baseline-report', type=Path)
    parser.add_argument('--source-geometry', default=None,
                        help='explicit geometry directory for report; otherwise require a unique candidate')
    args = parser.parse_args(argv)
    if args.action in ('prepare', 'execute'):
        prepare(args.source_mask_run, args.baseline_sfm, args.baseline_report, args.run)
        if args.action == 'prepare':
            return 0
    if args.action in ('run', 'execute'):
        result = execute(args.run)
        return 0 if result['status'] == 'EXPERIMENT_COMPLETE_NOT_ACCEPTED' else 1
    elif args.action == 'report':
        from .region_report import render
        render(args.run, source_geometry=args.source_geometry)
    elif args.action == 'tracking-worker':
        from .region_tracks import run
        run(args.run / 'PLAN.json', args.run / 'tracking')
    else:
        from .region_geometry import run
        plan = read(args.run / 'PLAN.json')
        run(Path(plan['baseline_sfm']), args.run / 'tracking' / 'REGION_TRACKS.json', args.run / 'geometry')
    return 0
