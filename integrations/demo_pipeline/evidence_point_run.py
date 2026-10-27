"""Reuse measured observations, compare admission and triangulation separately.

No decoding, segmentation, new tracking or source-observation interpolation.
Each new native policy runs once under the existing Windows Job supervisor.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import sys
from .balanced_point_run import read, save, binding, sha, check_bindings

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('two_sample_candidates','geometry_evidence')


def unique_bindings(items):
    bound = {}
    for item in items:
        key = str(Path(item['path']).resolve())
        if key in bound and bound[key] != item:
            raise ValueError('conflicting binding: '+key)
        bound[key] = item
    return list(bound.values())


def prepare(control: Path):
    prior, old_plan = read(control/'INPUTS.json'), read(control/'PLAN.json')
    if read(control/'RESULT.json').get('status') != 'COMPARISON_COMPLETE':
        raise ValueError('completed photometric control required')
    track_path = control/'photometric/TRACKS.json'
    tracks = read(track_path)
    if tracks.get('tracking_variant') != 'photometric_gap_v1':
        raise ValueError('frozen photometric observations required')
    if tracks.get('predicted_observations') != 0:
        raise ValueError('predicted observations forbidden')
    old_code = unique_bindings(old_plan['implementation'] + old_plan.get('control_implementation',[]))
    sources = unique_bindings(prior['bindings'] + read(control/'CV_BINDINGS.json'))
    drift = check_bindings(sources + old_code)
    if drift:
        raise ValueError('control input/implementation drift: '+str(drift))
    native = control/'photometric/native'
    refs = [control/n for n in ('INPUTS.json','PLAN.json','RESULT.json','SUMMARY.json',
        'TRACKING_RESULT.json','CV_BINDINGS.json','photometric/TRACKS.json',
        'photometric/native/PLAN.json','photometric/native/RESULT.json')]
    for model in read(native/'RESULT.json')['models']:
        binary = Path(model['binary_path']).resolve(strict=True)
        if not binary.is_relative_to(native.resolve()):
            raise ValueError('control model outside native directory')
        refs.extend(p for p in binary.iterdir() if p.is_file())
    sources = unique_bindings(sources + [binding(p) for p in refs])
    frequency = Counter(o['track_id'] for f in tracks['frames'] for o in f['observations'])
    census = dict(sampled_frames=len(tracks['frames']),candidate_ids=len(frequency),
        single_sample_ids=sum(n==1 for n in frequency.values()),
        exactly_two_sample_ids=sum(n==2 for n in frequency.values()),
        at_least_three_sample_ids=sum(n>=3 for n in frequency.values()),
        at_least_two_sample_ids=sum(n>=2 for n in frequency.values()),
        observation_count=sum(frequency.values()),predicted_observations=0,
        note='Distinct saved sample count, not consecutive decoded frames; identities remain candidates.')
    return sources, old_code, track_path, census


def launch(output: Path, action: str, policy=None):
    from .supervision import supervise
    label = policy or action
    directory = output/'processes'/label
    directory.mkdir(parents=True)
    tmp = directory/'tmp'; tmp.mkdir()
    script = ROOT/'scripts/select_geometry_tracks.py'
    argv = [sys.executable,'-B','-X','utf8',str(script),action,'--output',str(output)]
    if policy:
        argv += ['--policy',policy]
    env = {k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMP=str(tmp),TEMP=str(tmp))
    result = supervise(argv,cwd=ROOT,env_overrides=env,log_path=directory/'supervision.jsonl',
        stdout_path=directory/'stdout.txt',stderr_path=directory/'stderr.txt',
        wall_clock_seconds=240 if policy else 60,job_memory_limit_bytes=8<<30,
        run_dir=output,run_dir_threshold_bytes=1<<30,poll_seconds=.5,
        reclamation_seconds=15,label=label,script_sha256=sha(script))
    save(directory/'ENTRY.json',result)
    return result


def execute(control: Path, output: Path, offline_evidence=()):
    if output.exists() or output.is_relative_to(control) or control.is_relative_to(output):
        raise ValueError('fresh disjoint output required')
    sources, old_code, track_path, census = prepare(control)
    names = ('scripts/select_geometry_tracks.py','integrations/demo_pipeline/evidence_point_run.py',
        'integrations/demo_pipeline/evidence_native.py','integrations/demo_pipeline/evidence_point_report.py',
        'integrations/demo_pipeline/balanced_point_run.py','integrations/demo_pipeline/supervision.py',
        'tests/test_evidence_native.py','tests/test_evidence_point_run.py')
    offline = [binding(Path(p)) for p in offline_evidence]
    implementation = unique_bindings([binding(ROOT/n) for n in names] + old_code + offline)
    output.mkdir(parents=True)
    save(output/'INPUTS.json',dict(bindings=sources,source_bindings=sources,
        frames=read(control/'INPUTS.json')['frames'],tracks_path=str(track_path),census=census))
    save(output/'PLAN.json',dict(control_run=str(control),tracks_path=str(track_path),
        source_bindings=sources,implementation=implementation,offline_evidence=offline,
        policies=list(ARMS),census=census,tracking_calls=0,segmentation_calls=0,retries=0,
        changes=dict(two_sample_candidates='minimum measured saved samples 3 -> 2; ignore_two_view_tracks stays true',
            geometry_evidence='same minimum 2; ignore_two_view_tracks becomes false'),
        unchanged='Measured coordinates, track IDs, masks, images, local pair window 3, numeric geometry/camera policies.',
        geometry_evidence='All native points retained; same-fitted-model consistency checks are not held-out accuracy.',
        budget=dict(native_seconds_per_arm=240,audit_seconds=60,job_commit_bytes=8<<30,output_bytes=1<<30)))
    result = dict(status='RUNNING',stages={},geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown')
    save(output/'RESULT.json',result)
    try:
        for action,policy in [('audit-control',None)] + [('native',p) for p in ARMS]:
            label = policy or action
            if check_bindings(sources + implementation):
                raise ValueError('bound source changed before '+label)
            entry = launch(output,action,policy)
            result['stages'][label] = entry
            save(output/'RESULT.json',result)
            if entry.get('root_exit_code') != 0 or not entry.get('reclaim_confirmed'):
                raise RuntimeError('bounded stage failed: '+label)
        result['status'] = 'COMPARISON_COMPLETE'
    except Exception as error:
        result.update(status='FAILED',error=type(error).__name__+': '+str(error))
    finally:
        result['input_drift'] = check_bindings(sources)
        result['implementation_drift'] = check_bindings(implementation)
        if result['input_drift'] or result['implementation_drift']:
            result['status'] = 'FAILED'
        save(output/'RESULT.json',result)
    if result['status'] == 'COMPARISON_COMPLETE':
        from .evidence_point_report import build
        try:
            build(output)
        except Exception as error:
            result.update(status='REPORT_FAILED',error=type(error).__name__+': '+str(error))
            save(output/'RESULT.json',result)
    return 0 if result['status'] == 'COMPARISON_COMPLETE' else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','audit-control','native','report'])
    p.add_argument('--control',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--policy',choices=ARMS)
    p.add_argument('--offline-evidence',type=Path,action='append',default=[])
    args = p.parse_args(argv); output = args.output.resolve()
    if args.action == 'run':
        if args.control is None: p.error('--control required')
        return execute(args.control.resolve(strict=True),output,args.offline_evidence)
    plan = read(output/'PLAN.json')
    if check_bindings(plan['source_bindings'] + plan['implementation']):
        raise ValueError('frozen binding drift before child action')
    if args.action == 'audit-control':
        from .evidence_native import audit_existing
        audit_existing(Path(plan['control_run'])/'photometric/native/RESULT.json',output/'control_geometry_evidence.json')
    elif args.action == 'native':
        if args.policy is None: p.error('--policy required')
        from .evidence_native import run
        run(Path(plan['tracks_path']),output/args.policy/'native',args.policy)
    else:
        from .evidence_point_report import build
        build(output)
    return 0
