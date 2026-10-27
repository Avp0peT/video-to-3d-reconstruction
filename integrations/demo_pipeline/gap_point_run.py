"""Bound, reuse and compare actual images with short-gap candidate recovery.

The old continuous run and implementation are read-only controls. No predicted
coordinates become observations, no missing SAM result is filled, no thresholds
in the existing native geometric consumer are changed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import sys
import time

from .balanced_point_run import read, save, binding, sha, check_bindings, source_inputs

ROOT = Path(__file__).resolve().parents[2]
ARM = 'gap_tolerant'


def prepare(control: Path) -> tuple[dict, list]:
    """Bind decoded images and verify source sample pixels before real tracking."""
    from scripts.prepare_video import inspect_image
    prior = read(control/'INPUTS.json')
    plan = read(control/'PLAN.json')
    if read(control/'RESULT.json')['status'] != 'COMPARISON_COMPLETE':
        raise ValueError('completed continuous control required')
    drift = check_bindings(prior['bindings']) + check_bindings(plan['implementation'])
    if drift:
        raise ValueError('control drift: '+str(drift))
    inputs = source_inputs(Path(prior['pipeline']))
    if inputs != prior:
        raise ValueError('current source differs from control inputs')
    images = control/'decoded'
    paths = sorted(images.glob('frame_*.png'))
    if len(paths) != inputs['decoded_count'] or not 3 <= len(paths) <= 2000:
        raise ValueError('bounded complete decoded video required')
    for i, path in enumerate(paths):
        if path.name != f'frame_{i:06d}.png':
            raise ValueError('decode image sequence differs')
    for f in inputs['frames']:
        info = inspect_image(images/f"frame_{f['decode_index']:06d}.png")
        if info['decoded_rgb_sha256'] != f['decoded_rgb_sha256']:
            raise ValueError('control sampled pixels differ')
    refs = [control/n for n in ('INPUTS.json','PLAN.json','RESULT.json','DECODE.json','SUMMARY.json',
            'balanced/TRACKS.json','balanced/native/PLAN.json','balanced/native/RESULT.json')]
    for m in read(control/'balanced/native/RESULT.json').get('models',[]):
        path = Path(m['binary_path'])
        if not path.is_absolute():
            path = control/'balanced/native'/path
        refs.extend(sorted(path.glob('*.bin')))
    inputs['decoded_directory'] = str(images)
    inputs['bindings'] += [binding(p) for p in refs+paths]
    return inputs, plan['implementation']


def sample_population(tracker, frame: dict, gray, cv, profile, ever_recovered: set,
                      recovered_now: set) -> tuple[dict | None, dict]:
    """Only read post-mask actual active locations; pending entries stay absent."""
    import numpy as np
    from PIL import Image
    from .balanced_object_tracks import seed_balanced, coverage
    if frame['original_mask_status'] != 'OK':
        event = tracker.clear('sample_without_committed_mask')
        return None, dict(kind='sample_without_committed_mask',ended_ids=event['ids'])
    with Image.open(frame['mask_path']) as im:
        mask = np.asarray(im).copy()
    if mask.shape != gray.shape:
        raise ValueError('sample mask dimensions differ')
    filtered = tracker.filter_active(mask, quotas=True)
    continued = len(tracker.ids)
    born, _ = seed_balanced(gray,mask,tracker.xy,cv,profile)
    tracker.seed(born,frame['decode_index'])
    observations = [dict(track_id=int(tid),xy=xy.tolist(),measured=True,
                         recovered=int(tid) in ever_recovered,
                         recovered_this_decode=int(tid) in recovered_now)
                    for tid,xy in zip(tracker.ids,tracker.xy)]
    return {**frame,'observations':observations}, dict(input_index=frame['input_index'],
        decode_index=frame['decode_index'],continued=continued,born=len(born),candidates=len(observations),
        recovered_history_observations=sum(o['recovered'] for o in observations),
        recovered_this_decode=sum(o['recovered_this_decode'] for o in observations),
        coverage=coverage(mask,tracker.xy),filter=filtered)


def track(output: Path) -> dict:
    import numpy as np
    from PIL import Image
    from .temporal_review import load_cv
    from .balanced_object_tracks import Profile, coverage
    from .gap_tolerant_tracks import GapTracker, RecoveryProfile
    inputs = read(output/'INPUTS.json')
    profile, recovery = Profile(), RecoveryProfile()
    references = []
    cv = load_cv(references)
    if cv is None:
        raise RuntimeError('verified OpenCV missing')
    cv.setNumThreads(2)
    save(output/'CV_BINDINGS.json',references)
    tracker = GapTracker(cv,profile,recovery)
    samples = {f['decode_index']:f for f in inputs['frames']}
    frames, statistics = [], []
    rejections, kinds = Counter(), Counter()
    ever_recovered = set()
    recovery_events = 0
    started = time.monotonic()
    with (output/'recovery_events.jsonl').open('x',encoding='utf-8') as event_log, (output/'steps.jsonl').open('x',encoding='utf-8') as step_log:
        for di in range(inputs['decoded_count']):
            if time.monotonic()-started > 360:
                raise TimeoutError('bounded tracking elapsed 360s')
            path = Path(inputs['decoded_directory'])/f'frame_{di:06d}.png'
            with Image.open(path) as im:
                gray = cv.cvtColor(np.asarray(im.convert('RGB')),cv.COLOR_RGB2GRAY)
            before = len(tracker.ids)
            moved = tracker.update(gray,di)
            rejections.update(moved['rejections'])
            recovered = set(map(int,moved['recovered_ids']))
            ever_recovered.update(recovered)
            recovery_events += len(recovered)
            events = list(moved['events'])
            f = samples.get(di)
            sampled_stats = None
            if f:
                row, sampled_stats = sample_population(tracker,f,gray,cv,profile,ever_recovered,recovered)
                if row is not None:
                    frames.append(row)
                    statistics.append(sampled_stats)
                    events.append({'kind':'sample_filter',**sampled_stats['filter']})
                else:
                    events.append(sampled_stats)
                print(f"sample {f['input_index']}: measured={len(tracker.ids)} pending={len(tracker.pending)} recovered_step={len(recovered)}",flush=True)
            for event in events:
                event = {**event,'decode_index':di}
                kinds[event['kind']] += 1
                event_log.write(json.dumps(event,allow_nan=False)+'\n')
            event_log.flush()
            step_log.write(json.dumps(dict(decode_index=di,started=before,active=len(tracker.ids),
                pending=len(tracker.pending),recovered=len(recovered),rejections=moved['rejections'],
                mask_status=f['original_mask_status'] if f else 'UNKNOWN_NOT_SEGMENTED',sample=sampled_stats),allow_nan=False)+'\n')
            step_log.flush()
    frequency = Counter(o['track_id'] for f in frames for o in f['observations'])
    for f,s in zip(frames,statistics):
        with Image.open(f['mask_path']) as im:
            mask = np.asarray(im)
        supported = [o['xy'] for o in f['observations'] if frequency[o['track_id']]>=3]
        s.update(three_sample_candidates=len(supported),three_sample_coverage=coverage(mask,supported))
    destination = output/ARM
    destination.mkdir()
    document = dict(schema='balanced-object-observations-v1',arm='balanced',tracking_variant='gap_tolerant_v1',
        frames=frames,profile=asdict(profile),recovery_profile=asdict(recovery),statistics=statistics,
        source_bindings=inputs['bindings'],coordinates='OpenCV center zero; native consumer adds 0.5 once',
        tracking_identity='Candidate temporal identities including image-validated recovery; not ground truth.',
        recovered_label='This ID has had a measured recovery at or before this observation; not necessarily in the current decoded frame.',
        failures=dict(rejections),new_masks=0,predicted_observations=0,
        mask_gap_policy='Sample NO_MATCH clears; unknown intermediates are not SAM foreground observations.')
    save(destination/'TRACKS.json',document)
    drift = check_bindings(inputs['bindings'])
    result = dict(status='TRACKING_COMPLETE' if not drift else 'SOURCE_DRIFT',seconds=time.monotonic()-started,
        decoded_frames=inputs['decoded_count'],sampled_frames=len(frames),births=tracker.next_id,
        observations=sum(len(f['observations']) for f in frames),tracks_with_three_samples=sum(v>=3 for v in frequency.values()),
        recovery_events=recovery_events,recovered_unique_ids=len(ever_recovered),event_counts=dict(kinds),
        rejection_counts=dict(rejections),statistics=statistics,source_drift=drift,predicted_observations=0)
    save(output/'TRACKING_RESULT.json',result)
    if drift:
        raise ValueError('source drift')
    return result


def launch(output: Path, action: str, seconds: int) -> dict:
    from .supervision import supervise
    directory = output/'processes'/action
    directory.mkdir(parents=True)
    tmp = directory/'tmp'
    tmp.mkdir()
    script = ROOT/'scripts/recover_object_tracks.py'
    command = [sys.executable,'-B','-X','utf8',str(script),action,'--output',str(output)]
    env = {k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMP=str(tmp),TEMP=str(tmp))
    result = supervise(command,cwd=ROOT,env_overrides=env,log_path=directory/'supervision.jsonl',
        stdout_path=directory/'stdout.txt',stderr_path=directory/'stderr.txt',wall_clock_seconds=seconds,
        job_memory_limit_bytes=8<<30,run_dir=output,run_dir_threshold_bytes=1<<30,poll_seconds=.5,
        reclamation_seconds=15,label=action,script_sha256=sha(script))
    save(directory/'ENTRY.json',result)
    return result


def execute(control: Path, output: Path) -> int:
    if output.exists() or output.is_relative_to(control):
        raise ValueError('fresh independent output required')
    inputs, old_code = prepare(control)
    from .balanced_object_tracks import Profile
    from .gap_tolerant_tracks import RecoveryProfile
    names = ('scripts/recover_object_tracks.py','integrations/demo_pipeline/gap_point_run.py',
             'integrations/demo_pipeline/gap_tolerant_tracks.py','integrations/demo_pipeline/gap_point_report.py',
             'integrations/demo_pipeline/balanced_point_report.py','integrations/demo_pipeline/balanced_native.py',
             'integrations/demo_pipeline/balanced_object_tracks.py','integrations/demo_pipeline/balanced_point_run.py')
    implementation = [binding(ROOT/n) for n in names]
    output.mkdir(parents=True)
    save(output/'INPUTS.json',inputs)
    save(output/'PLAN.json',dict(control_run=str(control),profile=asdict(Profile()),recovery_profile=asdict(RecoveryProfile()),
        implementation=implementation,control_implementation=old_code,segmentation_calls=0,retries=0,
        native_policy='Existing balanced_native byte-identical; arm identifies unchanged balanced seed detector.',
        independent_variable='Bounded measured recovery and temporary missing state; births/quotas/native policy unchanged.',
        recovered_label='Recovered ID history at each actual observation; current-step status stored separately.',
        budget=dict(tracking_seconds=390,native_seconds=240,job_commit_bytes=8<<30,output_bytes=1<<30)))
    result = dict(status='RUNNING',stages={},geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown')
    save(output/'RESULT.json',result)
    try:
        for action,seconds in [('track',390),('native',240)]:
            entry = launch(output,action,seconds)
            result['stages'][action] = entry
            save(output/'RESULT.json',result)
            if entry.get('root_exit_code')!=0 or not entry.get('reclaim_confirmed'):
                raise RuntimeError('stage failed: '+action)
        result['status'] = 'COMPARISON_COMPLETE'
    except Exception as e:
        result.update(status='FAILED',error=type(e).__name__+': '+str(e))
    finally:
        result['input_drift'] = check_bindings(inputs['bindings'])
        result['implementation_drift'] = check_bindings(implementation+old_code)
        if result['input_drift'] or result['implementation_drift']:
            result['status']='FAILED'
        save(output/'RESULT.json',result)
    if (output/'TRACKING_RESULT.json').is_file():
        try:
            from .gap_point_report import build
            build(output)
        except Exception as e:
            save(output/'REPORT_FAILURE.json',dict(error=type(e).__name__+': '+str(e)))
            return 1
    return 0 if result['status']=='COMPARISON_COMPLETE' else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','track','native','report'])
    parser.add_argument('--control',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.action=='run':
        if args.control is None:
            parser.error('--control required')
        return execute(args.control.resolve(strict=True),output)
    if args.action=='track':
        track(output)
    elif args.action=='native':
        from .balanced_native import run
        result = run(output/ARM/'TRACKS.json',output/ARM/'native')
        print(json.dumps({k:result[k] for k in ('status','input_count','unique_registered_count')}))
    else:
        from .gap_point_report import build
        build(output)
    return 0
