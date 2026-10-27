"""Local photometric pre-normalization experiment with unmodified gap recovery.

Raw images, saved masks, seeding, quotas, sampling and native geometry stay fixed.
This is not a joint brightness/motion optimizer or proof of illumination causality.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import sys
import time

from .balanced_point_run import read, save, binding, sha, check_bindings
from .gap_point_run import sample_population

ROOT = Path(__file__).resolve().parents[2]
ARM = 'photometric'


def prepare(control: Path) -> tuple[dict, list, Path]:
    """Verify the completed gap run and all inherited inputs/code before use."""
    prior = read(control/'INPUTS.json')
    plan = read(control/'PLAN.json')
    if read(control/'RESULT.json')['status'] != 'COMPARISON_COMPLETE':
        raise ValueError('completed gap control required')
    tracks = read(control/'gap_tolerant/TRACKS.json')
    if tracks.get('tracking_variant') != 'gap_tolerant_v1':
        raise ValueError('gap-tolerant control contract required')
    code = plan['implementation'] + plan['control_implementation']
    cv_refs = read(control/'CV_BINDINGS.json')
    drift = check_bindings(prior['bindings'] + code + cv_refs)
    if drift:
        raise ValueError('control drift: '+str(drift))
    paths = sorted(Path(prior['decoded_directory']).glob('frame_*.png'))
    if len(paths) != prior['decoded_count'] or not 3 <= len(paths) <= 2000:
        raise ValueError('complete bounded decoded video required')
    for i,p in enumerate(paths):
        if p.name != f'frame_{i:06d}.png':
            raise ValueError('nonconsecutive decoded sequence')
    # The original run already verified sampled decoded RGB against canonical RGB.
    # All original decoded bytes and source declarations are rehashed above.
    refs = [control/n for n in ('INPUTS.json','PLAN.json','RESULT.json','SUMMARY.json',
        'TRACKING_RESULT.json','CV_BINDINGS.json','gap_tolerant/TRACKS.json',
        'gap_tolerant/native/PLAN.json','gap_tolerant/native/RESULT.json')]
    for model in read(control/'gap_tolerant/native/RESULT.json').get('models',[]):
        p = Path(model['binary_path'])
        if not p.is_absolute():
            p = control/'gap_tolerant/native'/p
        refs.extend(sorted(p.glob('*.bin')))
    bound = {r['path']:r for r in prior['bindings'] + cv_refs + [binding(p) for p in refs]}
    prior['bindings'] = list(bound.values())
    prior['expected_cv_bindings'] = cv_refs
    return prior, code, Path(plan['control_run'])


def track(output: Path) -> dict:
    import numpy as np
    from PIL import Image
    from .temporal_review import load_cv
    from .balanced_object_tracks import Profile, coverage, step_tracks
    from .gap_tolerant_tracks import RecoveryProfile
    from .photometric_tracks import PhotometricProfile, PhotometricGapTracker
    inputs, plan = read(output/'INPUTS.json'), read(output/'PLAN.json')
    profile = Profile(**plan['profile'])
    recovery = RecoveryProfile(**plan['recovery_profile'])
    photo = PhotometricProfile(**plan['photometric_profile'])
    references = []
    cv = load_cv(references)
    if cv is None or references != inputs['expected_cv_bindings']:
        raise RuntimeError('verified control OpenCV identity differs')
    cv.setNumThreads(2)
    save(output/'CV_BINDINGS.json',references)
    tracker = PhotometricGapTracker(cv,profile,recovery,photo)
    samples = {f['decode_index']:f for f in inputs['frames']}
    frames, statistics = [], []
    rejections, kinds, differential = Counter(), Counter(), Counter()
    ever_recovered, ever_photo = set(), set()
    recovery_events = 0
    previous = None
    started = time.monotonic()
    with (output/'recovery_events.jsonl').open('x',encoding='utf-8') as event_log, (output/'steps.jsonl').open('x',encoding='utf-8') as step_log:
        for di in range(inputs['decoded_count']):
            if time.monotonic()-started > 600:
                raise TimeoutError('bounded tracking elapsed 600s')
            with Image.open(Path(inputs['decoded_directory'])/f'frame_{di:06d}.png') as im:
                gray = cv.cvtColor(np.asarray(im.convert('RGB')),cv.COLOR_RGB2GRAY)
            old_ids, old_xy = tracker.ids.copy(), tracker.xy.copy()
            before = len(old_ids)
            # Diagnostic-only old method on exactly the same input positions.
            # It cannot update the live tracker or create observations.
            raw = step_tracks(previous,gray,old_xy,cv,profile) if previous is not None else None
            moved = tracker.update(gray,di)
            previous = gray
            raw_ids = set(map(int,old_ids[raw['accepted']])) if raw is not None else set()
            direct_ids = set(map(int,moved['strict_ids']))
            photo_only = direct_ids - raw_ids
            ever_photo.update(photo_only)
            shared = direct_ids & raw_ids
            differences = []
            if shared:
                new_positions = dict(zip(map(int,tracker.ids),tracker.xy))
                old_positions = dict(zip(map(int,old_ids),raw['xy']))
                differences = [float(np.linalg.norm(new_positions[i]-old_positions[i])) for i in shared]
            d = dict(tested=before,raw_accepted=len(raw_ids),photometric_accepted=len(direct_ids),
                photometric_only=len(photo_only),raw_only=len(raw_ids-direct_ids),common=len(shared),
                common_position_difference_median=float(np.median(differences)) if differences else None,
                common_position_difference_max=max(differences) if differences else None)
            for key in ('tested','raw_accepted','photometric_accepted','photometric_only','raw_only','common'):
                differential[key] += d[key]
            rejections.update(moved['rejections'])
            recovered = set(map(int,moved['recovered_ids']))
            ever_recovered.update(recovered)
            recovery_events += len(recovered)
            events = list(moved['events'])
            f, sampled_stats = samples.get(di), None
            if f:
                row, sampled_stats = sample_population(tracker,f,gray,cv,profile,ever_recovered,recovered)
                if row is not None:
                    for o in row['observations']:
                        o['photometric_history'] = o['track_id'] in ever_photo
                        o['photometric_only_this_decode'] = o['track_id'] in photo_only
                    frames.append(row)
                    statistics.append(sampled_stats)
                    events.append({'kind':'sample_filter',**sampled_stats['filter']})
                else:
                    events.append(sampled_stats)
                print(f"sample {f['input_index']}: measured={len(tracker.ids)} pending={len(tracker.pending)} photo_only_step={len(photo_only)}",flush=True)
            for event in events:
                event = {**event,'decode_index':di}
                kinds[event['kind']] += 1
                event_log.write(json.dumps(event,allow_nan=False)+'\n')
            event_log.flush()
            step_log.write(json.dumps(dict(decode_index=di,started=before,active=len(tracker.ids),
                pending=len(tracker.pending),recovered=len(recovered),rejections=moved['rejections'],
                photometric_differential=d,photometric_only_ids=sorted(photo_only),
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
    document = dict(schema='balanced-object-observations-v1',arm='balanced',tracking_variant='photometric_gap_v1',
        frames=frames,profile=asdict(profile),recovery_profile=asdict(recovery),photometric_profile=asdict(photo),statistics=statistics,
        source_bindings=inputs['bindings'],coordinates='OpenCV center zero; native consumer adds 0.5 once',
        tracking_identity='Candidate image-supported temporal identities; not ground truth.',
        photometric_label='ID history of direct photometric acceptance when raw LK rejected the same input position; not proof of illumination cause.',
        failures=dict(rejections),new_masks=0,predicted_observations=0,
        mask_gap_policy='Sample NO_MATCH clears; unknown intermediates are not SAM foreground observations.')
    save(destination/'TRACKS.json',document)
    drift = check_bindings(inputs['bindings'])
    result = dict(status='TRACKING_COMPLETE' if not drift else 'SOURCE_DRIFT',seconds=time.monotonic()-started,
        decoded_frames=inputs['decoded_count'],sampled_frames=len(frames),births=tracker.next_id,
        observations=sum(len(f['observations']) for f in frames),tracks_with_three_samples=sum(v>=3 for v in frequency.values()),
        recovery_events=recovery_events,recovered_unique_ids=len(ever_recovered),event_counts=dict(kinds),
        photometric_differential=dict(differential),photometric_history_unique_ids=len(ever_photo),
        photometric_history_observations=sum(o['photometric_history'] for f in frames for o in f['observations']),
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
    script = ROOT/'scripts/compensate_object_tracks.py'
    command = [sys.executable,'-B','-X','utf8',str(script),action,'--output',str(output)]
    env = {k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMP=str(tmp),TEMP=str(tmp))
    result = supervise(command,cwd=ROOT,env_overrides=env,log_path=directory/'supervision.jsonl',
        stdout_path=directory/'stdout.txt',stderr_path=directory/'stderr.txt',wall_clock_seconds=seconds,
        job_memory_limit_bytes=8<<30,run_dir=output,run_dir_threshold_bytes=1<<30,poll_seconds=.5,
        reclamation_seconds=15,label=action,script_sha256=sha(script))
    save(directory/'ENTRY.json',result)
    return result


def execute(control: Path, output: Path, offline_evidence=()) -> int:
    if output.exists() or output.is_relative_to(control):
        raise ValueError('fresh independent output required')
    inputs, old_code, seed_control = prepare(control)
    from .balanced_object_tracks import Profile
    from .gap_tolerant_tracks import RecoveryProfile
    from .photometric_tracks import PhotometricProfile
    names = ('scripts/compensate_object_tracks.py','integrations/demo_pipeline/photometric_point_run.py',
             'integrations/demo_pipeline/photometric_tracks.py','integrations/demo_pipeline/photometric_point_report.py',
             'tests/test_photometric_tracks.py','tests/test_photometric_point_run.py')
    offline = [binding(Path(p)) for p in offline_evidence]
    implementation = [binding(ROOT/n) for n in names] + old_code + offline
    output.mkdir(parents=True)
    save(output/'INPUTS.json',inputs)
    save(output/'PLAN.json',dict(control_run=str(control),seed_control_run=str(seed_control),profile=asdict(Profile()),
        recovery_profile=asdict(RecoveryProfile()),photometric_profile=asdict(PhotometricProfile()),
        implementation=implementation,control_implementation=old_code,offline_evidence=offline,segmentation_calls=0,retries=0,
        native_policy='Existing balanced_native unchanged, same options and geometry gates.',
        independent_variable='Local photometric pre-normalization for LK plus raw patch photometric validation. Not joint optimization; not an isolated exposure-causality experiment.',
        diagnostic='Raw LK also runs on identical live seeds per step for measurement only, without updating identities.',
        budget=dict(tracking_seconds=630,native_seconds=240,job_commit_bytes=8<<30,output_bytes=1<<30)))
    result = dict(status='RUNNING',stages={},geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown')
    save(output/'RESULT.json',result)
    try:
        for action,seconds in [('track',630),('native',240)]:
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
        result['implementation_drift'] = check_bindings(implementation)
        if result['input_drift'] or result['implementation_drift']:
            result['status']='FAILED'
        save(output/'RESULT.json',result)
    if (output/'TRACKING_RESULT.json').is_file():
        try:
            from .photometric_point_report import build
            build(output)
        except Exception as e:
            save(output/'REPORT_FAILURE.json',dict(error=type(e).__name__+': '+str(e)))
            return 1
    return 0 if result['status']=='COMPARISON_COMPLETE' else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','track','native','report'])
    parser.add_argument('--control',type=Path)
    parser.add_argument('--offline-evidence',type=Path,action='append',default=[])
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.action=='run':
        if args.control is None:
            parser.error('--control required')
        return execute(args.control.resolve(strict=True),output,args.offline_evidence)
    if args.action=='track':
        track(output)
    elif args.action=='native':
        from .balanced_native import run
        result = run(output/ARM/'TRACKS.json',output/ARM/'native')
        print(json.dumps({k:result[k] for k in ('status','input_count','unique_registered_count')}))
    else:
        from .photometric_point_report import build
        build(output)
    return 0
