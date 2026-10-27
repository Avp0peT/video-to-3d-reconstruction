"""Bounded, reproducible single-point diagnostic; no new matching or geometry."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import math
import sys
from .balanced_point_run import read,save,binding,check_bindings,sha
from .evidence_point_run import unique_bindings

ROOT=Path(__file__).resolve().parents[2]
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS')


def verify_endpoint(rgb,saved):
    import numpy as np
    from PIL import Image
    with Image.open(saved['image_path']) as im:canonical=np.asarray(im.convert('RGB'))
    height,width=rgb.shape[:2]
    digest=hashlib.sha256(f'{width}x{height}:RGB:'.encode('ascii')+rgb.tobytes()).hexdigest()
    if digest!=saved['decoded_rgb_sha256'] or not np.array_equal(rgb,canonical):
        raise ValueError('raw-to-canonical pixel identity failed')
    return dict(decode_index=saved['decode_index'],decoded_rgb_sha256=digest,equal=True)


def select(source,supplement,seed_index,seed_track,target_index,target_track,count):
    if type(count) is not int or not 2<=count<=128: raise ValueError('2..128 consecutive raw images required')
    inputs=read(source/'INPUTS.json');tracks=read(source/'photometric/TRACKS.json')
    inherited=read(source/'PLAN.json')['implementation']
    tracking_names={'photometric_tracks.py','balanced_object_tracks.py'}
    old_tracking=unique_bindings([r for r in inherited if Path(r['path']).name in tracking_names])
    if {Path(r['path']).name for r in old_tracking}!=tracking_names or check_bindings(old_tracking):
        raise ValueError('original measured-step implementation drift')
    frames={f['input_index']:f for f in inputs['frames']}
    observations={f['input_index']:f for f in tracks['frames']}
    def observation(index,track):
        f=frames[index];t=observations[index]
        if f['frame_id']!=t['frame_id'] or f['decode_index']!=t['decode_index']:
            raise ValueError('saved frame identity mismatch')
        rows=[o for o in t['observations'] if o['track_id']==track and o['measured'] is True]
        if len(rows)!=1: raise ValueError('one saved measured observation required')
        return dict(input_index=index,decode_index=f['decode_index'],frame_id=f['frame_id'],
                    track_id=track,xy=rows[0]['xy'],image_path=f['image_path'],image_sha256=f['image_sha256'],
                    decoded_rgb_sha256=f['decoded_rgb_sha256'])
    seed=observation(seed_index,seed_track);comparison=observation(target_index,target_track)
    comparison_source=read(supplement)
    if comparison_source['tracks_binding']!=binding(source/'photometric/TRACKS.json'):
        raise ValueError('comparison source observation ledger differs')
    pairs=[p for p in comparison_source['pairs'] if p['input_index_a']==seed_index and p['input_index_b']==target_index]
    matches=[m for p in pairs for m in p['matches'] if m['track_id_a']==seed_track and m['track_id_b']==target_track]
    if len(matches)!=1 or matches[0]['evidence']['xy_a']!=seed['xy'] or matches[0]['evidence']['xy_b']!=comparison['xy']:
        raise ValueError('comparison must bind the actual saved cross-ID proposal')
    if not seed['decode_index']<comparison['decode_index']<seed['decode_index']+count:
        raise ValueError('comparison outside consecutive diagnostic interval')
    comparison.update(used_as_algorithm_input=False,geometry_verified=matches[0]['evidence']['geometry_verified'])
    birth=next(f for f in tracks['frames'] if any(o['track_id']==seed_track for o in f['observations']))
    seed['first_saved_observation_input_index']=birth['input_index']
    seed['first_saved_observation_decode_index']=birth['decode_index']
    timeline=Path(inputs['video_root'])/'timeline.json'
    time_rows={f['decode_index']:f for f in read(timeline)['frames']}
    old={str(Path(r['path']).resolve()):r for r in inputs['bindings']}
    directory=Path(inputs['decoded_directory']).resolve(strict=True)
    selected=[];refs=[]
    for index in range(seed['decode_index'],seed['decode_index']+count):
        p=directory/f'frame_{index:06d}.png';actual=binding(p)
        if old.get(str(p))!=actual: raise ValueError('raw image differs from inherited binding: '+str(p))
        t=time_rows[index];tb=t['time_base'];sec=Fraction(t['pts'])*Fraction(tb[0],tb[1])
        selected.append(dict(decode_index=index,path=str(p),sha256=actual['sha256'],bytes=actual['bytes'],
            pts=t['pts'],time_base=tb,time_seconds=float(sec),mask_status='NOT_SEGMENTED_AT_RAW_FRAME'))
        refs.append(actual)
    for p in (source/'INPUTS.json',source/'PLAN.json',source/'photometric/TRACKS.json',source/'recovery_events.jsonl',
              source/'CV_BINDINGS.json',supplement,timeline,directory.parent/'DECODE.json'):
        refs.append(binding(p))
    for item in (seed,comparison):
        p=Path(item['image_path']).resolve(strict=True);actual=binding(p)
        if actual!=old.get(str(p)) or actual['sha256']!=item['image_sha256']:
            raise ValueError('canonical source drift')
        refs.append(actual)
    cvrefs=read(source/'CV_BINDINGS.json')
    if check_bindings(cvrefs):raise ValueError('bound OpenCV runtime drift')
    result=dict(schema='single-point-input-v1',source_run=str(source),source_video=inputs['source_video'],
        seed=seed,comparison=comparison,frames=selected,bindings=unique_bindings(refs+cvrefs+old_tracking),cv_bindings=cvrefs,
        selection_reason='One inspectable cross-ID proposal selected before running, not ground truth or best-of search.',
        normalized_video_assumption='Inherited square-pixel normalization; not a new camera calibration.',
        intermediate_mask_status='UNKNOWN_NOT_SEGMENTED',new_segmentation_calls=0)
    return result


def implementation():
    names=('scripts/trace_single_object_point.py','integrations/demo_pipeline/single_point_run.py',
        'integrations/demo_pipeline/single_point_trace.py','integrations/demo_pipeline/single_point_report.py',
        'integrations/demo_pipeline/photometric_tracks.py','integrations/demo_pipeline/balanced_object_tracks.py',
        'integrations/demo_pipeline/temporal_review.py','integrations/demo_pipeline/balanced_point_report.py',
        'integrations/demo_pipeline/balanced_point_run.py','integrations/demo_pipeline/evidence_point_run.py',
        'integrations/demo_pipeline/supervision.py','tests/test_single_point_trace.py',
        'tests/test_single_point_run.py','scripts/prepare_video.py')
    return [binding(ROOT/n) for n in names]


def launch(output,action):
    from .supervision import supervise
    d=output/'processes'/action;d.mkdir(parents=True);tmp=d/'tmp';tmp.mkdir()
    script=ROOT/'scripts/trace_single_object_point.py';env={k:'2' for k in THREADS}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TEMP=str(tmp),TMP=str(tmp))
    result=supervise([sys.executable,'-B','-X','utf8',str(script),action,'--output',str(output)],
        cwd=ROOT,env_overrides=env,log_path=d/'supervision.jsonl',stdout_path=d/'stdout.txt',stderr_path=d/'stderr.txt',
        wall_clock_seconds=120,job_memory_limit_bytes=4<<30,run_dir=output,run_dir_threshold_bytes=300<<20,
        poll_seconds=.5,reclamation_seconds=15,label='single_point_'+action,script_sha256=sha(script))
    save(d/'ENTRY.json',result);return result


def execute(args):
    output=args.output.resolve();source=args.source_run.resolve(strict=True);supplement=args.supplement.resolve(strict=True)
    if output.exists() or output.is_relative_to(source):raise ValueError('fresh disjoint diagnostic output required')
    inputs=select(source,supplement,args.seed_index,args.seed_track,args.target_index,args.target_track,args.frames)
    code=implementation();evidence=[binding(p) for p in args.offline_evidence]
    output.mkdir(parents=True)
    save(output/'INPUTS.json',inputs)
    save(output/'PLAN.json',dict(implementation=code,offline_evidence=evidence,
        real_tracking_entries=1,retries=0,frames=args.frames,threads=2,wall_seconds_per_stage=120,
        job_commit_bytes=4<<30,output_limit_bytes=300<<20,
        method='Unchanged photometric_step on every adjacent raw frame; diagnostic shadow after failure never recertified.',
        reference='https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html'))
    state=dict(status='RUNNING',stages={},geometry_accepted=False,physical_identity_verified=False,
               new_segmentation_calls=0,sfm_calls=0,mvs_calls=0)
    save(output/'RESULT.json',state)
    try:
        for action in ('track','report'):
            if check_bindings(inputs['bindings']+code+evidence):raise ValueError('frozen source/code drift')
            entry=launch(output,action);state['stages'][action]=entry;save(output/'RESULT.json',state)
            if entry.get('root_exit_code')!=0 or entry.get('reclaim_confirmed') is not True:
                raise RuntimeError('bounded stage failed: '+action)
        state['status']='DIAGNOSTIC_COMPLETE_NOT_IDENTITY_VERIFIED'
    except Exception as error:state.update(status='FAILED',error=type(error).__name__+': '+str(error))
    finally:
        state['source_drift']=check_bindings(inputs['bindings']);state['implementation_drift']=check_bindings(code+evidence)
        if state['source_drift'] or state['implementation_drift']:state['status']='FAILED'
        save(output/'RESULT.json',state)
    return 1 if state['status']=='FAILED' else 0


def track(output,inputs):
    import numpy as np
    from PIL import Image
    from .temporal_review import load_cv
    from .single_point_trace import trace
    refs=[];cv=load_cv(refs)
    if refs!=inputs['cv_bindings']:raise ValueError('actual OpenCV bindings differ')
    cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
    images=[];rgb_checks=[];by_decode={x['decode_index']:x for x in (inputs['seed'],inputs['comparison'])}
    for frame in inputs['frames']:
        with Image.open(frame['path']) as im:rgb=np.asarray(im.convert('RGB'))
        if frame['decode_index'] in by_decode:
            rgb_checks.append(verify_endpoint(rgb,by_decode[frame['decode_index']]))
        images.append(cv.cvtColor(rgb,cv.COLOR_RGB2GRAY))
    result=trace(images,[f['decode_index'] for f in inputs['frames']],inputs['seed']['xy'],cv)
    target=next(f for f in result['frames'] if f['decode_index']==inputs['comparison']['decode_index'])
    def distance(q):
        return None if q is None else math.dist(q,inputs['comparison']['xy'])
    result.update(rgb_checks=rgb_checks,cv_bindings=refs,
        comparison=dict(decode_index=target['decode_index'],old_target_xy=inputs['comparison']['xy'],
            diagnostic_candidate_xy=target['candidate_xy'],accepted_xy=target['accepted_xy'],
            diagnostic_distance_px=distance(target['candidate_xy']),accepted_distance_px=distance(target['accepted_xy']),
            identity_conclusion='UNKNOWN: unverified shadow distance cannot prove a correspondence wrong or right.'))
    save(output/'TRACE.json',result)
    print('frames',len(result['frames']),'first_failure',result['first_failure'],flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','track','report']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-run',type=Path);p.add_argument('--supplement',type=Path)
    for name in ('seed-index','seed-track','target-index','target-track'):p.add_argument('--'+name,type=int)
    p.add_argument('--frames',type=int,default=64);p.add_argument('--offline-evidence',type=Path,action='append',default=[])
    args=p.parse_args(argv)
    if args.action=='run':
        if any(getattr(args,k) is None for k in ('source_run','supplement','seed_index','seed_track','target_index','target_track')):
            p.error('source, supplement and explicit observation identities required')
        return execute(args)
    output=args.output.resolve(strict=True);plan=read(output/'PLAN.json');inputs=read(output/'INPUTS.json')
    if check_bindings(inputs['bindings']+plan['implementation']+plan['offline_evidence']):raise ValueError('child binding drift')
    if args.action=='track':track(output,inputs)
    else:
        from .single_point_report import render
        render(output)
    return 0
