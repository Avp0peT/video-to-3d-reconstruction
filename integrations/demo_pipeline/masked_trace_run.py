"""Reproducible mask supplementation and same-seed tracking comparison.

All substantive stages use the existing Windows Job supervisor. Old results are
read-only, and unsuccessful SAM observations never become synthetic masks.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys
from .balanced_point_run import read, save, binding, check_bindings, sha
from .single_point_run import ROOT, THREADS


def implementation():
    names=['scripts/trace_masked_object_point.py','integrations/demo_pipeline/masked_trace_run.py',
           'integrations/demo_pipeline/masked_trace_inputs.py','integrations/demo_pipeline/masked_point_trace.py',
           'integrations/demo_pipeline/masked_trace_report.py','integrations/demo_pipeline/photometric_tracks.py',
           'integrations/demo_pipeline/balanced_object_tracks.py','integrations/demo_pipeline/single_point_trace.py',
           'integrations/demo_pipeline/temporal_review.py','integrations/demo_pipeline/supervision.py',
           'scripts/demo_segment_worker.py','integrations/demo_pipeline/runtime.py']
    return [binding(ROOT/n) for n in names]


def invoke(out, label, argv, seconds, memory, extra_env=None):
    from .supervision import supervise
    d=out/'processes'/label
    d.mkdir(parents=True,exist_ok=False)
    tmp=d/'tmp'; tmp.mkdir()
    env={k:'2' for k in THREADS}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TEMP=str(tmp),TMP=str(tmp),
               PYTHONPATH=str(ROOT)+';'+str(ROOT/'src'))
    env.update(extra_env or {})
    save(d/'INTENT.json',dict(argv=argv,cwd=str(ROOT),env_overrides=env,seconds=seconds,job_commit_bytes=memory))
    r=supervise(argv,cwd=ROOT,env_overrides=env,log_path=d/'supervision.jsonl',stdout_path=d/'stdout.txt',
                stderr_path=d/'stderr.txt',wall_clock_seconds=seconds,job_memory_limit_bytes=memory,
                run_dir=out,run_dir_threshold_bytes=2<<30,poll_seconds=.5,reclamation_seconds=15,
                label='masked_point_'+label,script_sha256=sha(Path(__file__)))
    save(d/'ENTRY.json',r)
    print(label,'exit',r.get('root_exit_code'),'reclaimed',r.get('reclaim_confirmed'),flush=True)
    return r


def check(out):
    refs=read(out/'SOURCE_BINDINGS.json')+read(out/'PLAN.json')['implementation']
    drift=check_bindings(refs)
    if drift: raise ValueError('frozen input/code drift: '+repr(drift))


def prepare(args):
    from .masked_trace_inputs import prepare as prepare_inputs
    old=args.old_run.resolve(strict=True); out=args.output.resolve()
    prepared=prepare_inputs(old,out)
    shutil.copyfile(old/'INPUTS.json',out/'INPUTS.json')
    shutil.copyfile(old/'TRACE.json',out/'BASELINE_TRACE.json')
    oldconfig=args.sam_config.resolve(strict=True)
    prepared['sam_config']=str(oldconfig)
    save(out/'PREPARATION.json',prepared)
    refs=read(out/'SOURCE_BINDINGS.json')+[binding(old/'TRACE.json'),binding(out/'INPUTS.json'),
                                          binding(out/'BASELINE_TRACE.json'),binding(oldconfig)]
    save(out/'SOURCE_BINDINGS.json',refs)
    save(out/'PLAN.json',dict(schema='masked-point-plan-v1',implementation=implementation(),
         new_sam_frames=54,old_masks_reused=9,preserved_no_match=[576],sam_batches=1,retries=0,
         seed=read(out/'INPUTS.json')['seed'],cpu_comparisons=['current_mask_only','current_mask_and_velocity_init'],
         prediction_displacement_cap_px=96,prediction_requires_two_continuously_accepted_positions=True,
         candidate_merge_distance_px=2,mask_center_rounding='floor(x+0.5)',
         source='https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html',
         physical_identity_verified=False,geometry_accepted=False))
    save(out/'RESULT.json',dict(status='PREPARED',stages={},physical_identity_verified=False,geometry_accepted=False))
    print('prepared',out,flush=True)


def segment(out):
    check(out); prepared=read(out/'PREPARATION.json'); state=read(out/'RESULT.json')
    # A pre-existing launch receipt prevents accidental repeat even after a partial failure.
    if (out/'processes/photo_io').exists() or (out/'processes/sam').exists():
        raise FileExistsError('segmentation stages already attempted; inspect original receipts')
    try:
        entry=invoke(out,'photo_io',prepared['io_argv'],180,2<<30,prepared['io_env'])
        state['stages']['photo_io']=entry; save(out/'RESULT.json',state)
        if entry.get('root_exit_code')!=0 or not entry.get('reclaim_confirmed'): raise RuntimeError('photo IO failed')
        runs=list(Path(prepared['io_output']).glob('*/run.json'))
        if len(runs)!=1: raise ValueError('exactly one new photo IO run required')
        config=read(prepared['sam_config']); seg=out/'segmentation'; seg.mkdir()
        config.update(io_run=str(runs[0].parent),output_dir=str(seg),indices=prepared['indices'])
        save(seg/'CONFIG.json',config)
        state['io_run']=str(runs[0].parent); state['status']='SEGMENTING'; save(out/'RESULT.json',state)
        entry=invoke(out,'sam',[config['sam3_python'],'-B','-X','utf8',str(ROOT/'scripts/demo_segment_worker.py'),
                               '--config',str(seg/'CONFIG.json')],950,14<<30)
        state['stages']['sam']=entry
        # Partial/no-match runs remain consumable as explicit absent observations.
        if entry.get('root_exit_code') not in (0,1) or not entry.get('reclaim_confirmed'):
            raise RuntimeError('SAM stage failed; no automatic retry')
        from .masked_trace_inputs import collect
        masks=collect(out,io_run=runs[0].parent)
        state.update(status='MASKS_COLLECTED',observed_masks=masks['current_observed_count'],unknown_masks=masks['unknown_count'])
    except Exception as error:
        state.update(status='FAILED',error=type(error).__name__+': '+str(error))
        raise
    finally:
        state['source_drift']=check_bindings(read(out/'SOURCE_BINDINGS.json'))
        state['implementation_drift']=check_bindings(read(out/'PLAN.json')['implementation'])
        save(out/'RESULT.json',state)


def track(out):
    import numpy as np
    from PIL import Image
    from .temporal_review import load_cv
    from .masked_point_trace import trace
    check(out); inputs=read(out/'INPUTS.json'); masks=read(out/'MASKS.json')
    refs=[]; cv=load_cv(refs); cv.setNumThreads(2); cv.ocl.setUseOpenCL(False)
    if refs!=inputs['cv_bindings']: raise ValueError('OpenCV identity differs')
    if [r['decode_index'] for r in masks['frames']]!=[r['decode_index'] for r in inputs['frames']]:
        raise ValueError('ordered mask/frame mapping mismatch')
    images=[]; arrays=[]; maskrefs=[]
    for f,m in zip(inputs['frames'],masks['frames']):
        if sha(f['path'])!=f['sha256']: raise ValueError('raw image drift')
        with Image.open(f['path']) as im: images.append(cv.cvtColor(np.array(im.convert('RGB')),cv.COLOR_RGB2GRAY))
        if m['current_observed']:
            if sha(m['mask_path'])!=m['mask_sha256']: raise ValueError('mask drift')
            maskrefs.append(binding(m['mask_path']))
            with Image.open(m['mask_path']) as im: a=np.array(im)
            if im.mode!='L' or a.shape!=images[-1].shape or not np.isin(a,[0,255]).all(): raise ValueError('invalid selected mask')
            arrays.append(a)
        else: arrays.append(None)
    for motion,name in [(False,'TRACE_MASKED.json'),(True,'TRACE_MOTION.json')]:
        if (out/name).exists(): raise FileExistsError(out/name)
        result=trace(images,arrays,[f['decode_index'] for f in inputs['frames']],inputs['seed']['xy'],cv,
                     predict_motion=motion,times=[f['time_seconds'] for f in inputs['frames']])
        result.update(mask_manifest_binding=binding(out/'MASKS.json'),cv_bindings=refs)
        save(out/name,result)
        print(name,'first_failure',result['first_failure'],flush=True)
    if check_bindings(maskrefs): raise ValueError('mask drift after tracking')


def finish(out):
    check(out); state=read(out/'RESULT.json')
    if state['status']!='MASKS_COLLECTED': raise ValueError('collect actual masks first')
    script=ROOT/'scripts/trace_masked_object_point.py'
    try:
        for stage in ('track','report'):
            check(out)
            entry=invoke(out,stage,[sys.executable,'-B','-X','utf8',str(script),stage,'--output',str(out)],120,4<<30)
            state['stages'][stage]=entry; save(out/'RESULT.json',state)
            if entry.get('root_exit_code')!=0 or not entry.get('reclaim_confirmed'): raise RuntimeError('stage failed: '+stage)
        state['status']='DIAGNOSTIC_COMPLETE_NOT_IDENTITY_VERIFIED'
    except Exception as error:
        state.update(status='FAILED',error=type(error).__name__+': '+str(error)); raise
    finally:
        state['source_drift']=check_bindings(read(out/'SOURCE_BINDINGS.json'))
        state['implementation_drift']=check_bindings(read(out/'PLAN.json')['implementation'])
        save(out/'RESULT.json',state)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['run','prepare','segment','finish','track','report'])
    p.add_argument('--output',type=Path,required=True); p.add_argument('--old-run',type=Path);p.add_argument('--sam-config',type=Path)
    args=p.parse_args(argv)
    if args.action in ('prepare','run'):
        if args.old_run is None or args.sam_config is None: p.error('old-run and sam-config required')
        prepare(args)
    out=args.output.resolve(strict=True)
    if args.action in ('segment','run'): segment(out)
    if args.action in ('finish','run'): finish(out)
    if args.action=='track': track(out)
    if args.action=='report':
        check(out)
        from .masked_trace_report import render
        render(out)
    return 0
