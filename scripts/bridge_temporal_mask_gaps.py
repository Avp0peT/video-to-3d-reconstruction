"""Bounded two-anchor temporal support for a short missing-SAM interval."""
from pathlib import Path
import argparse
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from integrations.demo_pipeline.balanced_point_run import read,save,binding,check_bindings,sha


def verify(out):
    p=read(out/'PLAN.json')
    drift=check_bindings(p['source_bindings']+p['implementation'])
    if drift: raise ValueError('frozen input/code drift: '+repr(drift))


def prepare(source,out):
    if out.exists(): raise FileExistsError(out)
    source=source.resolve(strict=True); masks=read(source/'MASKS.json'); inputs=read(source/'INPUTS.json')
    oldplan=read(source/'PLAN.json')
    inherited=read(source/'SOURCE_BINDINGS.json')+oldplan['implementation']
    if check_bindings(inherited): raise ValueError('source experiment no longer bound')
    if [f['decode_index'] for f in masks['frames']]!=[f['decode_index'] for f in inputs['frames']]:
        raise ValueError('source mask/input order mismatch')
    rows=[]; refs=[binding(source/n) for n in ['MASKS.json','INPUTS.json','PLAN.json','ACCEPTANCE.json']]
    for f,m in zip(inputs['frames'],masks['frames']):
        if m['raw_path']!=f['path'] or m['raw_sha256']!=f['sha256']: raise ValueError('raw binding mismatch')
        if sha(f['path'])!=f['sha256']: raise ValueError('raw drift')
        refs.append(binding(f['path']))
        if m['current_observed']:
            if m['status']!='OK' or sha(m['mask_path'])!=m['mask_sha256']: raise ValueError('mask binding mismatch')
            refs.append(binding(m['mask_path']))
        rows.append(dict(decode_index=f['decode_index'],path=f['path'],sha256=f['sha256'],
                         time_seconds=f['time_seconds'],status=m['status'],mask_path=m['mask_path'],
                         mask_sha256=m['mask_sha256'],current_observed=m['current_observed']))
    gaps=[]; i=0
    while i<len(rows):
        if rows[i]['current_observed']: i+=1; continue
        j=i
        while j<len(rows) and not rows[j]['current_observed']: j+=1
        gaps.append((i,j));i=j
    eligible=[(a,b) for a,b in gaps if a>0 and b<len(rows) and b-a<=6]
    if len(eligible)!=1: raise ValueError('this bounded pilot requires exactly one bracketed gap of 1..6 frames')
    a,b=eligible[0]; n=b-a+2; control=None
    # Latest disjoint earlier interval, same length, with all actual SAM masks.
    for start in range(a-n,-1,-1):
        candidate=rows[start:start+n]
        if len(candidate)==n and all(f['current_observed'] for f in candidate): control=candidate;break
    if control is None: raise ValueError('same-length earlier leave-out control unavailable')
    selected=rows[a-1:b+1]
    if any(y['decode_index']!=x['decode_index']+1 for x,y in zip(selected,selected[1:])):
        raise ValueError('gap sequence is not consecutive')
    code=[binding(ROOT/n) for n in ['scripts/bridge_temporal_mask_gaps.py',
         'integrations/demo_pipeline/temporal_mask_gap.py','integrations/demo_pipeline/temporal_mask_gap_report.py',
         'integrations/demo_pipeline/temporal_review.py','integrations/demo_pipeline/masked_trace_run.py',
         'integrations/demo_pipeline/supervision.py','tests/test_temporal_mask_gap.py']]
    out.mkdir(parents=True)
    save(out/'INPUTS.json',dict(frames=selected,source_run=str(source)))
    save(out/'CONTROL_INPUTS.json',dict(frames=control,source_run=str(source),
          semantics='Interior actual masks withheld from propagation; only used for independent readback metrics.'))
    refs += [binding(out/'INPUTS.json'),binding(out/'CONTROL_INPUTS.json')]
    from integrations.demo_pipeline.temporal_mask_gap import PROFILE
    save(out/'PLAN.json',dict(schema='temporal-mask-gap-plan-v1',source_bindings=refs,implementation=code,
        cv_bindings=inputs['cv_bindings'],profile=PROFILE,actual_sam_changed=False,sam_calls=0,
        model_calls=0,point_identity_recovery=False,compute_entries=1,retries=0,
        leaveout_gate=dict(min_support_iou=.65,min_support_precision=.80),
        reference='https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html',
        book_boundary='Approximate correspondence and whole-image consistency; this flow propagation is an engineering hypothesis, not a Kozlov theorem.'))
    save(out/'RESULT.json',dict(status='PREPARED',stages={},geometry_accepted=False,point_identity_recovered=False))


def load_case(document,cv):
    import numpy as np
    from PIL import Image
    grays=[]; masks=[]
    for f in document['frames']:
        if sha(f['path'])!=f['sha256']: raise ValueError('raw drift at read')
        with Image.open(f['path']) as im: grays.append(cv.cvtColor(np.array(im.convert('RGB')),cv.COLOR_RGB2GRAY))
        if f['mask_path'] is None: masks.append(None);continue
        if sha(f['mask_path'])!=f['mask_sha256']: raise ValueError('mask drift at read')
        with Image.open(f['mask_path']) as im:
            a=np.array(im)
            if im.mode!='L' or a.shape!=grays[-1].shape or not np.isin(a,[0,255]).all(): raise ValueError('mask format')
            masks.append(a)
    return grays,masks


def compute(out):
    import numpy as np
    from PIL import Image
    from integrations.demo_pipeline.temporal_review import load_cv
    from integrations.demo_pipeline.temporal_mask_gap import propagate_gap
    verify(out); refs=[];cv=load_cv(refs);cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
    if refs!=read(out/'PLAN.json')['cv_bindings']: raise ValueError('OpenCV identity mismatch')
    control=read(out/'CONTROL_INPUTS.json'); cg,cm=load_case(control,cv)
    c=propagate_gap(cg,cm[0],cm[-1],cv)
    metrics=[]
    for i,(support,row) in enumerate(zip(c['support'],c['diagnostics']['gaps']),1):
        truth=cm[i]>0; intersection=int(np.count_nonzero(support & truth))
        union=int(np.count_nonzero(support | truth)); size=int(support.sum())
        iou=intersection/union if union else 0.; precision=intersection/size if size else 0.
        metrics.append(dict(decode_index=control['frames'][i]['decode_index'],support_iou=iou,
                            support_precision=precision,support_recall=intersection/int(truth.sum()),
                            passed=row['status']=='ACCEPTED_TEMPORAL_SUPPORT' and iou>=.65 and precision>=.80))
    allowed=all(m['passed'] for m in metrics)
    save(out/'CONTROL.json',dict(passed=allowed,metrics=metrics,diagnostics=c['diagnostics'],
         comparison_target='Withheld original SAM masks, not human ground truth'))
    primary=read(out/'INPUTS.json'); grays,masks=load_case(primary,cv)
    result=propagate_gap(grays,masks[0],masks[-1],cv)
    save(out/'FLOW_DIAGNOSTICS.json',result['diagnostics'])
    assets=out/'evidence_masks';assets.mkdir()
    def png(name,array):
        path=assets/name; labels=array.astype(np.uint8)*255;Image.fromarray(labels).save(path)
        with Image.open(path) as im:
            if im.mode!='L' or not np.array_equal(np.array(im),labels): raise ValueError('mask readback differs')
        return binding(path)
    frames=[]
    for i,f in enumerate(primary['frames']):
        idx=f['decode_index']; isgap=0<i<len(grays)-1
        row=result['diagnostics']['gaps'][i-1] if isgap else None
        permit=isgap and allowed and row['status']=='ACCEPTED_TEMPORAL_SUPPORT'
        reasons=[] if not isgap else list(row['reasons'])+([] if allowed else ['LEAVEOUT_CONTROL_FAILED'])
        union=result['predictions']['forward'][i] | result['predictions']['backward'][i]
        support=result['support'][i-1] if permit else np.zeros(grays[i].shape,bool)
        uncertainty=(union & ~support) if isgap else np.zeros(grays[i].shape,bool)
        frames.append(dict(decode_index=idx,status='ACTUAL_SAM_ANCHOR' if not isgap else ('TEMPORAL_SUPPORT_ONLY' if permit else 'UNRESOLVED'),
            source_sam_status=f['status'],support_allowed=bool(permit),rejection_reason='; '.join(reasons),
            forward_mask=png(f'{idx}_forward.png',result['predictions']['forward'][i]),
            backward_mask=png(f'{idx}_backward.png',result['predictions']['backward'][i]),
            forward_valid=png(f'{idx}_forward_valid.png',result['valid']['forward'][i]),
            backward_valid=png(f'{idx}_backward_valid.png',result['valid']['backward'][i]),
            support_mask=png(f'{idx}_support.png',support) if permit else None,
            uncertain_mask=png(f'{idx}_uncertain.png',uncertainty),
            support_pixels=int(support.sum()),uncertain_pixels=int(uncertainty.sum()),metrics=row))
    save(out/'GAP_RESULT.json',dict(schema='temporal-mask-gap-artifact-v1',frames=frames,
          supported_gap_frames=sum(r['support_allowed'] for r in frames),gap_frames=len(grays)-2,
          leaveout_control_passed=allowed,actual_sam_changed=False,original_no_match_preserved=True,
          point_identity_recovered=False,geometry_accepted=False,source_actual_mask_count=60,
          diagnostics_binding=binding(out/'FLOW_DIAGNOSTICS.json'),control_binding=binding(out/'CONTROL.json')))
    verify(out)


def child(action,out):
    verify(out)
    if action=='compute':compute(out)
    elif action=='report':
        from integrations.demo_pipeline.temporal_mask_gap_report import render
        render(out)
    elif action=='tests':
        from integrations.demo_pipeline.temporal_review import load_cv
        cv=load_cv([]);cv.setNumThreads(2);cv.ocl.setUseOpenCL(False)
        import pytest
        rc=pytest.main([str(ROOT/'tests/test_temporal_mask_gap.py'),'-q','-p','no:cacheprovider',
                       '--junitxml='+str(out/'tests.xml'),'--basetemp='+str(out/'pytest_tmp')])
        if rc:raise SystemExit(rc)
    verify(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','tests','compute','report'])
    p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if args.action!='run':child(args.action,out);return 0
    if args.source is None:p.error('--source required')
    prepare(args.source,out)
    from integrations.demo_pipeline.masked_trace_run import invoke
    state=read(out/'RESULT.json')
    try:
        for action in ['tests','compute','report']:
            verify(out)
            e=invoke(out,action,[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),action,'--output',str(out)],120,4<<30,
                     {'GAP_SUPERVISED':'1'})
            state['stages'][action]=e;save(out/'RESULT.json',state)
            if e.get('root_exit_code')!=0 or not e.get('reclaim_confirmed'):raise RuntimeError('bounded stage failed: '+action)
        gap=read(out/'GAP_RESULT.json')
        state.update(status='TEMPORAL_SUPPORT_EVALUATED',supported_gap_frames=gap['supported_gap_frames'],gap_frames=gap['gap_frames'])
    except Exception as error:
        state.update(status='FAILED',error=type(error).__name__+': '+str(error))
        raise
    finally:
        plan=read(out/'PLAN.json');state['source_drift']=check_bindings(plan['source_bindings']);state['implementation_drift']=check_bindings(plan['implementation'])
        if state['source_drift'] or state['implementation_drift']:state['status']='FAILED'
        save(out/'RESULT.json',state)
    return 0


if __name__=='__main__':raise SystemExit(main())
