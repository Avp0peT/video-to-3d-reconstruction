"""Automatically promote measured intermediate frames across component gaps."""
from pathlib import Path
import argparse
from integrations.demo_pipeline.pipeline import ROOT,LIGHT_REL,_invoke,entry_code,read,save,sha,utc
from .region_experiment import drift

CODE = ('region_bridge_experiment.py','region_bridge_inputs.py','region_bridge_assessment.py',
        'region_tracks.py','region_geometry.py','region_improvement_assessment.py','object_region_policy.py')


def prepare(source_run, output, selection_policy=None):
    source=Path(source_run).resolve(strict=True);out=Path(output).resolve()
    if out.exists() or source.is_relative_to(out):
        raise FileExistsError('fresh disjoint output required')
    previous=read(source/'PLAN.json')
    assessment=read(source/'assessment/ASSESSMENT.json')
    if not assessment['summary']['new']['largest_valid_component_views']:
        raise ValueError('no valid source component')
    if selection_policy is not None:
        from .region_bridge_inputs import choose_intermediates
        original=read(Path(previous['source_run'])/'PLAN.json')
        _, selected=choose_intermediates(original['timeline'],assessment['models'],selection_policy)
        if not selected['resource_limit_satisfied']:
            raise ValueError('existing endpoints exceed selection policy resource limit')
        selection_policy=selected['policy']
    refs=dict(previous['source_bindings'])
    for name in ('PLAN.json','assessment/ASSESSMENT.json','geometry/PLAN.json','geometry/RESULT.json',
                 'geometry/database.db','tracking/REGION_TRACKS.json'):
        refs[str(source/name)]=sha(source/name)
    for m in read(source/'geometry/RESULT.json')['models']:
        for key in ('binary_path','text_path','ply_path'):
            path=Path(m[key])
            for f in sorted(path.rglob('*')) if path.is_dir() else [path]:
                if f.is_file():refs[str(f)]=sha(f)
    if drift(refs):raise ValueError('source binding drift')
    bindings={str(ROOT/'integrations/video_masks'/p):sha(ROOT/'integrations/video_masks'/p) for p in CODE}
    for name in ('scripts/bridge_mask_regions.py','integrations/demo_pipeline/camera_domain_gate.py',
                 'integrations/demo_pipeline/pipeline.py','integrations/demo_pipeline/supervision.py'):
        bindings[str(ROOT/name)]=sha(ROOT/name)
    plan=dict(schema='glove-region-bridge-experiment-v1',source_run=str(source),created_utc=utc(),
              source_bindings=refs,code_bindings=bindings,
              selection_policy=selection_policy,
              native_max_frames=96,mask_changes=False,new_segmentation=False,
              changed_factors=['promote existing intermediate images into native keyframes',
                               'seed the same forward regional tracker at added keyframes'],
              preserved=['all original canonical keyframes and SIFT rows/matches',
                         'same camera/mapper/verification settings','same patch/FB/texture thresholds'],
              limits=dict(additional_keyframes=(selection_policy or {}).get('max_added',24),
                          consecutive_endpoint_gap=None if selection_policy else 6,cpu_threads=2,job_commit_bytes=4<<30),
              quality_status='NOT_EVALUATED',scale_status='unknown')
    out.mkdir(parents=True)
    save(out/'RUN_PLAN.json',plan)
    save(out/'SUMMARY.json',dict(status='PREPARED',stages={}))
    return plan


def execute(run):
    run=Path(run).resolve(strict=True);plan=read(run/'RUN_PLAN.json');state=read(run/'SUMMARY.json')
    if state['status']!='PREPARED' or (run/'logs').exists():raise ValueError('already attempted')
    if drift(plan['source_bindings']) or drift(plan['code_bindings']):raise ValueError('prelaunch drift')
    state.update(status='RUNNING',started_utc=utc());save(run/'SUMMARY.json',state)
    try:
        for label,seconds in [('input',180),('tracking',600),('geometry',600),('assessment',180)]:
            argv=[str(ROOT/LIGHT_REL),'-B','-X','utf8',str(ROOT/'scripts/bridge_mask_regions.py'),label+'-worker','--run',str(run)]
            state['stages'][label]=dict(status='RUNNING',started_utc=utc(),argv=argv);save(run/'SUMMARY.json',state)
            print('STAGE_BEGIN',label,flush=True)
            entry=_invoke(ROOT,run,label,argv,seconds=seconds,memory=4<<30)
            code=entry_code(entry)
            state['stages'][label].update(exit_code=code,reclaimed=entry.get('reclaim_confirmed'),status='COMPLETED' if code==0 else 'FAILED',finished_utc=utc());save(run/'SUMMARY.json',state)
            print('STAGE_END',label,code,flush=True)
            if code!=0:raise RuntimeError(label+' failed; no automatic retry')
        assessment=read(run/'assessment/BRIDGE_ASSESSMENT.json')
        if assessment.get('bridge_effect_evaluatable') is not True:
            raise RuntimeError('mapper scope mismatch: bridge effect not evaluatable')
        state['status']='BRIDGE_EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED'
    except Exception as exc:
        state.update(status='PARTIAL_DIAGNOSTIC',error=f'{type(exc).__name__}: {exc}')
    finally:
        state.update(input_drift=drift(plan['source_bindings']),code_drift=drift(plan['code_bindings']),finished_utc=utc())
        if state['input_drift'] or state['code_drift']:state['status']='DRIFT_REFUSED'
        save(run/'SUMMARY.json',state)
    return state


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','execute','run','input-worker','tracking-worker','geometry-worker','assessment-worker'])
    p.add_argument('--source-run',type=Path);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(argv)
    if a.action in ('prepare','execute'):
        if a.source_run is None:p.error('--source-run required')
        prepare(a.source_run,a.run)
        if a.action=='prepare':return 0
    if a.action in ('execute','run'):
        result=execute(a.run)
        return 0 if result['status']=='BRIDGE_EXPERIMENT_COMPLETE_NOT_GEOMETRY_ACCEPTED' else 1
    plan=read(a.run/'RUN_PLAN.json')
    if a.action=='input-worker':
        from .region_bridge_inputs import prepare as build
        build(plan['source_run'],a.run/'input',plan.get('selection_policy'))
    elif a.action=='tracking-worker':
        from .region_tracks import run
        run(a.run/'input/PLAN.json',a.run/'tracking')
    elif a.action=='geometry-worker':
        from .region_geometry import run
        source=read(a.run/'input/PLAN.json')['baseline_sfm']
        run(source,a.run/'tracking/REGION_TRACKS.json',a.run/'geometry',max_frames=96)
    else:
        from .region_bridge_assessment import assess
        assess(plan['source_run'],a.run)
    return 0
