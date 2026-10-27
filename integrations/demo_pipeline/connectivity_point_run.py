"""Automatic fixed comparison of saved-ID graph and radial calibration freedom."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
from .balanced_point_run import read, save, binding, sha, check_bindings
from .evidence_point_run import unique_bindings

ROOT=Path(__file__).resolve().parents[2]
ARMS=('all_shared_tracks','all_shared_tracks_zero_radial')


def prepare(control):
    prior=read(control/'INPUTS.json'); old_plan=read(control/'PLAN.json')
    if read(control/'RESULT.json').get('status')!='COMPARISON_COMPLETE':
        raise ValueError('completed evidence comparison required')
    tracks_path=Path(old_plan['tracks_path']).resolve(strict=True)
    tracks=read(tracks_path)
    if tracks.get('tracking_variant')!='photometric_gap_v1' or tracks.get('predicted_observations')!=0:
        raise ValueError('real frozen photometric observation ledger required')
    code=old_plan['implementation']
    sources=prior['bindings']
    if check_bindings(sources+code): raise ValueError('control source or implementation drift')
    native=control/'geometry_evidence/native'
    if read(native/'RESULT.json').get('policy_id')!='geometry_evidence': raise ValueError('control geometry policy differs')
    refs=[control/n for n in ('INPUTS.json','PLAN.json','RESULT.json','SUMMARY.json','DELIVERY_CHECK.json',
        'geometry_evidence/native/PLAN.json','geometry_evidence/native/RESULT.json','geometry_evidence/native/GEOMETRY_EVIDENCE.json')]
    for model in read(native/'RESULT.json')['models']:
        folder=Path(model['binary_path']).resolve(strict=True)
        if not folder.is_relative_to(native.resolve()): raise ValueError('control model outside native directory')
        refs.extend(p for p in folder.iterdir() if p.is_file())
    return unique_bindings(sources+[binding(p) for p in refs]),code,tracks_path,prior['frames']


def launch(output,action,policy=None):
    from .supervision import supervise
    label=policy or action; directory=output/'processes'/label; directory.mkdir(parents=True)
    tmp=directory/'tmp'; tmp.mkdir()
    script=ROOT/'scripts/connect_object_tracks.py'
    argv=[sys.executable,'-B','-X','utf8',str(script),action,'--output',str(output)]
    if policy: argv+=['--policy',policy]
    env={k:'2' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMP=str(tmp),TEMP=str(tmp))
    entry=supervise(argv,cwd=ROOT,env_overrides=env,log_path=directory/'supervision.jsonl',
        stdout_path=directory/'stdout.txt',stderr_path=directory/'stderr.txt',wall_clock_seconds=240 if policy else 60,
        job_memory_limit_bytes=8<<30,run_dir=output,run_dir_threshold_bytes=1<<30,poll_seconds=.5,
        reclamation_seconds=15,label=label,script_sha256=sha(script))
    save(directory/'ENTRY.json',entry)
    return entry


def execute(control,output,offline_evidence=()):
    if output.exists() or output.is_relative_to(control) or control.is_relative_to(output): raise ValueError('fresh disjoint output required')
    sources,old_code,tracks_path,frames=prepare(control)
    names=('scripts/connect_object_tracks.py','integrations/demo_pipeline/connectivity_point_run.py',
        'integrations/demo_pipeline/connectivity_native.py','integrations/demo_pipeline/connectivity_graph.py',
        'integrations/demo_pipeline/connectivity_point_report.py','integrations/demo_pipeline/supervision.py',
        'tests/test_connectivity_native.py','tests/test_connectivity_point_run.py')
    offline=[binding(Path(p)) for p in offline_evidence]
    implementation=unique_bindings([binding(ROOT/n) for n in names]+old_code+offline)
    output.mkdir(parents=True)
    from .connectivity_graph import audit
    graph_path=output/'GRAPH_AUDIT.json'
    audit(tracks_path,control/'geometry_evidence/native',graph_path)
    sources=unique_bindings(sources+[binding(graph_path)])
    save(output/'INPUTS.json',dict(bindings=sources,frames=frames,tracks_path=str(tracks_path)))
    save(output/'PLAN.json',dict(control_run=str(control),tracks_path=str(tracks_path),graph_audit_path=binding(graph_path),
        source_bindings=sources,implementation=implementation,offline_evidence=offline,policies=list(ARMS),
        changes=dict(all_shared_tracks='All image pairs with at least 15 existing real track IDs; no sequence-distance cap.',
            all_shared_tracks_zero_radial='Same all-shared pairs; freeze radial k=0 in absolute pose and local/global BA; focal still free.'),
        assumption='Zero radial coefficient is a diagnostic hypothesis, not camera calibration. No known focal length.',
        unchanged='Source coordinates/IDs/images/masks, >=2 samples, two-view triangulation allowed, numeric geometry gates, shared camera and initialization.',
        tracking_calls=0,segmentation_calls=0,retries=0,budget=dict(native_seconds_per_arm=240,audit_seconds=60,job_commit_bytes=8<<30,output_bytes=1<<30)))
    result=dict(status='RUNNING',stages={},geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown')
    save(output/'RESULT.json',result)
    try:
        for action,policy in [('audit-control',None)]+[('native',p) for p in ARMS]:
            if check_bindings(sources+implementation): raise ValueError('frozen binding drift before stage')
            entry=launch(output,action,policy); label=policy or action; result['stages'][label]=entry
            save(output/'RESULT.json',result)
            if entry.get('root_exit_code')!=0 or not entry.get('reclaim_confirmed'): raise RuntimeError('bounded stage failed: '+label)
        result['status']='COMPARISON_COMPLETE'
    except Exception as error:
        result.update(status='FAILED',error=type(error).__name__+': '+str(error))
    finally:
        result['input_drift']=check_bindings(sources); result['implementation_drift']=check_bindings(implementation)
        if result['input_drift'] or result['implementation_drift']: result['status']='FAILED'
        save(output/'RESULT.json',result)
    if result['status']=='COMPARISON_COMPLETE':
        try:
            from .connectivity_point_report import build
            build(output)
        except Exception as error:
            result.update(status='REPORT_FAILED',error=type(error).__name__+': '+str(error)); save(output/'RESULT.json',result)
    return 0 if result['status']=='COMPARISON_COMPLETE' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','audit-control','native','report'])
    parser.add_argument('--control',type=Path); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',choices=ARMS); parser.add_argument('--offline-evidence',type=Path,action='append',default=[])
    args=parser.parse_args(argv); output=args.output.resolve()
    if args.action=='run':
        if args.control is None: parser.error('--control required')
        return execute(args.control.resolve(strict=True),output,args.offline_evidence)
    plan=read(output/'PLAN.json')
    if check_bindings(plan['source_bindings']+plan['implementation']): raise ValueError('frozen binding drift before child')
    if args.action=='audit-control':
        from .connectivity_native import audit_existing
        audit_existing(Path(plan['control_run'])/'geometry_evidence/native/RESULT.json',output/'control_geometry_evidence.json')
    elif args.action=='native':
        if args.policy is None: parser.error('--policy required')
        from .connectivity_native import run
        run(Path(plan['tracks_path']),output/args.policy/'native',args.policy)
    else:
        from .connectivity_point_report import build
        build(output)
    return 0
