"""One bounded experiment: region correspondence across saved track identities."""
from pathlib import Path
import argparse
import sys
from .balanced_point_run import read, save, binding, sha, check_bindings
from .evidence_point_run import unique_bindings
from .connectivity_graph import components

ROOT = Path(__file__).resolve().parents[2]
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')


def prepare(control):
    old = read(control/'PLAN.json'); state = read(control/'RESULT.json')
    if state.get('status') != 'COMPARISON_COMPLETE':
        raise ValueError('completed connectivity comparison required')
    sources = read(control/'INPUTS.json')['bindings']; code = old['implementation']
    if check_bindings(sources+code): raise ValueError('control source or implementation drift')
    tracks = Path(old['tracks_path']).resolve(strict=True)
    native = control/'all_shared_tracks_zero_radial/native'
    result = read(native/'RESULT.json')
    if result.get('policy_id') != 'all_shared_tracks_zero_radial': raise ValueError('zero radial control required')
    refs = [control/n for n in ('PLAN.json','INPUTS.json','RESULT.json','GRAPH_AUDIT.json','SUMMARY.json','DELIVERY_CHECK.json')]
    refs += [native/n for n in ('PLAN.json','RESULT.json','GEOMETRY_EVIDENCE.json')]
    for model in result['models']:
        directory = Path(model['binary_path']).resolve(strict=True)
        if not directory.is_relative_to(native.resolve()): raise ValueError('model outside control')
        refs += [p for p in directory.iterdir() if p.is_file()]
    return unique_bindings(sources+[binding(p) for p in refs]), code, tracks, native


def graph_census(native, plan):
    nodes = [f['input_index'] for f in plan['frames']]
    ledger = {f['database_image_id']: f['input_index'] for f in plan['frames']}
    def summarize(pairs):
        edges = [(ledger[p['image_a']],ledger[p['image_b']]) for p in pairs]
        groups = components(nodes,edges)
        return dict(edges=len(edges),edge_pairs=edges,components=groups,isolated=sum(len(c)==1 for c in groups),
                    component_count=len(groups),largest_component=len(groups[0]) if groups else 0)
    return dict(raw=summarize(plan['candidate_pairs']),
                verified=summarize([p for p in native['pairs'] if p['inliers']>=15]),
                registered_unique=native['unique_registered_count'],native_models=len(native['models']),
                largest_native_model=max((m['registered_views'] for m in native['models']),default=0))


def graph_audit(control_result, new_result, supplement_path, output):
    paths=[Path(control_result),Path(control_result).with_name('PLAN.json'),Path(new_result),
           Path(new_result).with_name('PLAN.json'),Path(supplement_path)]
    refs=[binding(p) for p in paths]
    c,cp,n,np,s = [read(p) for p in paths]
    if cp['frames'] != np['frames']: raise ValueError('feature-row ledger changed')
    cg=graph_census(c,cp); ng=graph_census(n,np)
    old_members={i:k for k,g in enumerate(cg['verified']['components']) for i in g}
    submitted={(p['sequence_a'],p['sequence_b']):p for p in np['candidate_pairs']}
    verified={(p['image_a']-1,p['image_b']-1):p for p in n['pairs']}
    rows=[]
    for p in s['pairs']:
        a,b=p['sequence_a'],p['sequence_b']; fa,fb=np['frames'][a],np['frames'][b]
        ra={tid:i for i,tid in enumerate(fa['track_ids'])}; rb={tid:i for i,tid in enumerate(fb['track_ids'])}
        proposed={(ra[m['track_id_a']],rb[m['track_id_b']]) for m in p['matches']}
        actual={tuple(x) for x in submitted.get((a,b),{}).get('raw_feature_rows',[])}
        inliers={tuple(x) for x in verified.get((a,b),{}).get('inlier_feature_rows',[])}
        ia,ib=fa['input_index'],fb['input_index']
        rows.append(dict(sequence_a=a,sequence_b=b,input_index_a=ia,input_index_b=ib,
            proposed_crossid_matches=len(proposed),raw_crossid_matches=len(proposed&actual),
            verified_crossid_matches=len(proposed&actual&inliers),
            total_verified_inliers=len(inliers),cross_control_components=old_members[ia]!=old_members[ib],
            submitted=(a,b) in submitted))
    added = [r for r in rows if r['cross_control_components'] and r['total_verified_inliers']>=15]
    result=dict(schema='crossid-graph-audit-v1',control=cg,crossid=ng,supplement_pairs=rows,
        new_cross_component_verified_pairs=len(added),new_cross_component_verified_pair_records=added,
        new_verified_crossid_rows=sum(r['verified_crossid_matches'] for r in rows),
        node_count=len(cp['frames']),source_bindings=refs,source_drift=check_bindings(refs),
        note='15 verified rows define graph edges; matching components are not reconstructed models or ground truth.')
    if result['source_drift']: raise ValueError('graph source drift')
    save(output,result)
    return result


def launch(output, action):
    from .supervision import supervise
    directory=output/'processes'/action; directory.mkdir(parents=True)
    temp=directory/'tmp'; temp.mkdir()
    script=ROOT/'scripts/match_object_regions.py'
    env={k:'2' for k in THREADS}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',TMP=str(temp),TEMP=str(temp))
    argv=[sys.executable,'-B','-X','utf8',str(script),action,'--output',str(output)]
    result=supervise(argv,cwd=ROOT,env_overrides=env,log_path=directory/'supervision.jsonl',
        stdout_path=directory/'stdout.txt',stderr_path=directory/'stderr.txt',
        wall_clock_seconds=240 if action in ('match','native') else 120,job_memory_limit_bytes=8<<30,
        run_dir=output,run_dir_threshold_bytes=1<<30,poll_seconds=.5,reclamation_seconds=15,
        label='crossid_'+action,script_sha256=sha(script))
    save(directory/'ENTRY.json',result)
    return result


def execute(control, output, offline_evidence=()):
    if output.exists() or output.is_relative_to(control) or control.is_relative_to(output):
        raise ValueError('fresh disjoint output required')
    sources,old_code,tracks,native_control=prepare(control)
    names=('scripts/match_object_regions.py','integrations/demo_pipeline/crossid_region_run.py',
        'integrations/demo_pipeline/crossid_regions.py','integrations/demo_pipeline/crossid_native.py',
        'integrations/demo_pipeline/crossid_region_report.py','integrations/demo_pipeline/supervision.py',
        'tests/test_crossid_native.py','tests/test_crossid_regions.py','tests/test_crossid_region_run.py')
    evidence=[binding(Path(p)) for p in offline_evidence]
    code=unique_bindings(old_code+[binding(ROOT/n) for n in names]+evidence)
    output.mkdir(parents=True)
    save(output/'INPUTS.json',dict(bindings=sources,tracks_path=str(tracks)))
    save(output/'PLAN.json',dict(control_run=str(control),tracks_path=str(tracks),
        control_native_dir=str(native_control),source_bindings=sources,implementation=code,offline_evidence=evidence,
        method='Explicit image region row pairs across old track IDs; no coordinate edits or ID fusion.',
        unchanged='Same eligible >=2 observation rows; same zero radial/focal free solver and numeric gates.',
        stages=['match','native','report'],real_entries_per_stage=1,retries=0,
        budget=dict(match_seconds=240,native_seconds=240,report_seconds=120,job_commit_bytes=8<<30,output_bytes=1<<30),
        caveat='Zero radial is an assumption, not calibration; no physical identity or accuracy acceptance.'))
    result=dict(status='RUNNING',stages={},geometry_accepted=False,quality_status='NOT_EVALUATED',scale_status='unknown',
        tracks_path=str(tracks),supplement_matches_path=str(output/'regions/SUPPLEMENT.json'),
        native_result_path=str(output/'native/RESULT.json'),native_plan_path=str(output/'native/PLAN.json'),
        control_native_result_path=str(native_control/'RESULT.json'),control_native_plan_path=str(native_control/'PLAN.json'),
        graph_audit_path=str(output/'GRAPH_AUDIT.json'))
    save(output/'RESULT.json',result)
    try:
        for action in ('match','native','report'):
            if check_bindings(sources+code): raise ValueError('frozen binding drift')
            if action=='report':
                result['status']='EXPERIMENT_COMPLETE'
                save(output/'RESULT.json',result)
            entry=launch(output,action); result['stages'][action]=entry; save(output/'RESULT.json',result)
            if entry.get('root_exit_code')!=0 or not entry.get('reclaim_confirmed'):
                raise RuntimeError('bounded stage failed: '+action)
        result['status']='EXPERIMENT_COMPLETE'
    except Exception as error:
        result.update(status='FAILED',error=type(error).__name__+': '+str(error))
    finally:
        result['input_drift']=check_bindings(sources); result['implementation_drift']=check_bindings(code)
        if result['input_drift'] or result['implementation_drift']: result['status']='FAILED'
        save(output/'RESULT.json',result)
    return 0 if result['status']=='EXPERIMENT_COMPLETE' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','match','native','report'])
    parser.add_argument('--control',type=Path); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offline-evidence',type=Path,action='append',default=[])
    args=parser.parse_args(argv); output=args.output.resolve()
    if args.action=='run':
        if args.control is None: parser.error('--control required')
        return execute(args.control.resolve(strict=True),output,args.offline_evidence)
    plan=read(output/'PLAN.json'); state=read(output/'RESULT.json')
    if check_bindings(plan['source_bindings']+plan['implementation']): raise ValueError('frozen child bindings drift')
    if args.action=='match':
        from .crossid_regions import run
        run(plan['tracks_path'],output/'regions')
    elif args.action=='native':
        from .crossid_native import run
        run(plan['tracks_path'],output/'regions/SUPPLEMENT.json',output/'native')
    else:
        graph_audit(state['control_native_result_path'],state['native_result_path'],state['supplement_matches_path'],output/'GRAPH_AUDIT.json')
        from .crossid_region_report import render
        render(output)
    return 0
