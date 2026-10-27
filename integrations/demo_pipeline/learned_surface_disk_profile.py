"""Versioned disk4g profile: prepare, then launch exactly one fresh surface."""
import argparse
import copy
from pathlib import Path
import sys
from .balanced_point_run import binding,check_bindings,read,save,sha
from .learned_surface_continue import audit
from .learned_surface_run import ROOT,TOOL_ROOT,code_bindings,clean,require
from integrations.mvs_surface.supervision import run_supervised

PROFILE='learned-surface-disk4g-v1'
INNER_BYTES=4 << 30
OUTER_BYTES=6 << 30


def validate_disk_failure(failed,previous):
    failed=Path(failed).resolve(strict=True);previous=Path(previous).resolve(strict=True)
    inputs=read(failed/'INPUTS.json');outputs=read(failed/'OUTPUT_BINDINGS.json')
    require(Path(inputs['previous']).resolve()==previous,'failed run source differs')
    refs=inputs['source_bindings']+outputs+[binding(failed/'INPUTS.json'),binding(failed/'OUTPUT_BINDINGS.json')]
    require(not check_bindings(refs),'failed-run INPUTS/OUTPUT_BINDINGS drift')
    result=read(failed/'surface/result.json')
    require(result['status']=='FAILED' and len(result['stages'])==1
            and result['stages'][0]['name']=='dense','expected dense-only failure')
    supervisor_ref=result['stages'][0]['supervisor']
    require(not check_bindings([supervisor_ref]),'dense supervisor evidence drift')
    native=read(supervisor_ref['path'])
    require(native['status']==native['termination_reason']=='run_dir_threshold'
            and native['run_dir_threshold_bytes']==1 << 30 and native['reclaim_confirmed'] is True
            and native['errors']==[] and native['log_errors']==[], 'failure was not a cleanly reclaimed directory stop')
    require(any(s.get('run_dir_bytes',0)>1 << 30 for s in native['samples']), 'directory threshold crossing absent')
    require(native['wall_clock_seconds']==480 and native['job_memory_limit_bytes']==8 << 30
            and native['peak_job_memory_used_bytes']<8 << 30,'unexpected time/memory profile')
    outer=read(failed/'RESULT.json')['stages'][0]['supervisor']
    require(outer['status']=='completed' and outer['root_exit_code']==4 and outer['reclaim_confirmed'] is True
            and outer['errors']==[] and outer['log_errors']==[], 'failed attempt outer reclaim differs')
    return refs


def compare_disk_only(old,new):
    old=copy.deepcopy(old);new=copy.deepcopy(new)
    require(old['arguments']['output_bytes']==old['budgets']['output_bytes']==1 << 30,'prior disk budget differs')
    require(new['arguments']['output_bytes']==new['budgets']['output_bytes']==INNER_BYTES,'new disk budget differs')
    for plan in (old,new):
        plan['arguments'].pop('output_bytes');plan['budgets'].pop('output_bytes')
    require(old==new,'disk4g may change only output_bytes; algorithm/time/memory/source identity differs')


def prepare(previous,failed_run,output):
    previous=Path(previous).resolve(strict=True);failed=Path(failed_run).resolve(strict=True);out=Path(output).resolve()
    require(not out.exists(),'fresh disk4g output required')
    require(not out.is_relative_to(previous) and not out.is_relative_to(failed),'output must be outside old runs')
    refs,scene=audit(previous)
    refs+=validate_disk_failure(failed,previous)+code_bindings()+[binding(__file__),binding(sys.executable),
        binding(ROOT/'integrations/demo_pipeline/learned_surface_continue.py')]
    from integrations.mvs_surface.runner import make_plan,save_json,plan_files
    plan=make_plan(previous/'bridge/scene.mvs',previous/'bridge',previous/'masks/masks',previous/'masks/validity',
        TOOL_ROOT,total_seconds=600,stage_seconds=480,memory_bytes=8 << 30,output_bytes=INNER_BYTES,
        input_origin='unverified',ignore_label=0,photo_source=None)
    compare_disk_only(read(failed/'SURFACE_PLAN.json'),plan)
    refs+=plan_files(plan)
    require(not check_bindings(refs),'preparation input/code drift')
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/'SURFACE_PLAN.json',plan,exclusive=True)
    bootstrap=("import os,sys,runpy\n"
        +"os.environ['PYTHONPATH']="+repr(str(ROOT)+';'+str(ROOT/'src'))+"\n"
        +"for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='2'\n"
        +"sys.path[:0]="+repr([str(ROOT),str(ROOT/'src')])+"\n"
        +"sys.argv[0]="+repr(str(ROOT/'scripts/build_mvs_surface.py'))+"\n"
        +"runpy.run_path(sys.argv[0],run_name='__main__')\n")
    (out/'worker_bootstrap.py').write_text(bootstrap,encoding='utf-8')
    save(out/'LEARNED_SOURCE_CHAIN.json',dict(schema=PROFILE,previous=str(previous),failed_run=str(failed),
        source_kind='learned-experimental-identity-constrained',adapter_bindings_revalidated=True,
        reused_stages=['identity_export','bridge','masks'],old_native_cache_reused=False,
        runner_input_origin='unverified',legacy_experimental_inputs_claimed=False,
        physical_identity_verified=False,geometry_accepted=False,mask_consumption='NOT_VERIFIED',scene_views=len(scene['images'])))
    refs += [binding(out/name) for name in ('SURFACE_PLAN.json','worker_bootstrap.py','LEARNED_SOURCE_CHAIN.json')]
    save(out/'INPUTS.json',dict(schema=PROFILE,python=sys.executable,previous=str(previous),failed_run=str(failed),
        source_bindings=list({r['path']:r for r in refs}.values()),outer_wall_seconds=660,
        outer_memory_bytes=10 << 30,outer_output_bytes=OUTER_BYTES,monitored_root=str(out)))
    save(out/'PREPARED_BINDINGS.json',[binding(out/'INPUTS.json')])
    save(out/'RESULT.json',dict(schema=PROFILE,status='PREPARED',native_launches=0,geometry_accepted=False))
    return plan


def run(output):
    out=Path(output).resolve(strict=True)
    inputs=read(out/'INPUTS.json')
    require(inputs['schema']==PROFILE and inputs['monitored_root']==str(out),'disk4g input profile differs')
    refs=inputs['source_bindings']+read(out/'PREPARED_BINDINGS.json')+[binding(out/'PREPARED_BINDINGS.json')]
    require(not check_bindings(refs),'frozen disk4g input/code drift')
    require(not (out/'outer').exists() and not (out/'surface').exists(),'disk4g is single execution; old native caches forbidden')
    argv=[inputs['python'],'-B','-X','utf8',str(out/'worker_bootstrap.py'),'run','--plan',str(out/'SURFACE_PLAN.json'),
        '--plan-sha256',sha(out/'SURFACE_PLAN.json'),'--output',str(out/'surface')]
    ledger=[dict(stage='surface',status='INTENT_FROZEN',argv=argv,cwd=str(out),evidence=str(out/'outer'),
                 wall_seconds=660,memory_bytes=10 << 30,output_bytes=OUTER_BYTES)]
    save(out/'LAUNCH_LEDGER.json',ledger)
    (out/'outer').mkdir(exist_ok=False)
    result=dict(schema=PROFILE,status='FAILED',stages=ledger,geometry_accepted=False,
        quality_status='NOT_EVALUATED',physical_identity_verified=False,mask_consumption='NOT_VERIFIED',
        scale_status='unknown',cad_status='NOT_EVALUATED',old_native_cache_reused=False)
    try:
        entry=run_supervised(argv,cwd=out,output=out/'outer',wall_seconds=660,memory_bytes=10 << 30,
                             output_bytes=OUTER_BYTES,label='learned_surface_disk4g')
        save(out/'outer/ENTRY.json',entry)
        ledger[0].update(status='SUPERVISOR_RETURNED',supervisor=entry)
        save(out/'LAUNCH_LEDGER.json',ledger)
        require(clean(entry),'disk4g outer supervision was not clean')
        require(not check_bindings(refs),'disk4g post-execution input/code drift')
        surface=read(out/'surface/result.json')
        require(surface['status']=='SURFACE_GENERATED_NOT_ACCEPTED' and surface['exit_code']==0,'surface business gate failed')
        ledger[0]['status']='BUSINESS_GATE_PASSED'
        result.update(status='SURFACE_GENERATED_NOT_ACCEPTED',surface_result=binding(out/'surface/result.json'))
    except Exception as exc:
        result['error']=type(exc).__name__+': '+str(exc);ledger[0]['status']='FAILED'
    result['source_drift']=check_bindings(refs)
    if result['source_drift']:result['status']='INPUT_DRIFT'
    save(out/'LAUNCH_LEDGER.json',ledger);save(out/'RESULT.json',result)
    save(out/'OUTPUT_BINDINGS.json',[binding(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='OUTPUT_BINDINGS.json'])
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for name in ('previous','failed-run','output'):prep.add_argument('--'+name,required=True,type=Path)
    execute=sub.add_parser('run');execute.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    if a.action=='prepare':prepare(a.previous,a.failed_run,a.output);print('PREPARED disk4g');return 0
    result=run(a.output);print(result['status']);return 0 if result['status']=='SURFACE_GENERATED_NOT_ACCEPTED' else 1

if __name__=='__main__':raise SystemExit(main())
