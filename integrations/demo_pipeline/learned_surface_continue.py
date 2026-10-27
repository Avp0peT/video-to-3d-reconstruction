"""Continue a frozen learned surface adapter after completed export/bridge/masks."""
import argparse
from pathlib import Path
import sys
from .balanced_point_run import binding, check_bindings, read, save, sha
from .learned_surface_run import ROOT, TOOL_ROOT, clean, code_bindings, require
from .masked_trace_run import invoke


def audit(previous):
    previous=Path(previous).resolve(strict=True)
    inputs=read(previous/'INPUTS.json')
    refs=read(previous/'STAGE_BINDINGS.json')+inputs['source_bindings']
    require(not check_bindings(refs),'previous frozen inputs/code/stages drift')
    ledger=read(previous/'LAUNCH_LEDGER.json')
    require([r['stage'] for r in ledger]==['identity_export','bridge','masks'],'three completed stages required')
    for row in ledger:
        require(row['status']=='BUSINESS_GATE_PASSED' and clean(row['supervisor']), 'previous stage was not clean')
        require(read(previous/'processes'/row['stage']/'ENTRY.json')==row['supervisor'],'previous supervisor receipt differs')
    prior=read(previous/'RESULT.json')
    require(prior['status']=='FAILED' and prior['failed_stage']=='surface_plan','only failed plan preparation can continue')
    identity=read(previous/'identity_export/RESULT.json')
    bridge=read(previous/'bridge/RESULT.json')
    masks=read(previous/'masks/RESULT.json')
    require(identity['status']=='IDENTITY_CONSTRAINED_EXPERIMENTAL' and identity['source_unchanged'] is True,
            'identity export business status differs')
    require(bridge['status']=='PASS' and bridge['input_bindings_unchanged'] is True,'bridge business status differs')
    for name in ('UNDISTORT_COMPARISON.json','MVS_COMPARISON.json'):
        require(read(previous/'bridge'/name)['status']=='PASS','bridge camera/scene comparison failed')
    require(masks['status']=='AVAILABLE' and masks['available_views']==masks['registered_views']==54
            and masks['input_drift']==[], 'mask coverage differs')
    require(Path(masks['bridge']).resolve()==previous/'bridge'
            and Path(masks['identity_result']).resolve()==previous/'identity_export/RESULT.json', 'mask lineage differs')
    require(Path(identity['native_root']).resolve()==Path(inputs['native_run']).resolve(), 'native source differs')
    bp=read(previous/'bridge/PLAN.json')
    require(Path(bp['source_sparse']).resolve()==Path(identity['binary_path']).resolve()
            and Path(bp['source_images']).resolve()==Path(identity['image_root']).resolve(),'bridge lineage differs')
    refs+=identity['source_bindings']+identity['final_bindings']+masks['source_bindings']
    from integrations.mvs_surface.scene import read_scene
    scene=read_scene(previous/'bridge/scene.mvs')
    names={Path(i['name']).name for i in scene['images']}
    require(len(scene['images'])==54 and names=={i['name'] for i in identity['images']}
            =={r['native_name'] for r in masks['outputs']},'scene/export/mask view identity differs')
    require(all(i['name']=='undistorted/images/'+Path(i['name']).name for i in scene['images']),
            'reviewed bridge-relative scene path profile required')
    refs += [binding(previous/n) for n in ('INPUTS.json','STAGE_BINDINGS.json','RESULT.json','LAUNCH_LEDGER.json')]
    refs += [binding(previous/'processes'/row['stage']/'ENTRY.json') for row in ledger]
    require(not check_bindings(refs),'source chain changed during audit')
    return list({r['path']:r for r in refs}.values()),scene


def make_surface_plan(previous):
    from integrations.mvs_surface.runner import make_plan
    previous=Path(previous).resolve(strict=True)
    # Scene names already include undistorted/images/. The root must be bridge.
    return make_plan(previous/'bridge/scene.mvs',previous/'bridge',previous/'masks/masks',
        previous/'masks/validity',TOOL_ROOT,total_seconds=600,stage_seconds=480,
        memory_bytes=8 << 30,output_bytes=1 << 30,input_origin='unverified',ignore_label=0,photo_source=None)


def run(previous,output):
    previous=Path(previous).resolve(strict=True);out=Path(output).resolve()
    require(not out.exists(),'fresh continuation output required')
    require(not out.is_relative_to(previous),'continuation must not modify the previous run')
    refs,scene=audit(previous)
    refs += code_bindings()+[binding(__file__),binding(sys.executable)]
    refs=list({r['path']:r for r in refs}.values())
    require(not check_bindings(refs),'continuation input/code drift')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'INPUTS.json',dict(schema='learned-surface-continuation-inputs-v1',previous=str(previous),
         source_bindings=refs,reused_stages=['identity_export','bridge','masks'],python=sys.executable))
    refs.append(binding(out/'INPUTS.json'))
    ledger=[]
    result=dict(schema='learned-surface-continuation-result-v1',status='FAILED',geometry_accepted=False,
        quality_status='NOT_EVALUATED',mask_consumption='NOT_VERIFIED',physical_identity_verified=False,
        scale_status='unknown',cad_status='NOT_EVALUATED',stages=ledger)
    stage='surface_plan'
    try:
        plan=make_surface_plan(previous)
        from integrations.mvs_surface.runner import save_json,plan_files
        refs+=plan_files(plan)
        require(not check_bindings(refs),'inputs changed during plan preparation')
        path=out/'SURFACE_PLAN.json';save_json(path,plan,exclusive=True);refs.append(binding(path))
        save(out/'LEARNED_SOURCE_CHAIN.json',dict(schema='learned-surface-continuation-source-chain-v1',
            status='ADAPTER_SOURCE_BINDINGS_REVALIDATED',previous=str(previous),scene=binding(previous/'bridge/scene.mvs'),
            image_root=str(previous/'bridge'),scene_views=len(scene['images']),
            runner_input_origin='unverified',legacy_experimental_inputs_claimed=False,
            physical_identity_verified=False,geometry_accepted=False,mask_consumption='NOT_VERIFIED'))
        refs.append(binding(out/'LEARNED_SOURCE_CHAIN.json'))
        save(out/'STAGE_BINDINGS.json',refs)
        stage='surface'
        argv=[sys.executable,'-B','-X','utf8',str(ROOT/'scripts/build_mvs_surface.py'),'run',
            '--plan',str(path),'--plan-sha256',sha(path),'--output',str(out/'surface')]
        ledger.append(dict(stage=stage,status='INTENT_FROZEN',argv=argv,wall_seconds=660,job_commit_bytes=10 << 30))
        save(out/'LAUNCH_LEDGER.json',ledger)
        require(not check_bindings(refs),'pre-launch source/code drift')
        entry=invoke(out,'surface',argv,660,10 << 30)
        ledger[0].update(status='SUPERVISOR_RETURNED',supervisor=entry)
        save(out/'LAUNCH_LEDGER.json',ledger)
        require(clean(entry),'surface supervision was not clean')
        require(not check_bindings(refs),'post-launch source/code drift')
        surface=read(out/'surface/result.json')
        require(surface['status']=='SURFACE_GENERATED_NOT_ACCEPTED' and surface['exit_code']==0,'surface business status failed')
        ledger[0]['status']='BUSINESS_GATE_PASSED'
        result.update(status='SURFACE_GENERATED_NOT_ACCEPTED',surface_result=binding(out/'surface/result.json'))
    except Exception as exc:
        result.update(failed_stage=stage,error=type(exc).__name__+': '+str(exc))
        if ledger:ledger[-1]['status']='FAILED'
    result['source_drift']=check_bindings(refs)
    if result['source_drift']:result['status']='INPUT_DRIFT'
    save(out/'LAUNCH_LEDGER.json',ledger);save(out/'RESULT.json',result)
    save(out/'OUTPUT_BINDINGS.json',[binding(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='OUTPUT_BINDINGS.json'])
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    result=run(args.previous,args.output);print(result['status'],flush=True)
    return 0 if result['status']=='SURFACE_GENERATED_NOT_ACCEPTED' else 1

if __name__=='__main__':raise SystemExit(main())
