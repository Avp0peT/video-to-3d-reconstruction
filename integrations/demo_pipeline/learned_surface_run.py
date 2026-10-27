"""Single-pass learned experimental surface adapter; no acceptance upgrade."""
import argparse
from pathlib import Path
import sys
from .balanced_point_run import binding, check_bindings, read, save, sha
from .masked_trace_run import invoke

ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = ROOT / 'outputs/toolchains/openmvs-2.4.0-cpu/vc17/x64/Release'


def clean(entry):
    return (entry.get('status') == 'completed' and type(entry.get('root_exit_code')) is int
            and entry['root_exit_code'] == 0 and entry.get('reclaim_confirmed') is True
            and entry.get('errors') == [] and entry.get('log_errors') == [])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def code_bindings():
    paths = [ROOT / p for p in ('scripts/reconstruct_learned_surface.py',
        'integrations/demo_pipeline/learned_surface_run.py',
        'integrations/demo_pipeline/learned_identity_export.py',
        'integrations/demo_pipeline/learned_mvs_masks.py',
        'integrations/demo_pipeline/learned_silhouette_hull.py',
        'integrations/demo_pipeline/connectivity_native.py',
        'integrations/demo_pipeline/masked_trace_run.py',
        'integrations/demo_pipeline/supervision.py',
        'integrations/demo_pipeline/balanced_point_run.py',
        'scripts/prepare_mvs.py', 'scripts/build_mvs_surface.py')]
    for folder in ('integrations/mvs_surface', 'integrations/mvs_masks'):
        paths.extend(sorted((ROOT / folder).glob('*.py')))
    return [binding(p) for p in paths]


def run(diagnostic, native_run, masks, output):
    diagnostic, native_run, masks = [Path(p).resolve(strict=True) for p in (diagnostic, native_run, masks)]
    out = Path(output).resolve()
    require(not out.exists(), 'fresh output directory required; no retry or overwrite')
    initial = [binding(p) for p in (diagnostic, native_run/'PLAN.json', native_run/'RESULT.json', masks, Path(sys.executable))]
    initial += read(diagnostic)['source_bindings']
    initial += code_bindings()
    initial += [binding(TOOL_ROOT / n) for n in ('InterfaceCOLMAP.exe', 'DensifyPointCloud.exe', 'ReconstructMesh.exe')]
    require(not check_bindings(initial), 'initial source/code drift')
    out.mkdir(parents=True, exist_ok=False)
    refs = list({r['path']: r for r in initial}.values())
    frozen = dict(schema='learned-surface-adapter-inputs-v1', python=sys.executable,
        diagnostic=str(diagnostic), native_run=str(native_run), masks=str(masks),
        source_bindings=refs, source_kind='learned-experimental-identity-constrained',
        physical_identity_verified=False, geometry_accepted=False,
        limitation='Adapter audits its own learned source chain; no legacy experimental_inputs schema assertion.')
    save(out/'INPUTS.json', frozen)
    refs.append(binding(out/'INPUTS.json'))
    ledger = []
    result = dict(schema='learned-surface-adapter-result-v1', status='FAILED', stages=ledger,
        geometry_accepted=False, quality_status='NOT_EVALUATED', physical_identity_verified=False,
        mask_consumption='NOT_VERIFIED', scale_status='unknown', cad_status='NOT_EVALUATED')
    stage = 'admission'
    def unchanged():
        require(not check_bindings(refs), 'frozen source/code/generated input drift')
    def launch(label, args, seconds, memory):
        nonlocal stage
        stage = label
        unchanged()
        item = dict(stage=label, status='INTENT_FROZEN', argv=[sys.executable, '-B', '-X', 'utf8', *map(str,args)],
                    wall_seconds=seconds, job_commit_bytes=memory)
        ledger.append(item)
        save(out/'LAUNCH_LEDGER.json', ledger)
        entry = invoke(out, label, item['argv'], seconds, memory)
        item.update(status='SUPERVISOR_RETURNED', supervisor=entry)
        save(out/'LAUNCH_LEDGER.json', ledger)
        require(clean(entry), label + ' supervision was not clean')
        unchanged()
    def accept(folder):
        refs.extend(binding(p) for p in sorted(folder.rglob('*')) if p.is_file())
        ledger[-1]['status'] = 'BUSINESS_GATE_PASSED'
        save(out/'LAUNCH_LEDGER.json', ledger)
        save(out/'STAGE_BINDINGS.json', refs)
    try:
        export = out/'identity_export'
        launch('identity_export', ['-m', 'integrations.demo_pipeline.learned_identity_export',
            '--diagnostic', diagnostic, '--native-run', native_run, '--output', export], 60, 4 << 30)
        identity = read(export/'RESULT.json')
        require(identity['status']=='IDENTITY_CONSTRAINED_EXPERIMENTAL' and identity['source_unchanged'] is True,
                'identity export business gate failed')
        require(Path(identity['native_root']).resolve()==native_run and Path(identity['binary_path']).resolve()==export/'binary'
                and Path(identity['image_root']).resolve()==native_run/'images', 'export source identity mismatch')
        require(not check_bindings(identity['source_bindings']+identity['final_bindings']), 'export binding drift')
        accept(export)
        bridge = out/'bridge'
        launch('bridge', [ROOT/'scripts/prepare_mvs.py', '--sparse', export/'binary', '--image-root', native_run/'images',
            '--interface-colmap', TOOL_ROOT/'InterfaceCOLMAP.exe', '--output', bridge,
            '--wall-seconds', '210', '--max-output-mib', '1024'], 240, 8 << 30)
        bridged = read(bridge/'RESULT.json')
        require(bridged['status']=='PASS' and bridged['input_bindings_unchanged'] is True, 'bridge business gate failed')
        for name in ('UNDISTORT_COMPARISON.json','MVS_COMPARISON.json'):
            require(read(bridge/name)['status']=='PASS', 'bridge comparison failed')
        accept(bridge)
        consumer = out/'masks'
        launch('masks', ['-m','integrations.demo_pipeline.learned_mvs_masks','--bridge',bridge,
            '--identity-result',export/'RESULT.json','--native-plan',native_run/'PLAN.json',
            '--masks',masks,'--output',consumer],120,4 << 30)
        consumed = read(consumer/'RESULT.json')
        require(consumed['status']=='AVAILABLE' and consumed['available_views']==consumed['registered_views']==54
                and consumed['input_drift']==[], 'mask business gate failed')
        require(Path(consumed['identity_result']).resolve()==export/'RESULT.json'
                and Path(consumed['bridge']).resolve()==bridge, 'mask source identity mismatch')
        require(not check_bindings(consumed['source_bindings']), 'mask source binding drift')
        accept(consumer)
        stage = 'surface_plan'
        unchanged()
        from integrations.mvs_surface.runner import make_plan, save_json
        plan = make_plan(bridge/'scene.mvs', bridge/'undistorted/images', consumer/'masks',consumer/'validity',TOOL_ROOT,
            total_seconds=600,stage_seconds=480,memory_bytes=8 << 30,output_bytes=1 << 30,
            ignore_label=0,input_origin='unverified',photo_source=None)
        # Preserve the runner's returned plan verbatim, including source status.
        plan_path=out/'SURFACE_PLAN.json'
        save_json(plan_path,plan,exclusive=True)
        refs.append(binding(plan_path))
        save(out/'LEARNED_SOURCE_CHAIN.json',dict(schema='learned-surface-adapter-source-chain-v1',
            status='ADAPTER_SOURCE_BINDINGS_REVALIDATED', source_kind=frozen['source_kind'],
            identity_result=binding(export/'RESULT.json'), bridge_result=binding(bridge/'RESULT.json'),
            masks_result=binding(consumer/'RESULT.json'), surface_plan=binding(plan_path),
            runner_input_origin='unverified', legacy_experimental_inputs_claimed=False,
            physical_identity_verified=False,geometry_accepted=False,mask_consumption='NOT_VERIFIED'))
        refs.append(binding(out/'LEARNED_SOURCE_CHAIN.json'))
        save(out/'STAGE_BINDINGS.json',refs)
        launch('surface',[ROOT/'scripts/build_mvs_surface.py','run','--plan',plan_path,
            '--plan-sha256',sha(plan_path),'--output',out/'surface'],660,10 << 30)
        surface=read(out/'surface/result.json')
        require(surface['status']=='SURFACE_GENERATED_NOT_ACCEPTED' and surface['exit_code']==0,
                'surface business gate failed')
        accept(out/'surface')
        result.update(status='SURFACE_GENERATED_NOT_ACCEPTED',surface_result=binding(out/'surface/result.json'))
    except Exception as error:
        result.update(failed_stage=stage,error=type(error).__name__+': '+str(error))
        if ledger and ledger[-1]['status']!='BUSINESS_GATE_PASSED':
            ledger[-1]['status']='FAILED'
        save(out/'LAUNCH_LEDGER.json',ledger)
    result['source_drift']=check_bindings(refs)
    if result['source_drift']:
        result.update(status='INPUT_DRIFT')
    save(out/'RESULT.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic','native-run','masks','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args(argv)
    result=run(args.diagnostic,args.native_run,args.masks,args.output)
    print(result['status'],flush=True)
    return 0 if result['status']=='SURFACE_GENERATED_NOT_ACCEPTED' else 1
