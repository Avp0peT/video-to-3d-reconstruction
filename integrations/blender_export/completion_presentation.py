"""Two bounded isolated Blender entries for semantic completion presentation."""
from pathlib import Path
import json,sys
from .isolated import _binding,_snapshot,OUTPUTS
from .export import find_blender,_stage_ok
from integrations.mvs_surface.supervision import run_supervised

STYLES={'Inferred complete envelope','Observed-depth reconstruction'}
LIMIT=128<<20


def _save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def style_completion_blend(base_blend,payload_path,output,*,scratch_root):
    """Create a new styled file, then independently reopen and check it.

    Scratch must be under outputs/runtime_scratch, outside the artifact RUN.
    Caller remains responsible for any broader enclosing monitored process tree.
    Existing source files are bound before execution and never overwritten.
    """
    base=Path(base_blend).resolve(strict=True);payload=Path(payload_path).resolve(strict=True)
    out=Path(output).resolve();scratch=Path(scratch_root).resolve();runtime=(OUTPUTS/'runtime_scratch').resolve()
    if out.exists() or not out.is_relative_to(OUTPUTS.resolve()) or out.is_relative_to(runtime):raise ValueError('fresh artifact output outside runtime_scratch required')
    if not scratch.is_relative_to(runtime) or scratch==runtime:raise ValueError('private scratch root under outputs/runtime_scratch required')
    for path in (base,payload,out):
        if scratch.is_relative_to(path) or path.is_relative_to(scratch):raise ValueError('scratch and all input/output paths must be disjoint')
    if base.is_relative_to(out) or payload.is_relative_to(out):raise ValueError('output overlaps source inputs')
    if base.stat().st_size>LIMIT or payload.stat().st_size>LIMIT:raise ValueError('base/payload capacity exceeded')
    data=json.loads(payload.read_text(encoding='utf-8'));scenes=data.get('scenes')
    if not isinstance(scenes,list) or len(scenes)!=2 or {s.get('title') for s in scenes}!=STYLES:raise ValueError('exact observed/inferred payload titles required')
    if sum(len(s['vertices']) for s in scenes)>200000 or sum(len(s['faces']) for s in scenes)>400000:raise ValueError('payload geometry budget exceeded')
    worker=Path(__file__).with_name('completion_presentation_worker.py');legacy=worker.with_name('worker.py');helper=worker.with_name('isolated_launcher.py')
    executable=Path(find_blender()).resolve(strict=True)
    bound=[_binding(path) for path in (base,payload,worker,legacy,helper,Path(__file__).resolve(),Path(sys.executable).resolve(),executable)]
    for scene in scenes:
        source=_binding(scene['source_path'])
        if source['sha256']!=scene['sha256']:raise ValueError('payload mesh source differs')
        bound.append(source)
        for item in scene.get('evidence',[]):
            record=_binding(item['path'])
            if record['sha256']!=item['sha256']:raise ValueError('payload evidence source differs')
            bound.append(record)
    out.mkdir(parents=True,exist_ok=False);blend=out/'model.blend'
    report=dict(status='PARTIAL',geometry_accepted=False,geometry_changed=False,metadata_changed=False,base_blend=_binding(base),payload=_binding(payload),input_bindings=bound,stages=[],limits=dict(stage_seconds=90,memory_bytes=4<<30,artifact_bytes=LIMIT,maximum_native_launches=2),scratch_root=str(scratch),appearance='Inferred orange visible; observed blue hidden; not a geometry or completeness acceptance')
    try:
        for mode in ('create','verify'):
            stage=out/mode;stage.mkdir();temp=scratch/('presentation_'+mode)
            if temp.exists():raise ValueError('fresh stage scratch required')
            temp.mkdir(parents=True,exist_ok=False);before=_snapshot(temp)
            target=out/(mode.upper()+'.json')
            argv=[str(executable),'--background','--factory-startup','--disable-autoexec','--offline-mode','--threads','2','--python-exit-code','7','--python',str(worker),'--',mode,str(base),str(payload),report['payload']['sha256'],str(blend),str(target),str(legacy),_binding(legacy)['sha256']]
            request=dict(blender_argv=argv,cwd=str(out),scratch=str(temp),child_timeout_seconds=80.,result_path=str(stage/'child_result.json'))
            request_path=stage/'request.json';_save(request_path,request);request_binding=_binding(request_path)
            supervised=run_supervised([sys.executable,str(helper),str(request_path)],cwd=str(out),output=str(stage),wall_seconds=90,memory_bytes=4<<30,output_bytes=LIMIT,label='completion_presentation_'+mode)
            _save(stage/'supervision.json',supervised)
            record=dict(mode=mode,supervision=supervised,request_binding=request_binding,scratch_before=before,scratch_after=_snapshot(temp),scratch_observation='Endpoint snapshots only, not continuous scratch disk monitoring')
            report['stages'].append(record)
            if not _stage_ok(supervised):raise RuntimeError(mode+' did not exit successfully with confirmed reclamation')
            checked=json.loads(target.read_text(encoding='utf-8'))
            if checked.get('status')!='PASS' or checked.get('payload_sha256')!=report['payload']['sha256']:raise ValueError('worker report/payload mismatch')
            record['worker_report']=_binding(target)
            if mode=='create':created=checked;styled_binding=_binding(blend)
            else:
                if any(checked.get(key)!=created.get(key) for key in ('scenes','presentation')):raise ValueError('independent presentation readback changed')
                if _binding(blend)!=styled_binding:raise ValueError('styled blend changed during readback')
            if _binding(request_path)!=request_binding:raise ValueError('bound launch request changed')
            if any(_binding(item['path'])!=item for item in bound):raise ValueError('presentation input/code changed')
        if sum(path.stat().st_size for path in out.rglob('*') if path.is_file())>LIMIT:raise ValueError('presentation output exceeds artifact bound')
        report.update(status='SUCCESS',blend=_binding(blend),scenes=checked['scenes'],presentation=checked['presentation'],base_blend_unchanged=True,readback_verified=True)
        report_bytes=len(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'))
        if sum(path.stat().st_size for path in out.rglob('*') if path.is_file())+2*report_bytes>LIMIT:raise ValueError('artifact bound including final binding report exceeded')
    except Exception as exc:
        report.update(status='FAILED_PRESENTATION_NOT_ADOPTED',error=type(exc).__name__+': '+str(exc))
    _save(out/'BINDINGS.json',report)
    return report
