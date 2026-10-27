"""Private legacy-export adapter isolating volatile native temp from artifact trees."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import uuid
from integrations.mvs_surface.supervision import run_supervised

ROOT=Path(__file__).resolve().parents[2]
OUTPUTS=ROOT/'outputs'


def _binding(path):
    path=Path(path).resolve(strict=True);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def _save(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def _snapshot(path):
    count=0;size=0;entries=0
    for item in path.rglob('*'):
        entries+=1
        if entries>10000:raise RuntimeError('scratch endpoint snapshot entry budget exceeded')
        if item.is_symlink():raise ValueError('scratch symlinks are forbidden')
        if item.is_file():
            count+=1;size+=item.stat().st_size
            if count>10000 or size>1<<30:raise RuntimeError('scratch endpoint snapshot budget exceeded')
    return dict(files=count,entries=entries,bytes=size,max_entries=10000,max_bytes=1<<30)


def run_blender_isolated(argv,*,cwd,output,scratch_root,wall_seconds,memory_bytes,output_bytes,label):
    """Preserve original Job supervision, substituting a controlled helper child.

    Caller must place scratch outside any OUTER monitored run, too. This adapter
    can verify separation from its own cwd/output, not infer external observers.
    Scratch has bounded endpoint snapshots but NO continuous disk observation.
    """
    if not isinstance(argv,(list,tuple)) or not argv or any(not isinstance(x,str) or '\x00' in x for x in argv):raise ValueError('explicit NUL-free argv required')
    if type(wall_seconds) not in (int,float) or not 5<wall_seconds<=90:raise ValueError('stage wall bound must be >5 and <=90 seconds')
    if type(memory_bytes) is not int or not 0<memory_bytes<=4<<30 or type(output_bytes) is not int or not 0<output_bytes<=128<<20:raise ValueError('invalid memory/artifact bounds')
    if not isinstance(label,str) or not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in label):raise ValueError('safe stage label required')
    work=Path(cwd).resolve(strict=True);stage=Path(output).resolve(strict=True);scratch=Path(scratch_root).resolve()
    for path in (work,stage,scratch):
        if not path.is_relative_to(OUTPUTS.resolve()):raise ValueError('all paths must remain under repository outputs')
    if not stage.is_relative_to(work):raise ValueError('stage must lie inside monitored cwd')
    if scratch.is_relative_to(work) or work.is_relative_to(scratch):raise ValueError('scratch and monitored output must be disjoint, not ancestors')
    stage_scratch=scratch/label
    if stage_scratch.exists():raise ValueError('per-stage scratch must be fresh')
    helper=Path(__file__).with_name('isolated_launcher.py');legacy=Path(__file__).with_name('export.py');worker=Path(__file__).with_name('worker.py')
    bindings=dict(blender_executable=_binding(argv[0]),helper=_binding(helper),legacy_export=_binding(legacy),worker=_binding(worker),adapter=_binding(__file__),python_executable=_binding(sys.executable))
    if '--python' not in argv or argv.index('--python')+1>=len(argv) or Path(argv[argv.index('--python')+1]).resolve()!=worker.resolve():raise ValueError('only the bound legacy Blender worker is supported')
    stage_scratch.mkdir(parents=True,exist_ok=False)
    before=_snapshot(stage_scratch)
    request=dict(blender_argv=list(argv),cwd=str(work),scratch=str(stage_scratch),child_timeout_seconds=min(80.,float(wall_seconds)-5.),result_path=str(stage/'isolated_child_result.json'))
    request_path=stage/'isolated_request.json';_save(request_path,request)
    record=dict(schema='blender-temp-isolation-v1',original_blender_argv=list(argv),bindings=bindings,request_binding=_binding(request_path),
        environment_overrides={'TMP':str(stage_scratch),'TEMP':str(stage_scratch)},scratch_before=before,
        scratch_observation='Endpoint snapshots only; transient scratch growth is not continuously monitored. Scratch is retained, never deleted.',
        outer_scope='Caller must additionally ensure scratch lies outside every outer monitored RUN.',
        job_semantics='Helper and native child inherit original Job; no shell or breakaway; stdout/stderr inherited',
        limits=dict(stage_wall_seconds=wall_seconds,child_timeout_seconds=request['child_timeout_seconds'],memory_bytes=memory_bytes,artifact_output_bytes=output_bytes))
    _save(stage/'ISOLATION_PLAN.json',record)
    result=run_supervised([sys.executable,str(helper),str(request_path)],cwd=str(work),output=str(stage),wall_seconds=wall_seconds,memory_bytes=memory_bytes,output_bytes=output_bytes,label=label)
    # Preserve actual supervision even if the later scratch snapshot detects an error.
    _save(stage/'ISOLATION_SUPERVISION.json',result)
    after=_snapshot(stage_scratch)
    for bound in bindings.values():
        if _binding(bound['path'])!=bound:raise ValueError('isolation code or executable changed')
    if _binding(request_path)!=record['request_binding']:raise ValueError('isolation request changed')
    record.update(scratch_after=after,supervisor_root_exit_code=result.get('root_exit_code'))
    if Path(request['result_path']).is_file():record['child_result_binding']=_binding(request['result_path'])
    _save(stage/'ISOLATION_RESULT.json',record)
    return dict(result,temp_isolation=record)


def export_blend_isolated(spec,output,*,scratch_root,profile='standard'):
    """Use a fresh private legacy module; no global monkeypatching or core edits."""
    path=Path(__file__).with_name('export.py')
    spec_module=importlib.util.spec_from_file_location('_isolated_blender_export_'+uuid.uuid4().hex,path)
    module=importlib.util.module_from_spec(spec_module);spec_module.loader.exec_module(module)
    def supervise(argv,**kwargs):return run_blender_isolated(argv,scratch_root=scratch_root,**kwargs)
    module.run_supervised=supervise
    return module.export_blend(spec,output,profile=profile)
