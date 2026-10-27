"""Bind a mesh to a declared producer output; never certify photo provenance."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

MAX_JSON=1024**2
MAX_MESH=64*1024**2
SURFACE='mvs-surface-run-v1'
PROFILE='openmvs-2.4.0-cpu-preserve-raw-v1'
CONTINUATION='mvs-mesh-continuation-source-v1'

def continuation_ledger(result_path):
    """Recompute a completed one-shot continuation's file associations, never run it.

    This is deliberately a narrow contract for the accepted missing-log failure.
    Supervisor JSON is declared evidence, not authenticated native execution.
    """
    import re
    import sys
    root=Path(__file__).resolve().parents[2]
    before=list(sys.path)
    try:
        sys.path.insert(0,str(root))
        from integrations.mvs_surface import runner
        for name in ('integrations.mvs_surface.runner','integrations.mvs_surface.scene',
                     'integrations.mvs_surface.photo_inputs','integrations.viewer.ply_data'):
            module=sys.modules[name]
            expected=root.joinpath(*name.split('.')).with_suffix('.py')
            need(Path(module.__file__).resolve()==expected.resolve(),'imported source escapes project: '+name)
    finally:
        sys.path[:]=before
    checked={}
    def check(row,limit=512*1024**2):
        p=_path(row);actual=binding(p,limit)
        need(all(actual[k]==row[k] for k in actual),'continuation artifact drift: '+str(p))
        checked[actual['path']]=actual
        return actual
    def doc(path):
        value,b=document(path);checked[b['path']]=b;return value,b
    result,rb=doc(result_path);directory=Path(rb['path']).parent;package=directory.parent
    seal,sb=doc(package/'BINDINGS.json');auth,ab=doc(package/'MAIN_AUTHORIZATION.json')
    history,hb=doc(directory/'historical_plan_reference.json')
    need(result.get('contract_version')=='mvs-mesh-continuation-v1','continuation version mismatch')
    need(result.get('status')=='MESH_GENERATED_NOT_ACCEPTED' and type(result.get('exit_code')) is int and result['exit_code']==0,'continuation failed')
    for key,value in [('dense_launches',0),('native_launches',1),('known_native_launches',1),('launch_attempts',1)]:
        need(type(result.get(key)) is int and result[key]==value,'continuation launch count mismatch')
    need(result.get('original_input_drift')==result.get('execution_code_drift')==[],'continuation stability missing')
    need(result.get('scale_status')=='unknown' and result.get('quality_status')=='NOT_EVALUATED'
         and result.get('mask_consumption')=='NOT_VERIFIED' and result.get('CAD_status')=='NOT_RUN','continuation evidence scope changed')
    need(seal.get('schema')=='one-shot-mesh-continuation-preparation-v1' and seal.get('real_execution_authorized') is True
         and seal.get('implementation_finalized') is True and seal.get('b_log_review_accepted') is True
         and type(seal.get('allowed_native_entries')) is int and seal['allowed_native_entries']==1,'continuation seal missing')
    need(Path(seal['required_output']).resolve()==directory,'continuation output mismatch')
    entry=check(binding(package/'mesh_once.py',MAX_JSON))
    need(entry['sha256']==seal.get('entry_script_sha256')==auth.get('entry_script_sha256')
         and sb['sha256']==auth.get('authorized_binding_sha256') and auth.get('native_entries')==1
         and auth.get('root_mismatch_count')==0,'authorization binding mismatch')
    originals=seal.get('original_inputs');need(isinstance(originals,list) and 0<len(originals)<=1024,'original bindings missing')
    for row in originals:check(row)
    old,ob=doc(check(seal['old_result'])['path']);plan,pb=doc(history['path'])
    need(any(row==pb for row in originals) and history.get('sha256')==pb['sha256'],'historical plan not sealed')
    need(plan.get('contract_version')==SURFACE and plan.get('profile')==PROFILE
         and plan.get('input_origin')=='caller_declared_real' and len(plan.get('frames',[]))==43,'historical plan mismatch')
    need(history.get('implementation')==plan.get('implementation')==seal.get('historical_implementation'),'historical implementation mismatch')
    execution=seal.get('execution_implementation');need(isinstance(execution,list) and len(execution)==len(plan['implementation']),'execution seal incomplete')
    oldcode={r['path']:r for r in plan['implementation']};newcode={r['path']:r for r in execution}
    need(len(newcode)==len(execution) and set(newcode)==set(oldcode),'execution implementation set differs')
    fix=seal['accepted_B_fix'];final,_=doc(check(fix['final_hashes'])['path']);check(fix['handoff']);check(seal['b_log_review'])
    for row in final['artifacts']:check(row)
    products={r['path']:r for r in final['product_files']}
    for path,row in newcode.items():
        check(row)
        if row!=oldcode[path]:
            accepted=products.get(path,{})
            need(Path(path).name=='runner.py' and accepted.get('before_sha256')==oldcode[path]['sha256']
                 and all(accepted.get(k)==row[k] for k in ('path','bytes','sha256'))
                 and row['sha256']==fix['runner_sha256'],'unaccepted historical implementation change')
    failure='ValueError: dense mask log guard: required native DensifyPointCloud log missing'
    need(old.get('status')=='FAILED' and type(old.get('exit_code')) is int and old['exit_code']==4
         and old.get('error')==failure and old.get('input_drift')==[] and old.get('native_launches')==1
         and old.get('launch_attempts')==1 and old.get('launch_count_complete') is True,'historical failure differs')
    def stage(record,name):
        steps=record.get('stages');need(isinstance(steps,list) and len(steps)==1 and isinstance(steps[0],dict),'single stage required')
        s=steps[0];need(s.get('name')==name and type(s.get('returncode')) is int and s['returncode']==0,'stage exit mismatch')
        native,_=doc(check(s['supervisor'])['path'])
        need(native.get('status')=='completed' and type(native.get('root_exit_code')) is int and native['root_exit_code']==0
             and native.get('reclaim_confirmed') is True and native.get('errors')==native.get('log_errors')==[]
             and type(native.get('pid')) is int and native['pid']>0 and native['pid']==s.get('pid')
             and native.get('argv')==s.get('argv') and native.get('cwd')==s.get('cwd'),'supervisor association mismatch')
        evidence=Path(native['mvs_surface_adapter']['output']).resolve()
        need(evidence==Path(s['supervisor']['path']).parent.resolve(),'supervisor evidence location differs')
        for name in ('stdout.txt','stderr.txt'):check(binding(evidence/name,16*1024**2))
        return s
    dense=stage(old,'dense');mesh=stage(result,'mesh')
    need(dense.get('mask_log_guard',{}).get('error')==failure and mesh.get('status')=='GEOMETRY_READBACK_PASSED','stage disposition differs')
    reference=result['dense_reference'];need(reference['result']==ob and reference['supervisor']==dense['supervisor']
         and reference.get('mask_consumption')=='NOT_VERIFIED' and reference.get('mask_log_check')=='NEGATIVE_DIAGNOSTIC_CHECK_ONLY','dense reference differs')
    need(len(reference.get('original_logs',[]))==3,'dense original logs missing')
    oldroot=Path(ob['path']).parent
    oldnative,_=doc(dense['supervisor']['path'])
    oldevidence=Path(oldnative['mvs_surface_adapter']['output'])
    logpaths={str(_path(r)) for r in reference['original_logs']}
    expected_logs={str((oldevidence/n).resolve()) for n in ('stdout.txt','stderr.txt')}
    original_native=list(oldroot.glob('workDensifyPointCloud-*.log'))
    need(len(original_native)==1 and logpaths==expected_logs|{str(original_native[0].resolve())},'dense log association differs')
    for row in reference['original_logs']:
        check(row,16*1024**2);data=Path(row['path']).read_bytes()
        need(hashlib.sha256(data).hexdigest()==row['sha256'] and b'\x00' not in data,'log read drift/binary')
        need(not re.search(rb'warning:\s*can\s+not\s+load\s+the\s+segmentation\s+mask\b|error:\s*cannot\s+load\s+mask\s+image\b|\b(?:error|fatal):',data,re.I),'dense diagnostic found')
    oldwork=Path(ob['path']).parent/'work';work=directory/'work'
    source_rows=[plan['scene'],*plan['tools'],*((plan.get('photo_inputs') or {}).get('bindings',[])),
                 *(f[k] for f in plan['frames'] for k in ('image_file','mask','validity'))]
    for row in source_rows:
        if row['path'] not in oldcode:check(row)
    dense_bindings=[reference['outputs']['native_scene'],
                    *({k:r[k] for k in ('path','bytes','sha256')} for r in reference['outputs']['geometry'])]
    need(seal.get('dense_outputs')==dense_bindings,'dense staged source set differs')
    oldcommand=runner.commands(plan,oldwork)[0][1]
    oldcommand[oldcommand.index('-w')+1]=str(oldwork)
    need(dense.get('cwd')==str(oldwork) and dense.get('argv')==oldcommand,'historical dense command differs')
    need(runner._surface_outputs(oldwork,'dense')==reference['outputs'],'dense full readback differs')
    need(runner._surface_outputs(work,'mesh')==mesh.get('outputs'),'mesh full readback differs')
    need(mesh.get('cwd')==str(work) and mesh.get('argv')==runner.commands(plan,work)[1][1],'mesh command differs')
    expected=[(plan['scene'],work/'__mvs_input.mvs')]
    for frame in plan['frames']:
        expected.extend([(frame['image_file'],work/frame['image']['name']),
                         (frame['mask'],work/'__mvs_masks'/frame['mask_name']),
                         (frame['validity'],work/'__mvs_validity'/frame['validity_name'])])
    expected.extend((row,work/Path(row['path']).name) for row in seal['dense_outputs'])
    staged=result.get('staged');need(isinstance(staged,list) and len(staged)==len(expected)+1,'staged input set differs')
    copies={str(_path(row)):row for row in staged};need(len(copies)==len(staged),'duplicate staged path')
    for original,destination in expected:
        need(destination.resolve().is_relative_to(work.resolve()),'staged path escapes work')
        check(original);copy=check(copies[str(destination.resolve())])
        need(copy['bytes']==original['bytes'] and copy['sha256']==original['sha256'],'staged copy differs')
    config=check(copies[str((work/'__mvs_empty.cfg').resolve())]);need(config['bytes']==0,'configuration not empty')
    for row in [mesh['outputs']['native_scene'],*mesh['outputs']['geometry']]:check(row)
    logs=list(work.glob('ReconstructMesh-*.log'));need(bool(logs),'mesh native log missing')
    for path in logs:
        row=check(binding(path,16*1024**2));need(row['bytes']>0,'mesh native log empty')
    # Recheck every touched byte binding after all joins and geometry reads.
    for row in list(checked.values()):check(row)
    return {'schema_version':CONTINUATION,'result':rb,'preparation':sb,'authorization':ab,'history':hb,
            'historical_plan':pb,'entry':entry,'artifacts':sorted(checked.values(),key=lambda r:r['path']),
            'geometry':mesh['outputs']['geometry'],'scale_status':'unknown','quality_status':'NOT_EVALUATED',
            'mask_consumption':'NOT_VERIFIED','photo_lineage_status':'NOT_VERIFIED','native_execution_status':'NOT_VERIFIED',
            'evidence_scope':'Declared producer output association only; no native execution authentication or upstream photo chain.'}

def need(test,reason):
    if not test:raise ValueError('mesh source: '+reason)

def binding(path,limit):
    p=Path(path).resolve(strict=True);need(p.is_file(),'expected file')
    with p.open('rb') as stream:
        import os
        before=os.fstat(stream.fileno());need(before.st_size<=limit,'file byte limit exceeded')
        h=hashlib.sha256();total=0
        for block in iter(lambda:stream.read(1024**2),b''):
            total+=len(block);need(total<=limit,'file byte limit exceeded');h.update(block)
        after=os.fstat(stream.fileno())
    fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    need(fields(before)==fields(after)==fields(p.stat()) and total==before.st_size,'input changed while reading')
    return {'path':str(p),'bytes':total,'sha256':h.hexdigest()}

def document(path):
    b=binding(path,MAX_JSON)
    with Path(b['path']).open('rb') as stream:raw=stream.read(MAX_JSON+1)
    need(len(raw)<=MAX_JSON and hashlib.sha256(raw).hexdigest()==b['sha256'],'source record changed')
    def pairs(items):
        result={}
        for k,v in items:need(k not in result,'duplicate JSON key');result[k]=v
        return result
    def invalid(s):raise ValueError('mesh source: nonfinite JSON')
    def finite(s):
        x=float(s);need(math.isfinite(x),'nonfinite JSON');return x
    value=json.loads(raw.decode('utf-8-sig'),object_pairs_hook=pairs,parse_constant=invalid,parse_float=finite)
    need(isinstance(value,dict),'record must be a JSON object')
    return value,b

def _path(row):
    need(isinstance(row,dict) and isinstance(row.get('path'),str),'artifact path missing')
    p=Path(row['path']);need(p.is_absolute(),'artifact path must be absolute')
    need(type(row.get('bytes')) is int and row['bytes']>=0,'artifact bytes must be integer')
    import re
    need(isinstance(row.get('sha256'),str) and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None,'artifact SHA256 invalid')
    return p.resolve()

def _match(row,mesh):
    return str(_path(row))==mesh['path'] and row['bytes']==mesh['bytes'] and row['sha256']==mesh['sha256']

def validate_source(mesh_path,record_path,*,synthetic:bool,role:str|None=None)->dict:
    need(type(synthetic) is bool,'synthetic flag must be boolean')
    mesh=binding(mesh_path,MAX_MESH);record,source=document(record_path);plan_binding=None
    need(mesh['path']!=source['path'],'mesh and record must differ')
    if record.get('schema_version')=='mesh-prep-source-v1':
        need(synthetic and record.get('data_kind')=='synthetic','fixture requires synthetic input')
        need(record.get('evidence_scope')=='TEST_ONLY' and record.get('photo_reconstruction') is False,'fixture is not explicitly TEST_ONLY')
        need(isinstance(record.get('generation'),str) and bool(record['generation'].strip()),'fixture generation missing')
        need(role in (None,'fixture_mesh'),'unsupported fixture role');role='fixture_mesh'
        rows=record.get('artifacts');need(isinstance(rows,list) and 0<len(rows)<=256,'fixture artifact list invalid')
        paths=[]
        for row in rows:
            need(isinstance(row,dict) and row.get('role')=='fixture_mesh','fixture artifact role invalid')
            paths.append(str(_path(row)).casefold())
        need(len(set(paths))==len(paths),'duplicate artifact path')
        matches=[row for row in rows if _match(row,mesh)]
        need(len(matches)==1,'input mesh is not a unique fixture output')
        source_schema='mesh-prep-source-v1'
    elif record.get('schema_version')==CONTINUATION:
        need(not synthetic,'continuation cannot be relabelled synthetic')
        need(role in ('pre_cleanup_surface','mesh'),'continuation requires explicit raw/final role')
        actual=continuation_ledger(record.get('result',{}).get('path'))
        need(actual==record,'continuation ledger differs from recomputed evidence')
        rows=actual['geometry'];need(len(rows)==2 and sorted(r['role'] for r in rows)==['mesh','pre_cleanup_surface']
             and len({str(_path(r)).casefold() for r in rows})==2,'continuation raw/final roles differ')
        selected=next(r for r in rows if r['role']==role)
        need(_match(selected,mesh),'input mesh does not match continuation output role')
        plan_binding=actual['historical_plan'];source_schema=CONTINUATION
    elif record.get('contract_version')==SURFACE:
        need(not synthetic,'surface record cannot be relabelled synthetic')
        need(role in ('pre_cleanup_surface','mesh'),'surface requires explicit raw/final role')
        need(record.get('profile')==PROFILE,'surface profile mismatch')
        need(record.get('status')=='SURFACE_GENERATED_NOT_ACCEPTED' and type(record.get('exit_code')) is int
             and record['exit_code']==0,'surface run was not successful')
        need(record.get('input_drift')==[],'surface input stability was not verified')
        directory=Path(source['path']).parent
        plan,plan_binding=document(directory/'plan.json')
        need(plan.get('contract_version')==SURFACE and plan.get('profile')==PROFILE,'surface plan version mismatch')
        need(record.get('input_origin')==plan.get('input_origin')=='caller_declared_real','surface origin declaration mismatch')
        stages=record.get('stages')
        need(isinstance(stages,list) and len(stages)==2 and all(isinstance(s,dict) for s in stages)
             and [s.get('name') for s in stages]==['dense','mesh'],'surface stages missing or reordered')
        need(all(s.get('status')=='GEOMETRY_READBACK_PASSED' and type(s.get('returncode')) is int
                 and s['returncode']==0 for s in stages),'surface stage did not succeed')
        outputs=stages[1].get('outputs');need(isinstance(outputs,dict),'surface outputs missing')
        rows=outputs.get('geometry');need(isinstance(rows,list) and len(rows)==2,'raw/final surface outputs required')
        need(all(isinstance(r,dict) and isinstance(r.get('role'),str) for r in rows)
             and sorted(r['role'] for r in rows)==['mesh','pre_cleanup_surface'],'surface output roles invalid')
        paths=[]
        for row in rows:
            p=_path(row);need(p.is_relative_to((directory/'work').resolve()),'surface output escapes work directory')
            need(type(row.get('vertices')) is int and row['vertices']>0 and type(row.get('triangles')) is int
                 and row['triangles']>0,'surface geometry counts must be positive integers')
            paths.append(str(p).casefold())
        need(len(set(paths))==len(paths),'raw/final paths must differ')
        selected=next(row for row in rows if row['role']==role)
        need(_match(selected,mesh),'input mesh does not match requested producer output role')
        source_schema=SURFACE
    else:raise ValueError('mesh source: unsupported source record; arbitrary files are not producers')
    # Recheck the producer records after joining them to the selected mesh.
    need(binding(source['path'],MAX_JSON)==source,'source record drift')
    if plan_binding is not None:need(binding(plan_binding['path'],MAX_JSON)==plan_binding,'source plan drift')
    need(binding(mesh['path'],MAX_MESH)==mesh,'source mesh drift')
    return {'schema_version':'mesh-source-link-v1','status':'PRODUCER_OUTPUT_BOUND','source_schema':source_schema,
            'source_role':role,'synthetic_input':synthetic,'mesh':mesh,'source_record':source,'plan':plan_binding,
            'evidence_scope':'Declared producer output association only; no native execution authentication or upstream photo chain.',
            'photo_lineage_status':'NOT_VERIFIED','native_execution_status':'NOT_VERIFIED'}
