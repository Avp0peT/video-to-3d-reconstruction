"""TEST_DATA file joins only; hand-built supervisors do not prove execution."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from integrations.meshprep import provenance as p
from integrations.mvs_surface import runner

def b(path):
    data=path.read_bytes()
    return dict(path=str(path.resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')
    return b(path)

def file(path,data=b'TEST_DATA'):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return b(path)

@pytest.fixture
def continuation(tmp_path):
    package=tmp_path/'TEST_DATA';root=package/'mesh_only';work=root/'work';oldroot=tmp_path/'old';oldwork=oldroot/'work'
    scene=file(oldwork/'__mvs_input.mvs');tool=file(tmp_path/'tools/ReconstructMesh.exe')
    frames=[]
    for i in range(43):
        name=f'{i}.png';image=file(tmp_path/'images'/name);mask=file(tmp_path/'masks'/name);valid=file(tmp_path/'valid'/name)
        frames.append(dict(image_file=image,mask=mask,validity=valid,image={'name':'images/'+name},mask_name=name,validity_name=name))
    code=file(tmp_path/'code.py');plan=dict(contract_version=p.SURFACE,profile=p.PROFILE,input_origin='caller_declared_real',
        implementation=[code],frames=frames,scene=scene,tools=[tool],arguments={'tool_root':str(tool and tmp_path/'tools'),'ignore_label':0})
    pb=put(oldroot/'plan.json',plan)
    cloud=file(oldwork/'__mvs_dense.ply');archive=file(oldwork/'__mvs_dense.mvs')
    denseout=dict(native_scene=archive,geometry=[dict(**cloud,vertices=4,triangles=0,bbox=[[0,0,0],[1,1,1]],role='dense')])
    def stage(directory,name,argv,status):
        evidence=directory/'work/__mvs_runtime'/name
        for n in ('stdout.txt','stderr.txt'):file(evidence/n,b'')
        native=dict(status='completed',root_exit_code=0,reclaim_confirmed=True,errors=[],log_errors=[],pid=1,
                    argv=argv,cwd=str(directory/'work'),mvs_surface_adapter={'output':str(evidence)})
        return dict(name=name,returncode=0,pid=1,argv=argv,cwd=native['cwd'],supervisor=put(evidence/'supervisor_result.json',native),status=status)
    argv=runner.commands(plan,oldwork)[0][1];argv[argv.index('-w')+1]=str(oldwork)
    failure='ValueError: dense mask log guard: required native DensifyPointCloud log missing'
    dense=stage(oldroot,'dense',argv,'STOPPED');dense['mask_log_guard']={'error':failure}
    old=dict(status='FAILED',exit_code=4,error=failure,input_drift=[],native_launches=1,launch_attempts=1,launch_count_complete=True,stages=[dense])
    ob=put(oldroot/'result.json',old)
    original_log=file(oldroot/'workDensifyPointCloud-TEST.log',b'TEST_DATA completed')
    logs=[b(oldroot/'work/__mvs_runtime/dense'/n) for n in ('stdout.txt','stderr.txt')]+[original_log]
    staged=[]
    def copy_file(row,destination):staged.append(file(destination,Path(row['path']).read_bytes()))
    copy_file(scene,work/'__mvs_input.mvs')
    for f in frames:
        for k,n in [('image_file',f['image']['name']),('mask','__mvs_masks/'+f['mask_name']),('validity','__mvs_validity/'+f['validity_name'])]:copy_file(f[k],work/n)
    staged.append(file(work/'__mvs_empty.cfg',b''))
    for row in (archive,cloud):copy_file(row,work/Path(row['path']).name)
    raw=file(work/'__mvs_surface_raw.ply');mesh=file(work/'__mvs_surface.ply')
    output=dict(native_scene=file(work/'__mvs_surface.mvs'),geometry=[dict(**r,vertices=4,triangles=4,bbox=[[0,0,0],[1,1,1]],role=role) for r,role in [(raw,'pre_cleanup_surface'),(mesh,'mesh')]])
    step=stage(root,'mesh',runner.commands(plan,work)[1][1],'GEOMETRY_READBACK_PASSED');step['outputs']=output
    file(work/'ReconstructMesh-TEST.log',b'TEST_DATA complete')
    result=dict(contract_version='mvs-mesh-continuation-v1',status='MESH_GENERATED_NOT_ACCEPTED',exit_code=0,
        dense_launches=0,native_launches=1,known_native_launches=1,launch_attempts=1,original_input_drift=[],execution_code_drift=[],
        scale_status='unknown',quality_status='NOT_EVALUATED',mask_consumption='NOT_VERIFIED',CAD_status='NOT_RUN',stages=[step],staged=staged,
        dense_reference=dict(result=ob,supervisor=dense['supervisor'],outputs=denseout,original_logs=logs,mask_log_check='NEGATIVE_DIAGNOSTIC_CHECK_ONLY',mask_consumption='NOT_VERIFIED'))
    rp=root/'result.json';put(rp,result)
    put(root/'historical_plan_reference.json',dict(path=pb['path'],sha256=pb['sha256'],implementation=[code]))
    entry=file(package/'mesh_once.py');final=put(package/'fix.json',dict(artifacts=[],product_files=[]));handoff=file(package/'handoff.txt')
    seal=dict(schema='one-shot-mesh-continuation-preparation-v1',real_execution_authorized=True,implementation_finalized=True,b_log_review_accepted=True,
        allowed_native_entries=1,required_output=str(root),entry_script_sha256=entry['sha256'],original_inputs=[pb,ob],old_result=ob,
        historical_implementation=[code],execution_implementation=[code],dense_outputs=[archive,cloud],b_log_review=handoff,
        accepted_B_fix=dict(final_hashes=final,handoff=handoff,runner_sha256=code['sha256']))
    sb=put(package/'BINDINGS.json',seal)
    put(package/'MAIN_AUTHORIZATION.json',dict(entry_script_sha256=entry['sha256'],authorized_binding_sha256=sb['sha256'],native_entries=1,root_mismatch_count=0))
    return rp,result,mesh,denseout,output

def readback(monkeypatch,fixture):
    monkeypatch.setattr(runner,'_surface_outputs',lambda work,stage: copy.deepcopy(fixture[3] if stage=='dense' else fixture[4]))

def test_recomputed_ledger_preserves_unknowns_and_same_hash_roles(continuation,monkeypatch):
    readback(monkeypatch,continuation);rp,result,mesh,_,_=continuation
    ledger=p.continuation_ledger(rp);path=rp.parent/'source.json';put(path,ledger)
    value=p.validate_source(mesh['path'],path,synthetic=False,role='mesh')
    assert value['source_schema']==p.CONTINUATION
    assert value['photo_lineage_status']==value['native_execution_status']=='NOT_VERIFIED'
    assert ledger['mask_consumption']=='NOT_VERIFIED'
    with pytest.raises(ValueError):p.validate_source(mesh['path'],path,synthetic=True,role='mesh')
    with pytest.raises(ValueError):p.validate_source(mesh['path'],path,synthetic=False)

@pytest.mark.parametrize('fault',['bool-exit','dense-launch','drift','supervisor','staged','history','log','readback','ledger'])
def test_continuation_recomputes_evidence_instead_of_trusting_pass(continuation,monkeypatch,fault):
    readback(monkeypatch,continuation);rp,result,mesh,_,_=continuation
    ledger=p.continuation_ledger(rp);path=rp.parent/'source.json';put(path,ledger)
    if fault=='bool-exit':result['exit_code']=False;put(rp,result)
    elif fault=='dense-launch':result['dense_launches']=1;put(rp,result)
    elif fault=='drift':result['execution_code_drift']=['TEST_DATA drift'];put(rp,result)
    elif fault=='supervisor':Path(result['stages'][0]['supervisor']['path']).write_text('{}')
    elif fault=='staged':Path(result['staged'][1]['path']).write_bytes(b'changed')
    elif fault=='history':put(rp.parent/'historical_plan_reference.json',dict(path=str(rp),sha256=b(rp)['sha256'],implementation=[]))
    elif fault=='log':Path(result['dense_reference']['original_logs'][2]['path']).write_bytes(b'ERROR: cannot load mask image')
    elif fault=='readback':continuation[4]['geometry'][1]['triangles']=0
    else:ledger['mask_consumption']='SUPPORTED';put(path,ledger)
    with pytest.raises(ValueError):p.validate_source(mesh['path'],path,synthetic=False,role='mesh')

def test_input_face_limit_default_bounds_and_actual_reader_forwarding(tmp_path,monkeypatch):
    from scripts import prepare_mesh as prep
    parser=prep.parser();base=['--mesh','x.ply','--source-run','s.json','--output','o']
    assert parser.parse_args(base).max_input_faces==200000
    assert parser.parse_args(base+['--max-input-faces','400000']).max_input_faces==400000
    for value in ('0','400001','1.5'):
        with pytest.raises(SystemExit):parser.parse_args(base+['--max-input-faces',value])
    mesh=tmp_path/'x.ply';mesh.write_bytes(b'TEST_DATA');source=tmp_path/'s.json';put(source,{})
    monkeypatch.setattr(prep.source_checks,'validate_source',lambda *a,**k:{})
    def reader(path,**kwargs):
        assert kwargs=={'max_faces':400000}
        raise ValueError('TEST_DATA stop before geometry/native')
    monkeypatch.setattr(prep.ply,'read_ply',reader)
    args=SimpleNamespace(mesh=mesh,source_run=source,synthetic=False,source_role='mesh',max_input_faces=400000)
    report={}
    with pytest.raises(ValueError,match='TEST_DATA stop'):prep.prepare(args,tmp_path,report,lambda:None,lambda:None)
    assert report['max_input_faces']==400000

def test_isolated_absolute_source_loader_restores_search_path(tmp_path):
    root=Path(p.__file__).resolve().parents[2]
    executable=root/'outputs/toolchains/freecad-1.1.4/FreeCAD_1.1.4-Windows-x86_64-py311/bin/python.exe'
    if not executable.is_file():pytest.skip('exact accepted FreeCAD Python runtime not present')
    code='''import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('isolated_source',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
before=list(sys.path)
try:module.continuation_ledger(sys.argv[2])
except FileNotFoundError:pass
else:raise AssertionError('missing TEST_DATA result was accepted')
assert sys.path==before
assert 'vtkmodules' not in sys.modules and 'FreeCAD' not in sys.modules
'''
    result=subprocess.run([str(executable),'-I','-B','-c',code,str(Path(p.__file__).resolve()),str(tmp_path/'missing.json')],
                          capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr
