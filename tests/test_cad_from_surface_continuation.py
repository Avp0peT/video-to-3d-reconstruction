"""Synthetic formal continuation records; no native operations."""
from pathlib import Path
import json
import pytest
from integrations.demo_pipeline import cad


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def native(argv, root, code=0):
    return dict(status='completed',root_exit_code=code,reclaim_confirmed=True,pid=123,
                argv=argv,cwd=str(root),errors=[],log_errors=[])


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root=tmp_path;producer=root/'outputs/producer';producer.mkdir(parents=True)
    run=root/'outputs/continuation';run.mkdir()
    upstream={};bound=[]
    for name in ('input','sfm','segmentation'):
        entry=producer/'logs'/name/'ENTRY.json';argv=['synthetic',name]
        code=1 if name=='segmentation' else 0
        save(entry,native(argv,root,code));bound.append(cad.binding(entry,1<<20))
        upstream[name]=dict(status='PARTIAL' if code else 'SUCCESS',exit_code=code,
            reclaim_confirmed=True,argv=argv,entry_path=str(entry))
    save(producer/'pipeline.json',dict(status='PARTIAL',exit_code=1,stages=upstream))
    save(producer/'PLAN.json',{'synthetic_fixture':True})
    payload=producer/'input.bin';payload.write_bytes(b'synthetic original data');bound.append(cad.binding(payload,1<<20))
    save(producer/'SURFACE_RESUME_SOURCE.json',dict(schema='photo-surface-resume-source-v1',
        source_run=str(producer),producer_status='RUNNING',upstream_stages=upstream,bindings=bound))
    bound += [cad.binding(producer/name,1<<20) for name in ('pipeline.json','PLAN.json','SURFACE_RESUME_SOURCE.json')]
    python=root/'python.exe';python.write_bytes(b'synthetic python')
    freecad=root/'freecad.exe';freecad.write_bytes(b'synthetic freecad')
    scripts=root/'scripts';scripts.mkdir()
    for name in ('prepare_mesh.py','build_cad_delivery.py','mesh_to_cad.py','continue_photo_cad.py'):
        (scripts/name).write_text('# synthetic code')
    for relative in ('integrations/demo_pipeline/cad.py','integrations/demo_pipeline/surface.py','integrations/mvs_surface/supervision.py'):
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# synthetic code')
    frozen=dict(schema='photo-surface-continuation-v1',status='READY',source_run=str(producer),output=str(run),
        root=str(root),source_status='PARTIAL',upstream_stages=upstream,bindings=bound,
        python=str(python),sfm_run=str(producer/'sfm'),io_run=str(producer/'io'),model_id='0')
    save(run/'PLAN.json',frozen)
    work=run/'geometry/surface_pipeline';work.mkdir(parents=True)
    raw=work/'raw.ply';raw.write_bytes(b'synthetic raw surface')
    mesh=work/'mesh.ply';mesh.write_bytes(b'synthetic final mesh')
    blend=work/'model.blend';blend.write_bytes(b'synthetic blend')
    raw_ref=cad.binding(raw,1<<20);mesh_ref=cad.binding(mesh,1<<20);blend_ref=cad.binding(blend,1<<20)
    source_result=work/'surface/result.json'
    save(source_result,dict(status='SURFACE_GENERATED_NOT_ACCEPTED',exit_code=0,
        stages=[{'name':'dense'},{'name':'mesh','outputs':{'geometry':[
            dict(mesh_ref,role='mesh'),dict(raw_ref,role='pre_cleanup_surface')]}}]))
    blender_result=work/'blender/RESULT.json'
    data=dict(status='PASS',payload_sha256='synthetic-payload',scenes=[{'source_sha256':raw_ref['sha256']}])
    save(blender_result,dict(status='SUCCESS',blend=blend_ref,blend_sha256_before_readback=blend_ref['sha256'],
        payload_sha256='synthetic-payload',stages=[dict(mode=m,data=data,supervision=native(['synthetic',m],root)) for m in ('create','verify')]))
    surface=dict(status='SUCCESS',artifacts=[raw_ref],evidence_paths=[str(source_result)])
    blender=dict(status='SUCCESS',**blend_ref,reopen_verified=True,result_path=str(blender_result))
    names={'native_subset':'derive_general_subset.py','mvs_bridge':'prepare_mvs.py','mvs_masks':'prepare_mvs_masks.py',
           'surface_plan':'build_mvs_surface.py','surface_reconstruction':'build_mvs_surface.py','blender_export':'export_blender.py'}
    arguments={
        'native_subset':['--sfm-run',frozen['sfm_run'],'--io-run',frozen['io_run'],'--model-id','0','--output',str(work/'subset')],
        'mvs_bridge':['--image-root',frozen['sfm_run'],'--output',str(work/'bridge')],
        'mvs_masks':['--sfm-run',frozen['sfm_run'],'--bridge',str(work/'bridge'),'--output',str(work/'masks')],
        'surface_plan':['plan','--scene-path',str(work/'bridge/scene.mvs'),'--plan-output',str(work/'SURFACE_PLAN.json')],
        'surface_reconstruction':['run','--plan',str(work/'SURFACE_PLAN.json'),'--output',str(work/'surface')],
        'blender_export':['--spec',str(work/'SCENES.json'),'--output',str(work/'blender')]}
    stages={};steps={}
    for name,script in names.items():
        argv=[str(python),'-B','-X','utf8',str(root/'scripts'/script)]+arguments[name]
        entry=run/(name+'_ENTRY.json');value=native(argv,root)
        save(entry,value);save(run/'logs'/name/'ENTRY.json',value)
        stages[name]=dict(status='SUCCESS',exit_code=0,reclaim_confirmed=True,entry_path=str(entry))
        steps[name]={'status':'SUCCESS','exit_code':0}
    save(work/'RESULT.json',dict(status='SUCCESS',exit_code=0,surface=surface,blender=blender,stages=steps))
    meta=dict(schema='photo-surface-continuation-v1',status='PARTIAL',exit_code=1,deliverable_ready=True,
        source_run=str(producer),source_pipeline_status='PARTIAL',source_bindings=bound,
        reused_stage_statuses={k:v['status'] for k,v in upstream.items()},
        stages=stages,geometry_steps=steps,surface=surface,blender=blender)
    save(run/'CONTINUATION.json',meta)
    link=dict(mesh=mesh_ref,source_record=cad.binding(source_result,1<<20))
    monkeypatch.setattr(cad,'ROOT',root)
    # This fixture verifies orchestration/evidence binding, not native geometry.
    monkeypatch.setattr(cad,'validate_source',lambda *a,**k:link)
    return root,producer,run,root/'outputs/cad',python,freecad,meta,link


def planned(rig):
    root,producer,run,out,python,freecad,meta,link=rig
    return cad.plan(run,out,python=python,freecad_python=freecad,root=root)


def test_plan_accepts_formal_continuation_and_keeps_recipe(rig):
    frozen=planned(rig);root,producer,run,out,*_=rig
    assert frozen['source_kind']=='surface_continuation' and frozen['geometry_record'] is None
    assert frozen['recipe']['target_faces']==2000 and frozen['recipe']['method']=='quadric'
    assert not out.exists() and not (run/'pipeline.json').exists() and not (run/'CAD_SOURCE.json').exists()
    paths={r['path'] for r in frozen['bindings']}
    assert str(run/'CONTINUATION.json') in paths and str(producer/'SURFACE_RESUME_SOURCE.json') in paths


@pytest.mark.parametrize('fault',['schema','failed','missing_bindings','plan_bindings','checkpoint','stage_reclaim',
    'native_pid','wrong_script','wrong_source_arg','log_mismatch','source_drift','blend_drift','reopen','producer_reclaim'])
def test_different_failed_or_unbound_sources_refused(rig,fault):
    root,producer,run,out,python,freecad,meta,link=rig
    if fault=='schema':meta['schema']='other'
    if fault=='failed':meta['status']='FAILED'
    if fault=='missing_bindings':meta['source_bindings']=[]
    if fault=='plan_bindings':
        p=run/'PLAN.json';value=load(p);value['bindings']=[];save(p,value)
    if fault=='checkpoint':(producer/'SURFACE_RESUME_SOURCE.json').unlink()
    if fault=='stage_reclaim':meta['stages']['native_subset']['reclaim_confirmed']=False
    if fault in ('native_pid','wrong_script','wrong_source_arg','log_mismatch'):
        p=run/'native_subset_ENTRY.json';value=load(p)
        if fault=='native_pid':value['pid']=None
        if fault=='wrong_script':value['argv'][4]='foreign_script.py'
        if fault=='wrong_source_arg':value['argv'][6]='foreign_sfm'
        if fault=='log_mismatch':value['root_exit_code']=2
        save(p,value)
        if fault!='log_mismatch':save(run/'logs/native_subset/ENTRY.json',value)
    if fault=='source_drift':(producer/'input.bin').write_bytes(b'changed')
    if fault=='blend_drift':Path(meta['blender']['path']).write_bytes(b'changed')
    if fault=='reopen':meta['blender']['reopen_verified']=False
    if fault=='producer_reclaim':
        p=producer/'logs/input/ENTRY.json';value=load(p);value['reclaim_confirmed']=False;save(p,value)
    save(run/'CONTINUATION.json',meta)
    with pytest.raises((ValueError,OSError)):planned(rig)
    assert not out.exists()


def test_pipeline_source_remains_compatible(rig):
    root,producer,run,out,python,freecad,meta,link=rig
    (run/'CONTINUATION.json').unlink()
    save(run/'pipeline.json',dict(status='PARTIAL',deliverable_ready=True,
        stages={'surface':meta['surface'],'blender':meta['blender']}))
    assert planned(rig)['source_kind']=='pipeline'


def test_no_nested_output_or_ambiguous_source(rig):
    root,producer,run,out,python,freecad,meta,link=rig
    with pytest.raises(ValueError,match='overlaps'):cad.plan(run,run/'cad',root=root)
    save(run/'pipeline.json',{})
    with pytest.raises(ValueError,match='ambiguous'):planned(rig)


def test_execute_stable_rechecks_continuation_before_any_launch(rig,monkeypatch):
    frozen=planned(rig);root,producer,run,out,python,freecad,meta,link=rig
    calls=[]
    meta['stages']['native_subset']['reclaim_confirmed']=False;save(run/'CONTINUATION.json',meta)
    # Remove state-byte protection deliberately to prove semantic recheck too.
    frozen['bindings']=[r for r in frozen['bindings'] if r['path']!=str(run/'CONTINUATION.json')]
    result=cad.execute(frozen,supervisor=lambda *a,**k:calls.append(a))
    assert not calls and result['exit_code']==4 and result['status']=='SOURCE_DRIFT'


def test_download_verifier_rechecks_current_continuation(rig,monkeypatch):
    frozen=planned(rig);root,producer,run,out,python,freecad,meta,link=rig
    out.mkdir()
    # Isolate current source verification before prepared/native CAD evidence.
    save(out/'PLAN.json',frozen)
    save(out/'RESULT.json',dict(schema='photo-cad-result-v1',status='SUCCESS',exit_code=0,source_and_code_stable=True))
    (producer/'input.bin').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bound file changed'):cad.verify_cad_output(out,run)


def test_failed_preparation_remains_single_attempt(rig):
    frozen=planned(rig);calls=[]
    def supervisor(argv,**kw):calls.append(argv);return native(argv,kw['cwd'],2)
    result=cad.execute(frozen,supervisor=supervisor)
    assert len(calls)==1 and result['status']=='FAILED' and result['exit_code']==2
    assert result['source_and_code_stable'] is True and result['artifacts']==[]


def test_stubbed_cad_success_and_current_download_proof(rig,monkeypatch):
    frozen=planned(rig);root,producer,run,out,python,freecad,meta,link=rig
    prepared={'synthetic_validated':'prepared'};delivery={'synthetic_validated':'delivery'}
    monkeypatch.setattr(cad,'prepared_input',lambda *a,**k:prepared)
    monkeypatch.setattr(cad,'verify_delivery',lambda *a,**k:delivery)
    calls=[]
    def supervisor(argv,**kw):
        calls.append(argv)
        if len(calls)==1:
            save(out/'prepared/mesh_prep.json',dict(inputs={key:{k:link[key][k] for k in ('path','sha256')}
                for key in ('mesh','source_record')},source_validation=link))
        else:
            save(out/'cad/request.json',dict(prepared=prepared,prepared_run=str(out/'prepared')))
            refs=[]
            for name in ('model.FCStd','model.step'):
                p=out/'cad/cad'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic CAD')
                refs.append(cad.binding(p,1<<20))
            save(out/'cad/cad/conversion_report.json',dict(checks={'synthetic_stub':True},artifacts=refs))
        return native(argv,kw['cwd'])
    result=cad.execute(frozen,supervisor=supervisor)
    assert len(calls)==2 and result['status']=='SUCCESS' and result['exit_code']==0
    verified=cad.verify_cad_output(out,run)
    assert verified['source_run']==str(run) and len(verified['artifacts'])==2
    assert not (run/'pipeline.json').exists() and not (run/'CAD_SOURCE.json').exists()
    # Even omission of a direct frozen state row cannot bypass semantic recheck.
    p=out/'PLAN.json';value=load(p);value['bindings']=[r for r in value['bindings'] if r['path']!=str(run/'CONTINUATION.json')];save(p,value)
    meta['stages']['native_subset']['reclaim_confirmed']=False;save(run/'CONTINUATION.json',meta)
    with pytest.raises(ValueError,match='stage identity/reclaim'):cad.verify_cad_output(out,run)


def test_source_mode_cannot_be_downgraded_to_legacy(rig):
    frozen=planned(rig);frozen['source_kind']='pipeline';calls=[]
    result=cad.execute(frozen,supervisor=lambda *a,**k:calls.append(a))
    assert not calls and result['status']=='SOURCE_DRIFT' and 'source mode differs' in result['drift_error']
