"""Pure geometry/metadata and fake-supervisor tests; no native/model execution."""
import json
from pathlib import Path
import pytest
from integrations.demo_pipeline import measurement as m


@pytest.fixture
def source(tmp_path,monkeypatch):
    root=tmp_path;run=root/'outputs/source';cad=run/'cad';prepared=cad/'prepared';prepared.mkdir(parents=True)
    mesh=prepared/'prepared.ply'
    mesh.write_text('ply\nformat ascii 1.0\nelement vertex 4\nproperty double x\nproperty double y\nproperty double z\nelement face 4\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n')
    freecad=root/'freecad.exe';freecad.write_bytes(b'TEST_TOOL_NOT_RUN')
    for path in (run/'pipeline.json',cad/'RESULT.json',prepared/'mesh_prep.json',prepared/'selected_before.ply',prepared/'selection_map.json'):
        path.write_text('{}')
    (cad/'PLAN.json').write_text(json.dumps({'freecad_python':str(freecad)}))
    artifacts=[]
    for name in ('model.FCStd','model.step'):
        path=cad/name;path.write_bytes(b'TEST_CAD_BYTES');artifacts.append(m.file_binding(path))
    binding={'mesh':m.file_binding(mesh),'record':m.file_binding(prepared/'mesh_prep.json'),
             'synthetic_input':False,'data_kind':'real','artifacts':[m.file_binding(prepared/name) for name in ('prepared.ply','selected_before.ply','selection_map.json')], 'upstream':[]}
    monkeypatch.setattr(m,'ROOT',root)
    monkeypatch.setattr(m,'verify_cad_output',lambda output,source_run: {'artifacts':artifacts} if output==cad and source_run==run else (_ for _ in ()).throw(ValueError('foreign source')))
    monkeypatch.setattr(m,'prepared_input',lambda directory:binding)
    # New script bindings use real source files while product ROOT is isolated.
    original_binding=m.file_binding
    actual=Path(m.__file__).resolve().parents[2]
    def file_binding(path,*args):
        path=Path(path)
        if not path.exists() and path.is_relative_to(root/'scripts'):path=actual/'scripts'/path.name
        return original_binding(path,*args)
    monkeypatch.setattr(m,'file_binding',file_binding)
    return m.inspect_source(run),root


def reference(source):
    inspected,root=source;proof=root/'evidence.txt';proof.write_text('TEST_DATA_USER_DECLARED_MEASUREMENT')
    return m.create_request(inspected,[0,1],25,'mm','physical marks','vertex0 and vertex1 correspond to those marks',proof)


def test_inspection_same_source_and_bounded_points(source):
    inspected,_=source
    assert inspected['vertex_count']==4 and inspected['points'][1]==[1.,0.,0.]
    assert inspected['mesh']['sha256']==m.file_binding(inspected['mesh']['path'])['sha256']
    with pytest.raises(ValueError,match='foreign'):m.inspect_source(inspected['run'],Path(inspected['cad_output']).parent)


def test_request_and_real_export_html_preview(source):
    inspected,root=source;request=reference(source)
    assert request['schema_version']=='mesh-scale-request-v1' and request['data_kind']=='real'
    assert request['source_mesh_sha256']==inspected['mesh']['sha256']
    preview=m.make_preview(inspected,root/'outputs/preview')
    assert preview['viewer']['status']=='EXPORTED_NOT_UI_VERIFIED'
    html=Path(preview['path']).read_text()
    assert 'vertex-index' in html and 'Prepared mesh vertex indices' in html
    assert m.file_binding(preview['path'])==preview['binding']


@pytest.mark.parametrize('value',[True,False,0,-1,float('nan')])
def test_bad_length_and_missing_physical_evidence_rejected(source,value):
    inspected,root=source;proof=root/'proof';proof.write_text('TEST_DATA')
    with pytest.raises(ValueError):m.create_request(inspected,[0,1],value,'mm','marks','correspondence',proof)
    with pytest.raises(OSError):m.create_request(inspected,[0,1],25,'mm','marks','correspondence',root/'missing')


def test_wrong_meshhash_foreign_reference_and_existing_output_rejected(source):
    inspected,root=source;request=reference(source);request['source_mesh_sha256']='a'*64
    path=root/'reference.json';path.write_text(json.dumps(request))
    with pytest.raises(ValueError,match='different mesh'):m.plan(inspected['run'],path,root/'outputs/new')
    request['source_mesh_sha256']=inspected['mesh']['sha256'];request['reference']['evidence']['sha256']='b'*64;path.write_text(json.dumps(request))
    with pytest.raises(ValueError,match='evidence hash'):m.plan(inspected['run'],path,root/'outputs/new')
    path.write_text(json.dumps(reference(source)));out=root/'outputs/existing';out.mkdir()
    with pytest.raises(ValueError,match='already exists'):m.plan(inspected['run'],path,out)


@pytest.mark.parametrize('status,rc,expected',[('completed',2,2),('wall_clock_exceeded',None,5),('spawn_failure',None,4)])
def test_failed_delivery_once_preserves_source_and_exit(source,status,rc,expected):
    inspected,root=source;path=root/'reference.json';path.write_text(json.dumps(reference(source)))
    frozen=m.plan(inspected['run'],path,root/'outputs/new');calls=[]
    def supervisor(argv,**kw):calls.append(argv);return {'status':status,'root_exit_code':rc,'reclaim_confirmed':True}
    result=m.execute(frozen,supervisor=supervisor)
    assert len(calls)==1 and result['status']=='FAILED' and result['exit_code']==expected
    assert result['source_and_code_stable'] and result['artifacts']==[]


def test_foreign_scale_ledger_is_rejected(source,monkeypatch):
    inspected,root=source;path=root/'reference.json';path.write_text(json.dumps(reference(source)))
    frozen=m.plan(inspected['run'],path,root/'outputs/new');delivery=Path(frozen['output'])/'delivery';delivery.mkdir(parents=True)
    (delivery/'request.json').write_text(json.dumps({'prepared':inspected['prepared'],'prepared_run':inspected['prepared_dir'],'reference':str(root/'foreign-reference.json')}))
    monkeypatch.setattr(m,'verify_delivery',lambda directory:{'status':'BINDINGS_AND_REPORTS_VALID'})
    with pytest.raises(ValueError,match='foreign reference'):m._outputs(frozen)


def test_strict_native_false_exit_and_unknown_pid_rejected():
    assert not m.native_ok({'status':'completed','root_exit_code':False,'pid':12,'reclaim_confirmed':True,'argv':['exe'],'cwd':'cwd'},['exe'],'cwd')
    assert not m.native_ok({'status':'completed','root_exit_code':0,'pid':None,'reclaim_confirmed':True,'argv':['exe'],'cwd':'cwd'},['exe'],'cwd')


def test_internally_valid_foreign_ledger_cannot_replace_same_request_source(source,monkeypatch):
    inspected,root=source;path=root/'reference.json';path.write_text(json.dumps(reference(source)))
    frozen=m.plan(inspected['run'],path,root/'outputs/new');delivery=Path(frozen['output'])/'delivery';delivery.mkdir(parents=True)
    request={'prepared':inspected['prepared'],'prepared_run':inspected['prepared_dir'],'reference':str(path)}
    (delivery/'request.json').write_text(json.dumps(request))
    foreign=root/'foreign.ply';foreign.write_bytes(Path(inspected['mesh']['path']).read_bytes())
    ledger={'data_kind':'real','source_mesh':m.file_binding(foreign),'source_record':inspected['prepared']['record'],'request':m.file_binding(path)}
    monkeypatch.setattr(m,'verify_delivery',lambda directory:{'status':'BINDINGS_AND_REPORTS_VALID'})
    monkeypatch.setattr(m,'verify_ledger',lambda *a,**kw:ledger)
    with pytest.raises(ValueError,match='mismatched source_mesh'):m._outputs(frozen)


def test_fake_success_verified_bindings_and_source_drift_download_rejection(source,monkeypatch):
    inspected,root=source;path=root/'reference.json';path.write_text(json.dumps(reference(source)))
    frozen=m.plan(inspected['run'],path,root/'outputs/new');output=Path(frozen['output'])
    def outputs(plan):
        directory=output/'delivery/cad';directory.mkdir(parents=True,exist_ok=True)
        refs=[]
        for name in ('model.FCStd','model.step'):
            target=directory/name
            if not target.exists():target.write_bytes(b'TEST_DATA_SCALED_CAD')
            refs.append(m.file_binding(target))
        ledger=output/'fake_scale_ledger.json'
        if not ledger.exists():ledger.write_text('{}')
        return {'artifacts':refs,'scale_ledger':m.file_binding(ledger),'scale_status':'reference_scaled','length_unit':'mm',
                'independent_check_status':'NOT_EVALUATED','independent_checks':[]}
    monkeypatch.setattr(m,'_outputs',outputs)
    calls=[]
    def supervisor(argv,**kw):
        calls.append(argv);return {'status':'completed','root_exit_code':0,'reclaim_confirmed':True,
                                  'pid':321,'argv':argv,'cwd':kw['cwd'],'errors':[],'log_errors':[]}
    result=m.execute(frozen,supervisor=supervisor)
    assert len(calls)==1 and result['status']=='SUCCESS' and result['exit_code']==0
    verified=m.verify_result(output)
    assert verified['source_run']==inspected['run'] and verified['source_cad_output']==inspected['cad_output']
    assert verified['artifacts']==result['artifacts']
    Path(inspected['mesh']['path']).write_bytes(b'DRIFTED_SOURCE')
    with pytest.raises(ValueError,match='binding changed'):m.verify_result(output)


def test_independent_check_cannot_reuse_fit_endpoints(source):
    inspected,root=source;request=reference(source)
    check=dict(request['reference'],vertex_indices=[1,0])
    with pytest.raises(ValueError):m.create_request(inspected,[0,1],25,'mm','marks','correspondence',root/'evidence.txt',checks=[check])


@pytest.mark.parametrize('length,passes',[(25,True),(30,False)])
def test_independent_threshold_preflight(source,length,passes):
    inspected,root=source;request=reference(source)
    check=dict(request['reference'],vertex_indices=[0,2],length=length,allowed_absolute_error_mm=0.01)
    if passes:
        value=m.create_request(inspected,[0,1],25,'mm','marks','correspondence',root/'evidence.txt',checks=[check])
        assert value['checks']==[check]
    else:
        with pytest.raises(ValueError,match='exceeds declared threshold'):
            m.create_request(inspected,[0,1],25,'mm','marks','correspondence',root/'evidence.txt',checks=[check])
        request['checks']=[check];path=root/'reference.json';path.write_text(json.dumps(request))
        with pytest.raises(ValueError,match='exceeds declared threshold'):
            m.plan(inspected['run'],path,root/'outputs/rejected')
        assert not (root/'outputs/rejected').exists()


def test_no_checks_and_actual_ledger_failure_gate(source,monkeypatch):
    assert m._check_status([])=='NOT_EVALUATED'
    assert m._check_status([{'within_threshold':True}])=='PASS'
    inspected,root=source;path=root/'reference.json';path.write_text(json.dumps(reference(source)))
    frozen=m.plan(inspected['run'],path,root/'outputs/new');delivery=Path(frozen['output'])/'delivery';delivery.mkdir(parents=True)
    request={'prepared':inspected['prepared'],'prepared_run':inspected['prepared_dir'],'reference':str(path)}
    (delivery/'request.json').write_text(json.dumps(request))
    ledger={'data_kind':'real','source_mesh':inspected['mesh'],'source_record':inspected['prepared']['record'],
            'request':m.file_binding(path),'independent_checks':[{'within_threshold':False}]}
    monkeypatch.setattr(m,'verify_delivery',lambda directory:{'status':'BINDINGS_AND_REPORTS_VALID'})
    monkeypatch.setattr(m,'verify_ledger',lambda *a,**kw:ledger)
    with pytest.raises(ValueError,match='exceeds declared threshold'):m._outputs(frozen)
