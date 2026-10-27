"""Bounded orchestration failure tests; no native launch."""
from pathlib import Path
from unittest.mock import patch
import pytest
import json

from integrations.demo_pipeline import cad


def frozen(tmp_path):
    mesh=tmp_path/'original.ply';mesh.write_bytes(b'source bytes')
    source=tmp_path/'source.json';source.write_text('{}')
    return {'output':str(tmp_path/'new'),'mesh':str(mesh),'source_result':str(source),
            'source_link':{'mesh':cad.binding(mesh,1024),'source_record':cad.binding(source,1024)},'freecad_python':'existing-freecad','python':'existing-python',
            'bindings':[cad.binding(mesh,1024),cad.binding(source,1024)]}


def ok_native(argv,kwargs):
    return {'status':'completed','root_exit_code':0,'reclaim_confirmed':True,'pid':123,
            'argv':argv,'cwd':kwargs['cwd'],'errors':[],'log_errors':[]}


def write_prepared(plan):
    out=Path(plan['output'])/'prepared';out.mkdir(parents=True,exist_ok=True)
    (out/'mesh_prep.json').write_text(json.dumps({'inputs':{key:{field:plan['source_link'][key][field] for field in ('path','sha256')}
        for key in ('mesh','source_record')},'source_validation':plan['source_link']}))


@pytest.mark.parametrize('failure',[
    {'status':'completed','root_exit_code':2,'reclaim_confirmed':True},
    {'status':'wall_clock_exceeded','root_exit_code':None,'reclaim_confirmed':True},
    {'status':'completed','root_exit_code':0,'reclaim_confirmed':False},
])
def test_failed_or_timed_out_preparation_never_starts_cad(tmp_path,failure):
    plan=frozen(tmp_path);calls=[]
    def supervisor(argv,**kwargs):calls.append(argv);return failure
    with patch.object(cad,'validate_source',return_value=plan['source_link']):
        result=cad.execute(plan,supervisor=supervisor)
    assert result['status']=='FAILED' and len(calls)==1
    assert result['stages'][0]['name']=='prepare'
    assert result['artifacts']==[] and result['error'].startswith('RuntimeError: prepare failed')


def test_open_mesh_and_fake_success_rejected_by_existing_prepared_validator(tmp_path):
    plan=frozen(tmp_path);calls=[]
    def supervisor(argv,**kwargs):calls.append(argv);return ok_native(argv,kwargs)
    with patch.object(cad,'validate_source',return_value=plan['source_link']),patch.object(
            cad,'prepared_input',side_effect=ValueError('OPEN_MESH: closed solid required')):
        result=cad.execute(plan,supervisor=supervisor)
    assert len(calls)==1 and result['status']=='FAILED'
    assert 'OPEN_MESH' in result['error'] and result['source_and_code_stable'] is True


def test_source_drift_after_prepare_blocks_delivery(tmp_path):
    plan=frozen(tmp_path);calls=[]
    def supervisor(argv,**kwargs):
        calls.append(argv);Path(plan['mesh']).write_bytes(b'changed source')
        write_prepared(plan);return ok_native(argv,kwargs)
    with patch.object(cad,'validate_source',return_value=plan['source_link']),patch.object(cad,'prepared_input',return_value={}):
        result=cad.execute(plan,supervisor=supervisor)
    assert len(calls)==1 and result['status']=='SOURCE_DRIFT' and result['exit_code']==4
    assert result['source_and_code_stable'] is False


def test_delivery_false_success_is_not_promoted(tmp_path):
    plan=frozen(tmp_path);calls=[]
    def supervisor(argv,**kwargs):calls.append(argv);write_prepared(plan);return ok_native(argv,kwargs)
    with patch.object(cad,'validate_source',return_value=plan['source_link']),patch.object(cad,'prepared_input',return_value={}),patch.object(
            cad,'verify_delivery',side_effect=ValueError('reopened solid invalid')):
        result=cad.execute(plan,supervisor=supervisor)
    assert len(calls)==2 and result['status']=='FAILED' and 'reopened solid invalid' in result['error']
    assert result['artifacts']==[]


def test_paths_outside_run_or_project_are_rejected(tmp_path):
    run=tmp_path/'outputs'/'source';run.mkdir(parents=True)
    with pytest.raises(ValueError,match='output outside'):
        cad.plan(run,tmp_path/'outside',root=tmp_path)
    file=tmp_path/'external.json';file.write_text('{}')
    with pytest.raises(ValueError,match='escapes unified run'):
        cad.within(file,run)


def test_unverified_unified_success_is_rejected(tmp_path):
    run=tmp_path/'outputs'/'source';run.mkdir(parents=True)
    (run/'pipeline.json').write_text('{"status":"SUCCESS","deliverable_ready":false}')
    with pytest.raises(ValueError,match='no verified deliverable'):
        cad.plan(run,tmp_path/'outputs'/'new',root=tmp_path)


def test_backend_exception_preserves_original_cause(tmp_path):
    plan=frozen(tmp_path)
    def supervisor(*args,**kwargs):raise RuntimeError('Job backend unavailable')
    with patch.object(cad,'validate_source',return_value=plan['source_link']):result=cad.execute(plan,supervisor=supervisor)
    assert result['status']=='FAILED' and result['error']=='RuntimeError: Job backend unavailable'
    assert len(result['stages'])==1 and result['stages'][0]['status']=='STARTING'


def test_download_verifier_rejects_wrong_source_and_fake_success(tmp_path):
    output=tmp_path/'outputs'/'cad';output.mkdir(parents=True)
    source=tmp_path/'outputs'/'source';source.mkdir()
    other=tmp_path/'outputs'/'other';other.mkdir()
    (output/'PLAN.json').write_text(__import__('json').dumps({'schema':'photo-cad-continuation-v1',
        'run':str(source),'output':str(output)}))
    (output/'RESULT.json').write_text('{"schema":"photo-cad-result-v1","status":"SUCCESS","exit_code":0,"source_and_code_stable":false}')
    with patch.object(cad,'ROOT',tmp_path):
        with pytest.raises(ValueError,match='another source run'):cad.verify_cad_output(output,other)
        with pytest.raises(ValueError,match='no accepted result'):cad.verify_cad_output(output,source)


def test_post_success_artifact_failure_is_nonzero_and_keeps_cause(tmp_path):
    plan=frozen(tmp_path)
    def supervisor(argv,**kwargs):write_prepared(plan);return ok_native(argv,kwargs)
    report={'checks':{},'artifacts':[{'path':str(tmp_path/'absent.FCStd'),'bytes':1,'sha256':'a'*64}]}
    original=cad.document
    def document(path):
        if Path(path).name=='conversion_report.json':return report,{}
        return original(path)
    with patch.object(cad,'validate_source',return_value=plan['source_link']),patch.object(cad,'prepared_input',return_value={}),patch.object(
        cad,'delivery_bound',return_value={}),patch.object(cad,'document',side_effect=document):
        result=cad.execute(plan,supervisor=supervisor)
    assert result['status']=='FAILED' and result['exit_code']==4
    assert 'FileNotFoundError' in result['error']


def test_internal_valid_prepared_from_other_source_is_rejected(tmp_path):
    plan=frozen(tmp_path);write_prepared(plan)
    record=Path(plan['output'])/'prepared/mesh_prep.json'
    other=tmp_path/'other.ply';other.write_bytes(b'source bytes')
    data=json.loads(record.read_text());data['inputs']['mesh']['path']=str(other);record.write_text(json.dumps(data))
    with patch.object(cad,'prepared_input',return_value={'internally_valid':True}):
        with pytest.raises(ValueError,match='another producer'):cad.prepared_bound(plan)


def test_internal_valid_delivery_from_other_prepared_is_rejected(tmp_path):
    plan=frozen(tmp_path);out=Path(plan['output'])/'cad';out.mkdir(parents=True)
    (out/'request.json').write_text(json.dumps({'prepared':{'another':'valid'},'prepared_run':str(tmp_path/'foreign-prepared')}))
    with patch.object(cad,'verify_delivery',return_value={'status':'BINDINGS_AND_REPORTS_VALID'}):
        with pytest.raises(ValueError,match='another prepared'):cad.delivery_bound(plan,{'this':'prepared'})


@pytest.mark.parametrize('status,code',[('spawn_failure',4),('assign_failure',4),('wall_clock_exceeded',5),('run_dir_threshold',5)])
def test_only_actual_budget_statuses_map_to_five(tmp_path,status,code):
    plan=frozen(tmp_path)
    with patch.object(cad,'validate_source',return_value=plan['source_link']):
        result=cad.execute(plan,supervisor=lambda *a,**k:{'status':status,'root_exit_code':None,'reclaim_confirmed':True})
    assert result['exit_code']==code and result['status']=='FAILED'


def test_native_identity_argv_reclaim_and_errors_are_required():
    valid=ok_native(['exe'],{'cwd':'cwd'})
    assert cad.native_ok(valid,['exe'],'cwd')
    for field,value in [('argv',['other']),('cwd','other'),('pid',None),('reclaim_confirmed',False),('errors',['error'])]:
        assert not cad.native_ok(dict(valid,**{field:value}),['exe'],'cwd')


def test_download_verifier_rejects_forged_native_record(tmp_path):
    plan=frozen(tmp_path);source=tmp_path/'source_run';source.mkdir()
    output=tmp_path/'outputs'/'cad';output.mkdir(parents=True)
    plan.update(schema='photo-cad-continuation-v1',run=str(source),output=str(output))
    result={'schema':'photo-cad-result-v1','status':'SUCCESS','exit_code':0,'source_and_code_stable':True,
            'preparation':{},'stages':[]}
    with patch.object(cad,'ROOT',tmp_path):
        for name,argv,_ in cad.commands(plan):
            evidence=output/(name+'_supervision');evidence.mkdir()
            native=ok_native(argv,{'cwd':str(output)})
            if name=='prepare':native['reclaim_confirmed']=False
            (evidence/'result.json').write_text(json.dumps(native))
            result['stages'].append({'name':name,'argv':argv,'cwd':str(output),'exit_code':0,
                                     'reclaim_confirmed':True,'supervisor':str(evidence/'result.json')})
        (output/'PLAN.json').write_text(json.dumps(plan));(output/'RESULT.json').write_text(json.dumps(result))
        with patch.object(cad,'validate_source',return_value=plan['source_link']),patch.object(cad,'prepared_bound',return_value={}):
            with pytest.raises(ValueError,match='native supervision did not pass'):
                cad.verify_cad_output(output,source)


@pytest.mark.parametrize('fault',[None,'path','sha256','current_bytes','frozen_bytes'])
def test_prepared_real_schema_projection_and_identity_rejection(tmp_path,fault):
    # Read the actual PASS record that exposed the path+sha256 versus full
    # binding mismatch. Its source is never modified by this regression.
    actual_path=cad.ROOT/'outputs/photo_cad_ui/20261011T154719Z_2a5a03d3648c/delivery/prepared/mesh_prep.json'
    if not actual_path.is_file():pytest.skip('historical real-produced schema record unavailable')
    produced=json.loads(actual_path.read_text(encoding='utf-8-sig'))
    schema={key:tuple(row) for key,row in produced['inputs'].items()}
    assert all(set(fields)=={'path','sha256'} for fields in schema.values())
    plan=frozen(tmp_path);write_prepared(plan)
    record=Path(plan['output'])/'prepared/mesh_prep.json'
    projected={'inputs':{key:{field:plan['source_link'][key][field] for field in schema[key]}
                         for key in schema},'source_validation':plan['source_link']}
    if fault=='path':
        other=tmp_path/'other.ply';other.write_bytes(Path(plan['mesh']).read_bytes());projected['inputs']['mesh']['path']=str(other)
    elif fault=='sha256':projected['inputs']['mesh']['sha256']='a'*64
    elif fault=='current_bytes':Path(plan['mesh']).write_bytes(b'changed bytes')
    elif fault=='frozen_bytes':plan['source_link']['mesh']['bytes']+=1
    record.write_text(json.dumps(projected))
    with patch.object(cad,'prepared_input',return_value={'internally_valid':True}):
        if fault is None:assert cad.prepared_bound(plan)=={'internally_valid':True}
        else:
            with pytest.raises(ValueError,match='another producer'):cad.prepared_bound(plan)


def test_existing_real_preparation_is_same_frozen_source_read_only():
    directory=cad.ROOT/'outputs/photo_cad_ui/20261011T154719Z_2a5a03d3648c/delivery'
    if not directory.is_dir():pytest.skip('historical produced preparation unavailable')
    plan=json.loads((directory/'PLAN.json').read_text(encoding='utf-8-sig'))
    checked=cad.prepared_bound(plan)
    assert checked['record']['path']==str(directory/'prepared/mesh_prep.json')


def checkpoint_fixture(tmp_path):
    run=tmp_path/'outputs/source';run.mkdir(parents=True)
    stages={'surface':{'status':'SUCCESS','artifacts':[]},
            'blender':{'status':'SUCCESS','reopen_verified':True,'sha256':'a'*64}}
    state={'target':'cad','status':'GEOMETRY_READY_FOR_CAD','geometry_deliverable_ready':True,
           'deliverable_ready':False,'stages':stages}
    path=run/'CAD_SOURCE.json'
    path.write_text(json.dumps(dict(state,schema='photo-cad-source-v1',source_run=str(run))))
    (run/'pipeline.json').write_text(json.dumps(state))
    return run,path,state


def test_immutable_checkpoint_survives_unrelated_final_state_update(tmp_path):
    run,path,state=checkpoint_fixture(tmp_path)
    first,row=cad.geometry_checkpoint(run,path)
    state.update(status='PARTIAL',deliverable_ready=True,finished_utc='later')
    state['stages']['cad']={'status':'SUCCESS'}
    state['stages']['report']={'status':'SUCCESS'}
    (run/'pipeline.json').write_text(json.dumps(state))
    assert cad.geometry_checkpoint(run,path)==(first,row)
    assert row==cad.binding(path,8<<20)


@pytest.mark.parametrize('fault',['source_run','schema','target','status','ready','current_geometry','current_target'])
def test_checkpoint_rejects_foreign_state_or_changed_geometry(tmp_path,fault):
    run,path,state=checkpoint_fixture(tmp_path)
    value=json.loads(path.read_text())
    if fault=='source_run':value['source_run']=str(tmp_path/'other')
    elif fault=='schema':value['schema']='other'
    elif fault=='target':value['target']='blender'
    elif fault=='status':value['status']='SUCCESS'
    elif fault=='ready':value['deliverable_ready']=True
    elif fault=='current_geometry':state['stages']['blender']['sha256']='b'*64
    elif fault=='current_target':state['target']='report'
    path.write_text(json.dumps(value));(run/'pipeline.json').write_text(json.dumps(state))
    with pytest.raises(ValueError):cad.geometry_checkpoint(run,path)


def test_checkpoint_rejects_any_other_name_and_nested_output_without_explicit_mode(tmp_path):
    run,path,state=checkpoint_fixture(tmp_path)
    other=run/'another.json';other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError,match='this run/CAD_SOURCE'):cad.geometry_checkpoint(run,other)
    with pytest.raises(ValueError,match='overlaps'):
        cad.plan(run,run/'cad',root=tmp_path)
    with pytest.raises(ValueError,match='this run/cad'):
        cad.plan(run,run/'another',geometry_record=path,root=tmp_path)
