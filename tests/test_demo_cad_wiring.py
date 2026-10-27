"""Unified CAD orchestration only; no native execution."""
from pathlib import Path
import pytest
from test_photo_to_blender_pipeline import rig, native_fixture, renderer, good_surface
from integrations.demo_pipeline import pipeline as p


def cad_options(opts):
    opts.target='cad';opts.freecad_python=opts.python;opts.research_code6=False


def cad_artifacts(output):
    directory=output/'cad/cad';directory.mkdir(parents=True)
    refs=[]
    for name in ('model.FCStd','model.step'):
        path=directory/name;path.write_bytes(b'TEST_DATA_FACETED_CAD')
        refs.append({'path':str(path),'bytes':path.stat().st_size,'sha256':p.sha(path)})
    return refs


def test_cad_target_declares_tools_and_geometry_checkpoint(rig):
    root,opts=rig;cad_options(opts);calls=[];invoke,_=native_fixture()
    def builder(**kw):
        calls.append(kw)
        state=p.read(kw['run']/'pipeline.json');checkpoint=p.read(kw['geometry_record'])
        assert state['status']=='GEOMETRY_READY_FOR_CAD' and not state['deliverable_ready']
        assert checkpoint['schema']=='photo-cad-source-v1' and checkpoint['source_run']==str(kw['run'])
        assert checkpoint['stages']['surface']==state['stages']['surface']
        assert checkpoint['stages']['blender']==state['stages']['blender']
        assert checkpoint['geometry_deliverable_ready'] is True
        assert kw['output']==kw['run']/'cad'
        return {'status':'SUCCESS','exit_code':0,'artifacts':cad_artifacts(kw['output'])}
    planned=p.plan(opts,root)
    assert planned['dependencies']['freecad_python'] and planned['stages'].count('cad')==1
    run,code=p.run_pipeline(opts,root=root,executor=invoke,renderer=renderer,
                            surface_builder=good_surface,cad_builder=builder)
    final=p.read(run/'pipeline.json')
    assert len(calls)==1 and code==0 and final['target_achieved'] and final['deliverable_ready']
    assert final['status']=='SUCCESS' and p.read(run/'CAD_SOURCE.json')['status']=='GEOMETRY_READY_FOR_CAD'


@pytest.mark.parametrize('reason',['OPEN_MESH','INVALID_MESH_TOPOLOGY','backend unavailable'])
def test_cad_failure_preserves_blender_but_target_unachieved(rig,reason):
    root,opts=rig;cad_options(opts);invoke,_=native_fixture();calls=[]
    def builder(**kw):calls.append(kw);return {'status':'FAILED','exit_code':2,'error':reason}
    run,code=p.run_pipeline(opts,root=root,executor=invoke,renderer=renderer,
                            surface_builder=good_surface,cad_builder=builder)
    state=p.read(run/'pipeline.json')
    assert len(calls)==1 and code==2 and state['stages']['cad']['status']=='FAILED'
    assert state['stages']['blender']['status']=='SUCCESS' and state['geometry_deliverable_ready']
    assert not state['deliverable_ready'] and not state['target_achieved']
    assert reason in state['errors']


def test_cad_missing_tool_stops_before_input(rig):
    root,opts=rig;cad_options(opts);opts.freecad_python=root/'missing.exe';invoke,calls=native_fixture()
    run,code=p.run_pipeline(opts,root=root,executor=invoke,renderer=renderer,
                            cad_builder=lambda **kw:pytest.fail('CAD launched with missing dependency'))
    assert code==3 and calls==[] and not p.read(run/'pipeline.json')['target_achieved']


@pytest.mark.parametrize('target',['blender','report'])
def test_other_targets_never_start_cad(rig,target):
    root,opts=rig;opts.target=target;opts.research_code6=False;invoke,calls=native_fixture()
    run,code=p.run_pipeline(opts,root=root,executor=invoke,renderer=renderer,
        surface_builder=good_surface,cad_builder=lambda **kw:pytest.fail('CAD launched for another target'))
    assert 'cad' not in p.read(run/'pipeline.json')['stages'] and calls==['input','sfm','segmentation']
    assert not (run/'CAD_SOURCE.json').exists()


def test_default_cad_cli_once_and_same_source_verification(rig,monkeypatch):
    root,opts=rig;cad_options(opts);base,calls=native_fixture();cad_calls=[];verified=[]
    from integrations.demo_pipeline import cad
    def invoke(root,run,name,argv,**kwargs):
        if name!='cad_continuation':return base(root,run,name,argv,**kwargs)
        cad_calls.append(argv)
        assert argv[argv.index('--run')+1]==str(run)
        assert argv[argv.index('--geometry-record')+1]==str(run/'CAD_SOURCE.json')
        assert argv[argv.index('--freecad-python')+1]==str(opts.python)
        assert kwargs['seconds']==240 and kwargs['memory']==3<<30
        cad_artifacts(run/'cad')
        return {'status':'completed','root_exit_code':0,'reclaim_confirmed':True,'errors':[],'log_errors':[]}
    def verify(output,source):
        verified.append((output,source))
        refs=[]
        for path in (output/'cad/cad').iterdir():refs.append({'path':str(path),'bytes':path.stat().st_size,'sha256':p.sha(path)})
        return {'status':'CURRENT_VERIFIED_CAD_OUTPUT','artifacts':refs}
    monkeypatch.setattr(cad,'verify_cad_output',verify)
    run,code=p.run_pipeline(opts,root=root,executor=invoke,renderer=renderer,surface_builder=good_surface)
    assert len(cad_calls)==1 and verified==[(run/'cad',run)] and code==0
    assert p.read(run/'pipeline.json')['deliverable_ready']
    # A successfully generated solid does not prove whole-object shape quality.
    from integrations.demo_pipeline.quality import read_for_display
    state=p.read(run/'pipeline.json')
    quality=read_for_display(run,state)
    assert state['quality_reference']['sha256']==p.sha(run/'QUALITY.json')
    assert state['object_goal_achieved'] is False
    assert quality['object_shape_status']=='NOT_EVALUATED'
    assert quality['full_object_accepted'] is False
    assert 'quality_warning' not in quality  # terminal status did not invalidate the facts


def test_blender_unverified_never_enters_cad(rig):
    root,opts=rig;cad_options(opts);invoke,_=native_fixture()
    def invalid(**kw):
        result=good_surface(**kw);result['blender']['reopen_verified']=False;return result
    run,code=p.run_pipeline(opts,root=root,executor=invoke,renderer=renderer,surface_builder=invalid,
                           cad_builder=lambda **kw:pytest.fail('unverified geometry followed by CAD'))
    assert code==4 and not (run/'CAD_SOURCE.json').exists()


def test_partial_segmentation_with_verified_cad_remains_partial_ready(rig):
    root,opts=rig;cad_options(opts);invoke,_=native_fixture(seg='PARTIAL')
    def builder(**kw):return {'status':'SUCCESS','exit_code':0,'artifacts':cad_artifacts(kw['output'])}
    run,code=p.run_pipeline(opts,root=root,executor=invoke,renderer=renderer,surface_builder=good_surface,cad_builder=builder)
    state=p.read(run/'pipeline.json')
    assert code==1 and state['status']=='PARTIAL' and state['deliverable_ready']
    assert state['stages']['cad']['output']==str(run/'cad')


def test_cad_exception_marks_stage_failed_and_keeps_geometry(rig):
    root,opts=rig;cad_options(opts);invoke,_=native_fixture()
    def builder(**kw):raise RuntimeError('OPEN_MESH rejected')
    run,code=p.run_pipeline(opts,root=root,executor=invoke,renderer=renderer,surface_builder=good_surface,cad_builder=builder)
    state=p.read(run/'pipeline.json')
    assert code==4 and state['stages']['cad']['status']=='FAILED' and 'OPEN_MESH' in state['stages']['cad']['error']
    assert state['geometry_deliverable_ready'] and not state['deliverable_ready']
