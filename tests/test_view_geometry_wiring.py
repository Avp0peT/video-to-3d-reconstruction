"""Synthetic wiring evidence only; no native models or reconstruction executed."""
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_photo_to_blender_pipeline import rig as base_rig,native_fixture,renderer,good_surface
from integrations.demo_pipeline import pipeline as p
from integrations.demo_pipeline.report import _view_geometry_outputs

@pytest.fixture
def rig(tmp_path_factory,monkeypatch):
    # Short fixture alias keeps nested native paths below Win32 MAX_PATH.
    return base_rig.__wrapped__(tmp_path_factory.mktemp('v'),monkeypatch)

def binding(path):
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=p.sha(path))

def native_with_model():
    base,calls=native_fixture()
    def execute(root,run,name,argv,**kw):
        result=base(root,run,name,argv,**kw)
        if name=='sfm':
            native=Path(argv[argv.index('--output')+1])/'native'
            model=native/'sparse/0';model.mkdir(parents=True)
            for filename in ('cameras.bin','images.bin','points3D.bin'):(model/filename).write_bytes(b'SYNTHETIC_NOT_COLMAP')
            record=p.read(native/'run.json');record['models']=[dict(model_id='0',usable=True,roundtrip_verified=True,num_registered_images=3,num_points3D=1,binary_path='sparse/0')];p.save(native/'run.json',record)
        return result
    return execute,calls

def diagnostic(**kw):
    output=kw['output'];output.mkdir()
    for name in ('PLAN.json','TRACKS.json','report.html'):(output/name).write_text('SYNTHETIC_WIRING_ONLY')
    sfm=kw['sfm_run'];record=p.read(sfm/'run.json')
    result=dict(schema='native-view-geometry-v1',status='PARTIAL_DIAGNOSTIC',source_model_id=kw['model_id'],source_sfm_status=record['status'],counts={'synthetic':1},geometry_acceptance=False,scale_status='unknown',source_bindings=[binding(sfm/'run.json')]+[binding(sfm/'sparse/0'/name) for name in ('cameras.bin','images.bin','points3D.bin')],artifacts=[binding(output/name) for name in ('PLAN.json','TRACKS.json','report.html')])
    p.save(output/'RESULT.json',result);return result

def test_default_and_disable(rig):
    root,options=rig
    assert p.plan(options,root)['stages'].index('view_geometry')==3
    options.view_geometry_audit=False
    assert 'view_geometry' not in p.plan(options,root)['stages']
    execute,calls=native_with_model()
    p.run_pipeline(options,root=root,executor=execute,renderer=renderer,surface_builder=good_surface,view_geometry_builder=lambda **k:pytest.fail('disabled'))
    assert 'view_geometry' not in calls

def test_injected_diagnostic_runs_after_object_first_sfm_and_retains_geometry(rig):
    root,options=rig;execute,calls=native_with_model()
    def build(**kw):
        assert calls==['input','segmentation','sfm'];assert kw['model_id']=='0';return diagnostic(**kw)
    run,code=p.run_pipeline(options,root=root,executor=execute,renderer=renderer,surface_builder=good_surface,view_geometry_builder=build)
    state=p.read(run/'pipeline.json');record=p.read(run/'REPORT_DATA.json')['view_geometry']
    assert code==0 and state['deliverable_ready'] and record['status']=='PARTIAL_DIAGNOSTIC'
    warnings=[];html,verified=_view_geometry_outputs(run,run/'report',{'view_geometry':record},state,warnings)
    assert verified['available'] and 'TRACKS.json' in html and not warnings
    (run/'view_geometry/TRACKS.json').write_text('DRIFT')
    html,verified=_view_geometry_outputs(run,run/'report',{'view_geometry':record},state,[])
    assert not verified['available'] and 'href=' not in html

@pytest.mark.parametrize('fault',['model','source','artifact','acceptance','missing_plan','duplicate_source'])
def test_invalid_binding_is_failed_diagnostic_and_geometry_survives(rig,fault):
    root,options=rig;execute,calls=native_with_model()
    def build(**kw):
        result=diagnostic(**kw)
        if fault=='model':result['source_model_id']='wrong'
        if fault=='source':result['source_bindings'][0]['sha256']='0'*64
        if fault=='artifact':result['artifacts'][0]['sha256']='0'*64
        if fault=='acceptance':result['geometry_acceptance']=True
        if fault=='missing_plan':result['artifacts']=result['artifacts'][1:]
        if fault=='duplicate_source':result['source_bindings'].append(result['source_bindings'][0])
        p.save(kw['output']/'RESULT.json',result);return result
    run,code=p.run_pipeline(options,root=root,executor=execute,renderer=renderer,surface_builder=good_surface,view_geometry_builder=build)
    state=p.read(run/'pipeline.json')
    assert code==0 and state['deliverable_ready'] and state['stages']['view_geometry']['status']=='FAILED'
    assert 'result' not in state['stages']['view_geometry']

@pytest.mark.parametrize('reason,reclaim,expected',[('wall_clock_exceeded',True,5),('user_cancelled',True,130),('reclaim_unconfirmed',False,4),('normal_failure',True,0)])
def test_bounded_cli_and_stop_semantics(rig,reason,reclaim,expected):
    root,options=rig;execute,calls=native_with_model()
    def wrapper(root,run,name,argv,**kw):
        if name=='view_geometry':
            assert kw['seconds']==90 and kw['memory']==1<<30
            assert Path(argv[argv.index('--sfm-run')+1]).name=='native'
            assert argv[argv.index('--model-id')+1]=='0'
            return dict(status='completed' if reason=='normal_failure' else 'terminated',root_exit_code=2,reclaim_confirmed=reclaim,termination_reason=reason,errors=[],log_errors=[])
        return execute(root,run,name,argv,**kw)
    run,code=p.run_pipeline(options,root=root,executor=wrapper,renderer=renderer,surface_builder=good_surface)
    assert code==expected
    assert 'segmentation' in calls
    assert p.read(run/'pipeline.json')['stages']['segmentation']['status']=='SUCCESS'
    if reason!='normal_failure':
        assert p.read(run/'pipeline.json')['stages']['surface']['status']=='NOT_RUN'
