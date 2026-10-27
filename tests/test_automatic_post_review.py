"""Offline post-review wiring tests; no model/native workers."""
import json
from pathlib import Path
import pytest
from integrations.demo_pipeline.pipeline import Options,plan,run_post_review,run_pipeline,save,read
from integrations.demo_pipeline.report import _post_review_outputs


def outcome(output,status):
    output.mkdir(parents=True);save(output/'RESULT.json',{'status':status});(output/'index.html').write_text('diagnostic only')


@pytest.mark.parametrize('code,status',[(0,'DIAGNOSTIC_COMPLETE'),(1,'INSUFFICIENT_DATA'),(2,'FAILED')])
def test_subprocess_contract_and_renderer_bindings(tmp_path,code,status):
    run=tmp_path/'run';run.mkdir();calls=[]
    def invoke(root,run,stage,argv,**kw):
        calls.append((stage,argv,kw));outcome(run/'post_review',status)
        return {'status':'completed','root_exit_code':code,'reclaim_confirmed':True,'errors':[],'log_errors':[]}
    detail=run_post_review(tmp_path,run,'frozen.exe',invoke)
    assert detail['status']==status and not detail['geometry_acceptance']
    stage,argv,kw=calls[0];assert stage=='post_review' and argv[0]=='frozen.exe'
    assert kw==dict(seconds=120,memory=1<<30,additional_roots=[])
    warnings=[];html,verified=_post_review_outputs(run,run/'report',{'post_review':detail},{},warnings)
    assert verified['available'] and not warnings and '非建模验收' in html
    (run/'post_review/index.html').write_text('drift')
    _,verified=_post_review_outputs(run,run/'report',{'post_review':detail},{},warnings)
    assert not verified['available'] and warnings


def test_unreclaimed_or_return_mismatch_refused(tmp_path):
    run=tmp_path/'run';run.mkdir()
    with pytest.raises(ValueError,match='reclamation'):
        run_post_review(tmp_path,run,'frozen',lambda *a,**k: {'status':'completed','root_exit_code':0,'reclaim_confirmed':False})
    def builder(**kw):
        outcome(kw['output'],'FAILED');return {'status':'completed','root_exit_code':0,'reclaim_confirmed':True}
    with pytest.raises(ValueError,match='disagrees'):
        run_post_review(tmp_path,run,'frozen',None,builder)


def test_default_runs_even_without_model_and_disabled_does_not(tmp_path):
    root=tmp_path/'repo';root.mkdir();source=tmp_path/'photos';source.mkdir()
    (source/'one.jpg').write_bytes(b'fixture');(source/'two.jpg').write_bytes(b'fixture2')
    opts=Options(source,'paper',output=root/'outputs/run',python=root/'missing.exe',target='report')
    assert plan(opts,root)['stages'][-2:]==['post_review','report']
    calls=[]
    def invoke(root,run,name,argv,**kw):
        assert name=='post_review';calls.append(name);outcome(run/'post_review','INSUFFICIENT_DATA')
        return {'status':'completed','root_exit_code':1,'reclaim_confirmed':True}
    def render(run):
        (run/'report').mkdir();p=run/'report/index.html';p.write_text('fixture report');return p
    run,code=run_pipeline(opts,root=root,executor=invoke,renderer=render)
    assert calls==['post_review'] and read(run/'pipeline.json')['stages']['post_review']['status']=='INSUFFICIENT_DATA'
    assert code==2
    opts.post_review=False;opts.output=root/'outputs/off'
    assert 'post_review' not in plan(opts,root)['stages']
    run_pipeline(opts,root=root,executor=invoke,renderer=render)
    assert calls==['post_review']


def test_cancel_does_not_launch_post_review(tmp_path):
    root=tmp_path/'repo';root.mkdir();source=tmp_path/'photos';source.mkdir()
    for i in range(2):(source/f'{i}.jpg').write_bytes(b'fixture')
    python=root/'python.exe';python.write_bytes(b'fixture');git=root/'git.exe';git.write_bytes(b'fixture')
    options=Options(source,'paper',output=root/'outputs/run',python=python,git_exe=git,target='report')
    calls=[]
    def invoke(root,run,stage,argv,**kw):
        calls.append(stage);raise KeyboardInterrupt()
    def render(run):
        (run/'report').mkdir();p=run/'report/index.html';p.write_text('report');return p
    run,code=run_pipeline(options,root=root,executor=invoke,renderer=render)
    assert calls==['input'] and code==130
    assert read(run/'pipeline.json')['stages']['post_review']['status']=='NOT_RUN_CANCELLED'


@pytest.mark.parametrize('cause',['wall_clock_exceeded','total_threshold','run_dir_threshold'])
def test_resource_stop_does_not_start_post_review(tmp_path,cause):
    root=tmp_path/'repo';root.mkdir();source=tmp_path/'photos';source.mkdir()
    for i in range(2):(source/f'{i}.jpg').write_bytes(b'fixture')
    python=root/'python.exe';python.write_bytes(b'fixture');git=root/'git.exe';git.write_bytes(b'fixture')
    options=Options(source,'paper',output=root/'outputs/run',python=python,git_exe=git,target='report')
    calls=[]
    def invoke(root,run,stage,argv,**kw):
        calls.append(stage)
        return {'status':'stopped','termination_reason':cause,'reclaim_confirmed':True}
    def render(run):
        (run/'report').mkdir();p=run/'report/index.html';p.write_text('retained cause report');return p
    run,code=run_pipeline(options,root=root,executor=invoke,renderer=render)
    state=read(run/'pipeline.json')
    assert calls==['input'] and code==5
    assert state['stages']['post_review']['status']=='NOT_RUN_RESOURCE_LIMIT'
    assert (run/'report/index.html').is_file()
