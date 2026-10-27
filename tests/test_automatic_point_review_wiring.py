import pytest
from integrations.demo_pipeline.pipeline import run_point_review, save
from integrations.demo_pipeline.report import _point_review_outputs

@pytest.mark.parametrize('code,status',[(0,'AUTO_POINT_REVIEW_COMPLETE'),(1,'INSUFFICIENT_DATA'),(2,'FAILED')])
def test_point_contract_and_bound_report(tmp_path,code,status):
    run=tmp_path/'run';run.mkdir();calls=[]
    def invoke(root,run,stage,argv,**limits):
        calls.append((stage,argv,limits));out=run/'point_review';out.mkdir()
        save(out/'RESULT.json',dict(status=status,geometry_accepted=False))
        (out/'index.html').write_text('diagnostic')
        return dict(status='completed',root_exit_code=code,reclaim_confirmed=True)
    detail=run_point_review(tmp_path,run,'frozen.exe',invoke)
    assert detail['status']==status and detail['geometry_accepted'] is False
    assert calls[0][0]=='point_review' and calls[0][1][-4:]==['--pipeline',str(run),'--output',str(run/'point_review')]
    assert calls[0][2]==dict(seconds=180,memory=2<<30,additional_roots=[])
    warnings=[]
    assert _point_review_outputs(run,run/'report',{'point_review':detail},{},warnings)[1]['available']
    (run/'point_review/index.html').write_text('drift')
    assert not _point_review_outputs(run,run/'report',{'point_review':detail},{},warnings)[1]['available']

@pytest.mark.parametrize('entry',[
    dict(status='stopped',termination_reason='wall_clock_exceeded',reclaim_confirmed=True),
    dict(status='stopped',termination_reason='job_memory_limit',reclaim_confirmed=True),
    dict(status='completed',root_exit_code=0,reclaim_confirmed=False),
    dict(status='stopped',termination_reason='user_cancelled',reclaim_confirmed=True),
])
def test_point_stop_blocks_following_diagnostic(tmp_path,entry):
    detail=run_point_review(tmp_path,tmp_path,'frozen',lambda *a,**k:entry)
    assert detail['blocks_post_review'] and 'record' not in detail

def test_acceptance_contract_refused(tmp_path):
    def builder(**kw):
        kw['output'].mkdir();save(kw['output']/'RESULT.json',dict(status='AUTO_POINT_REVIEW_COMPLETE',geometry_accepted=True))
        return dict(status='completed',root_exit_code=0,reclaim_confirmed=True)
    with pytest.raises(ValueError,match='acceptance'):
        run_point_review(tmp_path,tmp_path,'frozen',None,builder)

@pytest.mark.parametrize('stopped',[False,True])
def test_video_pipeline_point_before_post_and_resource_stop(tmp_path,monkeypatch,stopped):
    from integrations.demo_pipeline import pipeline as mod
    root=tmp_path/'repo';root.mkdir();source=tmp_path/'photos';source.mkdir()
    for i in range(2):(source/f'{i}.jpg').write_bytes(b'fixture')
    tool=root/'tool.exe';tool.write_bytes(b'fixture')
    opts=mod.Options(source,'object',output=root/'outputs/run',python=tool,git_exe=tool,target='report')
    original_plan=mod.plan
    def video_plan(*args):
        frozen=original_plan(*args);frozen['source']['kind']='video'
        frozen['dependencies'].update(ffmpeg=True,ffprobe=True,sam3_python=True,sam3_package=True,checkpoint=True)
        frozen['video_tools']=dict(ffmpeg=str(tool),ffprobe=str(tool))
        return frozen
    monkeypatch.setattr(mod,'plan',video_plan)
    calls=[]
    def invoke(root,run,stage,argv,**limits):
        calls.append(stage)
        if stage=='input':
            io=run/'input/io';io.mkdir(parents=True);save(io/'run.json',dict(status='SUCCESS',contract_version='photo-io-v1'))
            save(run/'input/RESULT.json',dict(status='INPUTS_PREPARED',photo_io=dict(run=dict(path=str(io/'run.json'),sha256=mod.sha(io/'run.json')),identities=[dict(input_index=0),dict(input_index=1)])))
        elif stage=='segmentation':
            out=run/'segmentation/product';out.mkdir();save(out/'manifest.json',dict(status='SUCCESS',frames=[]))
            save(run/'segmentation/PRODUCT_OUTCOME.json',dict(status='SUCCESS',run_dir=str(out),counts={}))
        elif stage=='point_review':
            if stopped:return dict(status='stopped',termination_reason='job_memory_limit',reclaim_confirmed=True)
            out=run/'point_review';out.mkdir();save(out/'RESULT.json',dict(status='INSUFFICIENT_DATA',geometry_accepted=False));(out/'index.html').write_text('points')
        elif stage=='post_review':
            out=run/'post_review';out.mkdir();save(out/'RESULT.json',dict(status='INSUFFICIENT_DATA'));(out/'index.html').write_text('post')
        return dict(status='completed',root_exit_code=1 if stage in ('point_review','post_review') else 0,reclaim_confirmed=True)
    def render(run):
        out=run/'report';out.mkdir();(out/'index.html').write_text('report');return out/'index.html'
    run,_=mod.run_pipeline(opts,root=root,executor=invoke,renderer=render)
    assert 'point_review' in calls
    if stopped:
        assert 'post_review' not in calls
        assert mod.read(run/'pipeline.json')['stages']['post_review']['status']=='NOT_RUN_RESOURCE_LIMIT'
    else:assert calls.index('point_review')<calls.index('post_review')
