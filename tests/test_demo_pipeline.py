"""Orchestration tests use explicit test data, never launch native workers."""
from pathlib import Path
import json, sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.demo_pipeline.pipeline import Options,plan,run_pipeline,save,read,sha,make_viewer,entry_code

@pytest.fixture
def rig(tmp_path):
    root=tmp_path/'repo';root.mkdir();source=tmp_path/'photos';source.mkdir()
    for i in range(2):(source/f'{i}.jpg').write_bytes(b'TEST_DATA_IMAGE'+bytes([i]))
    light=root/'python.exe';light.write_bytes(b'TEST_DATA')
    package=root/'package';package.mkdir();save(package/'BUNDLE.json',{})
    weight=root/'weight.pt';weight.write_bytes(b'TEST_DATA')
    options=Options(source,'test object',output=root/'outputs/demo',python=light,sam3_python=light,sam3_package=package,checkpoint=weight,target='report')
    return root,options

def executor_for(*, input_rc=0, sfm_status='NO_MODEL', seg_status='SUCCESS', bad_reclaim=None, delete_source=None):
    calls=[]
    def execute(root,run,stage,argv,**kwargs):
        calls.append(stage)
        if stage=='input' and not input_rc:
            io=run/'input/photo_io/fixture';io.mkdir(parents=True)
            save(io/'run.json',{'status':'SUCCESS','contract_version':'photo-io-v1'})
            save(run/'input/RESULT.json',{'status':'INPUTS_PREPARED','photo_io':{'run':{'path':str(io/'run.json'),'sha256':sha(io/'run.json')},'identities':[{'input_index':2},{'input_index':7}]}})
        if stage=='sfm':
            path=Path(argv[argv.index('--output')+1])/'test_run';path.mkdir(parents=True)
            save(path/'run.json',{'status':sfm_status,'counts':{'registered_unique_images':0,'unique_images':2},'models':[]})
        if stage=='segmentation':
            cfg=read(run/'segmentation/CONFIG.json');assert cfg['indices']==[2,7]
            product=run/'segmentation/product';product.mkdir()
            save(product/'manifest.json',{'status':seg_status,'frames':[{'input_index':i,'status':'OK','selected_mask':{'path':f'{i}.mask.png'}} for i in (2,7)]})
            save(run/'segmentation/PRODUCT_OUTCOME.json',{'run_dir':str(product),'status':seg_status,'exit_code':0 if seg_status=='SUCCESS' else 4,'counts':{'requested':2,'success':2 if seg_status=='SUCCESS' else 0}})
            if delete_source:delete_source.unlink()
        if stage=='post_review':
            output=run/'post_review';output.mkdir()
            save(output/'RESULT.json',{'status':'INSUFFICIENT_DATA','reason':'TEST_DATA: no native model'})
            (output/'index.html').write_text('TEST_DATA_DIAGNOSTIC',encoding='utf-8')
        return {'status':'completed','root_exit_code':input_rc if stage=='input' else 1 if stage=='post_review' or (stage=='sfm' and sfm_status=='NO_MODEL') else 0,
                'reclaim_confirmed':stage!=bad_reclaim,'errors':[],'log_errors':[]}
    return execute,calls

def render(run):
    path=run/'report';path.mkdir();(path/'index.html').write_text('TEST_DATA_REPORT');return path/'index.html'

def test_no_model_continues_segmentation_and_reports_partial(rig):
    root,opts=rig;execute,calls=executor_for()
    run,code=run_pipeline(opts,root=root,executor=execute,renderer=render)
    state=read(run/'pipeline.json')
    assert calls==['input','segmentation','sfm','post_review'] and code==1
    assert state['stages']['sfm']['status']=='NO_MODEL' and state['stages']['segmentation']['status']=='SUCCESS'
    assert state['status']=='PARTIAL' and state['stages']['report']['status']=='SUCCESS'
    assert (run/'report/index.html').is_file()
    assert state['stages']['post_review']['status']=='INSUFFICIENT_DATA'
    assert state['stages']['post_review']['record']['sha256']==sha(run/'post_review/RESULT.json')


def test_missing_git_rejected_before_input_and_sfm(rig,monkeypatch):
    from integrations.demo_pipeline import runtime
    root,opts=rig;execute,calls=executor_for()
    monkeypatch.setattr(runtime,'discover_git',lambda explicit=None:None)
    frozen=plan(opts,root)
    assert frozen['dependencies']['git_exe'] is False and frozen['git_exe'] is None
    run,code=run_pipeline(opts,root=root,executor=execute,renderer=render)
    assert code==3 and calls==['post_review']
    assert read(run/'pipeline.json')['stages']['sfm']['status']=='NOT_RUN'


def test_explicit_git_is_frozen_and_passed_to_segmentation_config(rig):
    root,opts=rig;git=root/'Git/git.exe';git.parent.mkdir();git.write_bytes(b'TEST_GIT')
    opts.git_exe=git
    execute,calls=executor_for()
    frozen=plan(opts,root)
    assert frozen['git_exe']==str(git.resolve()) and frozen['dependencies']['git_exe']
    run,code=run_pipeline(opts,root=root,executor=execute,renderer=render)
    assert read(run/'segmentation/CONFIG.json')['git_exe']==str(git.resolve())
    assert calls==['input','segmentation','sfm','post_review']

def test_input_failure_stops_native_but_keeps_report(rig):
    root,opts=rig;execute,calls=executor_for(input_rc=2)
    run,code=run_pipeline(opts,root=root,executor=execute,renderer=render)
    assert code==2 and calls==['input','post_review'] and read(run/'pipeline.json')['status']=='FAILED'
    assert (run/'report/index.html').is_file()

def test_unconfirmed_reclaim_stops_following_stage(rig):
    root,opts=rig;execute,calls=executor_for(bad_reclaim='sfm')
    run,code=run_pipeline(opts,root=root,executor=execute,renderer=render)
    assert calls==['input','segmentation','sfm'] and code==4
    assert read(run/'pipeline.json')['stages']['segmentation']['status']=='SUCCESS'
    assert read(run/'pipeline.json')['stages'].get('surface',{}).get('status','NOT_RUN')=='NOT_RUN'
    assert read(run/'pipeline.json')['stages'].get('view_geometry',{}).get('status','NOT_RUN')=='NOT_RUN'

def test_missing_input_at_final_check_still_reports(rig):
    root,opts=rig;execute,calls=executor_for(delete_source=opts.source/'0.jpg')
    run,code=run_pipeline(opts,root=root,executor=execute,renderer=render)
    assert code==2 and (run/'report/index.html').is_file()

def test_existing_output_refuses_all_stage_launches(rig):
    root,opts=rig;opts.output.mkdir(parents=True);execute,calls=executor_for()
    with pytest.raises(FileExistsError):run_pipeline(opts,root=root,executor=execute,renderer=render)
    assert not calls

def test_input_output_overlap_is_rejected(rig):
    root,opts=rig
    parent=root/'outputs/photos';parent.mkdir(parents=True)
    for i in range(2):(parent/f'{i}.jpg').write_bytes(b'TEST_DATA')
    opts.source=parent;opts.output=parent/'result'
    with pytest.raises(ValueError,match='重叠'):run_pipeline(opts,root=root,executor=lambda *a,**k:pytest.fail('launched'),renderer=render)

def test_plan_does_not_create_output(rig):
    root,opts=rig;result=plan(opts,root)
    assert result['automatic_retry'] is False and not opts.output.exists()

def test_report_render_failure_is_not_pass(rig):
    root,opts=rig;execute,_=executor_for()
    def fail(run):raise OSError('disk full')
    run,code=run_pipeline(opts,root=root,executor=execute,renderer=fail)
    assert code==4 and read(run/'pipeline.json')['stages']['report']['status']=='FAILED'


def test_upstream_checkpoint_wiring_is_producer_time_not_historical(rig,monkeypatch):
    from integrations.demo_pipeline import resume
    root,opts=rig;opts.automatic_recovery=False
    execute,calls=executor_for(sfm_status='SUCCESS')
    observed=[]
    def execute_with_product(*args,**kwargs):
        result=execute(*args,**kwargs)
        if args[2]=='segmentation':
            path=args[1]/'segmentation/PRODUCT_OUTCOME.json'
            data=read(path);data['run_dir']=str(args[1]/'segmentation/product');save(path,data)
        return result
    def checkpoint(run,**kwargs):
        state=read(run/'pipeline.json')
        observed.append((state['status'],state['exit_code'],state['stages']['segmentation']['status']))
        return {'path':str(run/'TEST_CHECKPOINT'),'sha256':'fixture','bytes':1}
    monkeypatch.setattr(resume,'write_resume_checkpoint',checkpoint)
    run,code=run_pipeline(opts,root=root,executor=execute_with_product,renderer=render)
    assert observed==[('RUNNING',None,'SUCCESS')]
    assert read(run/'pipeline.json')['resume_checkpoint']['status']=='COMMITTED'
    assert calls==['input','segmentation','sfm','post_review']


@pytest.mark.parametrize('native_code',[3,4,5])
def test_native_failure_exit_is_preserved_even_with_report(rig,native_code):
    root,opts=rig;execute,_=executor_for(seg_status='RESOURCE_LIMIT' if native_code==5 else 'BACKEND_ERROR')
    def failing(*args,**kwargs):
        entry=execute(*args,**kwargs)
        if args[2]=='segmentation':entry['root_exit_code']=native_code
        return entry
    run,code=run_pipeline(opts,root=root,executor=failing,renderer=render)
    assert code==native_code
    assert read(run/'pipeline.json')['stages']['report']['status']=='SUCCESS'


@pytest.mark.parametrize('reason,expected',[('wall_clock_exceeded',5),('total_threshold',5),('run_dir_threshold',5),('user_cancelled',130)])
def test_resource_stop_prevents_next_native_stage(rig,reason,expected):
    root,opts=rig;execute,calls=executor_for()
    def stopped(*args,**kwargs):
        entry=execute(*args,**kwargs)
        if args[2]=='sfm':entry.update(status=reason,termination_reason=reason)
        return entry
    run,code=run_pipeline(opts,root=root,executor=stopped,renderer=render)
    assert code==expected and calls==['input','segmentation','sfm']
    assert read(run/'pipeline.json')['stages']['segmentation']['status']=='SUCCESS'
    assert read(run/'pipeline.json')['stages'].get('surface',{}).get('status','NOT_RUN')=='NOT_RUN'
    assert read(run/'pipeline.json')['stages'].get('view_geometry',{}).get('status','NOT_RUN')=='NOT_RUN'


def test_video_entry_uses_declared_count_and_absolute_tools(rig,monkeypatch):
    root,opts=rig;video=opts.source.parent/'input.mp4';video.write_bytes(b'TEST_DATA_VIDEO')
    opts.source=video;opts.frames=8
    (root/'ffmpeg.exe').write_bytes(b'TEST_TOOL');(root/'ffprobe.exe').write_bytes(b'TEST_TOOL')
    monkeypatch.chdir(root);opts.ffmpeg='ffmpeg.exe';opts.ffprobe='ffprobe.exe'
    execute,_=executor_for();captured=[]
    def record(*args,**kwargs):
        captured.append(args[3]);return execute(*args,**kwargs)
    run,code=run_pipeline(opts,root=root,executor=record,renderer=render)
    command=captured[0]
    assert command[3]=='video' and command[command.index('--count')+1]=='8'
    for flag in ('--ffmpeg','--ffprobe'):assert Path(command[command.index(flag)+1]).is_absolute()
    assert read(run/'pipeline.json')['source_kind']=='video'

@pytest.mark.parametrize('entry',[
    {'status':'completed','root_exit_code':0,'reclaim_confirmed':False},
    {'status':'completed','root_exit_code':0,'reclaim_confirmed':True,'errors':['x']},
    {'status':'completed','root_exit_code':0,'reclaim_confirmed':True,'log_errors':['x']},
    {'status':'supervisor_error','root_exit_code':0,'reclaim_confirmed':True},
])
def test_supervisor_error_never_pass(entry):assert entry_code(entry)!=0

def test_real_viewer_export_from_explicit_synthetic_sidecar(tmp_path):
    # TEST_DATA: rendering only, not evidence of photo reconstruction.
    sfm=tmp_path/'sfm';run=tmp_path/'run';sfm.mkdir();run.mkdir()
    save(sfm/'desc.json',{'source':{'points3D':{'5':{'xyz':[1.25,-2.0,3.0],'color':[10,20,30]},'9':{'xyz':[2,0,0],'color':[100,110,120]}}}})
    save(sfm/'run.json',{'status':'SUCCESS','counts':{'registered_unique_images':2,'unique_images':2},'models':[
        {'usable':True,'roundtrip_verified':True,'num_points3D':2,'num_registered_images':2,'model_id':'0','descriptions_path':'desc.json'}]})
    result=make_viewer(sfm,run)
    assert result['status']=='SUCCESS' and result['stats']['points3d']==2
    assert '1.25 -2 3 10 20 30' in (run/'pointcloud/component_00.ply').read_text()
    assert (run/'pointcloud/viewer.html').is_file()

def test_viewer_does_not_accept_escaped_description(tmp_path):
    sfm=tmp_path/'sfm';run=tmp_path/'run';sfm.mkdir();run.mkdir();save(tmp_path/'outside.json',{})
    save(sfm/'run.json',{'status':'SUCCESS','models':[{'usable':True,'roundtrip_verified':True,'descriptions_path':'../outside.json'}]})
    with pytest.raises(ValueError,match='escapes'):make_viewer(sfm,run)
