"""Whole-video orchestration, using explicit fake executors only."""
from pathlib import Path
import json
import pytest
from integrations.demo_pipeline import pipeline as mod


def rig(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    source=tmp_path/'video.mp4';source.write_bytes(b'FAKE_VIDEO_NOT_DECODED')
    tools={}
    for name in ('python.exe','git.exe','ffmpeg.exe','ffprobe.exe','weight.pt'):
        p=root/name;p.write_bytes(b'FAKE_TOOL_NOT_EXECUTED');tools[name]=p
    package=root/'package';package.mkdir();mod.save(package/'BUNDLE.json',{})
    opts=mod.Options(source,'paper roll',output=root/'outputs/result',all_frames=True,
        python=tools['python.exe'],sam3_python=tools['python.exe'],sam3_package=package,
        checkpoint=tools['weight.pt'],git_exe=tools['git.exe'],ffmpeg=str(tools['ffmpeg.exe']),
        ffprobe=str(tools['ffprobe.exe']),target='report',research_code6=False,
        view_geometry_audit=False,post_review=False,point_selection='rcover-v1')
    return root,opts


def render(run):
    p=run/'report';p.mkdir();(p/'index.html').write_text('FAKE REPORT')
    return p/'index.html'


def test_plan_whole_video_is_explicit_and_bounded(tmp_path):
    root,opts=rig(tmp_path);p=mod.plan(opts,root)
    assert p['video_selection']=='all_decoded_frames' and p['video_frames'] is None
    assert p['video_frame_limit']==192 and p['segmentation_max_batches']==2
    assert p['sfm_profile']=='cad-object-video-v1' and not opts.output.exists()
    opts.all_frames=False;p=mod.plan(opts,root)
    assert p['video_frame_limit']==96 and p['sfm_profile']=='cad-object-v1'
    opts.frames=192
    with pytest.raises(ValueError):mod.plan(opts,root)


@pytest.mark.parametrize('stop_first',[False,True])
def test_two_batches_exact_cover_or_stop_without_following_native(tmp_path,monkeypatch,stop_first):
    from integrations.demo_pipeline import segmentation_batches
    root,opts=rig(tmp_path);calls=[];merged=[]
    def merge(paths,output,expected_indices):
        assert len(paths)==2 and expected_indices==list(range(192))
        merged.append(True);output.mkdir()
        frames=[dict(input_index=i,status='OK' if i in (0,100) else 'NO_MATCH',
            selected_mask={'path':'fake.png'} if i in (0,100) else None) for i in expected_indices]
        mod.save(output/'manifest.json',dict(status='PARTIAL',frames=frames))
        return dict(status='PARTIAL',counts={'requested':192,'success':2},run_dir=str(output),exit_code=1)
    monkeypatch.setattr(segmentation_batches,'merge_manifests',merge)
    def execute(root,run,stage,argv,**kw):
        calls.append(stage);code=0
        if stage=='input':
            assert '--all-frames' in argv
            io=run/'input/photo_io/fake';io.mkdir(parents=True)
            mod.save(io/'run.json',dict(status='SUCCESS'))
            mod.save(run/'input/RESULT.json',dict(status='INPUTS_PREPARED',photo_io={
                'run':{'path':str(io/'run.json'),'sha256':mod.sha(io/'run.json')},
                'identities':[{'input_index':i} for i in range(192)]}))
        elif stage.startswith('segmentation_batch_'):
            cfg=mod.read(Path(argv[argv.index('--config')+1]))
            n=int(stage[-2:]);assert cfg['indices']==list(range(n*96,(n+1)*96))
            d=Path(cfg['output_dir']);p=d/'product';p.mkdir()
            mod.save(p/'manifest.json',{'fixture':True})
            code=5 if stop_first and n==0 else 1
            mod.save(d/'PRODUCT_OUTCOME.json',dict(status='RESOURCE_LIMIT' if code==5 else 'PARTIAL',
                run_dir=str(p),counts={'requested':96},exit_code=code))
        elif stage=='sfm':
            assert argv[argv.index('--profile')+1]=='cad-object-video-v1'
            assert argv[argv.index('--point-selection')+1]=='rcover-v1'
            assert Path(argv[argv.index('--object-mask-manifest')+1])==run/'segmentation/merged/manifest.json'
            p=Path(argv[argv.index('--output')+1])/'fixture';p.mkdir(parents=True)
            mod.save(p/'run.json',dict(status='NO_MODEL',counts={},models=[]));code=1
        else:raise AssertionError(stage)
        return dict(status='completed',root_exit_code=code,reclaim_confirmed=True,errors=[],log_errors=[])
    run,code=mod.run_pipeline(opts,root=root,executor=execute,renderer=render)
    if stop_first:
        assert calls==['input','segmentation_batch_00'] and code==5 and not merged
    else:
        assert calls==['input','segmentation_batch_00','segmentation_batch_01','sfm'] and code==1
        assert merged and mod.read(run/'pipeline.json')['stages']['segmentation']['batch_count']==2
