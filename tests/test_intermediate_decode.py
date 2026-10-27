"""No native process is launched by these extraction contract tests."""
import hashlib
import json
import time
from pathlib import Path
import pytest
from PIL import Image
from integrations.demo_pipeline import intermediate_decode as mod


def recipe(source, output):
    return ['ffmpeg','-hide_banner','-nostdin','-n','-loglevel','info','-copyts','-xerror',
            '-threads','2','-protocol_whitelist','file','-format_whitelist','mov,matroska,avi',
            '-err_detect','explode','-i',str(source),'-map','0:0','-an','-sn','-dn',
            '-filter_threads','2','-vf',"select='eq(n,0)+eq(n,2)'"+mod.SUFFIX,
            '-fps_mode','passthrough','-enc_time_base','filter','-avoid_negative_ts','disabled',
            '-c:v','png','-threads','2','-start_number','0','-f','image2',str(output)]


@pytest.fixture
def bound_case(tmp_path, monkeypatch):
    root=tmp_path/'video';root.mkdir()
    def save(path, value):
        path.write_text(json.dumps(value),encoding='utf-8')
    source=tmp_path/'source.mp4';source.write_bytes(b'bound source')
    tool=tmp_path/'ffmpeg.exe';tool.write_bytes(b'bound tool')
    endpoint=tmp_path/'endpoint.png';Image.new('RGB',(4,4),(10,20,30)).save(endpoint)
    info=mod.inspect_image(endpoint)
    timeline=dict(stream_index=0,width=4,height=4,rotation_degrees=0,time_base=[1,24],
                  frames=[dict(decode_index=i,pts=i,time_base=[1,24]) for i in range(3)])
    save(root/'timeline.json',timeline)
    argv=recipe(source,tmp_path/'old_%06d.png');argv[0]=str(tool)
    save(root/'extract.process.json',dict(argv=argv,returncode=0,env_overrides={}))
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    video=dict(contract_version='video-photo-v1',status='SUCCESS',source_video=str(source),
               source_sha256=h(source),source_sha256_after=h(source),decoded_count=3,
               timeline={k:v for k,v in timeline.items() if k!='frames'},tools={str(tool):h(tool)},
               frames=[dict(decode_index=0,pts=0,time_base=[1,24],status='PHOTO_IO_OK',
                            width=4,height=4,export_pts_verified=True,decoded_rgb_sha256=info['decoded_rgb_sha256'])])
    save(root/'RESULT.json',video)
    receipt=tmp_path/'RESULT.json'
    save(receipt,dict(contract_version='object-input-v1',status='INPUTS_PREPARED',input_kind='video',
                      reused_video_run=str(root),source_bindings=[dict(path=str(root/'RESULT.json'),sha256=h(root/'RESULT.json'))]))
    discovery=dict(source=str(source),stages=dict(input=dict(argv=['--output',str(tmp_path)])))
    frames=[dict(decode_index=0,pts=0,time_base=[1,24],path=str(endpoint),sha256=h(endpoint))]
    calls=[]
    def fake_run(argv, run, name, deadline, dataset, overrides):
        plan=json.loads((run.parent/'EXTRACTION_PLAN.json').read_text())
        assert plan['argv']==argv and plan['decode_indices']==[0,1]
        calls.append(argv)
        expression=argv[argv.index('-vf')+1]
        import re
        ids=[int(i) for i in re.findall(r'eq\(n,(\d+)\)',expression)]
        for n,i in enumerate(ids):
            Image.new('RGB',(4,4),(10,20,30) if i==0 else (40,50,60)).save(dataset/f'frame_{n:06d}.png')
        log=run/'extract.stderr.txt'
        log.write_text('[showinfo] config in time_base: 1/24\n'+''.join(
            f'[showinfo] n: {n} pts: {i} s:4x4\n' for n,i in enumerate(ids)))
        (run/'extract.process.json').write_text('{}')
        return run/'extract.stdout.txt',log
    monkeypatch.setattr(mod,'bounded_run',fake_run)
    return discovery,frames,tmp_path/'new',root,source,tool,calls


def run(case, indices=(0,1)):
    d,f,o,*_=case
    return mod.decode_intermediates(d,f,indices,o,[],deadline=time.monotonic()+10)


def test_extracts_originals_and_verifies_endpoint(bound_case):
    mapping,receipt=run(bound_case)
    assert list(mapping)==[0,1]
    assert mapping[0]['endpoint_verified'] is True
    assert mapping[1]['original_mask_status']=='UNKNOWN_NOT_SEGMENTED'
    assert mapping[1]['mask'] is None and mapping[1]['gray'].shape==(4,4)
    assert receipt['verified_endpoint_indices']==[0]
    json.dumps(receipt)
    assert len(bound_case[-1])==1


@pytest.mark.parametrize('indices',[(True,),(-1,),(0,0),(3,),(),tuple(range(65))])
def test_invalid_indices_never_launch(bound_case,indices):
    with pytest.raises(ValueError):run(bound_case,indices)
    assert not bound_case[-1]


@pytest.mark.parametrize('field', ['source','tool','recipe','endpoint'])
def test_binding_or_recipe_drift_never_launches(bound_case,field):
    d,f,o,root,source,tool,calls=bound_case
    if field in ('source','tool'):
        (source if field=='source' else tool).write_bytes(b'drift')
    elif field=='recipe':
        p=root/'extract.process.json';old=json.loads(p.read_text());old['argv'].insert(1,'-noautorotate');p.write_text(json.dumps(old))
    else:
        Image.new('RGB',(4,4),(200,0,0)).save(f[0]['path'])
        f[0]['sha256']=hashlib.sha256(Path(f[0]['path']).read_bytes()).hexdigest()
    with pytest.raises(ValueError):run(bound_case)
    assert not calls


def test_wrong_redecoded_endpoint_rejected(bound_case,monkeypatch):
    original=mod.bounded_run
    def wrong(*args,**kwargs):
        result=original(*args,**kwargs)
        Image.new('RGB',(4,4),(0,0,0)).save(args[4]/'frame_000000.png')
        return result
    monkeypatch.setattr(mod,'bounded_run',wrong)
    with pytest.raises(ValueError,match='redecoded endpoint RGB'):run(bound_case)


def test_expired_deadline_is_timeout(bound_case):
    d,f,o,*_=bound_case
    with pytest.raises(TimeoutError):
        mod.decode_intermediates(d,f,[0,1],o,[],deadline=time.monotonic()-1)
    assert not bound_case[-1]


@pytest.mark.parametrize('message,kind',[
    ('shared process deadline exceeded at extract',TimeoutError),
    ('process logs exceeded bound at extract',MemoryError),
    ('output size exceeded bound at extract',MemoryError),
    ('unknown limit',mod.LimitError),
])
def test_limits_preserve_resource_classification(bound_case,monkeypatch,message,kind):
    def fail(*args,**kwargs):raise mod.LimitError(message)
    monkeypatch.setattr(mod,'bounded_run',fail)
    with pytest.raises(kind):run(bound_case)


def test_extracted_pngs_bound_for_downstream_drift_detection(bound_case):
    d,f,o,*_=bound_case;refs=[]
    mapping,_=mod.decode_intermediates(d,f,[0,1],o,refs,deadline=time.monotonic()+10)
    paths={r['path'] for r in refs}
    assert all(row['path'] in paths for row in mapping.values())
    Image.new('RGB',(4,4),(1,2,3)).save(mapping[1]['path'])
    declared=next(r for r in refs if r['path']==mapping[1]['path'])
    with pytest.raises(ValueError,match='binding differs'):
        mod.bind(mapping[1]['path'],[],declared)


def test_native_failure_remains_runtime_error(bound_case,monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('extract exited 1')
    monkeypatch.setattr(mod,'bounded_run',fail)
    with pytest.raises(RuntimeError,match='extract exited'):run(bound_case)
