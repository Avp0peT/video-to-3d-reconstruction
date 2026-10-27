"""Identity-bound positive box wiring; all model/processor behavior is mocked."""
from __future__ import annotations
from contextlib import contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'integrations/sam3'))
from sam3_adapter import backends, binding, runner, sourceio, worker
from sam3_adapter.visual_prompt import (VisualPromptError, validate_visual_prompt, validate_bound_frame,
                                        normalized_box, visual_prompt_sha256)

def document(width=1024,height=683,frame_id='f_example',sha='a'*64):
    return dict(kind='positive_boxes',image_domain='photo_io_canonical',annotation_source='ai_assisted',
                frames=[dict(input_index=59,frame_id=frame_id,image_sha256=sha,width=width,height=height,
                             box_xyxy=[230,70,705,637] if width==1024 else [3,2,35,20])])

def source_fixture(tmp_path):
    root=tmp_path/'io'; (root/'images').mkdir(parents=True)
    image=root/'images/000059.png'; Image.new('RGB',(40,24),(11,22,33)).save(image)
    raw_hash='b'*64
    row=dict(input_index=59,frame_id='f_fixture59',image_path='raw/59.png',image_sha256=raw_hash,
             image_output='images/000059.png',canonical_width=40,canonical_height=24,status='OK')
    (root/'frames.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
    (root/'run.json').write_text(json.dumps(dict(contract_version='photo-io-v1',run_id='fixture_visual',
        dataset_id='fixture',data_kind='real',status='SUCCESS',input_count=1,ok_count=1,error_count=0))+'\n',encoding='utf-8')
    return root,image,document(40,24,row['frame_id'],binding.sha256_file(image))

def request(tmp_path,source,prompt):
    return runner.RunRequest(io_run=source,indices=[59],object_id='skull-01',text_prompt=None,
        visual_prompt=prompt,output_root=tmp_path/'out',repo_dir=tmp_path/'repo',device='cpu',
        monitor_interval_seconds=0)

def test_frozen_box_normalization_and_no_mutation():
    value=document(); original=copy.deepcopy(value)
    validated=validate_visual_prompt(value,[59],None)
    assert validated==original and validated is not value
    assert normalized_box(validated['frames'][0])==[467.5/1024,353.5/683,475/1024,567/683]
    validated['frames'][0]['box_xyxy'][0]=0
    assert value==original

@pytest.mark.parametrize('field,value',[
    ('input_index',True),('input_index',-1),('input_index',60),('frame_id',''),('image_sha256','b'*63),
    ('image_sha256','A'*64),('width',True),('width',0),('height',683.0),
    ('box_xyxy',[True,70,705,637]),('box_xyxy',[230.,70,705,637]),
    ('box_xyxy',[float('nan'),70,705,637]),('box_xyxy',[float('inf'),70,705,637]),
    ('box_xyxy',[230,70,230,637]),('box_xyxy',[705,70,230,637]),
    ('box_xyxy',[-1,70,705,637]),('box_xyxy',[230,70,1025,637]),('box_xyxy',[230,70,705,684]),
    ('box_xyxy',[230,70,705])])
def test_malformed_frame_rejected(field,value):
    prompt=document();prompt['frames'][0][field]=value
    with pytest.raises(VisualPromptError):validate_visual_prompt(prompt,[59],None)

@pytest.mark.parametrize('change', ['negative','points','domain','source','missing','extra','duplicate','partial','other_index','text','bool_index'])
def test_unsupported_prompts_rejected(change):
    prompt=document();indices=[59];text=None
    if change=='negative':prompt['kind']='negative_boxes'
    if change=='points':prompt['frames'][0]['points']=[[1,2]]
    if change=='domain':prompt['image_domain']='undistorted'
    if change=='source':prompt['annotation_source']='human_truth'
    if change=='missing':del prompt['frames'][0]['frame_id']
    if change=='extra':prompt['label']=True
    if change=='duplicate':prompt['frames']*=2;indices=[59,60]
    if change=='partial':indices=[59,60]
    if change=='other_index':indices=[60]
    if change=='text':text='human skull'
    if change=='bool_index':indices=[True]
    with pytest.raises(VisualPromptError):validate_visual_prompt(prompt,indices,text)

def test_canonical_hash_is_distinct_from_original_and_bound(tmp_path):
    root,image,prompt=source_fixture(tmp_path)
    frame=sourceio.resolve_frame(sourceio.load_source_io(root),59)
    assert frame.photo_io_image_sha256!=frame.source_image_sha256==prompt['frames'][0]['image_sha256']
    validate_bound_frame(prompt,frame)
    prompt['frames'][0]['image_sha256']=frame.photo_io_image_sha256
    with pytest.raises(VisualPromptError,match='image_sha256'):validate_bound_frame(prompt,frame)

@pytest.mark.parametrize('key,value', [('frame_id','wrong'),('width',41),('height',25),('image_sha256','0'*64)])
def test_bound_identity_drift_rejected(tmp_path,key,value):
    root,image,prompt=source_fixture(tmp_path)
    frame=sourceio.resolve_frame(sourceio.load_source_io(root),59)
    prompt['frames'][0][key]=value
    with pytest.raises(VisualPromptError,match='identity drift'):validate_bound_frame(prompt,frame)

def test_request_json_roundtrip_and_independent_backend_config(tmp_path):
    root,image,prompt=source_fixture(tmp_path);req=request(tmp_path,root,prompt)
    payload=json.loads(json.dumps(runner._serialize_request(req),allow_nan=False))
    restored=runner.RunRequest(**payload)
    assert restored.text_prompt is None and restored.visual_prompt==prompt
    backend=runner._build_backend(restored)
    options=backend.effective_options()
    assert options['prompt_mode']=='positive_boxes'
    assert options['visual_prompt_sha256']==visual_prompt_sha256(prompt)
    assert options['autocast_scope']=='set_image+add_geometric_prompt'
    assert options['enable_inst_interactivity'] is False
    assert options['autocast_dtype'] is None # this fixture uses CPU
    payload['visual_prompt']['frames'][0]['box_xyxy'][0]=4
    assert backend.visual_prompt['frames'][0]['box_xyxy'][0]==3

def test_text_config_digest_matches_frozen_original(tmp_path):
    # Complete pre-box text configuration contract; no local RUN/old source import.
    for device in ('cpu','cuda'):
        cuda=device=='cuda'
        before={
            'builder':'sam3.model_builder.build_sam3_image_model','version':'sam3',
            'repo_dir':str(tmp_path.resolve()),
            'bpe_path':str(tmp_path.resolve()/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'),
            'bpe_sha256':None,'device':device,'eval_mode':True,'load_from_HF':False,
            'enable_segmentation':True,'enable_inst_interactivity':False,'compile':False,
            'inference_mode':True,'autocast_enabled':cuda,'autocast_device_type':'cuda' if cuda else None,
            'autocast_dtype':'bfloat16' if cuda else None,'autocast_scope':'set_image+set_text_prompt',
            'processor_resolution':1008,'processor_confidence_threshold':0.5,
            'mask_binarization':'model_side_sigmoid_logits_gt_0.5_after_bilinear_resample',
            'score_semantics':'presence_calibrated_sigmoid_product_not_probability',
            'image_builder_consumes_config_file':False,
        }
        after=backends.Sam3Backend(tmp_path,None,None,device,0.5).effective_options()
        expected_digest=hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                                                   allow_nan=False).encode('utf-8')).hexdigest()
        assert before==after and expected_digest==binding.effective_options_digest(after)
        assert 'prompt_mode' not in after and 'visual_prompt_sha256' not in after

class MockTorch:
    bfloat16=object()
    def __init__(self):
        self.active=[]; self.cuda=SimpleNamespace(empty_cache=lambda:None)
    @contextmanager
    def inference_mode(self):
        self.active.append('inference')
        try:yield
        finally:assert self.active.pop()=='inference'
    @contextmanager
    def autocast(self,**kwargs):
        assert kwargs==dict(device_type='cuda',dtype=self.bfloat16)
        self.active.append('bf16')
        try:yield
        finally:assert self.active.pop()=='bf16'

def backend_rig(tmp_path,monkeypatch):
    root,image,prompt=source_fixture(tmp_path)
    backend=backends.Sam3Backend(tmp_path,None,None,'cuda',0.5,visual_prompt=prompt)
    torch=MockTorch();trace=[];state={'masks':None,'scores':None}
    class Processor:
        def set_image(self,value):
            assert torch.active==['inference','bf16']; trace.append('image'); return state
        def add_geometric_prompt(self,*,box,label,state):
            assert torch.active==['inference','bf16']; assert box==normalized_box(prompt['frames'][0]) and label is True
            trace.append('box');return state
        def set_text_prompt(self,**kwargs):raise AssertionError('box-only called text prompt')
    backend._torch=torch;backend._processor=Processor()
    monkeypatch.setattr(backends,'candidates_from_processor_output',lambda *args,**kwargs:[])
    return root,image,prompt,backend,torch,trace

def test_real_segment_box_path_only_once_with_shared_bf16_context(tmp_path,monkeypatch):
    root,image,prompt,backend,torch,trace=backend_rig(tmp_path,monkeypatch)
    assert backend.segment(image,None,'skull-01',40,24,59)==[]
    assert trace==['image','box'] and torch.active==[]
    assert backend.effective_options()['autocast_dtype']=='bfloat16'

def test_segment_canonical_file_drift_stops_before_forward(tmp_path,monkeypatch):
    root,image,prompt,backend,torch,trace=backend_rig(tmp_path,monkeypatch)
    Image.new('RGB',(40,24),(99,88,77)).save(image)
    with pytest.raises(backends.BackendError,match='SHA256 drift'):backend.segment(image,None,'o',40,24,59)
    assert trace==[] and torch.active==[]

@pytest.mark.parametrize('kind',['text','index','dimensions','box_error'])
def test_segment_does_not_fallback(tmp_path,monkeypatch,kind):
    root,image,prompt,backend,torch,trace=backend_rig(tmp_path,monkeypatch)
    if kind=='box_error':
        def fail(**kwargs):raise RuntimeError('geometric failure')
        backend._processor.add_geometric_prompt=fail
    with pytest.raises(backends.BackendError):
        backend.segment(image,'skull' if kind=='text' else None,'o',41 if kind=='dimensions' else 40,24,60 if kind=='index' else 59)
    assert 'box' not in trace and torch.active==[]

def test_parent_invalid_box_refuses_backend_before_resources(tmp_path,monkeypatch):
    root,image,prompt=source_fixture(tmp_path);prompt['frames'][0]['frame_id']='wrong'
    def denied(*args,**kwargs):raise AssertionError('invalid input reached backend or resources')
    monkeypatch.setattr(runner,'_build_backend',denied);monkeypatch.setattr(runner,'ResourceMonitor',denied)
    result=runner.execute(request(tmp_path,root,prompt))
    assert result.status=='INPUT_ERROR' and result.exit_code==2
    assert result.manifest['object']['text_prompt'] is None and result.manifest['object']['visual_prompt']==prompt
    assert result.manifest['frames'][0]['errors'][0]['code']=='VISUAL_PROMPT_INVALID'

@pytest.mark.parametrize('candidate_count,status', [(0,'NO_MATCH'),(1,'OK'),(2,'NEEDS_SELECTION')])
def test_box_parent_manifest_and_selection_still_use_original_policy(tmp_path,monkeypatch,candidate_count,status):
    root,image,prompt,backend,torch,trace=backend_rig(tmp_path,monkeypatch)
    candidates=[backends.Candidate('c%02d'%i, bytes([255])*(40*24),40,24,.8,'test-only') for i in range(candidate_count)]
    monkeypatch.setattr(backends,'candidates_from_processor_output',lambda *a,**k:candidates)
    backend.load=lambda:None; backend.resolve_weights=lambda:None
    backend.describe=lambda:dict(name='sam3',code_commit='MOCK_ONLY',checkpoint_sha256='MOCK_ONLY',
        config_sha256=binding.effective_options_digest(backend.effective_options()),device='cuda')
    backend.bindings=lambda:dict(config={'effective_options':backend.effective_options()})
    monkeypatch.setattr(runner,'_build_backend',lambda req:backend)
    class Monitor:
        def __init__(self,**kwargs):pass
        def can_prove_headroom(self):return True,'test only'
        def sample(self):return dict(utc='fixture',process_commit_bytes=1)
        def violations(self,sample=None):return []
        def channel_report(self):return {'test_only':True}
    monkeypatch.setattr(runner,'ResourceMonitor',Monitor)
    outcome=runner.execute(request(tmp_path,root,prompt))
    manifest=outcome.manifest;frame=manifest['frames'][0]
    assert manifest['object']['text_prompt'] is None and manifest['object']['visual_prompt']==prompt
    assert manifest['object']['submitted_options']['visual_prompt']=='positive_boxes'
    assert frame['status']==status and len(frame['candidates'])==candidate_count
    assert trace==['image','box'] and torch.active==[]
    assert (frame['selected_mask'] is not None)==(status=='OK')
    assert manifest['quality_status']==manifest['cross_view_status']=='NOT_EVALUATED'
    if status=='OK':
        assert frame['selection']['method']=='unique_candidate'
        assert frame['photo_io_image_sha256']!=frame['source_image_sha256']==prompt['frames'][0]['image_sha256']
        assert (outcome.run_dir/'overlays/000059.png').is_file()

def test_valid_child_box_commits_actual_candidate_and_overlay_once(tmp_path,monkeypatch):
    root,image,prompt,backend,torch,trace=backend_rig(tmp_path,monkeypatch)
    backend.load=lambda:None;backend.resolve_weights=lambda:None
    candidate=backends.Candidate('c00',bytes([255])*(40*24),40,24,.8,'fixture_only')
    monkeypatch.setattr(backends,'candidates_from_processor_output',lambda *a,**k:[candidate])
    monkeypatch.setattr(runner,'_build_backend',lambda req:backend)
    job=dict(request=runner._serialize_request(request(tmp_path,root,prompt)),
        events_path=str(tmp_path/'events.jsonl'),run_dir=str(tmp_path),monitor_device=None,
        frames=[dict(input_index=59,source_image=str(image),width=40,height=24)])
    worker.write_job(tmp_path/'job.json',job)
    assert worker.child_main(tmp_path/'job.json')==0
    events,errors=worker.read_events_strict(tmp_path/'events.jsonl',allow_trailing_partial=False)
    assert errors==[]
    frames=[e for e in events if e['type']==worker.EVENT_FRAME]
    assert len(frames)==1 and frames[0]['input_index']==59 and frames[0]['status']=='OK'
    assert len(frames[0]['candidates'])==1 and frames[0]['selection']['method']=='unique_candidate'
    assert trace==['image','box'] and torch.active==[] and (tmp_path/'overlays/000059.png').is_file()

@pytest.mark.parametrize('tamper',['path','identity','duplicate','actual_bytes'])
def test_child_rebinds_box_before_monitor_or_model(tmp_path,monkeypatch,tamper):
    root,image,prompt=source_fixture(tmp_path);req=request(tmp_path,root,prompt)
    item=dict(input_index=59,source_image=str(image),width=40,height=24)
    if tamper=='path':item['source_image']=str(tmp_path/'other.png')
    if tamper=='identity':prompt['frames'][0]['image_sha256']='0'*64
    if tamper=='actual_bytes':Image.new('RGB',(40,24),(0,0,0)).save(image)
    job=dict(request=runner._serialize_request(req),events_path=str(tmp_path/'events.jsonl'),run_dir=str(tmp_path),
             monitor_device='cuda',frames=[item,item] if tamper=='duplicate' else [item])
    worker.write_job(tmp_path/'job.json',job)
    from sam3_adapter import resources
    def denied(*a,**k):raise AssertionError('invalid job reached model/monitor')
    monkeypatch.setattr(resources,'ResourceMonitor',denied);monkeypatch.setattr(runner,'_build_backend',denied)
    assert worker.child_main(tmp_path/'job.json')==2
    events,errors=worker.read_events_strict(tmp_path/'events.jsonl',allow_trailing_partial=False)
    assert errors==[] and events[-1]['kind']=='INPUT_ERROR' and events[-1]['stage']=='visual_prompt'

def cli_module():
    spec=importlib.util.spec_from_file_location('box_cli',ROOT/'scripts/segment_sam3.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_cli_loads_frozen_format_without_probe_or_ignored_fields(tmp_path,monkeypatch):
    cli=cli_module();prompt=document();p=tmp_path/'box.json';p.write_text(json.dumps(prompt),encoding='utf-8')
    seen={}
    def execute(req):
        seen['request']=req;return SimpleNamespace(run_dir=None,status='SUCCESS',exit_code=0,manifest=None)
    monkeypatch.setattr(cli,'execute',execute)
    assert cli.main(['--io-run',str(tmp_path),'--indices','59','--object-id','o','--visual-prompt',str(p),
        '--output',str(tmp_path/'out'),'--checkpoint',str(tmp_path/'weights'),'--checkpoint-sha256','a'*64])==0
    assert seen['request'].visual_prompt==prompt and seen['request'].text_prompt is None
    assert seen['request'].host_probe['status']=='NOT_RUN'
    with pytest.raises(SystemExit):cli.build_parser().parse_args(['--io-run','.','--indices','59','--object-id','o',
        '--output','out','--visual-prompt',str(p),'--text-prompt','skull'])

@pytest.mark.parametrize('text', ['{"kind":NaN}', '{"kind":"positive_boxes","kind":"points"}'])
def test_cli_strict_json_rejects_before_execute(tmp_path,monkeypatch,text):
    cli=cli_module();p=tmp_path/'box.json';p.write_text(text,encoding='utf-8')
    monkeypatch.setattr(cli,'execute',lambda req:pytest.fail('invalid prompt reached execution'))
    with pytest.raises(SystemExit):cli.main(['--io-run','.','--indices','59','--object-id','o',
        '--output',str(tmp_path/'out'),'--visual-prompt',str(p)])
