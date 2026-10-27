"""Real receipt/file bindings for the video diagnostic input boundary; no decoder."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from integrations.demo_pipeline import temporal_review as review


def write(path, value, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(('\n'.join(json.dumps(r) for r in value) if lines else json.dumps(value)), encoding='utf-8')


def binding(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


@pytest.fixture
def chain(tmp_path):
    pipeline = tmp_path/'pipeline'; t01 = pipeline/'input/video/photo_io/run'
    video_root = pipeline/'input/video'; seg = pipeline/'segmentation/product'; sfm = tmp_path/'sfm'
    source = tmp_path/'source.mp4'; source.write_bytes(b'fixture video identity; never decoded')
    t01.mkdir(parents=True); seg.mkdir(parents=True); sfm.mkdir()
    run = dict(contract_version='photo-io-v1', run_id='t01-fixture', status='SUCCESS', input_count=2, ok_count=2, error_count=0)
    write(t01/'run.json', run)
    rows=[]; identities=[]; canonical=[]; masks=[]; vrows=[]; frows=[]
    for n in range(2):
        rgb=np.full((24,32,3), 40+n*30, dtype=np.uint8)
        image=t01/f'image{n}.png'; Image.fromarray(rgb).save(image)
        ref=binding(image); raw_sha=hashlib.sha256(f'raw camera bytes {n}'.encode()).hexdigest()
        digest=hashlib.sha256(b'32x24:RGB:'+rgb.tobytes()).hexdigest(); fid=f'frame-{n}'
        rows.append(dict(input_index=n,status='OK',frame_id=fid,image_output=image.name,image_sha256=raw_sha,canonical_width=32,canonical_height=24))
        canonical.append(ref); identities.append(dict(input_index=n,frame_id=fid,decoded_rgb_sha256=digest,prepared_dimensions=[32,24]))
        mp=seg/f'mask{n}.png'; Image.fromarray(np.full((24,32),255,np.uint8)).save(mp); mh=binding(mp)['sha256']
        masks.append(dict(input_index=n,frame_id=fid,source_image=str(image),source_image_sha256=ref['sha256'],photo_io_image_sha256=raw_sha,image_domain='photo_io_canonical',width=32,height=24,status='OK',selected_candidate_id='c0',selected_mask=dict(path=mp.name,sha256=mh),candidates=[dict(candidate_id='c0',mask_path=mp.name,mask_sha256=mh)]))
        vrows.append(dict(status='PHOTO_IO_OK',photo_input_index=n,output_index=n,decode_index=n*3,pts=n*3,time_base=[1,30],time_seconds=n/10,photo_frame_id=fid,decoded_rgb_sha256=digest))
        frows.append(dict(input_index=n,frame_id=fid,canonical_image_sha256=ref['sha256'],upstream_io_run_id=run['run_id'],registrations=[dict(model_id=0,image_id=7+n)]))
    write(t01/'frames.jsonl',rows,True)
    photo=dict(run=binding(t01/'run.json'),frames=binding(t01/'frames.jsonl'),identities=identities,canonical_images=canonical,count=2)
    video=dict(contract_version='video-photo-v1',status='SUCCESS',source_video=str(source),source_sha256=binding(source)['sha256'],source_sha256_after=binding(source)['sha256'],photo_io=dict(run=str(t01/'run.json'),sha256=photo['run']['sha256'],frames_sha256=photo['frames']['sha256']),frames=vrows,decoded_count=4,selection={'fixture':True})
    manifest=dict(contract_version='object-mask-v1',source_io=dict(path=str(t01),run_id=run['run_id'],run_json_sha256=photo['run']['sha256'],frames_jsonl_sha256=photo['frames']['sha256']),frames=masks,object={'prompt':'roll'})
    receipt=dict(contract_version='object-input-v1',status='INPUTS_PREPARED',input_kind='video',reused_video_run=str(video_root),source_bindings=[],photo_io=photo)
    discovery=dict(source_kind='video',source=str(source),stages=dict(input=dict(argv=['prepare','--output',str(pipeline/'input')]),segmentation=dict(run_dir=str(seg)),sfm=dict(run_json=str(sfm/'run.json'))))
    write(sfm/'run.json',dict(upstream_io_run_id=run['run_id'],models=[dict(model_id=0,registered_image_ids=[7,8])]))
    write(sfm/'frames_map.jsonl',frows,True)
    def commit():
        states=Counter(r['status'] for r in manifest['frames'])
        manifest['counts']=dict(requested=len(masks),success=states['OK'],no_match=states['NO_MATCH'],needs_selection=states['NEEDS_SELECTION'],not_run=states['NOT_RUN'],error=sum(states[s] for s in ('INPUT_ERROR','BACKEND_ERROR','RESOURCE_LIMIT')))
        write(video_root/'RESULT.json',video)
        receipt['source_bindings']=[binding(video_root/'RESULT.json')]
        write(pipeline/'input/RESULT.json',receipt); write(seg/'manifest.json',manifest); write(pipeline/'pipeline.json',discovery)
    commit()
    return dict(pipeline=pipeline,discovery=discovery,video=video,receipt=receipt,manifest=manifest,sfm=sfm,frows=frows,commit=commit,t01=t01)


def load(c):
    return review.load_inputs(c['pipeline'],c['discovery'],[])


def test_complete_chain_allows_raw_canonical_difference_and_is_read_only(chain):
    paths=list(chain['pipeline'].rglob('*'))+list(chain['sfm'].rglob('*'))
    before={p:p.read_bytes() for p in paths if p.is_file()}
    frames,grays,masks,meta=load(chain)
    assert len(frames)==len(grays)==len(masks)==2
    assert all(f['model_ids']==['0'] for f in frames)
    assert meta['new_decoding'] is meta['new_segmentation'] is False
    assert all(m.dtype==np.uint8 and m.shape==(24,32) for m in masks)
    assert chain['manifest']['frames'][0]['photo_io_image_sha256'] != frames[0]['sha256']
    assert before=={p:p.read_bytes() for p in before}


def test_canonical_declared_hash_rejected(chain):
    chain['receipt']['photo_io']['canonical_images'][0]['sha256']='0'*64; chain['commit']()
    with pytest.raises(ValueError,match='binding differs'): load(chain)


@pytest.mark.parametrize('change',[{'time_seconds':.11},{'pts':True},{'time_base':[0,30]},{'decode_index':0},{'pts':0,'time_seconds':0}])
def test_bad_timeline_rejected(chain,change):
    chain['video']['frames'][1].update(change); chain['commit']()
    with pytest.raises(ValueError): load(chain)


def test_wrong_selected_candidate_rejected(chain):
    chain['manifest']['frames'][0]['selected_candidate_id']='other'; chain['commit']()
    with pytest.raises(ValueError,match='selected mask/candidate'): load(chain)


@pytest.mark.parametrize('status',['NO_MATCH','NEEDS_SELECTION'])
def test_non_ok_candidate_is_not_silently_selected(chain,status):
    row=chain['manifest']['frames'][0]; row['status']=status
    # An unselected candidate remains present; it must not become a mask.
    row['selected_mask']=None; row['selected_candidate_id']=None
    row['candidates'][0]['mask_path']='does-not-exist.png'; chain['commit']()
    frames,_,masks,meta=load(chain)
    assert masks[0] is None and masks[1] is not None
    assert frames[0]['mask_binding'] is None and frames[0]['mask_pixels']==0
    assert meta['source_mask_counts'][status]==1


@pytest.mark.parametrize('field,value',[('frame_id','wrong'),('canonical_image_sha256','0'*64),('upstream_io_run_id','other')])
def test_sfm_frame_identity_rejected(chain,field,value):
    chain['frows'][0][field]=value; write(chain['sfm']/'frames_map.jsonl',chain['frows'],True)
    with pytest.raises(ValueError,match='SfM frame identity'): load(chain)


def test_sfm_registration_membership_rejected(chain):
    chain['frows'][0]['registrations'][0]['image_id']=100
    write(chain['sfm']/'frames_map.jsonl',chain['frows'],True)
    with pytest.raises(ValueError,match='registration not bound'): load(chain)


@pytest.mark.parametrize('kind',['missing_toolchain','not_video','no_manifest'])
def test_not_available_never_imports_cv_or_tracks(chain,monkeypatch,tmp_path,kind):
    def forbidden(*args,**kwargs): raise AssertionError('native boundary entered')
    monkeypatch.setattr(review.importlib,'import_module',forbidden)
    monkeypatch.setattr(review,'track_sequence',forbidden)
    monkeypatch.setattr(review,'CV_ROOT',tmp_path/'absent-toolchain')
    if kind=='not_video': chain['discovery']['source_kind']='photos'
    if kind=='no_manifest': chain['discovery']['stages'].pop('segmentation')
    result=review.review_video(chain['pipeline'],tmp_path/'review',chain['discovery'])
    assert result['status']=='NOT_AVAILABLE' and result['model_executions']==0
    assert result['source_unchanged'] is True and result['correspondence_verified'] is False
    page=Path(result['report_path']).read_text(encoding='utf-8')
    assert 'href="TRACKS.json"' not in page and 'href="GEOMETRY.json"' not in page
    assert not (tmp_path/'review/TRACKS.json').exists() and not (tmp_path/'review/GEOMETRY.json').exists()


@pytest.mark.parametrize('destination',['existing','input','internal','sfm','segmentation','source'])
def test_output_scope_rejected_before_writes(chain,tmp_path,destination):
    paths=dict(existing=tmp_path/'existing',input=chain['pipeline']/'input/new',internal=chain['pipeline']/'arbitrary',sfm=chain['sfm']/'review',segmentation=Path(chain['discovery']['stages']['segmentation']['run_dir'])/'review',source=Path(chain['discovery']['source'])/'review')
    target=paths[destination]
    if destination=='existing': target.mkdir(); (target/'sentinel').write_bytes(b'unchanged')
    with pytest.raises(ValueError): review.review_video(chain['pipeline'],target,chain['discovery'])
    if destination=='existing': assert (target/'sentinel').read_bytes()==b'unchanged'
    else: assert not target.exists()


def test_reserved_automatic_output_is_allowed_without_cv(chain,monkeypatch,tmp_path):
    monkeypatch.setattr(review,'CV_ROOT',tmp_path/'absent-toolchain')
    output=chain['pipeline']/'post_review/temporal_tracks'
    result=review.review_video(chain['pipeline'],output,chain['discovery'])
    assert result['status']=='NOT_AVAILABLE' and (output/'RESULT.json').is_file()


def rebind_t01(chain):
    photo=chain['receipt']['photo_io']
    photo['run']=binding(chain['t01']/'run.json')
    chain['video']['photo_io']['sha256']=photo['run']['sha256']
    chain['manifest']['source_io']['run_json_sha256']=photo['run']['sha256']
    chain['commit']()


def test_t01_wrong_contract_rejected_after_valid_binding(chain):
    p=chain['t01']/'run.json'; row=json.loads(p.read_text()); row['contract_version']='other'; write(p,row)
    rebind_t01(chain)
    with pytest.raises(ValueError,match='photo-io-v1'): load(chain)


def test_t01_frames_must_belong_to_bound_run(chain,tmp_path):
    p=tmp_path/'foreign/frames.jsonl'; p.parent.mkdir(); p.write_bytes((chain['t01']/'frames.jsonl').read_bytes())
    chain['receipt']['photo_io']['frames']=binding(p); chain['commit']()
    with pytest.raises(ValueError,match='belong to bound run'): load(chain)


@pytest.mark.parametrize('source',['t01','identities','masks'])
def test_boolean_identity_key_rejected(chain,source):
    if source=='identities': chain['receipt']['photo_io']['identities'][0]['input_index']=False
    elif source=='masks': chain['manifest']['frames'][0]['input_index']=False
    else:
        p=chain['t01']/'frames.jsonl'; rows=[json.loads(s) for s in p.read_text().splitlines()]
        rows[0]['input_index']=False; write(p,rows,True)
        ref=binding(p); chain['receipt']['photo_io']['frames']=ref
        chain['video']['photo_io']['frames_sha256']=ref['sha256']; chain['manifest']['source_io']['frames_jsonl_sha256']=ref['sha256']
    chain['commit']()
    with pytest.raises(ValueError,match='integer'): load(chain)


@pytest.mark.parametrize('kind',['video','photos'])
def test_default_review_video_seam_without_sfm_retains_main_status(chain,tmp_path,monkeypatch,kind):
    from integrations.demo_pipeline import post_review
    chain['discovery']['source_kind']=kind; chain['discovery']['stages'].pop('sfm'); chain['commit']()
    calls=[]
    def seam(pipeline,output,discovery=None):
        calls.append((pipeline,output,discovery))
        output.mkdir(); (output/'index.html').write_text('candidate report')
        return dict(status='DIAGNOSTIC_COMPLETE',report_path=str(output/'index.html'))
    monkeypatch.setattr(review,'review_video',seam)
    result=post_review.review(chain['pipeline'],tmp_path/'automatic')
    assert result['status']=='INSUFFICIENT_DATA'
    assert result['reason']=='No committed native SfM run in pipeline'
    assert result['native_model_executions']==0 and result['correspondence_verified'] is False
    report=Path(result['report_path']).read_text(encoding='utf-8')
    if kind=='video':
        assert len(calls)==1 and calls[0][2]==chain['discovery']
        assert result['temporal_tracks']['status']=='DIAGNOSTIC_COMPLETE'
        assert 'href="temporal_tracks/index.html"' in report
    else:
        assert calls==[] and 'temporal_tracks' not in result


@pytest.mark.parametrize('key,value',[('requested',3),('success',1),('no_match',True),('error',False)])
def test_bad_manifest_counts_rejected(chain,key,value):
    chain['manifest']['counts'][key]=value
    write(Path(chain['discovery']['stages']['segmentation']['run_dir'])/'manifest.json',chain['manifest'])
    with pytest.raises(ValueError,match='counts'): load(chain)


@pytest.mark.parametrize('status',['UNKNOWN',True])
def test_unknown_manifest_status_rejected(chain,status):
    chain['manifest']['frames'][0]['status']=status; chain['commit']()
    with pytest.raises(ValueError,match='mask frame status'): load(chain)


@pytest.mark.parametrize('retained',['selected_mask','selected_candidate_id'])
def test_non_ok_selected_candidate_rejected(chain,retained):
    row=chain['manifest']['frames'][0]; row['status']='NO_MATCH'
    if retained=='selected_mask': row['selected_candidate_id']=None
    else: row['selected_mask']=None
    chain['commit']()
    with pytest.raises(ValueError,match='non-OK'): load(chain)


@pytest.mark.parametrize('key,value',[('input_count',3),('ok_count',1),('error_count',True)])
def test_invalid_t01_counts_rejected(chain,key,value):
    p=chain['t01']/'run.json'; row=json.loads(p.read_text()); row[key]=value; write(p,row)
    rebind_t01(chain)
    with pytest.raises(ValueError,match='counts'): load(chain)
