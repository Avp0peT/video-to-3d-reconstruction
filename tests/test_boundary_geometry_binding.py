import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from integrations.video_masks import boundary_geometry_binding as b


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def image(path, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (4, 4), (value, value, value)).save(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, monkeypatch):
    s, l, u = [tmp_path/x for x in ('surface', 'local', 'union')]
    mask=u/'final_masks/000000.png'; mask.parent.mkdir(parents=True)
    Image.new('L', (4,4),255).save(mask)
    sha=hashlib.sha256(mask.read_bytes()).hexdigest()
    rgb=s/'export/images/a.png'; rgbsha=image(rgb)
    cam=dict(model='PINHOLE',width=4,height=4,params=[2.,2.,2.,2.])
    native=dict(name='undistorted/images/a.png',global_id=7,width=4,height=4,K=[[2,0,1.5],[0,2,1.5],[0,0,1]],R=np.eye(3).tolist(),C=[0,0,0])
    f=dict(decode_index=0,input_index=12,exported_name='a.png',target_image_id=7,status='OK',readback_verified=True,source_mask_path=str(mask),source_mask_sha256=sha,source_camera=cam,target_camera=cam,mask=dict(path=str(mask),sha256=sha),validity=dict(path=str(mask),sha256=sha))
    e=dict(decode_index=0,input_index=12,exported_name='a.png',source_mask_path=str(mask),source_mask_sha256=sha,source_image_path=str(rgb),source_image_sha256=rgbsha,exported_png_path=str(rgb),exported_png_sha256=rgbsha,source_camera=cam,source_pose=dict(rotation_matrix=np.eye(3).tolist(),translation=[0,0,0]))
    save(s/'export/FRAME_MAP.json',dict(schema='region-surface-frame-map-v1',frames=[e]))
    save(s/'export/EXPORT.json',dict(schema='region-surface-export-v1',source_unchanged=True,frame_map_sha256=hashlib.sha256((s/'export/FRAME_MAP.json').read_bytes()).hexdigest()))
    scene=s/'bridge/scene.mvs';scene.parent.mkdir(parents=True);scene.write_bytes(b'scene')
    save(s/'masks/RESULT.json',dict(schema='region-surface-masks-v1',coverage_status='COMPLETE',adapter_source_chain='VERIFIED',scene_binding=dict(sha256=hashlib.sha256(b'scene').hexdigest()),frames=[f]))
    save(s/'surface/result.json',dict(status='SURFACE_GENERATED_NOT_ACCEPTED'))
    l.mkdir();(l/'retained.ply').write_bytes(b'mesh')
    for name in ('retained.mapping.npz','LINEAGE.npz'):(l/name).write_bytes(b'mapping')
    save(l/'RESULT.json',dict(schema='local-boundary-surface-v1',status='LOCAL_BOUNDARY_CANDIDATE_NOT_ACCEPTED',source_unchanged=True,geometry_accepted=False,source_run=str(s),outputs=dict(retained=dict(path=str(l/'retained.ply'),sha256=hashlib.sha256(b'mesh').hexdigest(),vertices=3,faces=1))))
    save(u/'RESULT.json',dict(status='FULL_TIMELINE_FUSED_EXPORTED',mode='union'))
    save(u/'ASSEMBLY.json',dict(rows=[dict(decode_index=0,mask_path=str(mask),mask_sha256=sha)]))
    monkeypatch.setattr(b,'read_scene',lambda _:dict(transform=np.eye(4),images=[native]))
    monkeypatch.setattr(b,'read_ply',lambda *a,**k:dict(vertices=np.zeros((3,3)),faces=np.array([[0,1,2]])))
    return s,l,u,f,e,native


def test_context_joins_by_decode_and_keeps_missing_depth_unknown(tmp_path,monkeypatch):
    s,l,u,*_=fixture(tmp_path,monkeypatch)
    c=b.prepare_context(s,l)
    assert c['union_run']==str(u) and list(c['frames'])==[0]
    assert c['frames'][0]['depth_path'] is None and c['frames'][0]['camera_scene']['global_id']==7


def test_non_union_is_explicit_not_applicable(tmp_path,monkeypatch):
    s,l,u,*_=fixture(tmp_path,monkeypatch)
    save(u/'RESULT.json',dict(status='FULL_TIMELINE_FUSED_EXPORTED',mode='supported-holes'))
    assert b.prepare_context(s,l)['status']=='NOT_APPLICABLE'


def test_mesh_hash_drift_refused(tmp_path,monkeypatch):
    s,l,*_=fixture(tmp_path,monkeypatch);(l/'retained.ply').write_bytes(b'changed')
    with pytest.raises(ValueError,match='binding mismatch'):b.prepare_context(s,l)


@pytest.mark.parametrize('change,match', [('K','centre'),('global_id','identity'),('C','pose')])
def test_camera_convention_and_identity_refused(tmp_path,monkeypatch,change,match):
    *_,f,e,n=fixture(tmp_path,monkeypatch)
    if change=='K':n['K'][0][2]+=.5
    elif change=='global_id':n[change]+=1
    else:n[change][0]=1
    with pytest.raises(ValueError,match=match):b._camera_check(f,n,e)


def test_warp_refuses_lossy_rgb_mismatch_before_mapping(tmp_path,monkeypatch):
    s,l,u,*_=fixture(tmp_path,monkeypatch);v=b.prepare_context(s,l)['frames'][0];r=tmp_path/'review'
    save(r/'RESULT.json',dict(status='AUTOMATIC_BOUNDARY_REVIEW_READY_NOT_ACCEPTED',source_unchanged=True,source_run=str(u),mask_promotion=False))
    native=r/'native/000000.png';ns=image(native,40)
    canonical=r/'canonical/000000.jpg';cs=image(canonical,50)
    video=r/'video';video.write_bytes(b'video')
    save(r/'DECODE_BINDINGS.json',dict(source_video=dict(path=str(video),sha256=hashlib.sha256(b'video').hexdigest()),frames=[dict(decode_index=0,ordinal=0,sha256=ns,canonical_sha256=cs)]))
    with pytest.raises(ValueError,match='SOURCE_RGB_PIXEL_MISMATCH'):b.warp_evidence(v,r,[])


def derivation_fixture(tmp_path):
    u=tmp_path/'union'; video=tmp_path/'video.mp4';video.write_bytes(b'video')
    vs=hashlib.sha256(b'video').hexdigest();v=tmp_path/'video_run';io=v/'photo_io/run01'
    canonical=io/'images/000000.png';cs=image(canonical,30)
    raw=tmp_path/'dataset/frames/000000.png';rs=image(raw,30)
    row=dict(input_index=0,image_output='images/000000.png',frame_id='frame0',status='OK',orientation_applied=1,
             exif_orientation_original=None,raw_to_canonical=np.eye(3).tolist(),raw_width=4,raw_height=4,canonical_width=4,canonical_height=4)
    save(io/'run.json',{});(io/'frames.jsonl').write_text(json.dumps(row)+'\n')
    fr=dict(decode_index=0,pts=10,time_base=[1,90],export_pts_verified=True,status='PHOTO_IO_OK',photo_input_index=0,
            photo_frame_id='frame0',width=4,height=4,sha256=rs,decoded_rgb_sha256=hashlib.sha256(b'4x4:RGB:'+b._image(raw).tobytes()).hexdigest(),image='frames/000000.png',output_index=0)
    timeline=dict(rotation_degrees=0,display_matrix=None,sample_aspect_ratio='1:1',stream_index=0,width=8,height=8,frames=[dict(decode_index=0,pts=10,time_base=[1,90])])
    save(v/'timeline.json',timeline)
    save(v/'RESULT.json',dict(status='SUCCESS',source_sha256=vs,source_sha256_after=vs,source_video=str(video),dataset_root=str(tmp_path/'dataset'),frames=[fr],photo_io=dict(run=str(io/'run.json'),sha256=hashlib.sha256((io/'run.json').read_bytes()).hexdigest(),frames_sha256=hashlib.sha256((io/'frames.jsonl').read_bytes()).hexdigest())))
    argv=['ffmpeg','-i',str(video),'-map','0:0','-fps_mode','passthrough','-vf',"select='eq(n,0)',scale=w=min(1024\\,iw):h=min(1024\\,ih):force_original_aspect_ratio=decrease,format=rgb24,showinfo"]
    save(v/'extract.process.json',dict(returncode=0,argv=argv))
    save(u/'PLAN.json',dict(video_result_path=str(v/'RESULT.json'),video_result_sha256=hashlib.sha256((v/'RESULT.json').read_bytes()).hexdigest(),timeline_path=str(v/'timeline.json'),timeline_sha256=hashlib.sha256((v/'timeline.json').read_bytes()).hexdigest(),source_video_path=str(video),source_video_sha256=vs,width=4,height=4))
    view=dict(decode_index=0,frame=dict(canonical_image_path=str(canonical),canonical_image_sha256=cs,source_image_path=str(canonical)))
    return u,v,view


def test_bound_resampling_chain_requires_no_rgb_tolerance(tmp_path):
    u,v,view=derivation_fixture(tmp_path)
    proof=b._prove_derivation(view,u,[])
    assert proof['status']=='VERIFIED_BOUND_ORDINAL_PTS_RESAMPLING_CHAIN'
    assert proof['resampling_diff_known'] and proof['pts']==10


def test_extract_crop_cannot_pass_derivation_proof(tmp_path):
    u,v,view=derivation_fixture(tmp_path)
    process=json.loads((v/'extract.process.json').read_text());process['argv'][-1]+=',crop=2:2'
    save(v/'extract.process.json',process)
    with pytest.raises(ValueError,match='filter'):b._prove_derivation(view,u,[])


def test_extra_intermediate_frame_has_no_endpoint_identity(tmp_path):
    u,v,view=derivation_fixture(tmp_path);view['decode_index']=1
    with pytest.raises(ValueError,match='UNVERIFIED_FRAME_IDENTITY'):b._prove_derivation(view,u,[])


def test_missing_union_metadata_is_not_applicable(tmp_path,monkeypatch):
    s,l,u,*_=fixture(tmp_path,monkeypatch);(u/'RESULT.json').unlink()
    assert b.prepare_context(s,l)['status']=='NOT_APPLICABLE'


@pytest.mark.parametrize('text',['{"x":1,"x":2}', '{"x":NaN}'])
def test_metadata_json_rejects_ambiguous_values(tmp_path,text):
    p=tmp_path/'bad.json';p.write_text(text)
    with pytest.raises(ValueError):b._read(p,[])


def test_binding_cache_deduplicates_and_detects_drift(tmp_path):
    p=tmp_path/'input';p.write_bytes(b'a');refs=[]
    b._bind(p,refs);b._bind(p,refs);assert len(refs)==1
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed since binding'):b._bind(p,refs)


def test_rgb_scope_never_shrinks_consumer_foreground_validity(tmp_path,monkeypatch):
    from types import SimpleNamespace
    s,l,u,*_=fixture(tmp_path,monkeypatch);view=b.prepare_context(s,l)['frames'][0];r=tmp_path/'review'
    save(r/'RESULT.json',dict(status='AUTOMATIC_BOUNDARY_REVIEW_READY_NOT_ACCEPTED',source_unchanged=True,source_run=str(u),mask_promotion=False))
    ns=image(r/'native/000000.png');cs=image(r/'canonical/000000.jpg')
    video=r/'video';video.write_bytes(b'video')
    save(r/'DECODE_BINDINGS.json',dict(source_video=dict(path=str(video),sha256=hashlib.sha256(b'video').hexdigest()),frames=[dict(decode_index=0,ordinal=0,sha256=ns,canonical_sha256=cs)]))
    base=np.full((4,4),255,np.uint8);base[0,0]=0
    bp=r/'base.png';Image.fromarray(base).save(bp)
    save(r/'gaps/PLAN.json',dict(frames=[dict(decode_index=0,base=str(bp),union=view['mask_record']['source_mask_path'],image=str(r/'canonical/000000.jpg'))]))
    target=np.zeros((4,4),bool);target[0,0]=True;code=target.astype(np.uint8)
    (r/'evidence').mkdir();np.savez(r/'evidence/000000.npz',target=target,evidence_code=code)
    (r/'gaps/arrays').mkdir();unresolved=np.zeros((4,4),bool);unresolved[1,1]=True
    np.savez(r/'gaps/arrays/000000.npz',narrow_external=target,unresolved_background=unresolved)
    def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
    save(r/'evidence/000000.json',dict(decode_index=0,input_sha256=dict(rgb_native=digest(b._image(r/'native/000000.png')),base_canonical=digest(base!=0),literal_union=digest(np.ones((4,4),bool))),stats=dict(source_target_pixels=1,target_pixel_counts=dict(SUPPORTED_RGB_CONSISTENCY=1,WEAK_RGB_EVIDENCE=0,UNKNOWN=0))))
    monkeypatch.setattr(b,'_camera',lambda x:x)
    monkeypatch.setattr(b,'warp_binary_mask',lambda a,*args:SimpleNamespace(mask=a.copy(),valid=np.full((4,4),255,np.uint8)))
    (s/'bridge/undistorted/images').mkdir(parents=True);image(s/'bridge/undistorted/images/a.png')
    result=b.warp_evidence(view,r,[])
    assert result['validity'].all() and result['supported_gap'].sum()==1
    assert result['identity']['rgb_pixel_equal'] is True
