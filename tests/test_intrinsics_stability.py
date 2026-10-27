import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline import intrinsics_stability as m


def dump(path,data):
    path.write_text(json.dumps(data),encoding='utf-8')


@pytest.fixture
def world(tmp_path):
    root=tmp_path/'sfm';root.mkdir();(root/'images').mkdir()
    frames=[]
    for iid,index in ((1,19),(2,21)):
        image=root/f'images/{index}.png';Image.new('RGB',(100,60),(iid,2,3)).save(image)
        frames.append(dict(input_index=index,database_image_id=iid,staged_image_name=f'images/{index}.png',
            frame_id=f'frame{index}',upstream_io_run_id='io1',canonical_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    (root/'frames_map.jsonl').write_text('\n'.join(json.dumps(r) for r in frames))
    dump(root/'run.json',dict(contract_version='sfm-native-object-v1',upstream_io_run_id='io1',
        effective_options=dict(focal_length_init='1.2 * max(width,height)',camera_sharing='single'),models=[]))
    db=sqlite3.connect(root/'database.db')
    db.executescript('''CREATE TABLE images(image_id,name,camera_id);
CREATE TABLE cameras(camera_id,model,width,height,params,prior_focal_length);
CREATE TABLE keypoints(image_id,rows,cols,data);
CREATE TABLE two_view_geometries(pair_id,rows,cols,data,config,F,H);''')
    db.execute('INSERT INTO cameras VALUES(1,2,100,60,?,0)',(np.array([120,50,30,0],dtype='<f8').tobytes(),))
    for iid,f in enumerate(frames,1):
        db.execute('INSERT INTO images VALUES(?,?,1)',(iid,f['staged_image_name']))
        xy=np.array([[20+i*3+iid,10+i*2] for i in range(8)],dtype='<f4')
        db.execute('INSERT INTO keypoints VALUES(?,8,2,?)',(iid,xy.tobytes()))
    F=np.array([[0,0,0],[0,0,-1],[0,1,0]],dtype='<f8')
    matches=np.column_stack((np.arange(8),np.arange(8))).astype('<u4')
    db.execute('INSERT INTO two_view_geometries VALUES(?,8,2,?,3,?,NULL)',(2147483649,matches.tobytes(),F.tobytes()))
    db.commit();db.close()
    return root,tmp_path/'report',frames


def change(root,statement,args=()):
    db=sqlite3.connect(root/'database.db');db.execute(statement,args);db.commit();db.close()


def test_readonly_profile_preserves_unknown_and_cannot_authorize_ba(world):
    root,out,_=world;before=(root/'database.db').read_bytes()
    r=m.audit_intrinsics(root,out)
    assert r['status']=='INTRINSICS_DIAGNOSTIC_COMPLETE' and r['source_unchanged']
    assert r['counts']['profiled_pairs']==1 and len(r['pairs'][0]['profile']['curve'])==101
    assert r['pairs'][0]['profile']['focal_observability_status']=='FLAT_PROFILE'
    assert r['pairs'][0]['sampson']['max_pixels']==0
    assert r['preprocessing']['optical_zoom']=='UNKNOWN'
    assert r['decision']['recommended_camera_params'] is None and not r['decision']['new_ba_authorized']
    assert (root/'database.db').read_bytes()==before and not Path(str(root/'database.db')+'-wal').exists()
    assert (out/'index.html').is_file()
    with pytest.raises(ValueError,match='fresh'):m.audit_intrinsics(root,out)


@pytest.mark.parametrize('config',[4,5,6,7])
def test_ambiguous_geometry_never_profiled(world,config):
    root,out,_=world;change(root,'UPDATE two_view_geometries SET config=?,F=NULL',(config,))
    r=m.audit_intrinsics(root,out)
    assert r['counts']['profiled_pairs']==0
    assert r['pairs'][0]['reason']=='AMBIGUOUS_GEOMETRY_CONFIGURATION'


def test_distinct_camera_ids_not_silently_shared(world):
    root,out,_=world
    change(root,'INSERT INTO cameras SELECT 2,model,width,height,params,prior_focal_length FROM cameras')
    change(root,'UPDATE images SET camera_id=2 WHERE image_id=2')
    assert m.audit_intrinsics(root,out)['pairs'][0]['reason']=='SHARED_CAMERA_PROFILE_NOT_APPLICABLE'


@pytest.mark.parametrize('damage',['wal','hash','identity','keypoint_index','nonfinite','infinity_json','escape'])
def test_bad_inputs_refused_before_output(world,damage):
    root,out,frames=world
    if damage=='wal':Path(str(root/'database.db')+'-wal').write_bytes(b'active')
    if damage=='hash':(root/'images/19.png').write_bytes(b'changed')
    if damage=='identity':change(root,'UPDATE images SET name=? WHERE image_id=1',('wrong.png',))
    if damage=='keypoint_index':change(root,'UPDATE two_view_geometries SET data=?',(np.array([[99,0]]*8,dtype='<u4').tobytes(),))
    if damage=='nonfinite':change(root,'UPDATE two_view_geometries SET F=?',(np.full((3,3),np.nan,dtype='<f8').tobytes(),))
    if damage=='infinity_json':(root/'run.json').write_text('{"oops":1e999}')
    if damage=='escape':
        (root.parent/'outside.png').write_bytes((root/'images/19.png').read_bytes())
        frames[0]['staged_image_name']='../outside.png'
        (root/'frames_map.jsonl').write_text('\n'.join(json.dumps(r) for r in frames))
        change(root,'UPDATE images SET name=? WHERE image_id=1',('../outside.png',))
    with pytest.raises((ValueError,KeyError)):m.audit_intrinsics(root,out)
    assert not out.exists()


def test_drift_during_read_rejected(world,monkeypatch):
    root,out,_=world;real=m.preprocessing
    def mutate(*args,**kwargs):
        r=real(*args,**kwargs)
        with (root/'run.json').open('a') as f:f.write(' ')
        return r
    monkeypatch.setattr(m,'preprocessing',mutate)
    with pytest.raises(ValueError,match='drift'):m.audit_intrinsics(root,out)
    assert not out.exists()


def make_video(root,frames):
    pipeline=root.parent/'pipeline';vroot=pipeline/'input/video';io=vroot/'photo_io/io1'
    (io/'images').mkdir(parents=True)
    rows=[]
    for f in frames:
        name=f['staged_image_name'];(io/name).write_bytes((root/name).read_bytes())
        rows.append(dict(input_index=f['input_index'],frame_id=f['frame_id'],status='OK',
            image_output=name,canonical_width=100,canonical_height=60,raw_width=100,raw_height=60,
            image_sha256='raw image digest need not equal canonical digest',raw_to_canonical=[[1,0,0],[0,1,0],[0,0,1]]))
    dump(io/'run.json',dict(contract_version='photo-io-v1',run_id='io1'))
    (io/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    source=root.parent/'video.mp4';source.write_bytes(b'bound fixture, not decoded video')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    dump(pipeline/'pipeline.json',dict(source_kind='video',source=str(source),stages=dict(
        sfm=dict(run_json=str(root/'run.json'),argv=['script','--io-run',str(io)]),
        input=dict(argv=['script','--output',str(pipeline/'input')]))))
    timeline=dict(width=200,height=120,sample_aspect_ratio='1:1',rotation_degrees=0,display_matrix=None)
    dump(vroot/'RESULT.json',dict(contract_version='video-photo-v1',status='SUCCESS',source_video=str(source),timeline=timeline,
        source_sha256=sha(source),source_sha256_after=sha(source),photo_io=dict(run=str(io/'run.json'),
        sha256=sha(io/'run.json'),frames_sha256=sha(io/'frames.jsonl')),
        frames=[dict(photo_input_index=f['input_index'],photo_frame_id=f['frame_id'],width=100,height=60,status='PHOTO_IO_OK') for f in frames]))
    dump(vroot/'timeline.json',timeline)
    dump(vroot/'extract.process.json',dict(argv=['ffmpeg','-i',str(source),'-vf','scale=100:60,format=rgb24']))
    return pipeline,io


def test_scaling_is_bound_but_does_not_establish_optical_zoom(world):
    root,out,frames=world;pipeline,io=make_video(root,frames)
    r=m.audit_intrinsics(root,out,pipeline=pipeline)['preprocessing']
    assert r['status']=='VIDEO_PREPROCESSING_BOUND' and r['raw_to_canonical_identity']
    assert r['output_to_decoded_size_ratios']==[(.5,.5)] and r['equal_axis_size_ratios']
    assert r['optical_zoom']=='UNKNOWN' and r['physical_intrinsics']=='UNKNOWN'
    assert not r['calibration_accepted'] and not r['crop_filter_recorded']


def test_video_wrong_frame_identity_refused(world):
    root,out,frames=world;pipeline,io=make_video(root,frames)
    path=pipeline/'input/video/RESULT.json';v=json.loads(path.read_text());v['frames'][0]['photo_frame_id']='wrong';dump(path,v)
    with pytest.raises(ValueError,match='video frame identity'):m.audit_intrinsics(root,out,pipeline=pipeline)
    assert not out.exists()


@pytest.mark.parametrize('field',['timeline','extraction_source'])
def test_unrelated_metadata_is_not_provenance(world,field):
    root,out,frames=world;pipeline,io=make_video(root,frames)
    if field=='timeline':
        path=pipeline/'input/video/timeline.json';value=json.loads(path.read_text());value['width']=400;dump(path,value)
    else:
        other=root.parent/'other.mp4';other.write_bytes(b'wrong source')
        dump(pipeline/'input/video/extract.process.json',dict(argv=['ffmpeg','-i',str(other),'-vf','scale=100:60']))
    with pytest.raises(ValueError,match='timeline metadata|different source video'):
        m.audit_intrinsics(root,out,pipeline=pipeline)
    assert not out.exists()


@pytest.mark.parametrize('fail',[False,True])
def test_automatic_review_runs_intrinsics_even_without_model(world,monkeypatch,fail):
    from integrations.demo_pipeline import post_review
    root,out,frames=world;pipeline,io=make_video(root,frames)
    p=json.loads((pipeline/'pipeline.json').read_text());p['source_kind']='photos';dump(pipeline/'pipeline.json',p)
    calls=[]
    def fake(*args,**kwargs):
        calls.append(args)
        if fail:raise ValueError('missing matrix')
        return dict(status='INTRINSICS_DIAGNOSTIC_COMPLETE',decision=dict(calibration_accepted=False),counts={},source_bindings=[],implementation=[])
    monkeypatch.setattr(m,'audit_intrinsics',fake)
    monkeypatch.setattr(post_review,'load_sources',lambda *args:(None,'No native model'))
    r=post_review.review(pipeline,out)
    assert r['status']=='INSUFFICIENT_DATA' and len(calls)==1
    assert r['intrinsics_stability']['status']==('NOT_EVALUATED' if fail else 'INTRINSICS_DIAGNOSTIC_COMPLETE')
    if fail:assert 'missing matrix' in r['intrinsics_stability']['reason']
