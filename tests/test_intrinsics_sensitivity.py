import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline import intrinsics_stability as previous
from integrations.demo_pipeline import intrinsics_sensitivity as m


def dump(path,value):path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def world(tmp_path):
    root=tmp_path/'sfm';(root/'images').mkdir(parents=True)
    frames=[]
    for iid,index in ((1,19),(2,21)):
        path=root/f'images/{index}.png';Image.new('RGB',(100,60),(iid,2,3)).save(path)
        frames.append(dict(input_index=index,database_image_id=iid,frame_id=f'frame{index}',upstream_io_run_id='io1',
            staged_image_name=f'images/{index}.png',canonical_image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (root/'frames_map.jsonl').write_text('\n'.join(json.dumps(r) for r in frames))
    dump(root/'run.json',dict(contract_version='sfm-native-object-v1',upstream_io_run_id='io1',models=[]))
    d=sqlite3.connect(root/'database.db')
    d.executescript('''CREATE TABLE images(image_id,name,camera_id);
CREATE TABLE cameras(camera_id,model,width,height,params,prior_focal_length);
CREATE TABLE keypoints(image_id,rows,cols,data);
CREATE TABLE two_view_geometries(pair_id,rows,cols,data,config,F,H);''')
    d.execute('INSERT INTO cameras VALUES(1,2,100,60,?,0)',(np.array([120,50,30,0],dtype='<f8').tobytes(),))
    xy=np.array([[10+8*x,10+10*y] for y in range(4) for x in range(8)],dtype='<f4')
    for iid,f in enumerate(frames,1):
        d.execute('INSERT INTO images VALUES(?,?,1)',(iid,f['staged_image_name']))
        points=(xy+np.array([2,1],dtype='<f4')*(iid-1)).astype('<f4')
        d.execute('INSERT INTO keypoints VALUES(?,32,2,?)',(iid,points.tobytes()))
    F=np.array([[0,0,1],[0,0,-2],[-1,2,0]],dtype='<f8')
    ids=np.column_stack((np.arange(32),np.arange(32))).astype('<u4')
    d.execute('INSERT INTO two_view_geometries VALUES(?,32,2,?,3,?,NULL)',(2147483649,ids.tobytes(),F.tobytes()))
    d.commit();d.close()
    previous.audit_intrinsics(root,tmp_path/'prior')
    return root,tmp_path/'prior/INTRINSICS.json',tmp_path/'result'


def test_real_sqlite_path_preserves_sources_and_holds_out(world):
    root,source,out=world;before=(root/'database.db').read_bytes()
    r=m.audit_sensitivity(source,out)
    assert r['status']=='INTRINSICS_SENSITIVITY_COMPLETE' and r['source_unchanged']
    assert (root/'database.db').read_bytes()==before and (out/'index.html').exists()
    analysis=r['pairs'][0]['analysis']
    assert len(analysis['principal_profiles'])==5
    zero=analysis['distortion_scenarios'][1]
    assert zero['homography_compatible'] is True
    for fold in zero['folds']:
        assert len(fold['training_feature_pairs'])==len(fold['heldout_feature_pairs'])==16
        assert set(map(tuple,fold['training_feature_pairs'])).isdisjoint(map(tuple,fold['heldout_feature_pairs']))
    assert r['decision']['adequate_triangulation_parallax']=='NOT_ESTABLISHED'
    assert r['decision']['recommended_camera_params'] is None and not r['decision']['new_ba_authorized']
    with pytest.raises(ValueError,match='fresh'):m.audit_sensitivity(source,out)


@pytest.mark.parametrize('damage',['source_bytes','missing_db_binding','pair_identity','F_values','journal','prior_status','source_overlap'])
def test_changed_or_unbound_input_refused(world,damage):
    root,source,out=world;r=json.loads(source.read_text())
    if damage=='source_bytes':(root/'images/19.png').write_bytes(b'drift')
    if damage=='missing_db_binding':r['source_bindings']=[x for x in r['source_bindings'] if not x['path'].endswith('database.db')]
    if damage=='pair_identity':r['pairs'][0]['input_indices']=[1,2]
    if damage=='F_values':r['pairs'][0]['fundamental'][0][0]=9
    if damage=='journal':Path(str(root/'database.db')+'-wal').write_bytes(b'active')
    if damage=='prior_status':r['status']='FAILED'
    if damage=='source_overlap':out=root/'inside'
    dump(source,r)
    with pytest.raises(ValueError):m.audit_sensitivity(source,out)
    assert not out.exists()


def test_interpretation_missing_is_not_negative_evidence():
    p=dict(input_indices=[1,2],analysis=dict(status='SENSITIVITY_EVALUATED',principal_profiles=[],
        distortion_scenarios=[dict(k=0.0,homography_compatible=None)]))
    r=m.summarize([p])
    assert r['homography_compatibility']['0.0']=={'not_evaluated':1}
    assert r['principal_point_sensitivity'][0]['ratio'] is None


def test_automatic_review_invokes_sensitivity_after_intrinsics_without_model(tmp_path,monkeypatch):
    from integrations.demo_pipeline import post_review as post
    pipeline=tmp_path/'pipeline';pipeline.mkdir()
    data={'stages':{'sfm':{'run_json':str(tmp_path/'sfm/run.json')}}}
    dump(pipeline/'pipeline.json',data)
    calls=[]
    def first(*args,**kwargs):
        calls.append('intrinsics');return dict(status='INTRINSICS_DIAGNOSTIC_COMPLETE',decision={},counts={},source_bindings=[],implementation=[])
    def second(source,out):
        calls.append('sensitivity');assert source.name=='INTRINSICS.json'
        return dict(status='INTRINSICS_SENSITIVITY_COMPLETE',decision={},summary={},source_bindings=[],implementation=[])
    monkeypatch.setattr(previous,'audit_intrinsics',first);monkeypatch.setattr(m,'audit_sensitivity',second)
    monkeypatch.setattr(post,'load_sources',lambda *a:(None,'no model'))
    r=post.review(pipeline,tmp_path/'report')
    assert calls==['intrinsics','sensitivity'] and r['status']=='INSUFFICIENT_DATA'
    assert r['intrinsics_sensitivity']['status']=='INTRINSICS_SENSITIVITY_COMPLETE'
