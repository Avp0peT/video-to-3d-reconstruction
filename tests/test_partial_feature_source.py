import hashlib
import json
import sqlite3

import numpy as np
import pytest
from PIL import Image

from integrations.demo_pipeline.partial_feature_source import load_feature_frames


def fixture(tmp_path,n=3):
    root=tmp_path/'sfm';root.mkdir()
    image=root/'actual.png';Image.new('RGB',(40,40),'gray').save(image)
    mask=root/'mask.png';Image.new('L',(40,40),255).save(mask)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    mb=dict(input_index=19,image_name='actual.png',source_path=str(mask),staged_path=str(mask),bytes=mask.stat().st_size,sha256=digest(mask))
    sfm=dict(upstream_io_run_id='source',effective_options={'foreground':{'mask_bindings':[mb]}})
    (root/'run.json').write_text(json.dumps(sfm))
    mapped=dict(input_index=19,frame_id='frame19',staged_image_name='actual.png',database_image_id=7,canonical_image_sha256=digest(image),upstream_io_run_id='source')
    (root/'frames_map.jsonl').write_text(json.dumps(mapped)+'\n')
    with sqlite3.connect(root/'database.db') as c:
        c.executescript('CREATE TABLE images(image_id INTEGER,name TEXT);CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);CREATE TABLE descriptors(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB,type INTEGER);INSERT INTO images VALUES(7,"actual.png");')
        keys=np.tile(np.array([[10.5,11.5]],dtype='<f4'),(n,1))
        c.execute('INSERT INTO keypoints VALUES(7,?,2,?)',(n,keys.tobytes()))
        c.execute('INSERT INTO descriptors VALUES(7,?,128,?,0)',(n,np.arange(n*128,dtype=np.uint8).tobytes()))
    frames=[dict(input_index=19,frame_id='frame19',sha256=digest(image),path=str(image),width=40,height=40,original_mask_status='OK',mask_binding=dict(path=str(mask),bytes=mask.stat().st_size,sha256=digest(mask)))]
    return dict(stages={'sfm':{'run_json':str(root/'run.json')}}),frames,[np.full((40,40),255,dtype=np.uint8)],root


def test_native_rows_and_deterministic_bounded_selection(tmp_path):
    p,f,m,root=fixture(tmp_path,300);refs=[]
    before=(root/'database.db').read_bytes()
    result=load_feature_frames(p,f,m,refs);row=result['frames'][0]
    assert row['frame_index']==0 and row['input_index']==19 and row['db_image_id']==7
    assert len(row['points'])==256 and row['points'][0]==dict(id='7:0',feature_row=0,xy=[10.5,11.5])
    assert row['points'][-1]['id']=='7:299'
    assert row['descriptors'].shape==(256,128) and row['descriptors'].dtype==np.uint8
    assert len(row['selection']['omitted_eligible_rows'])==44
    assert (root/'database.db').read_bytes()==before and refs
    again=load_feature_frames(p,f,m,[])
    assert row['points']==again['frames'][0]['points']


@pytest.mark.parametrize('change',[
    'UPDATE descriptors SET rows=2', 'UPDATE descriptors SET data=zeroblob(1)',
    'UPDATE descriptors SET cols=64','UPDATE descriptors SET type=1',
    'UPDATE keypoints SET cols=3','UPDATE keypoints SET data=zeroblob(1)',
    'UPDATE images SET name="guessed.png"','UPDATE images SET image_id=8',
    'INSERT INTO images VALUES(7,"actual.png")',
])
def test_database_wrong_identity_and_bad_blobs_rejected(tmp_path,change):
    p,f,m,root=fixture(tmp_path)
    with sqlite3.connect(root/'database.db') as c:c.execute(change)
    with pytest.raises(ValueError):load_feature_frames(p,f,m,[])


@pytest.mark.parametrize('change',['bool','frame','hash','maskpixels','path','duplicate'])
def test_frame_and_mask_binding_rejections(tmp_path,change):
    p,f,m,root=fixture(tmp_path)
    if change=='bool':f[0]['input_index']=True
    elif change=='frame':f[0]['frame_id']='other'
    elif change=='hash':f[0]['sha256']='0'*64
    elif change=='maskpixels':m[0][0,0]=0
    elif change=='path':f[0]['mask_binding']['path']=str(root/'actual.png')
    else:f.append(dict(f[0]));m.append(m[0].copy())
    with pytest.raises(ValueError):load_feature_frames(p,f,m,[])


def test_nonfinite_rejected_and_outside_mask_excluded(tmp_path):
    p,f,m,root=fixture(tmp_path)
    with sqlite3.connect(root/'database.db') as c:
        keys=np.array([[1,1],[50,50],[10.5,11.5]],dtype='<f4')
        c.execute('UPDATE keypoints SET data=?',(keys.tobytes(),))
    result=load_feature_frames(p,f,m,[])
    assert result['frames'][0]['selection']['selected_rows']==[0,2]
    with sqlite3.connect(root/'database.db') as c:
        keys[0,0]=np.nan;c.execute('UPDATE keypoints SET data=?',(keys.tobytes(),))
    with pytest.raises(ValueError,match='nonfinite'):load_feature_frames(p,f,m,[])
