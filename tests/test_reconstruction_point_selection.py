import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import pytest
from PIL import Image
from kozlov_lab.reconstruction import point_selection as ps


def fixture(tmp_path, points=((10.,10.),(10.,10.),(80.,80.))):
    mask=tmp_path/'mask.png'; Image.new('L',(100,100),255).save(mask)
    raw=mask.read_bytes()
    bindings=[dict(image_name='one.png',source_path=str(mask),staged_path=str(mask),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())]
    db=tmp_path/'database.db'
    with sqlite3.connect(db) as c:
        c.executescript('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT,camera_id INTEGER); CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY,width INTEGER,height INTEGER); CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB); CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB,type INTEGER); CREATE TABLE matches(pair_id INTEGER,rows INTEGER); CREATE TABLE two_view_geometries(pair_id INTEGER,rows INTEGER); INSERT INTO images VALUES(1,"one.png",1); INSERT INTO cameras VALUES(1,100,100);')
        c.execute('INSERT INTO keypoints VALUES(1,?,2,?)',(len(points),b''.join(struct.pack('<ff',*p) for p in points)))
        c.execute('INSERT INTO descriptors VALUES(1,?,128,?,0)',(len(points),b''.join(bytes([i%256])*128 for i in range(len(points)))))
    return db,bindings,tmp_path/'audit.json'


def test_select_preserves_rows_and_alias_witnesses(tmp_path):
    db,b,e=fixture(tmp_path)
    result=ps.apply_point_selection(db,b,e)
    assert result['status']=='VERIFIED' and result['before_total']==3 and result['after_total']==2
    image=result['per_image'][0]
    assert image['selected_original_rows']==[0,2]
    assert image['original_to_new']==[0,None,1]
    assert image['witness_original_rows']==[0,0,2]
    with sqlite3.connect(db) as c:
        assert c.execute('SELECT data FROM descriptors').fetchone()[0]==bytes(128)+bytes([2])*128
    assert ps.selection_policy('native') is None
    with pytest.raises(ValueError):ps.selection_policy('unknown')


@pytest.mark.parametrize('table',['matches','two_view_geometries'])
def test_even_zero_row_match_record_rejected(tmp_path,table):
    db,b,e=fixture(tmp_path)
    with sqlite3.connect(db) as c:c.execute('INSERT INTO '+table+' VALUES(1,0)')
    before=db.read_bytes()
    with pytest.raises(ValueError,match='matching records'):ps.apply_point_selection(db,b,e)
    assert db.read_bytes()==before


@pytest.mark.parametrize('change',["UPDATE descriptors SET rows=2","UPDATE descriptors SET type=1","UPDATE descriptors SET cols=64","UPDATE descriptors SET data=x'00'"])
def test_bad_descriptor_rejected(tmp_path,change):
    db,b,e=fixture(tmp_path)
    with sqlite3.connect(db) as c:c.execute(change)
    before=db.read_bytes()
    with pytest.raises(ValueError,match='descriptor'):ps.apply_point_selection(db,b,e)
    assert db.read_bytes()==before


def test_evidence_failure_rolls_back(tmp_path,monkeypatch):
    db,b,e=fixture(tmp_path);before=db.read_bytes();save=ps._save
    def fail(path,value):
        if path==e:raise OSError('simulated evidence failure')
        save(path,value)
    monkeypatch.setattr(ps,'_save',fail)
    with pytest.raises(OSError):ps.apply_point_selection(db,b,e)
    assert db.read_bytes()==before


def test_old_mask_hash_rejected(tmp_path):
    db,b,e=fixture(tmp_path)
    Image.new('L',(100,100),0).save(b[0]['staged_path'])
    with pytest.raises(ValueError,match='binding drift'):ps.apply_point_selection(db,b,e)


@pytest.mark.parametrize('points',[[(float('nan'),0)],[(0,float('inf'))]])
def test_nonfinite_rejected(tmp_path,points):
    db,b,e=fixture(tmp_path,points)
    with pytest.raises(ValueError,match='nonfinite'):ps.apply_point_selection(db,b,e)


def test_outside_and_boundary_points_excluded(tmp_path):
    db,b,e=fixture(tmp_path,[(-.1,0),(100,10),(0,0),(99.9,99.9)])
    r=ps.apply_point_selection(db,b,e)
    assert r['per_image'][0]['eligible_original_rows']==[2,3]


def test_missing_masks_rejected(tmp_path):
    db,b,e=fixture(tmp_path)
    with pytest.raises(ValueError):ps.apply_point_selection(db,[],e)


def test_postcommit_publication_failure_is_explicit(tmp_path,monkeypatch):
    db,b,e=fixture(tmp_path)
    def fail(*args):raise OSError('publication failed')
    monkeypatch.setattr(ps.os,'replace',fail)
    with pytest.raises(OSError):ps.apply_point_selection(db,b,e)
    assert json.loads(e.read_text())['status']=='PREPARED_NOT_COMMITTED'
    failed=json.loads(e.with_name(e.name+'.failed.json').read_text())
    assert failed['status']=='FAILED' and failed['database_committed'] is True


@pytest.mark.parametrize('mutation',['duplicate','rows','shape','labels'])
def test_input_boundaries(tmp_path,mutation):
    db,b,e=fixture(tmp_path)
    if mutation=='duplicate':b=b+b
    elif mutation=='rows':
        with sqlite3.connect(db) as c:c.execute('UPDATE keypoints SET rows=20001')
    elif mutation=='shape':
        with sqlite3.connect(db) as c:c.execute('UPDATE cameras SET width=99')
    else:
        Image.new('L',(100,100),1).save(b[0]['staged_path'])
        raw=Path(b[0]['staged_path']).read_bytes()
        b[0].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):ps.apply_point_selection(db,b,e)
