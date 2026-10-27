"""Saved SQLite/PNG input contract; never imports or executes pycolmap."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrations.demo_pipeline import partial_reconstruction as p


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved(tmp_path):
    root = tmp_path/'source'
    root.mkdir()
    rows, masks = [], []
    for iid in (1, 2, 3):
        image, mask = root/f'view_{iid}.png', root/f'mask_{iid}.png'
        Image.new('RGB', (32, 48), (30*iid, 20, 50)).save(image)
        Image.new('L', (32, 48), 255).save(mask)
        rows.append(dict(input_index=18+iid, database_image_id=iid,
                         foreground_selected=True, upstream_io_run_id='fixture-io',
                         staged_image_name=image.name, canonical_image_sha256=digest(image),
                         status='SELECTED'))
        masks.append(dict(input_index=18+iid, image_name=image.name,
                          staged_path=str(mask), sha256=digest(mask)))
    rows.append(dict(input_index=25, database_image_id=None, status='NO_MATCH', reason='fixture missing'))
    run = dict(contract_version='sfm-native-object-v1',
               backend=dict(name='pycolmap', version='4.2.1'), upstream_io_run_id='fixture-io',
               effective_options=dict(foreground=dict(mask_bindings=masks)))
    (root/'run.json').write_text(json.dumps(run), encoding='utf-8')
    (root/'frames_map.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    with sqlite3.connect(root/'database.db') as con:
        con.executescript('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT,camera_id INTEGER);'
                          'CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY,model INTEGER,width INTEGER,height INTEGER,params BLOB);'
                          'CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);'
                          'CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);')
        con.execute('INSERT INTO cameras VALUES(1,2,32,48,?)', (np.array([40,16,24,0],dtype='<f8').tobytes(),))
        xy = np.array([[2+i%5*5, 3+i//5*9] for i in range(15)], dtype='<f4')
        for iid in (1,2,3):
            con.execute('INSERT INTO images VALUES(?,?,1)', (iid, f'view_{iid}.png'))
            con.execute('INSERT INTO keypoints VALUES(?,15,2,?)', (iid, xy.tobytes()))
        matches = np.column_stack((np.arange(15), np.arange(15))).astype('<u4').tobytes()
        for a,b in ((1,2),(2,3)):
            con.execute('INSERT INTO two_view_geometries VALUES(?,15,2,?)', (a*2147483647+b,matches))
    return root


def test_prepare_real_saved_inputs(saved):
    before={f.name:digest(f) for f in saved.iterdir()}
    plan, xy=p.prepare(saved)
    assert plan == p.prepare(saved/'run.json')[0]
    assert plan['graph']['groups'][0]['input_indices']==[19,20,21]
    assert plan['graph']['groups'][0]['scheduled']
    assert plan['graph']['groups'][0]['three_view_tracks']==15
    assert len(plan['graph']['tracks'])==15 and set(xy)=={1,2,3}
    assert all(f['camera_id']==1 for f in plan['frames'])
    assert plan['excluded_source_rows']==[dict(input_index=25,status='NO_MATCH',reason='fixture missing')]
    assert {f.name:digest(f) for f in saved.iterdir()}==before


@pytest.mark.parametrize('name',['view_1.png','mask_1.png'])
def test_hash_drift_refused(saved,name):
    with (saved/name).open('ab') as stream: stream.write(b'drift')
    with pytest.raises(ValueError,match='source hash differs'):p.prepare(saved)


@pytest.mark.parametrize('mutation', ['extra_camera','unknown_camera','width','height'])
def test_camera_binding_refused(saved,mutation):
    with sqlite3.connect(saved/'database.db') as con:
        if mutation=='extra_camera':
            con.execute('INSERT INTO cameras SELECT 2,model,width,height,params FROM cameras')
            con.execute('UPDATE images SET camera_id=2 WHERE image_id=3')
        elif mutation=='unknown_camera':con.execute('UPDATE images SET camera_id=2 WHERE image_id=3')
        else:con.execute(f'UPDATE cameras SET {mutation}={mutation}+1')
    with pytest.raises(ValueError,match='shared camera|size differs'):p.prepare(saved)


def test_non_pycolmap_refused(saved):
    path=saved/'run.json';run=json.loads(path.read_text());run['backend']['name']='other'
    path.write_text(json.dumps(run),encoding='utf-8')
    with pytest.raises(ValueError,match='pycolmap 4.2.1'):p.prepare(saved)


@pytest.mark.parametrize('bad_row',[15,4294967295])
def test_match_row_outside_source_refused(saved,bad_row):
    matches=np.column_stack((np.arange(15),np.arange(15))).astype('<u4');matches[-1,1]=bad_row
    with sqlite3.connect(saved/'database.db') as con:
        con.execute('UPDATE two_view_geometries SET data=? WHERE pair_id=?',(matches.tobytes(),2147483649))
    with pytest.raises(ValueError,match='feature row outside'):p.prepare(saved)


def test_duplicate_match_rows_refused(saved):
    matches=np.column_stack((np.arange(15),np.arange(15))).astype('<u4')
    matches=np.vstack((matches,matches[0]))
    with sqlite3.connect(saved/'database.db') as con:
        con.execute('UPDATE two_view_geometries SET rows=16,data=? WHERE pair_id=?',(matches.tobytes(),2147483649))
    with pytest.raises(ValueError,match='duplicate'):p.prepare(saved)
