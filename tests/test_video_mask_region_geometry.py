"""TEST_DATA only: supplemental row mapping and immutable source checks."""
import json
import sqlite3
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from integrations.video_masks.region_geometry import digest,prepare,supplement_database

def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'images').mkdir()
    db=sqlite3.connect(source/'database.db')
    db.executescript('CREATE TABLE images(image_id INTEGER,name TEXT);CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB);CREATE TABLE matches(pair_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);CREATE TABLE descriptors(image_id INTEGER,data BLOB);CREATE TABLE two_view_geometries(pair_id INTEGER);')
    frames=[];maps=[]
    for idx,iid in enumerate((9,3)):
        image=source/'images'/f'{idx}.png';mask=tmp_path/f'mask{idx}.png'
        Image.new('RGB',(8,6)).save(image);Image.new('L',(8,6),255).save(mask)
        db.execute('INSERT INTO images VALUES (?,?)',(iid,f'images/{idx}.png'))
        db.execute('INSERT INTO keypoints VALUES (?,?,?,?)',(iid,1,4,np.array([[2,3,1,0]],np.float32).tobytes()))
        maps.append(dict(input_index=idx,database_image_id=iid,staged_image_name=f'images/{idx}.png',canonical_image_sha256=digest(image)))
        frames.append(dict(input_index=idx,decode_index=idx*2,image_path=str(image),mask_path=str(mask),image_sha256=digest(image),mask_sha256=digest(mask),width=8,height=6,observations=[dict(track_id=7,xy=[1.25,2.0]),dict(track_id=10+idx,xy=[3.,3.])]))
    db.execute('INSERT INTO matches VALUES (?,?,?,?)',(3*2147483647+9,1,2,np.array([[0,0]],np.uint32).tobytes()))
    db.execute('INSERT INTO descriptors VALUES (?,?)',(9,b'original'));db.execute('INSERT INTO two_view_geometries VALUES (1)');db.commit();db.close()
    (source/'frames_map.jsonl').write_text('\n'.join(map(json.dumps,maps)))
    (source/'run.json').write_text('{}')
    planfile=tmp_path/'PLAN.json';planfile.write_text(json.dumps(dict(schema='glove-region-plan-v1',timeline=frames)))
    tracks=tmp_path/'tracks.json';tracks.write_text(json.dumps(dict(schema='glove-region-tracks-v1',coordinate_convention='opencv_integer_centers',frames=frames,source_bindings=[dict(path=str(planfile),sha256=digest(planfile))])))
    return source,tracks

def test_append_rows_reverse_pair_and_source_isolation(tmp_path):
    source,tracks=fixture(tmp_path);before=digest(source/'database.db');plan=prepare(source,tracks)
    assert [f['image_id'] for f in plan['frames']]==[9,3]
    assert list(plan['frames'][0]['points'])==[7]
    pairs,allpairs=supplement_database(plan,tmp_path/'new.db')
    assert pairs[0]['region_rows']==[[1,1]]
    assert pairs[0]['sift_raw']==1 and allpairs==[(3,9,2)]
    with sqlite3.connect(tmp_path/'new.db') as db:
        rows,cols,data=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=9').fetchone()
        points=np.frombuffer(data,np.float32).reshape(rows,cols)
        np.testing.assert_array_equal(points[0],[2,3,1,0]);np.testing.assert_array_equal(points[1,:2],[1.75,2.5])
        assert db.execute('SELECT count(*) FROM descriptors').fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM two_view_geometries').fetchone()[0]==0
    assert digest(source/'database.db')==before


def test_identity_policy_cannot_be_bypassed_before_geometry(tmp_path):
    from integrations.video_masks.track_identity import default_policy
    source,tracks=fixture(tmp_path)
    doc=json.loads(tracks.read_text());planfile=Path(doc['source_bindings'][0]['path'])
    plan=json.loads(planfile.read_text());plan['identity_policy']=default_policy()
    planfile.write_text(json.dumps(plan));doc['source_bindings'][0]['sha256']=digest(planfile)
    tracks.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='identity policy differs'):prepare(source,tracks)
    doc['identity_policy']=default_policy()
    for frame in doc['frames']:
        frame['observations'].append(dict(track_id=99,xy=[1.4,2.0]))
    tracks.write_text(json.dumps(doc))
    before=digest(source/'database.db')
    with pytest.raises(ValueError,match='unresolved target identity'):prepare(source,tracks)
    assert digest(source/'database.db')==before
    for frame in doc['frames']:
        frame['observations'][-1]['xy']=[5.,4.]
    tracks.write_text(json.dumps(doc))
    assert len(prepare(source,tracks)['frames'][0]['points'])==2

@pytest.mark.parametrize('change',['hash','coordinate','duplicate','outside','nonfinite','imageid','decode'])
def test_reject_bad_bindings(tmp_path,change):
    source,tracks=fixture(tmp_path);doc=json.loads(tracks.read_text())
    if change=='hash':doc['frames'][0]['mask_sha256']='bad'
    if change=='decode':doc['frames'][0]['decode_index']=88
    if change=='coordinate':doc['coordinate_convention']='native'
    if change=='duplicate':doc['frames'][0]['observations']*=2
    if change=='outside':doc['frames'][0]['observations'][0]['xy']=[100,1]
    if change=='nonfinite':doc['frames'][0]['observations'][0]['xy']=[float('nan'),1]
    if change=='imageid':
        with sqlite3.connect(source/'database.db') as db:db.execute('UPDATE images SET name="wrong" WHERE image_id=9')
    tracks.write_text(json.dumps(doc))
    with pytest.raises(ValueError):prepare(source,tracks)


def test_sqlite_empty_null_match_rows(tmp_path):
    source,tracks=fixture(tmp_path)
    pairid=3*2147483647+9
    with sqlite3.connect(source/'database.db') as db:
        db.execute('UPDATE matches SET rows=0,cols=2,data=NULL WHERE pair_id=?',(pairid,))
        assert db.execute('SELECT data FROM matches').fetchone()[0] is None
    original=digest(source/'database.db')
    pairs,allpairs=supplement_database(prepare(source,tracks),tmp_path/'new.db')
    assert pairs[0]['sift_raw']==0 and pairs[0]['region_raw']==1
    assert allpairs==[(3,9,1)]
    with sqlite3.connect(tmp_path/'new.db') as db:
        rows,cols,data=db.execute('SELECT rows,cols,data FROM matches').fetchone()
        assert (rows,cols)==(1,2)
        np.testing.assert_array_equal(np.frombuffer(data,np.uint32),[1,1])
    assert digest(source/'database.db')==original

@pytest.mark.parametrize('rows,cols,data',[(1,2,None),(0,3,None),(1,2,b'bad'),(-1,2,b'')])
def test_reject_corrupt_sqlite_match_rows(tmp_path,rows,cols,data):
    source,tracks=fixture(tmp_path)
    with sqlite3.connect(source/'database.db') as db:
        db.execute('UPDATE matches SET rows=?,cols=?,data=?',(rows,cols,data))
    plan=prepare(source,tracks)
    with pytest.raises(ValueError,match='baseline raw match'):
        supplement_database(plan,tmp_path/'new.db')


def test_frame_limit_default_and_versioned_expansion():
    from integrations.video_masks.region_geometry import _validate_frame_limit
    _validate_frame_limit(64,64,{})
    with pytest.raises(ValueError):_validate_frame_limit(65,64,{})
    policy=dict(endpoint_expansion_policy=dict(id='glove-region-bridge-keyframes-v1'))
    _validate_frame_limit(65,96,policy)
    _validate_frame_limit(96,96,policy)
    for maximum,source in [(96,{}),(96,dict(endpoint_expansion_policy=dict(id='wrong'))),(65,policy),(95,policy),(97,policy),(True,policy)]:
        with pytest.raises(ValueError):_validate_frame_limit(65,maximum,source)
    with pytest.raises(ValueError):_validate_frame_limit(97,96,policy)


def test_prepare_default_limit_is_recorded(tmp_path):
    source,tracks=fixture(tmp_path)
    assert prepare(source,tracks)['max_frames']==64
    with pytest.raises(ValueError,match='versioned bridge policy'):
        prepare(source,tracks,max_frames=96)


@pytest.mark.parametrize('empty_data',[None,b''])
def test_sqlite_empty_keypoint_rows(tmp_path,empty_data):
    source,tracks=fixture(tmp_path)
    with sqlite3.connect(source/'database.db') as db:
        db.execute('UPDATE keypoints SET rows=0,cols=4,data=? WHERE image_id=9',(empty_data,))
    original=digest(source/'database.db')
    plan=prepare(source,tracks)
    pairs,_=supplement_database(plan,tmp_path/'new.db')
    assert pairs[0]['region_rows']==[[1,0]]
    with sqlite3.connect(tmp_path/'new.db') as db:
        rows,cols,data=db.execute('SELECT rows,cols,data FROM keypoints WHERE image_id=9').fetchone()
        assert (rows,cols)==(1,4)
        np.testing.assert_array_equal(np.frombuffer(data,np.float32).reshape(rows,cols)[0,:2],[1.75,2.5])
    assert digest(source/'database.db')==original

@pytest.mark.parametrize('rows,cols,data',[(1,4,None),(0,3,None),(1,4,b'bad'),(-1,4,b'')])
def test_reject_corrupt_keypoint_rows(tmp_path,rows,cols,data):
    source,tracks=fixture(tmp_path)
    with sqlite3.connect(source/'database.db') as db:
        db.execute('UPDATE keypoints SET rows=?,cols=?,data=? WHERE image_id=9',(rows,cols,data))
    with pytest.raises(ValueError,match='baseline keypoint'):
        supplement_database(prepare(source,tracks),tmp_path/'new.db')



def test_mapper_selection_default_unchanged_and_expanded_76():
    from types import SimpleNamespace
    from integrations.video_masks.region_geometry import _bind_mapper_image_names
    names=[f'images/{i:06d}.png' for i in range(64)]
    mapper=SimpleNamespace(image_names=names.copy())
    frames=[dict(name=n) for n in names]
    receipt=_bind_mapper_image_names(mapper,frames,64,{})
    assert mapper.image_names==names and receipt['added']==[] and receipt['removed']==[]
    more=[f'images/bridge_{i}.png' for i in range(12)]
    receipt=_bind_mapper_image_names(mapper,frames+[dict(name=n) for n in more],96,dict(id='glove-region-bridge-keyframes-v1'))
    assert mapper.image_names==names+more and receipt['added']==sorted(more) and not receipt['removed']

@pytest.mark.parametrize('case',['missing_bound','extra_inherited','duplicate_bound','no_policy'])
def test_mapper_selection_rejects_unbound_names(case):
    from types import SimpleNamespace
    from integrations.video_masks.region_geometry import _bind_mapper_image_names
    mapper=SimpleNamespace(image_names=['a','b']);frames=[dict(name='a'),dict(name='b'),dict(name='new')]
    policy=dict(id='glove-region-bridge-keyframes-v1')
    if case=='missing_bound':frames=[dict(name='a'),dict(name='new')]
    if case=='extra_inherited':mapper.image_names=['a','b','unbound']
    if case=='duplicate_bound':frames.append(dict(name='new'))
    if case=='no_policy':policy={}
    with pytest.raises(ValueError):_bind_mapper_image_names(mapper,frames,96,policy)
