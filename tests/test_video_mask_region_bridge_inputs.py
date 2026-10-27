"""TEST_DATA only: selection and standalone preparation without SfM."""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from integrations.video_masks import region_bridge_inputs as bridge
from integrations.video_masks.region_tracks import PROFILE


def test_selection_valid_largest_components_and_deterministic_cap():
    timeline=[dict(decode_index=n,**(dict(input_index=n//9) if n%9==0 else {})) for n in range(145)]
    models=[dict(label='new',model_id=i,camera_domain_status='CAMERA_DOMAIN_VALID',registered_indices=[i]) for i in range(17)]
    models.append(dict(label='new',model_id=90,camera_domain_status='CAMERA_DOMAIN_REJECTED',registered_indices=list(range(17))))
    added,ledger=bridge.select_intermediates(timeline,models)
    assert len(added)==24 and added[:2]==[3,6] and added[-1]==105
    assert ledger['group_by_original_position'][0]==0
    assert bridge.select_intermediates(timeline,list(reversed(models)))[0]==added
    assert all(n%9 for n in added)


def test_gap_bound_and_no_valid_component_no_added():
    timeline=[dict(decode_index=n*9,input_index=n) for n in range(9)]
    models=[dict(label='new',model_id=1,camera_domain_status='CAMERA_DOMAIN_VALID',registered_indices=[0]),
            dict(label='new',model_id=2,camera_domain_status='CAMERA_DOMAIN_VALID',registered_indices=[8])]
    assert bridge.select_intermediates(timeline,models)[0]==[]
    assert bridge.select_intermediates(timeline,[])[0]==[]


def fixture(tmp_path):
    pc=pytest.importorskip('pycolmap')
    baseline=tmp_path/'baseline';baseline.mkdir();(baseline/'images').mkdir()
    original=tmp_path/'original';original.mkdir();source=tmp_path/'source';source.mkdir()
    (source/'assessment').mkdir();(source/'geometry').mkdir();(source/'tracking').mkdir()
    timeline=[]
    for n in range(7):
        image=tmp_path/f'{n}.jpg';mask=tmp_path/f'{n}_mask.png'
        Image.fromarray(np.full((32,32,3),n*20,np.uint8)).save(image)
        Image.fromarray(np.full((32,32),255,np.uint8)).save(mask)
        row=dict(decode_index=n,image_path=str(image),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),mask_path=str(mask),mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),width=32,height=32)
        if n in (0,6):row['input_index']=n//6
        timeline.append(row)
    db=pc.Database.open(baseline/'database.db')
    cid=db.write_camera(pc.Camera(model='SIMPLE_PINHOLE',width=32,height=32,params=[30,16,16]))
    sensor=pc.sensor_t(pc.SensorType.CAMERA,cid)
    rig=pc.Rig();rig.add_ref_sensor(sensor);rig_id=db.write_rig(rig)
    maps=[]
    for i,n in enumerate((0,6),1):
        name=f'images/{i}.jpg';(baseline/name).write_bytes(Path(timeline[n]['image_path']).read_bytes())
        db.write_image(pc.Image(name=name,camera_id=cid,image_id=i),use_image_id=True)
        frame=pc.Frame(rig_id=rig_id);frame.add_data_id(pc.data_t(sensor,i));db.write_frame(frame)
        db.write_keypoints(i,np.array([[12,12,1,0],[20,20,1,0]],np.float32))
        maps.append(dict(input_index=i-1,database_image_id=i,staged_image_name=name,canonical_image_sha256=timeline[n]['image_sha256']))
    db.write_matches(1,2,np.array([[0,0]],np.uint32));db.close()
    (baseline/'run.json').write_text(json.dumps(dict(run_id='TEST_DATA_OLD_RUN',models=[dict(model_id=42)],counts=dict(points=999),effective_options=dict(native=dict(mapper=dict(frozen=True))),status='PARTIAL')))
    (baseline/'frames_map.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in maps))
    (original/'PLAN.json').write_text(json.dumps(dict(schema='glove-region-plan-v1',profile=PROFILE,timeline=timeline,baseline_sfm=str(baseline),source_bindings={})))
    (source/'PLAN.json').write_text(json.dumps(dict(source_run=str(original))))
    models=[dict(label='new',model_id=i,camera_domain_status='CAMERA_DOMAIN_VALID',registered_indices=[i]) for i in (0,1)]
    (source/'assessment'/'ASSESSMENT.json').write_text(json.dumps(dict(models=models)))
    for path in (source/'geometry'/'RESULT.json',source/'geometry'/'PLAN.json',source/'tracking'/'REGION_TRACKS.json'):path.write_text('{}')
    return source,baseline


def test_prepare_preserves_native_database_and_actual_input_bytes(tmp_path):
    source,baseline=fixture(tmp_path)
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    plan=bridge.prepare(source,tmp_path/'out')
    assert plan['added_decode_indices']==[2,4]
    assert [r['input_index'] for r in plan['timeline'] if 'input_index' in r]==[0,2,4,6]
    sfm=Path(plan['baseline_sfm'])
    maps=[json.loads(s) for s in (sfm/'frames_map.jsonl').read_text().splitlines()]
    assert [m['original_input_index'] for m in maps]==[0,None,None,1]
    assert json.loads((sfm/'run.json').read_text())['status']=='PREPARED_ONLY'
    prepared=json.loads((sfm/'run.json').read_text())
    assert prepared['schema']=='glove-region-prepared-native-input-v1'
    assert prepared['run_id']!='TEST_DATA_OLD_RUN' and prepared['source_run_id']=='TEST_DATA_OLD_RUN'
    assert prepared['source_run_sha256']==hashlib.sha256((baseline/'run.json').read_bytes()).hexdigest()
    assert prepared['models']==[] and prepared['geometry_executed'] is False
    assert 'counts' not in prepared and 'contract_version' not in prepared
    assert json.loads((sfm/'run.json').read_text())['effective_options']==json.loads((baseline/'run.json').read_text())['effective_options']
    with sqlite3.connect(sfm/'database.db') as db:
        assert list(db.execute('SELECT image_id,rows,cols FROM keypoints ORDER BY image_id'))==[(1,2,4),(2,2,4),(3,0,4),(4,0,4)]
        assert db.execute('SELECT COUNT(*) FROM matches').fetchone()[0]==1
        assert db.execute('SELECT COUNT(*) FROM descriptors').fetchone()[0]==0
    for m in maps:
        assert hashlib.sha256((sfm/m['staged_image_name']).read_bytes()).hexdigest()==m['canonical_image_sha256']
    import pycolmap as pc
    db=pc.Database.open(sfm/'database.db')
    try:
        frames={f.frame_id:f for f in db.read_all_frames()}
        rigs={r.rig_id:r for r in db.read_all_rigs()}
        assert len(frames)==4 and len(rigs)==1
        for image in db.read_all_images():
            assert image.has_frame_id()
            frame=frames[image.frame_id]
            assert image.data_id in frame.data_ids
            assert image.data_id.sensor_id==pc.sensor_t(pc.SensorType.CAMERA,image.camera_id)
            assert image.data_id.sensor_id in rigs[frame.rig_id].sensor_ids()
        options=pc.DatabaseCacheOptions(load_all_images=True,min_num_matches=0)
        cache=pc.DatabaseCache.create(db,options)
        assert len(cache.images)==4
        assert all(im.has_frame_id() for im in cache.images.values())
    finally:db.close()
    with sqlite3.connect(baseline/'database.db') as olddb,sqlite3.connect(sfm/'database.db') as newdb:
        for table in ('rigs','rig_sensors','frames','frame_data'):
            assert set(olddb.execute('SELECT * FROM '+table))<=set(newdb.execute('SELECT * FROM '+table))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())


def test_prepare_refuses_source_sha_drift(tmp_path):
    source,_=fixture(tmp_path)
    original=tmp_path/'original'/'PLAN.json';doc=json.loads(original.read_text())
    doc['timeline'][2]['image_sha256']='0'*64;original.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='binding drift'):
        bridge.prepare(source,tmp_path/'out')
