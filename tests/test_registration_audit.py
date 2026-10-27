import json
from pathlib import Path
import sqlite3
import struct
import pytest
from integrations.demo_pipeline import registration_audit as audit


def test_log_parser_observed_attempts_unknown_inliers():
    text='Registering image #1 (num_reg_frames=2)\n=> Image sees 120 / 289 points\n=> Could not register, trying another image.\nRegistering image #1 (num_reg_frames=2)\nRegistering image with structure-less fallback\n=> Image sees 385 / 385 correspondences\n=> Could not register, trying another image.\n'
    r=audit.parse_registration_attempts(text,1)
    assert r['observed_attempt_count']==2 and all(a['status']=='FAILED' for a in r['attempts'])
    assert r['attempts'][0]['logged_visible_points']['numerator']==120
    assert r['attempts'][1]['structureless_fallback']
    assert all(a['absolute_pose_inlier_count'] is None and a['internal_reason_status']=='UNKNOWN' for a in r['attempts'])
    assert audit.parse_registration_attempts(None,1)['observed_attempt_count'] is None
    assert audit.parse_registration_attempts('Registering image #10 (x)',1)['observed_attempt_count']==0


def fixture(tmp_path,log=True):
    root=tmp_path/'native';root.mkdir();binary=root/'model';binary.mkdir();logs=root/'logs';logs.mkdir()
    rows=[dict(input_index=i+19,upstream_io_run_id='io',database_image_id=i+1,staged_image_name='images/'+str(i+1)+'.png',
        registrations=[] if i==0 else [dict(model_id='0',image_id=i+1,camera_id=1)],status='UNREGISTERED' if i==0 else 'REGISTERED') for i in range(3)]
    fmap=root/'frames_map.jsonl';fmap.write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
    run=dict(contract_version='sfm-native-object-v1',upstream_io_run_id='io',backend=dict(name='pycolmap',version='4.2.1'),
        models=[dict(model_id='0',binary_path='model',usable=True,roundtrip_verified=True,registered_image_ids=[2,3],num_points3D=1)])
    run_path=root/'run.json';run_path.write_text(json.dumps(run),encoding='utf-8')
    with sqlite3.connect(root/'database.db') as c:
        c.executescript('CREATE TABLE images(image_id INTEGER,name TEXT,camera_id INTEGER); CREATE TABLE cameras(camera_id INTEGER,model INTEGER,width INTEGER,height INTEGER,params BLOB,prior_focal_length INTEGER); CREATE TABLE keypoints(image_id INTEGER,rows INTEGER,cols INTEGER,data BLOB); CREATE TABLE matches(pair_id INTEGER,rows INTEGER); CREATE TABLE two_view_geometries(pair_id INTEGER,rows INTEGER,cols INTEGER,data BLOB,config INTEGER);')
        c.execute('INSERT INTO cameras VALUES(1,2,100,100,?,0)',(struct.pack('<4d',100,50,50,0),))
        for i in range(1,4):
            c.execute('INSERT INTO images VALUES(?,?,1)',(i,'images/'+str(i)+'.png'))
            c.execute('INSERT INTO keypoints VALUES(?,1,2,?)',(i,struct.pack('<ff',1,2)))
        for target in (2,3):
            pid=2147483647+target;c.execute('INSERT INTO matches VALUES(?,2)',(pid,))
            c.execute('INSERT INTO two_view_geometries VALUES(?,1,2,?,3)',(pid,struct.pack('<II',0,0)))
    image_bytes=struct.pack('<Q',2)
    for i in (2,3):image_bytes+=struct.pack('<idddddddi',i,1,0,0,0,0,0,0,1)+('images/'+str(i)+'.png').encode()+b'\0'+struct.pack('<Qddq',1,1,2,7)
    (binary/'images.bin').write_bytes(image_bytes)
    (binary/'points3D.bin').write_bytes(struct.pack('<Q',1)+struct.pack('<QdddBBBd',7,0,0,5,0,0,0,0)+struct.pack('<Qiiii',2,2,0,3,0))
    (binary/'cameras.bin').write_bytes(struct.pack('<QiiQQdddd',1,1,2,100,100,100,50,50,2))
    if log:(logs/'incremental_mapping.stderr.txt').write_text('Registering image #1 (num_reg_frames=2)\n=> Image sees 1 / 1 points\n=> Could not register, trying another image.\n',encoding='utf-8')
    (logs/'incremental_mapping.result.json').write_text(json.dumps(dict(actual_options=dict(pipeline=dict(max_extra_param=1.,min_focal_length_ratio=.1,max_focal_length_ratio=10.,min_num_matches=1,ignore_watermarks=False,mapper=dict(abs_pose_min_num_inliers=30))))),encoding='utf-8')
    (logs/'incremental_mapping.spec.json').write_text(json.dumps(dict(camera_mode='SINGLE')),encoding='utf-8')
    return run_path


def test_actual_synthetic_native_bytes_readback_source_unchanged(tmp_path):
    run=fixture(tmp_path);database=run.parent/'database.db';before=database.read_bytes();refs=[]
    r=audit.audit_registration(run,19,tmp_path/'report',refs)
    assert r['counts']['observed_registration_attempts']==1 and r['counts']['positive_geometric_pairs']==2
    assert r['final_snapshot_correspondences'][0]['distinct_source_rows']==1
    assert r['final_snapshot_correspondences'][0]['distinct_point3D_ids']==1
    assert r['suspected_mechanisms'][0]['causality']=='SUSPECTED_MECHANISM_NOT_PROVEN'
    assert r['registration_failure_cause']=='UNKNOWN_INTERNAL_BRANCH_NOT_RETAINED'
    assert r['recorded_camera_mode']=='SINGLE' and database.read_bytes()==before
    gate=r['final_snapshot_correspondences'][0]['static_registration_gate']
    assert gate['structure_based']['potential_unique_2D3D']==1
    assert gate['structure_based']['retained_count']==0
    assert gate['structureless']['retained_count']==0
    assert gate['camera_checks'][1]['extra_params_bogus'] is True
    assert gate['historical_failure_proven'] is False
    assert (tmp_path/'report/REGISTRATION.json').is_file() and refs
    json.dumps(r,allow_nan=False)


def test_missing_log_is_unknown_not_zero_and_parent_refs_owned_by_parent(tmp_path):
    run=fixture(tmp_path,False)
    refs=[dict(path=str(tmp_path/'parent-owned-unavailable'),bytes=99,sha256='not-ours')]
    r=audit.audit_registration(run,19,tmp_path/'report',refs)
    assert r['attempt_evidence']['status']=='UNKNOWN_LOG_UNAVAILABLE'
    assert r['counts']['observed_registration_attempts'] is None and r['counts']['observed_failed_attempts'] is None


def test_source_drift_refuses_result(tmp_path,monkeypatch):
    run=fixture(tmp_path);original=audit.parse_registration_attempts
    def changed(*a,**k):
        result=original(*a,**k);run.write_text(run.read_text()+' ',encoding='utf-8');return result
    monkeypatch.setattr(audit,'parse_registration_attempts',changed)
    with pytest.raises(ValueError,match='drift'):audit.audit_registration(run,19,tmp_path/'report',[])
    assert not (tmp_path/'report').exists()


def test_version_and_row_mismatch_refuse(tmp_path):
    run=fixture(tmp_path);r=json.loads(run.read_text());r['backend']['version']='other';run.write_text(json.dumps(r),encoding='utf-8')
    with pytest.raises(ValueError,match='4.2.1'):audit.audit_registration(run,19,tmp_path/'report',[])
    r['backend']['version']='4.2.1';run.write_text(json.dumps(r),encoding='utf-8')
    with sqlite3.connect(run.parent/'database.db') as db:db.execute('UPDATE keypoints SET data=? WHERE image_id=2',(struct.pack('<ff',99,99),))
    with pytest.raises(ValueError,match='row xy'):audit.audit_registration(run,19,tmp_path/'report',[])


def test_fresh_output_and_bool_index_refused(tmp_path):
    run=fixture(tmp_path);out=tmp_path/'report';out.mkdir()
    with pytest.raises(ValueError):audit.audit_registration(run,19,out,[])
    with pytest.raises(ValueError):audit.audit_registration(run,True,tmp_path/'new',[])


def test_missing_actual_options_preserves_unknown_gate(tmp_path):
    run=fixture(tmp_path)
    (run.parent/'logs/incremental_mapping.result.json').write_text('{}',encoding='utf-8')
    r=audit.audit_registration(run,19,tmp_path/'report',[])
    assert r['final_snapshot_correspondences'][0]['static_registration_gate']['status']=='UNKNOWN_MAPPER_OPTIONS'
    assert r['registration_failure_cause']=='UNKNOWN_INTERNAL_BRANCH_NOT_RETAINED'
