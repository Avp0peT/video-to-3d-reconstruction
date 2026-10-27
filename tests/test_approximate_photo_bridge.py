"""Synthetic native-format camera/track integration, without pixel inference."""
import hashlib
import importlib
import json
from pathlib import Path
import struct
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.demo_pipeline import parallel_photo_audit as prior


def camera_fixture(tmp_path):
    root=tmp_path/'source';binary=root/'models/0/binary'
    binary.mkdir(parents=True);(root/'images').mkdir()
    rng=np.random.default_rng(1996)
    xyz=rng.uniform((-0.7,-0.6,-0.6),(0.7,0.6,0.6),(18,3))
    cameras=[]
    for ident,angle,center,f,k in ((1,-0.07,(0.,0.,-8.),1000.,0.01),(2,0.13,(1.,0.2,-8.),950.,-0.005)):
        cosine,sine=np.cos(angle),np.sin(angle)
        rotation=np.array(((cosine,0.,sine),(0.,1.,0.),(-sine,0.,cosine)))
        translation=-rotation@np.array(center)
        quaternion=(float(np.cos(angle/2)),0.,float(np.sin(angle/2)),0.)
        params=(f,512.,341.5,k)
        cameras.append((ident,rotation,translation,quaternion,params))
    image_bytes=struct.pack('<Q',2);camera_bytes=struct.pack('<Q',2);frames=[]
    for ident,rotation,translation,quaternion,params in cameras:
        f,cx,cy,k=params
        camera_bytes+=struct.pack('<iiQQdddd',ident,2,1024,683,*params)
        name=f'images/{ident}.png';photo=root/name
        photo.write_bytes(b'fixed synthetic photograph byte fixture '+str(ident).encode())
        image_bytes+=struct.pack('<idddddddi',ident,*quaternion,*translation,ident)+name.encode()+b'\0'+struct.pack('<Q',18)
        for pid,p in enumerate(xyz,1):
            cp=rotation@p+translation;u,v=cp[:2]/cp[2];factor=1+k*(u*u+v*v)
            image_bytes+=struct.pack('<ddq',f*u*factor+cx,f*v*factor+cy,pid)
        frames.append(dict(upstream_io_run_id='synthetic_camera_io',input_index=ident-1,frame_id=f'frame{ident}',
            staged_image_name=name,canonical_image_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),
            registrations=[dict(model_id='0',image_id=ident,camera_id=ident)],status='REGISTERED'))
    point_bytes=struct.pack('<Q',18)
    for pid,p in enumerate(xyz,1):
        point_bytes+=struct.pack('<QdddBBBdQ',pid,*p,1,2,3,0.,2)+struct.pack('<iiii',1,pid-1,2,pid-1)
    (binary/'images.bin').write_bytes(image_bytes)
    (binary/'points3D.bin').write_bytes(point_bytes)
    (binary/'cameras.bin').write_bytes(camera_bytes)
    (root/'frames_map.jsonl').write_text('\n'.join(json.dumps(row) for row in frames),encoding='utf-8')
    (root/'run.json').write_text(json.dumps(dict(contract_version='sfm-native-object-v1',status='SUCCESS',upstream_io_run_id='synthetic_camera_io',
        models=[dict(model_id='0',binary_path='models/0/binary',registered_image_ids=[1,2],num_points3D=18,usable=True,roundtrip_verified=True)])),encoding='utf-8')
    old=tmp_path/'prior';prior.audit(root,'0',old)
    frozen=old/'INPUT.json';data=json.loads(frozen.read_text(encoding='utf-8'))
    assert data['point_ids']==[1,2,3,4,5,6] and data['common_tracks']==18
    return root,binary,frozen,xyz


def bridge():
    return importlib.import_module('integrations.demo_pipeline.approximate_photo_bridge')



@pytest.mark.parametrize('changed',['photo','native'])
def test_prior_source_binding_drift_rejected_before_output(tmp_path,changed):
    root,binary,frozen,truth=camera_fixture(tmp_path)
    if changed=='photo':(root/'images/1.png').write_bytes(b'drifted bound photograph')
    else:
        path=binary/'cameras.bin';path.write_bytes(path.read_bytes()+b'bad')
    output=tmp_path/'rejected'
    with pytest.raises(ValueError):bridge().experiment(frozen,output)
    assert not output.exists()


def test_frozen_diagnostic_six_cannot_be_changed_or_reselected(tmp_path):
    root,binary,frozen,truth=camera_fixture(tmp_path)
    data=json.loads(frozen.read_text(encoding='utf-8'))
    data['point_ids']=[7,8,9,10,11,12]
    frozen.write_text(json.dumps(data),encoding='utf-8')
    output=tmp_path/'reselected'
    with pytest.raises(ValueError):bridge().experiment(frozen,output)
    assert not output.exists()


def test_bridge_output_never_overwritten(tmp_path):
    _,_,frozen,_=camera_fixture(tmp_path)
    output=tmp_path/'existing';output.mkdir();marker=output/'evidence.bin';marker.write_bytes(b'original evidence')
    with pytest.raises(ValueError):bridge().experiment(frozen,output)
    assert marker.read_bytes()==b'original evidence'
    assert list(output.iterdir())==[marker]



def test_calibrated_six_candidate_and_full_training_split_against_truth(tmp_path):
    root,binary,frozen,truth=camera_fixture(tmp_path)
    original=frozen.read_bytes();output=tmp_path/'experiment'
    result=bridge().experiment(frozen,output)
    assert result['source_unchanged'] is True and frozen.read_bytes()==original
    assert result['point_ids']==[1,2,3,4,5,6]
    assert result['candidate']['status']=='CALIBRATED_SIX_POINT_CANDIDATE'
    assert result['rectified_exact']['status']=='CONSTRUCTED'
    assert result['rectified_exact']['independent']['status']=='VERIFIED'
    assert result['candidate']['vertex_count']==6 and result['candidate']['face_count']==0
    assert result['independent_new_reconstruction'] is False and result['mesh_created'] is False and result['cad_created'] is False
    plan=json.loads((output/'PLAN.json').read_text(encoding='utf-8'))
    assert plan['six_point_ids']==[1,2,3,4,5,6]
    assert plan['training_ids']==[7,9,11,13,15,17]
    assert plan['validation_ids']==[8,10,12,14,16,18]
    assert set(plan['training_ids']).isdisjoint(plan['six_point_ids'])
    assert set(plan['validation_ids']).isdisjoint(plan['six_point_ids'])
    assert set(plan['training_ids']).isdisjoint(plan['validation_ids'])
    assert len(plan['common_ids'])==18 and result['train_count']==result['validation_count']==6
    assert plan['calibration'][0]['pose']!=plan['calibration'][1]['pose']
    ply=Path(result['candidate']['path']);text=ply.read_text(encoding='ascii')
    assert 'element vertex 6' in text and 'ENGINEERING_CANDIDATE_NOT_ACCEPTED' in text
    candidate=np.array([[float(v) for v in row.split()] for row in text.split('end_header\n',1)[1].splitlines()])
    assert candidate.shape==(6,3)
    error=np.linalg.norm(candidate-truth[:6],axis=1)
    # Fixed normalized2^18 quantization, baseline about1 and depth8. A0.01
    # world-unit tolerance is conservative for these synthetic points;
    # it is a test-control check, never a real-photo millimetre accuracy claim.
    assert max(error)<0.01,error
    assert json.loads((output/'RESULT.json').read_text(encoding='utf-8'))==json.loads(json.dumps(result))
    unchanged_files={p:p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(ValueError):bridge().experiment(frozen,output)
    assert all(p.read_bytes()==content for p,content in unchanged_files.items())


def test_truncated_frozen_six_observation_ledger_rejected_before_output(tmp_path):
    _,_,frozen,_=camera_fixture(tmp_path)
    data=json.loads(frozen.read_text(encoding='utf-8'));data['observations']=data['observations'][:-1]
    frozen.write_text(json.dumps(data),encoding='utf-8');output=tmp_path/'truncated'
    with pytest.raises(ValueError):bridge().experiment(frozen,output)
    assert not output.exists()



def test_frozen_xy_cannot_disagree_with_binary64_identity(tmp_path):
    _,_,frozen,_=camera_fixture(tmp_path)
    data=json.loads(frozen.read_text(encoding='utf-8'))
    data['observations'][0]['observations'][0]['xy'][0]+=1
    frozen.write_text(json.dumps(data),encoding='utf-8');output=tmp_path/'conflicting_xy'
    with pytest.raises(ValueError):bridge().experiment(frozen,output)
    assert not output.exists()


def test_duplicate_source_binding_is_not_silently_collapsed(tmp_path):
    _,_,frozen,_=camera_fixture(tmp_path)
    data=json.loads(frozen.read_text(encoding='utf-8'))
    data['source_bindings'].append(data['source_bindings'][0])
    frozen.write_text(json.dumps(data),encoding='utf-8');output=tmp_path/'duplicate_binding'
    with pytest.raises(ValueError):bridge().experiment(frozen,output)
    assert not output.exists()
