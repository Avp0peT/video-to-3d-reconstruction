import copy
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integrations.demo_pipeline import parallel_photo_audit as a

BODY = [(0,0,0),(1,0,0),(0,1,0),(1,1,1),(2,3,2),(3,1,-1)]
FIRST = {i+1:(float(p[0]),float(p[1])) for i,p in enumerate(BODY)}
SECOND = {i+1:(float(p[0]+p[2]),float(p[1]+2*p[2])) for i,p in enumerate(BODY)}


def native(tmp_path, *, damage=None):
    root = tmp_path / 'source'
    root.mkdir()
    binary = root / 'models' / '0' / 'binary'
    binary.mkdir(parents=True)
    (root / 'images').mkdir()
    image_bytes = struct.pack('<Q',2)
    frames = []
    for ident, view in ((1,FIRST),(2,SECOND)):
        name = f'images/{ident}.png'
        photo = root / name
        photo.write_bytes(b'bound fixture image ' + str(ident).encode())
        image_bytes += struct.pack('<idddddddi',ident,1,0,0,0,0,0,0,1) + name.encode() + b'\0' + struct.pack('<Q',7 if damage=='multi' and ident==1 else 6)
        for pid,(x,y) in view.items():
            image_bytes += struct.pack('<ddq',x,y,pid)
        if damage=='multi' and ident==1:
            image_bytes += struct.pack('<ddq',.1,.1,1)
        frames.append(dict(upstream_io_run_id='io1', input_index=ident-1, frame_id=f'frame{ident}',
                           staged_image_name=name, canonical_image_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),
                           registrations=[dict(model_id='0',image_id=ident,camera_id=1)], status='REGISTERED'))
    point_bytes = struct.pack('<Q',6)
    for pid in FIRST:
        point_bytes += struct.pack('<QdddBBBdQ',pid,900+pid,800,700,1,2,3,0,3 if damage=='multi' and pid==1 else 2)
        point_bytes += struct.pack('<iiii',1,pid-1,2,pid-1 if damage != 'track' else 0)
        if damage=='multi' and pid==1:
            point_bytes += struct.pack('<ii',1,6)
    (binary/'images.bin').write_bytes(image_bytes + (b'bad' if damage == 'trailing' else b''))
    (binary/'points3D.bin').write_bytes(point_bytes)
    (binary/'cameras.bin').write_bytes(struct.pack('<QiiQQdddd',1,1,2,1024,683,1000,512,341.5,0))
    if damage == 'camera':
        frames[0]['registrations'][0]['camera_id'] = 2
    (root/'frames_map.jsonl').write_text('\n'.join(json.dumps(x) for x in frames),encoding='utf-8')
    run = dict(contract_version='sfm-native-object-v1', status='SUCCESS', upstream_io_run_id='io1',
               models=[dict(model_id='0',binary_path='models/0/binary',registered_image_ids=[1,2],
                            num_points3D=6,usable=True,roundtrip_verified=True)])
    (root/'run.json').write_text(json.dumps(run),encoding='utf-8')
    return root, binary


def test_full_synthetic_native_chain_and_readback(tmp_path):
    root, binary = native(tmp_path)
    out = tmp_path/'result'
    result = a.audit(root,'0',out)
    assert result['exact_status'] == 'CONSTRUCTED'
    assert result['independent_check']['status'] == 'VERIFIED'
    assert result['synthetic_control']['status'] == 'CONSTRUCTED'
    assert result['photo_mesh_created'] is False
    data = json.loads((out/'INPUT.json').read_text(encoding='utf-8'))
    assert data['observations'][0]['observations'][0]['xy'] == [0.,0.]
    assert data['observations'][0]['observations'][0]['xy'] != [901.,800.]
    assert json.loads((out/'RESULT.json').read_text(encoding='utf-8')) == result
    before = (out/'INPUT.json').read_bytes()
    with pytest.raises(ValueError,match='no overwrite'):
        a.audit(root,'0',out)
    assert (out/'INPUT.json').read_bytes() == before


@pytest.mark.parametrize('damage,match',[('track','identity mismatch'),('trailing','trailing native'),('camera','frame camera')])
def test_input_identity_corruption_rejected_before_output(tmp_path,damage,match):
    root,_ = native(tmp_path,damage=damage)
    out = tmp_path/'rejected'
    with pytest.raises(ValueError,match=match):
        a.audit(root,'0',out)
    assert not out.exists()


def test_image_drift_rejected(tmp_path):
    root,_ = native(tmp_path)
    (root/'images/1.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='byte mismatch'):
        a.audit(root,'0',tmp_path/'out')


def test_native_truncation_and_nonfinite(tmp_path):
    _,binary = native(tmp_path)
    p = binary/'images.bin'
    raw = p.read_bytes()
    p.write_bytes(raw[:-1])
    with pytest.raises(ValueError,match='truncation|truncated'):
        a.read_native(p,binary/'points3D.bin')
    # First observation follows count, image header, null name, observation count.
    offset = 8 + 64 + len('images/1.png') + 1 + 8
    altered = bytearray(raw)
    struct.pack_into('<d',altered,offset,float('nan'))
    p.write_bytes(altered)
    with pytest.raises(ValueError,match='observation'):
        a.read_native(p,binary/'points3D.bin')


def test_selection_independent_of_coordinate_geometry_and_sorted_ids():
    images = {1:{},2:{},3:{}}
    points = {i:dict(track={1:i,2:i,3:i}) for i in (60,40,20,10,30,50)}
    pair,ids,counts = a.choose_pair(images,points)
    assert pair == (1,2) and ids == [10,20,30,40,50,60]
    assert [c['shared_tracks'] for c in counts] == [6,6,6]


def test_multiple_observations_preserved_but_not_silently_chosen(tmp_path):
    _,binary = native(tmp_path,damage='multi')
    images,points = a.read_native(binary/'images.bin',binary/'points3D.bin')
    assert points[1]['ambiguous_observations'] == {'1':[0,6]}
    assert points[1]['track_elements'] == {(1,0),(1,6),(2,0)}
    assert 1 not in points[1]['track']
    with pytest.raises(ValueError,match='six common'):
        a.choose_pair(images,points)


def test_binary64_preserved_not_decimal_rounding():
    images = {1:dict(observations=[(123.456,2.5,1)]),2:dict(observations=[(5.,6.,1)])}
    first,_ = a.views_for((1,2),[1],images,{1:dict(track={1:0,2:0})})
    assert first[1][0] == F.from_float(123.456)
    assert first[1][0] != F('123.456')


def test_tls_positive_and_perturbed_observations():
    good = a.numerical_residual(FIRST,SECOND)
    assert good['rms_joint_distance_px'] < 1e-12
    changed = dict(SECOND)
    changed[6] = (changed[6][0]+.25,changed[6][1]-.31)
    bad = a.numerical_residual(FIRST,changed)
    assert bad['rms_joint_distance_px'] > .001
    assert bad['acceptance_threshold'] is None and bad['model_accepted'] is None


def test_exact_negative_is_verified_not_reconstructed():
    first = {i:tuple(F(v) for v in p) for i,p in FIRST.items()}
    second = {i:tuple(F(v) for v in p) for i,p in SECOND.items()}
    second[6] = (second[6][0]+F(1,4),second[6][1]-F(31,100))
    result,_,check = a._run_exact(first,second)
    assert result['status'] == 'INCOMPATIBLE_EXACT_PARALLEL'
    assert check['status'] == 'VERIFIED'
    assert result['algorithm_called'] == 'NOT_CONSTRUCTED'


def test_escape_and_bool_model_ids_rejected(tmp_path):
    root,_ = native(tmp_path)
    with pytest.raises(ValueError,match='escapes'):
        a.under(root,'..')
    run = json.loads((root/'run.json').read_text())
    run['models'][0]['registered_image_ids'] = [True,2]
    (root/'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError,match='typed unique'):
        a.audit(root,'0',tmp_path/'out')


def test_duplicate_json_key_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        a.loads('{"schema":1,"schema":2}')


def test_input_change_during_algorithm_rejected(tmp_path,monkeypatch):
    root,_ = native(tmp_path)
    original = a._run_exact
    calls = []
    def changed(*args):
        calls.append(1)
        if len(calls)==1:
            (root/'images/1.png').write_bytes(b'drift after freeze')
        return original(*args)
    monkeypatch.setattr(a,'_run_exact',changed)
    with pytest.raises(ValueError,match='stability'):
        a.audit(root,'0',tmp_path/'out')
    assert not (tmp_path/'out/RESULT.json').exists()
