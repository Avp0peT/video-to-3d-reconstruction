"""Saved-camera boundary tests using fake native data only."""
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from integrations.demo_pipeline import point_review_cameras as subject


@pytest.fixture
def saved(tmp_path, monkeypatch):
    root = tmp_path/'sfm'; binary = root/'models/0/binary'
    binary.mkdir(parents=True)
    for name in ('cameras.bin', 'images.bin', 'points3D.bin', 'rigs.bin', 'frames.bin'):
        (binary/name).write_bytes(b'fake binary fixture')
    module = tmp_path/'fake_native.py'; module.write_text('# fake only')
    frame = dict(input_index=0, frame_id='frame0', sha256='image-hash', width=32, height=24,
                 mask_binding=dict(sha256='mask-hash'))
    row = dict(input_index=0, frame_id='frame0', canonical_image_sha256='image-hash',
               canonical_mask_sha256='mask-hash', upstream_io_run_id='upstream',
               status='REGISTERED', staged_image_name='images/0.png',
               registrations=[dict(model_id=0, image_id=7, camera_id=3)])
    run = dict(contract_version='sfm-native-object-v1', status='SUCCESS', upstream_io_run_id='upstream',
               backend=dict(name='pycolmap', version='4.2.1'),
               effective_options=dict(foreground=dict(mask_bindings=[dict(input_index=0, sha256='mask-hash')])),
               models=[dict(model_id=0, binary_path='models/0/binary', registered_image_ids=[7], roundtrip_verified=True)])
    camera = NS(width=32, height=24, model_name='SIMPLE_RADIAL', params=np.array([30.,16.,12.,0.]))
    pose = NS(rotation=NS(matrix=lambda: np.eye(3)), translation=np.array([1.,2.,3.]))
    image = NS(name='images/0.png', camera_id=3, cam_from_world=lambda: pose)
    native = NS(reg_image_ids=lambda: [7], image=lambda iid: image, cameras={3: camera})
    calls = []
    def reconstruct(path):
        calls.append(path)
        return native
    pc = NS(__version__='4.2.1', __file__=str(module), Reconstruction=reconstruct)
    monkeypatch.setattr(subject, '_native_module', lambda: pc)
    discovery = dict(stages=dict(sfm=dict(run_json=str(root/'run.json'))))
    def commit():
        (root/'run.json').write_text(json.dumps(run), encoding='utf-8')
        (root/'frames_map.jsonl').write_text(json.dumps(row)+'\n', encoding='utf-8')
    commit()
    return NS(**locals())


def test_reads_current_run_and_preserves_zero_radial(saved):
    refs = []
    result = subject.load_registered_cameras(saved.discovery, [saved.frame], refs)
    assert result['status'] == 'LOADED' and result['native_reads'] == 1
    camera = result['models'][0]['cameras'][0]
    assert camera['params'] == [30.,16.,12.,0.]
    assert camera['t'] == [1.,2.,3.] and camera['R'] == np.eye(3).tolist()
    assert camera['pose_convention'] == 'world_to_camera'
    assert saved.calls == [str(saved.binary.resolve())]
    assert {r['path'] for r in refs} >= {str(saved.root/'run.json'), str(saved.binary/'cameras.bin')}
    assert all('sha256' in r for r in result['models'][0]['binding']['binaries'])


def test_no_model_never_loads_native(saved):
    saved.run.update(status='NO_MODEL', models=[]); saved.commit()
    assert subject.load_registered_cameras(saved.discovery, [saved.frame], [])['status'] == 'NO_MODEL'
    assert subject.load_registered_cameras({}, [], [])['models'] == []
    assert saved.calls == []


@pytest.mark.parametrize('field,value', [('frame_id','other'), ('canonical_image_sha256','other'),
                                        ('canonical_mask_sha256','other'), ('upstream_io_run_id','other')])
def test_rejects_identity_drift_before_native(saved, field, value):
    saved.row[field] = value; saved.commit()
    with pytest.raises(ValueError, match='identity differs'):
        subject.load_registered_cameras(saved.discovery, [saved.frame], [])
    assert saved.calls == []


def test_rejects_native_identity(saved):
    saved.image.name = 'other.png'
    with pytest.raises(ValueError, match='name/camera'):
        subject.load_registered_cameras(saved.discovery, [saved.frame], [])


def test_unsupported_lens_is_explicit_and_unmodified(saved):
    saved.camera.model_name = 'OPENCV'; saved.camera.params = np.arange(8.)
    result = subject.load_registered_cameras(saved.discovery, [saved.frame], [])
    assert result['status'] == 'UNSUPPORTED'
    camera = result['unsupported_models'][0]['unsupported_cameras'][0]
    assert result['models'] == []
    assert camera['status'] == 'UNSUPPORTED' and camera['params'] == list(range(8))


@pytest.mark.parametrize('lens,params', [('SIMPLE_RADIAL',[30.,16.,12.,0.1]),
                                       ('PINHOLE',[30.,30.,16.,12.]), ('SIMPLE_PINHOLE',[30.,16.,12.])])
def test_only_zero_radial_enters_computable_models(saved, lens, params):
    saved.camera.model_name = lens; saved.camera.params = np.array(params)
    result = subject.load_registered_cameras(saved.discovery, [saved.frame], [])
    assert result['status'] == 'UNSUPPORTED' and result['models'] == []
    assert result['unsupported_models'][0]['unsupported_cameras'][0]['params'] == params


def test_detects_binary_change_during_native_read(saved):
    def mutate(path):
        (saved.binary/'images.bin').write_bytes(b'drift')
        return saved.native
    saved.pc.Reconstruction = mutate
    with pytest.raises(ValueError, match='binding differs'):
        subject.load_registered_cameras(saved.discovery, [saved.frame], [])


def test_wrong_registration_list_and_path_escape(saved):
    saved.run['models'][0]['registered_image_ids'] = [8]; saved.commit()
    with pytest.raises(ValueError, match='registered image IDs'):
        subject.load_registered_cameras(saved.discovery, [saved.frame], [])
    saved.run['models'][0].update(registered_image_ids=[7], binary_path='../outside'); saved.commit()
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.load_registered_cameras(saved.discovery, [saved.frame], [])


def test_native_failure_retains_every_previously_read_binding(saved):
    refs = [dict(path='preexisting', sha256='unrelated')]
    def fail(path):
        raise RuntimeError('fake native read failure')
    saved.pc.Reconstruction = fail
    with pytest.raises(RuntimeError, match='fake native'):
        subject.load_registered_cameras(saved.discovery, [saved.frame], refs)
    paths = {ref['path'] for ref in refs}
    assert str(saved.root/'run.json') in paths
    assert str(saved.root/'frames_map.jsonl') in paths
    assert str(saved.module) in paths
    assert all(str(saved.binary/name) in paths for name in
               ('cameras.bin','images.bin','points3D.bin','rigs.bin','frames.bin'))
    assert all('bytes' in ref and 'sha256' in ref for ref in refs[1:])


def test_run_binding_ignores_preexisting_refs_and_sam_mask_needs_no_photo_io_mask(saved):
    saved.row['canonical_mask_sha256'] = None; saved.commit()
    refs = [dict(path='preexisting', sha256='unrelated')]
    result = subject.load_registered_cameras(saved.discovery, [saved.frame], refs)
    assert result['status'] == 'LOADED'
    assert result['models'][0]['binding']['run']['path'] == str(saved.root/'run.json')


def test_identity_failure_retains_run_and_frame_map(saved):
    refs = []
    saved.row['frame_id'] = 'changed'; saved.commit()
    with pytest.raises(ValueError):
        subject.load_registered_cameras(saved.discovery, [saved.frame], refs)
    assert [r['path'] for r in refs] == [str(saved.root/'run.json'), str(saved.root/'frames_map.jsonl')]


def test_pinhole_contract_preserves_unsupported_native_lens(saved):
    saved.run['contract_version']='sfm-native-object-pinhole-v1'
    saved.camera.model_name='SIMPLE_PINHOLE'
    saved.camera.params=np.array([30.,16.,12.])
    saved.commit()
    result=subject.load_registered_cameras(saved.discovery,[saved.frame],[])
    assert result['status']=='UNSUPPORTED' and result['native_reads']==1
    camera=result['unsupported_models'][0]['unsupported_cameras'][0]
    assert camera['camera_model']=='SIMPLE_PINHOLE' and camera['params']==[30.,16.,12.]
