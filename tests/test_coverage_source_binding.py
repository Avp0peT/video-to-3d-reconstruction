"""Source identity checks with fake native readback; no native reconstruction."""
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from integrations.demo_pipeline.coverage_expansion import load_local_models
from integrations.demo_pipeline.partial_reconstruction import write_ply
from integrations.demo_pipeline.registration_probe import _binding


def dump(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def bound_source(tmp_path, monkeypatch):
    local = tmp_path/'local'; folder = local/'group_01'/'model_0'
    folder.mkdir(parents=True)
    sfm = tmp_path/'sfm'; sfm.mkdir(); run = sfm/'run.json'
    xyz = np.array([.1, .2, 5.]); observations = []; frames = []; old = []; masks = []
    images = {}
    for i, center in enumerate((-.6, 0., .6), 1):
        t = np.array([-center, 0., 0.]); q = xyz+t
        pixels = (800.*q[:2]/q[2]+[512., 384.]).tolist()
        observations.append(dict(image_id=i, feature_row=0, input_index=i,
            params=[800., 512., 384., 0.], R=np.eye(3).tolist(), t=t.tolist(),
            raw_pixels=pixels, foreground=True))
        frames.append(dict(input_index=i, frame_id=f'frame-{i}', sha256=f'image-{i}',
            original_mask_status='OK', mask_binding=dict(sha256=f'mask-{i}')))
        old.append(dict(input_index=i, frame_id=f'frame-{i}', canonical_image_sha256=f'image-{i}',
            database_image_id=i, staged_image_name=f'image_{i}.png'))
        masks.append(dict(input_index=i, sha256=f'mask-{i}'))
        pose = SimpleNamespace(rotation=SimpleNamespace(matrix=lambda: np.eye(3)), translation=t)
        images[i] = SimpleNamespace(camera_id=1, name=f'image_{i}.png', cam_from_world=lambda p=pose:p,
            points2D=[SimpleNamespace(xy=np.array(pixels))])
    ledger = [dict(point_id=1, xyz=xyz.tolist(), color=[10,20,30],
        accepted_conditionally=True, observations=observations)]
    dump(run, dict(effective_options=dict(foreground=dict(mask_bindings=masks))))
    fmap = sfm/'frames_map.jsonl'
    fmap.write_text('\n'.join(json.dumps(r) for r in old), encoding='utf-8')
    dump(folder/'POINT_LEDGER.json', ledger)
    write_ply(folder/'supported.ply', ledger)
    result = dict(status='PARTIAL_CONDITIONAL_MODEL', source_unchanged=True,
        source_bindings=[_binding(run), _binding(fmap)], groups=[dict(group_id=1, models=[dict(
            model_id=0, status='PARTIAL_CONDITIONAL_MODEL', ba_converged=True,
            roundtrip_differences=[], registered_input_indices=[1,2,3],
            ply_readback=dict(binding=_binding(folder/'supported.ply')))])])
    dump(local/'RESULT.json', result)
    native = SimpleNamespace(reg_image_ids=lambda:[1,2,3], image=lambda i:images[i],
        cameras={1:SimpleNamespace(model_name='SIMPLE_RADIAL', params=[800.,512.,384.,0.])},
        points3D={1:SimpleNamespace(xyz=xyz, color=np.array([10,20,30]),
            track=SimpleNamespace(elements=[SimpleNamespace(image_id=i,point2D_idx=0) for i in (1,2,3)]))})
    fakefile=tmp_path/'fake_pycolmap.py'; fakefile.write_text('# fake readback only',encoding='ascii')
    fake=SimpleNamespace(__version__='4.2.1',__file__=str(fakefile),Reconstruction=lambda p:native)
    monkeypatch.setitem(sys.modules,'pycolmap',fake)
    return SimpleNamespace(local=local,folder=folder,run=run,frames=frames,ledger=ledger,native=native)


def test_matching_sources_pass_real_strict_gate(bound_source):
    s=bound_source
    assert len(load_local_models(s.local,s.frames,[],s.run)) == 1


def test_wrong_sfm_source_rejected(bound_source,tmp_path):
    s=bound_source; wrong=tmp_path/'other-run.json'; wrong.write_text('{}',encoding='ascii')
    with pytest.raises(ValueError,match='different SfM'):
        load_local_models(s.local,s.frames,[],wrong)


@pytest.mark.parametrize('field',['frame_id','sha256','mask'])
def test_current_frame_or_mask_identity_rejected(bound_source,field):
    s=bound_source; frames=copy.deepcopy(s.frames)
    if field=='mask':frames[0]['mask_binding']['sha256']='different'
    else:frames[0][field]='different'
    with pytest.raises(ValueError,match='image or mask identity'):
        load_local_models(s.local,frames,[],s.run)


@pytest.mark.parametrize('change',['xyz','color','camera','foreground','pixels','feature_row','input_index'])
def test_tampered_ledger_rejected(bound_source,change):
    s=bound_source; ledger=copy.deepcopy(s.ledger)
    if change=='xyz':ledger[0]['xyz'][0]+=1.
    if change=='color':ledger[0]['color'][0]+=1
    if change=='camera':ledger[0]['observations'][0]['t'][0]+=.1
    if change=='foreground':ledger[0]['observations'][0]['foreground']=False
    if change=='pixels':ledger[0]['observations'][0]['raw_pixels'][0]+=.05
    if change=='feature_row':ledger[0]['observations'][0]['feature_row']=1
    if change=='input_index':
        for observation in ledger[0]['observations']:
            observation['input_index']=observation['input_index'] % 3 + 1
    dump(s.folder/'POINT_LEDGER.json',ledger)
    with pytest.raises(ValueError):
        load_local_models(s.local,s.frames,[],s.run)
