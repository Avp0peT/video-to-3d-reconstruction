"""Native integration with known projected coordinates, never real photographs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from integrations.demo_pipeline import view_geometry as v
from test_multi_pair_ablation import multi_camera_fixture
from test_mismatch_identifiability import sample_pair


def hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize("contract", ["sfm-native-object-v1", "sfm-native-object-pinhole-v1"])
def test_native_complete_and_bounded_coverage(tmp_path,monkeypatch,contract):
    pinhole=contract.endswith("pinhole-v1")
    source,binary,_,truth=multi_camera_fixture(tmp_path,count=9,
        camera_model="SIMPLE_PINHOLE" if pinhole else "SIMPLE_RADIAL")
    if pinhole:
        path=source/"run.json"
        doc=json.loads(path.read_text());doc["contract_version"]=contract;path.write_text(json.dumps(doc))
        cameras,_=v.bridge.read_calibration(binary/"cameras.bin",binary/"images.bin")
        assert all(c['native_model_id']==0 and len(c['native_params'])==3 and c['params'][3]==0. for c in cameras.values())
    before=hashes(source)
    monkeypatch.setattr(v,'MAX_TRACKS',5)
    out=tmp_path/'new'
    result=v.audit(source,'0',out)
    if pinhole:
        assert all(c['native_model_id']==0 and len(c['native_params'])==3 for c in result['source_cameras'].values())
    assert hashes(source)==before
    assert result['status']=='PARTIAL_DIAGNOSTIC'
    assert result['counts']['native_tracks']==9
    assert result['counts']['selected_tracks']==5
    assert result['counts']['not_sampled_tracks']==4
    plan=json.loads((out/'PLAN.json').read_text(encoding='utf-8'))
    assert plan['selected_point_ids']==[1,2,3,4,5]
    assert plan['not_sampled_point_ids']==[6,7,8,9]
    rows=json.loads((out/'TRACKS.json').read_text(encoding='utf-8'))['tracks']
    for row in rows:
        assert row['selected']['pair_image_ids']==[1,6]
        assert row['ordered_pair_control']['pair_image_ids']==[1,2]
        assert row['status']=='WITHIN_DIAGNOSTIC_LINE_NOT_VERIFIED'
        assert row['geometry_acceptance'] is False
        assert row['all_native_views_checked'] is True
        assert np.allclose(row['selected']['fit']['xyz'],truth[row['point_id']-1],atol=1e-7)
        assert len(row['selected']['per_view'])==6
        assert len(row['selected']['additional_image_ids'])==4
        assert row['selected']['all_observations']['max']<1e-7
    for ref in result['artifacts']+result['source_bindings']:
        assert hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest()==ref['sha256']
    assert json.loads((out/'RESULT.json').read_text(encoding='utf-8'))==result
    assert not list(out.glob('*.ply'))
    with pytest.raises(ValueError):v.audit(source,'0',out)


@pytest.mark.parametrize('bad',['status','model','counts','escaped'])
def test_bad_native_identity_refused_before_output(tmp_path,bad):
    source,_,_,_=multi_camera_fixture(tmp_path,count=7)
    path=source/'run.json';doc=json.loads(path.read_text(encoding='utf-8'))
    if bad=='status':doc['status']='NO_MODEL'
    elif bad=='model':doc['models'][0]['roundtrip_verified']=False
    elif bad=='counts':doc['models'][0]['num_points3D']+=1
    else:doc['models'][0]['binary_path']='../prior'
    path.write_text(json.dumps(doc),encoding='utf-8')
    with pytest.raises((ValueError,FileNotFoundError)):v.audit(source,'0',tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_selected_pair_fit_never_receives_extra_views(monkeypatch):
    positive,_=sample_pair()
    obs=sum(positive['observations'].values(),[])
    before=deepcopy(obs)
    seen=[];original=v.fixed.fit
    def spy(data,*args,**kwargs):
        seen.append([o['image_id'] for o in data]);return original(data,*args,**kwargs)
    monkeypatch.setattr(v.fixed,'fit',spy)
    result=v.check_candidate(obs,[1,2])
    assert seen==[[1,2]] and obs==before
    assert result['additional_image_ids']==[3,4]
    assert result['status']=='WITHIN_DIAGNOSTIC_LINE_NOT_VERIFIED'
    obs[-1]['raw_pixels'][0]+=50
    changed=v.check_candidate(obs,[1,2])
    assert changed['fit']['xyz']==result['fit']['xyz']
    assert changed['all_observations']['max']>49
    assert changed['status']=='OBSERVATION_INCONSISTENCY_DETECTED'


def test_pair_only_ambiguous_and_excess_views_never_pass():
    positive,_=sample_pair();obs=positive['observations']['pair']
    row=v.audit_track(1,obs)
    assert row['status']=='PAIR_ONLY_NOT_VERIFIED'
    ambiguous=v.audit_track(1,obs,{'3':[1,2]})
    assert ambiguous['status']=='INCOMPLETE_AMBIGUOUS_VIEWS'
    assert ambiguous['all_native_views_checked'] is False
    excess=[dict(obs[0],image_id=i) for i in range(65)]
    assert v.audit_track(1,excess)['status']=='NOT_EVALUATED_VIEW_COUNT'
    assert v.audit_track(1,[])['status']=='NOT_EVALUATED_VIEW_COUNT'


def test_full_check_retains_bad_projection_and_other_view_evidence():
    p,_=sample_pair();obs=sum(p['observations'].values(),[])
    obs[-1]['t']=[0.,0.,-10.]
    row=v.check_candidate(obs,[1,2])
    assert row['status']=='PROJECTION_REFUSED'
    assert len(row['per_view'])==4
    assert row['per_view'][-1]['diagnostic']['status']=='PROJECTION_REFUSED'
    assert row['per_view'][0]['diagnostic']['status']=='EVALUATED'


def test_shared_soft_budget_keeps_every_selected_slot(tmp_path,monkeypatch):
    source,_,_,_=multi_camera_fixture(tmp_path,count=7)
    clock=iter([0.,100.,100.,100.,100.,100.,100.,100.])
    monkeypatch.setattr(v.time,'monotonic',lambda:next(clock))
    out=tmp_path/'budget';r=v.audit(source,'0',out)
    assert r['counts']['track_statuses']=={'NOT_EVALUATED_BUDGET':7}
    assert r['status']=='PARTIAL_DIAGNOSTIC'
    rows=json.loads((out/'TRACKS.json').read_text(encoding='utf-8'))['tracks']
    assert [row['point_id'] for row in rows]==list(range(1,8))


def test_source_drift_after_plan_fails_without_result(tmp_path,monkeypatch):
    source,_,_,_=multi_camera_fixture(tmp_path,count=7)
    original=v.audit_track
    def changed(*args,**kwargs):
        row=original(*args,**kwargs)
        with (source/'run.json').open('a',encoding='utf-8') as stream:stream.write(' ')
        return row
    monkeypatch.setattr(v,'audit_track',changed)
    out=tmp_path/'drift'
    with pytest.raises(ValueError,match='drift'):v.audit(source,'0',out)
    assert (out/'TRACKS.json').exists()
    assert not (out/'RESULT.json').exists()
