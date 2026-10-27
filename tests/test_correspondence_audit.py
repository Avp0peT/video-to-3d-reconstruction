from copy import deepcopy
import hashlib
import json
import numpy as np

from integrations.demo_pipeline import correspondence_audit as audit
from kozlov_lab.stereo.baseline import multiview_fixed as fixed


def scene(count=5):
    xyz = [0.1,0.2,4.0]; observations=[]
    for i in range(count):
        o=dict(image_id=i+1,source_track_id=9,point2D_index=10+i,
               raw_pixels=[0.,0.],params=[800.,320.,240.,0.],
               R=np.eye(3).tolist(),t=[-i*.3,0.,0.],max_ideal_radius=3.)
        o['raw_pixels']=fixed.project(xyz,o)['pixels'].tolist()
        o['raw_binary64_hex']=[v.hex() for v in o['raw_pixels']]
        observations.append(o)
    return dict(point_id=9,C_xyz=xyz,J_xyz=xyz,observations=observations,
                image_pair=[1,2],ambiguous_observations={},original_track_elements=[])


def test_synthetic_projection_truth_and_immutable_source():
    source=scene(); original=deepcopy(source)
    row=audit.audit_point(source)
    assert source==original
    assert row['status']=='consistent_not_verified'
    assert row['train_image_ids']==[1,2,4] and row['heldout_image_ids']==[3,5]
    assert np.allclose(row['methods']['J_conditional']['xyz'],source['C_xyz'],atol=1e-8)
    assert row['methods']['J_conditional']['heldout']['max']<1e-8
    assert row['views'][2]['measured_xy']==source['observations'][2]['raw_pixels']


def test_heldout_pixels_never_affect_fit():
    source=scene(); changed=deepcopy(source)
    changed['observations'][2]['raw_pixels'][0]+=70
    before=audit.audit_point(source); after=audit.audit_point(changed)
    assert before['conditional_fit']==after['conditional_fit']
    assert after['status']=='inconsistency'
    assert after['methods']['J_conditional']['heldout']['max']>60


def test_wrong_correspondence_retained_and_detected():
    source=scene(); source['observations'][1]['raw_pixels'][1]+=80
    row=audit.audit_point(source)
    assert row['methods']['C']['status']=='inconsistency'
    assert row['status']=='inconsistency'
    assert row['views'][1]['measured_xy'][1]==source['observations'][1]['raw_pixels'][1]
    assert row['train_image_ids']==[1,2,4]


def test_two_views_and_ambiguous_never_verified():
    row=audit.audit_point(scene(2))
    assert row['status']=='insufficient' and row['methods']['J_conditional']['xyz'] is None
    source=scene(); source['ambiguous_observations']={'6':[5,7]}
    row=audit.audit_point(source)
    assert row['status']=='ambiguous'
    assert all(m['status']=='ambiguous' for m in row['methods'].values())


def test_budget_preserves_exact_point_and_diagnostic_denominators():
    row=audit.audit_point(scene(),run_fit=False)
    assert row['point_id']==9 and row['status']=='NOT_RUN_BUDGET'
    assert row['methods']['J_conditional']['full']['status']=='NOT_EVALUATED'
    summary=audit.summarize([row])
    assert summary['support_histogram']=={'5':1}
    assert summary['paired']['heldout']['count']==0


def test_native_fixture_local_surface_indices_and_unchanged_sources(tmp_path):
    from test_multi_pair_ablation import multi_camera_fixture
    from integrations.demo_pipeline import expanded_comparison
    _,_,frozen,_=multi_camera_fixture(tmp_path,count=6)
    previous=tmp_path/'previous'; previous.mkdir()
    expanded_comparison.generate(frozen,previous/'expanded')
    records=json.loads((previous/'expanded/PAIRED_POINTS.json').read_text())['points']
    (previous/'foreground').mkdir()
    (previous/'foreground/SHARED_FOREGROUND_POINTS.json').write_text(json.dumps({'points':records}))
    # Surface image indices belong to the bridge, not native COLMAP image IDs.
    identities=[dict(index=i,point_id=p['point_id'],scene_views=[0,2,3]) for i,p in enumerate(records)]
    for m in ('C','J'):
        folder=previous/'surface_inputs'/m; folder.mkdir(parents=True)
        (folder/'POINT_IDENTITY.json').write_text(json.dumps(identities))
    sealed = [previous/'surface_inputs'/m/'POINT_IDENTITY.json' for m in ('C','J')]
    sealed += [previous/'foreground/SHARED_FOREGROUND_POINTS.json',previous/'expanded/PLAN.json']
    (previous/'SHA256SUMS.txt').write_text('\n'.join(
        hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(previous).as_posix() for p in sealed))
    def hashes():
        return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file() and 'audit_output' not in p.parts}
    before=hashes(); result=audit.audit(previous,tmp_path/'audit_output')
    assert hashes()==before
    assert result['requested_count']==6 and result['SOURCE_immutable']
    saved=json.loads((tmp_path/'audit_output/PLAN.json').read_text())
    assert saved['frame_bindings'] and saved['conditional_not_independent']
    import pytest
    records[0]['C_xyz'][0]+=1
    (previous/'foreground/SHARED_FOREGROUND_POINTS.json').write_text(json.dumps({'points':records}))
    with pytest.raises(ValueError,match='sealed previous'):
        audit.audit(previous,tmp_path/'tampered_output')
    assert not (tmp_path/'tampered_output').exists()


def test_output_binding_does_not_inherit_input_size_limit(tmp_path,monkeypatch):
    path=tmp_path/'output'; path.write_bytes(b'large-relative-to-mocked-limit')
    monkeypatch.setattr(audit.native,'MAX_BYTES',1)
    import pytest
    with pytest.raises(ValueError,match='64 MiB'): audit.native.binding(path)
    assert audit._output_binding(path)['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_component_uses_raw_observations_without_paper_candidate(tmp_path):
    from test_multi_pair_ablation import multi_camera_fixture
    source,_,_,_=multi_camera_fixture(tmp_path,count=6)
    run=json.loads((source/'run.json').read_text())
    model=str(run['models'][0]['model_id'])
    result=audit.audit_native(source,model,tmp_path/'native_audit')
    assert result['old_xyz_used'] is False and result['C_algorithm_called'] is False
    rows=json.loads((tmp_path/'native_audit/TRACKS.json').read_text())['tracks']
    assert len(rows)==6
    assert all(r['methods']['C']['status']=='NOT_AVAILABLE' and r['methods']['C']['xyz'] is None for r in rows)
    assert all(r['baseline_pair_image_ids']==[1,2] and r['heldout_image_ids']==[3,5] for r in rows)
