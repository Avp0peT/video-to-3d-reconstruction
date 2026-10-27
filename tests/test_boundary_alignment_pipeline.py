"""Bound synthetic adapter tests; never execute real reconstruction inputs."""
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from integrations.mvs_surface import boundary_alignment_pipeline as p


def fixture(tmp_path,monkeypatch):
    inputs=tmp_path/'inputs'; inputs.mkdir(); prior=tmp_path/'prior'; prior.mkdir()
    frames=[]; maps={}; bindings=[]; shape=(64,64)
    for i in range(7):
        image=inputs/f'image{i}.png'; mask=inputs/f'mask{i}.png'; valid=inputs/f'valid{i}.png'
        Image.fromarray(np.full((*shape,3),100,np.uint8)).save(image)
        Image.fromarray(np.full(shape,255,np.uint8)).save(mask)
        Image.fromarray(np.full(shape,255,np.uint8)).save(valid)
        d=SimpleNamespace(view_ids=[i],K=np.array([[40.,0,32.],[0,40.,32.],[0,0,1.]]),R=np.eye(3),C=np.array([i*.01,0,0]),depth=np.full(shape,2.,np.float32))
        frame=dict(image=dict(global_id=i,width=64,height=64,K=d.K.tolist(),R=d.R.tolist(),C=d.C.tolist()),image_file=p.binding(image),mask=p.binding(mask),validity=p.binding(valid))
        frames.append(frame)
        bindings.extend([frame['image_file'],frame['mask'],frame['validity']])
        if i!=4:
            file=inputs/f'depth{i}.dmap'; file.write_bytes(b'SYNTHETIC'+bytes([i])); maps[str(file)]=d; bindings.append(p.binding(file))
    historic=inputs/'historical.json'; p.save(historic,dict(frames=frames)); bindings.append(p.binding(historic))
    p.save(prior/'PLAN.json',dict(reference_id=0,source_ids=[3,1,2,4,5,6],source_plan_historical_binding=p.binding(historic),input_bindings=bindings))
    p.save(prior/'RESULT.json',dict(status='MULTIVIEW_SILHOUETTE_ESTIMATES_COMPLETE_NOT_GEOMETRY_ACCEPTED'))
    roi=np.zeros(shape,bool); roi[24:40,24:40]=True
    np.savez_compressed(prior/'SILHOUETTE_INTERVALS.npz',roi=roi)
    monkeypatch.setattr(p,'read_dmap',lambda path:maps[str(path)])
    authenticated=[]
    def authenticate(d,cam):
        authenticated.append(cam['global_id']); np.testing.assert_array_equal(d.K,np.asarray(cam['K']))
    monkeypatch.setattr(p,'authenticate_depth',authenticate)
    protected={f:f.read_bytes() for folder in [inputs,prior] for f in folder.iterdir() if f.is_file()}
    return prior,tmp_path/'review',protected,authenticated


def test_actual_uniform_kernel_integration_missing_depth_and_readonly(tmp_path,monkeypatch):
    prior,out,protected,authenticated=fixture(tmp_path,monkeypatch)
    result=p.run(prior,out)
    assert result['status']=='BOUNDARY_ALIGNMENT_REVIEW_COMPLETE_NOT_ADOPTED'
    assert result['controls']['total']==0
    assert result['source_depth_available']['4'] is False
    assert sorted(authenticated)==[0,1,2,3,5,6]
    for iid in [3,1,2,4,5,6]:
        row=result['views'][str(iid)]
        assert row['tracking']['status']=='NOT_EVALUABLE'
        assert row['hypothetical_roi_projection'] is None
        assert (out/f'view_{iid}.png').exists()
        assert row['original_roi_projection']['projected']==256
    assert not result['geometry_changed'] and not result['masks_changed']
    assert result['cause_verdict']=='NOT_ESTABLISHED'
    for path,data in protected.items(): assert path.read_bytes()==data
    assert '未应用' in (out/'index.html').read_text(encoding='utf-8')
    plan=p.load(out/'PLAN.json'); assert plan['roi_array_binding']['shape']==[64,64]
    assert plan['parameters']['reference_controls']['body_roi_exclusion_radius']==result['controls']['metadata']['parameters']['body_roi_exclusion_radius']==6
    assert plan['source_ids']==[3,1,2,4,5,6]
    with np.load(out/'edge_4.npz',allow_pickle=False) as a: assert a['region_hint'].dtype==bool


def test_nonempty_hypothesis_is_diagnostic_only(tmp_path,monkeypatch):
    prior,out,protected,_=fixture(tmp_path,monkeypatch)
    original=p.audit_control_alignment
    def hypothetical(ref,source,controls,check):
        result=original(ref,source,controls,check)
        result['proposal_translation_xy']=[1.,0.]; result['status']='EVALUATED_NOT_ADOPTED'
        return result
    monkeypatch.setattr(p,'audit_control_alignment',hypothetical)
    result=p.run(prior,out)
    for iid in result['source_ids']:
        assert result['views'][str(iid)]['hypothetical_roi_projection'] is not None
        assert Image.open(out/f'view_{iid}.png').size==(256,114)
    for path,data in protected.items(): assert path.read_bytes()==data
    assert not result['geometry_changed']


def test_npz_nan_roundtrip(tmp_path):
    result=dict(values=np.array([np.nan,1.]),positions=np.array([[np.nan,np.nan]]),metadata='unknown is NaN')
    assert p._npz(tmp_path/'nan.npz',result)==dict(metadata='unknown is NaN')
    with np.load(tmp_path/'nan.npz',allow_pickle=False) as read: np.testing.assert_array_equal(read['values'],result['values'])


def test_modified_bound_mask_fails_before_kernels(tmp_path,monkeypatch):
    prior,out,_,_=fixture(tmp_path,monkeypatch)
    (prior.parent/'inputs/mask4.png').write_bytes(b'changed')
    monkeypatch.setattr(p,'select_reference_controls',lambda *a,**k:pytest.fail('kernel must not run'))
    with pytest.raises(ValueError,match='exact input binding drift'): p.run(prior,out)
    assert p.load(out/'RESULT.json')['status']=='FAILED_NOT_ADOPTED'


def test_bad_saved_roi_and_output_protection(tmp_path,monkeypatch):
    prior,out,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='fresh disjoint'): p.run(prior,prior/'nested')
    np.savez_compressed(prior/'SILHOUETTE_INTERVALS.npz',roi=np.ones((4,4),bool))
    with pytest.raises(ValueError,match='saved ROI'): p.run(prior,out)
    assert p.load(out/'RESULT.json')['status']=='FAILED_NOT_ADOPTED'


def test_failure_recorded_after_plan_freeze(tmp_path,monkeypatch):
    prior,out,_,_=fixture(tmp_path,monkeypatch)
    def fail(*a,**k):
        assert (out/'PLAN.json').exists()
        raise RuntimeError('synthetic kernel failure')
    monkeypatch.setattr(p,'select_reference_controls',fail)
    with pytest.raises(RuntimeError,match='synthetic kernel failure'): p.run(prior,out)
    assert p.load(out/'RESULT.json')['status']=='FAILED_NOT_ADOPTED'


def test_reference_depth_cannot_be_silently_unknown(tmp_path,monkeypatch):
    prior,out,_,_=fixture(tmp_path,monkeypatch)
    plan=p.load(prior/'PLAN.json')
    plan['input_bindings']=[b for b in plan['input_bindings'] if not b['path'].endswith('depth0.dmap')]
    p.save(prior/'PLAN.json',plan)
    with pytest.raises(ValueError,match='reference native depth required'): p.run(prior,out)
