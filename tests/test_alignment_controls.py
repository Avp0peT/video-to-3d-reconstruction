from copy import deepcopy
import cv2
import numpy as np
import pytest
from integrations.mvs_surface.alignment_controls import select_reference_controls,audit_control_alignment


def scene(uniform=False):
    cv2.setNumThreads(2)
    rng=np.random.default_rng(442)
    image=rng.integers(0,256,(192,192,3),np.uint8)
    image=cv2.GaussianBlur(image,(3,3),.7)
    if uniform: image[:]=100
    shifted=cv2.warpAffine(image,np.float32([[1,0,3],[0,1,-2]]),(192,192),borderMode=cv2.BORDER_REFLECT)
    def view(i,im):
        return dict(view_id=i,image=im,mask=np.ones((192,192),bool),validity=np.ones((192,192),bool),depth=np.full((192,192),2.,np.float32),K=np.array([[150.,0,96],[0,150.,96],[0,0,1]]),R=np.eye(3),C=np.zeros(3))
    roi=np.zeros((192,192),bool);roi[72:120,72:120]=True
    return view(0,image),view(1,shifted),roi


def test_textured_translation_image_only_unseeded_and_no_mutation(monkeypatch):
    ref,src,roi=scene();before=deepcopy((ref,src))
    controls=select_reference_controls(ref,roi)
    lk=cv2.calcOpticalFlowPyrLK;calls=[]
    def spy(*args,**kwargs):
        calls.append((args[3],kwargs['flags']))
        return lk(*args,**kwargs)
    monkeypatch.setattr(cv2,'calcOpticalFlowPyrLK',spy)
    result=audit_control_alignment(ref,src,controls)
    assert calls == [(None,0),(None,0)]
    np.testing.assert_allclose(result['proposal_translation_xy'],[3,-2],atol=.08)
    assert result['per_group']['body_test']['median_after'] < .1
    assert result['per_group']['body_test']['median_before'] > 3
    assert result['per_group']['roi']['accepted'] > 0
    assert controls['control_xy'].shape[0]<=448
    for a,b in zip((ref,src),before):
        for key in ('depth','image','mask','validity','K','R','C'): np.testing.assert_array_equal(a[key],b[key])
    body=controls['group']==0
    train=set(controls['cell_id'][body&controls['train_split']])
    test=set(controls['cell_id'][body&~controls['train_split']])
    assert train.isdisjoint(test)
    xy=controls['control_xy'].astype(int)
    assert not roi[xy[body,1],xy[body,0]].any()
    assert roi[xy[~body,1],xy[~body,0]].all()


def test_roi_controls_cannot_fit_translation():
    ref,src,roi=scene();controls=select_reference_controls(ref,roi)
    controls={k:(v[controls['group']==1] if isinstance(v,np.ndarray) else v) for k,v in controls.items()}
    result=audit_control_alignment(ref,src,controls)
    assert result['per_group']['roi']['accepted']>0
    assert result['proposal_translation_xy'] is None
    assert result['status']=='NOT_EVALUABLE'


@pytest.mark.parametrize('mode',['uniform','missing_depth','unknown_source','invalid_mask'])
def test_insufficient_evidence_rejects(mode):
    ref,src,roi=scene(uniform=mode=='uniform')
    if mode=='missing_depth': ref['depth'][:]=0
    if mode=='unknown_source': src['validity'][:]=False
    if mode=='invalid_mask': src['mask'][:]=False
    controls=select_reference_controls(ref,roi)
    result=audit_control_alignment(ref,src,controls)
    assert not result['accepted'].any()
    assert result['proposal_translation_xy'] is None
    assert result['per_group']['body_test']['median_before'] is None


def test_reference_invalidity_and_malformed_mask_rejected():
    ref,src,roi=scene()
    ref['validity'][:]=False
    assert not len(select_reference_controls(ref,roi)['control_xy'])
    src['mask']=np.ones((192,192),np.uint8)
    with pytest.raises(ValueError,match='invalid view mask'):
        audit_control_alignment(ref,src,select_reference_controls(ref,roi))


def test_changed_depth_at_existing_control_is_unknown():
    ref,src,roi=scene();controls=select_reference_controls(ref,roi)
    ref['depth'][:]=0
    result=audit_control_alignment(ref,src,controls)
    assert not result['prediction_positive'].any()
    assert not result['accepted'].any()


def test_tile_margin_roi_footprints_and_rejection_counts():
    ref,src,roi=scene();controls=select_reference_controls(ref,roi)
    xy=controls['control_xy'].astype(int)
    body=controls['group']==0
    expanded=cv2.dilate(roi.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)
    assert not expanded[xy[body,1],xy[body,0]].any()
    if body.any() and (~body).any():
        distance=np.max(np.abs(xy[body,None,:]-xy[None,~body,:]),axis=2)
        assert (distance>6).all()
    for px,py in xy[body]:
        assert 4 <= px%24 < 20 and 4 <= py%24 < 20
    # Force a subpixel-shifted tracked corner footprint to be unknown.
    src['validity'][20:80,20:80]=False
    result=audit_control_alignment(ref,src,controls)
    assert result['gate_not_passed']['known_foreground_patch'] > 0
    assert sum(result['per_group'][name]['attempted'] for name in ('body_train','body_test','roi')) == len(xy)
    assert result['gate_count_semantics'].startswith('Overlapping')
    for name in ('body_train','body_test','roi'):
        row=result['per_group'][name]
        assert row['attempted'] == row['accepted']+row['rejected']
