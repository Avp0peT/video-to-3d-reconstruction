"""Measured seed admission, fixed-camera fitting and independent candidate gate."""
from copy import deepcopy
import numpy as np
import pytest
from test_original_anchor_review import scene
from integrations.demo_pipeline import original_anchor_review as original
from integrations.demo_pipeline import seed_pair_validation as seed
from integrations.demo_pipeline.anchor_candidate_gate import classify_original_candidates


def inputs():
    frames,masks,tracking,cameras,truth=scene(3)
    for f in frames:
        f.update(path=f'synthetic://image/{f["input_index"]}',sha256='b'*64)
    observations=tracking['tracks'][0]['observations']
    for o in observations:o['tracking_status']='flow_tracked'
    observations[0].update(tracking_status='seed',strong=False,fb_error_px=None,lk_error=None)
    grays=[np.full(m.shape,i,np.uint8) for i,m in enumerate(masks)]
    class MeasuredCV:
        TERM_CRITERIA_COUNT=1;TERM_CRITERIA_EPS=2
        def __init__(self):self.calls=0
        def calcOpticalFlowPyrLK(self,a,b,p,n,**kw):
            self.calls+=1
            point=observations[1 if self.calls%2 else 0]['xy']
            return (np.asarray(point,np.float32)-.5).reshape(1,1,2),np.ones((1,1),np.uint8),np.array([[1.]],np.float32)
    return frames,masks,tracking,cameras,truth,grays,MeasuredCV()


def pipeline(data):
    f,m,t,c,truth,g,cv=data
    admission=original.review_original_anchors(f,m,t,c,evaluate=False)
    ledger=seed.validate_seed_pairs(f,g,m,t,admission,cv,camera_result=c)
    assert ledger['counts']['verified']==1
    return ledger,original.review_original_anchors(f,m,t,c,seed_validation=ledger)


def test_actual_pair_evidence_adds_third_view_without_relabelling_or_changing_thresholds():
    data=inputs();before=deepcopy(data[2]);ledger,diagnostics=pipeline(data)
    assert data[2]==before and data[-1].calls==2
    group=diagnostics['groups'][0]
    assert len(group['admitted_observations'])==3
    assert group['observations'][0]['original_tracking']['strong'] is False
    assert group['observations'][0]['original_tracking']['fb_error_px'] is None
    assert group['admitted_observations'][0]['evidence_kind']=='VERIFIED_SEED_PAIR'
    np.testing.assert_allclose(group['fit']['xyz'],data[4],atol=1e-5)
    assert len(group['leave_one_view_out'])==3
    assert all('not independent validation' in f['scope'] for f in group['leave_one_view_out'])
    classified=classify_original_candidates(diagnostics)
    assert classified['rows'][0]['tier']=='STRICT_CONDITIONAL'
    assert classified['rows'][0]['sources'][0]['evidence_kind']=='VERIFIED_SEED_PAIR'
    assert classified['geometry_accepted'] is False


def test_admission_discovery_does_not_fit_any_point(monkeypatch):
    data=inputs()
    def forbidden(*a,**k):raise AssertionError('no numerical fitting during admission')
    monkeypatch.setattr(original.fixed,'fit',forbidden)
    result=original.review_original_anchors(*data[:4],evaluate=False)
    assert result['status']=='ORIGINAL_ANCHOR_ADMISSION_ONLY' and result['numeric_calls']==0
    assert len(result['groups'][0]['admitted_observations'])==2


@pytest.mark.parametrize('kind',['image_hash','mask_hash','camera','raw_xy','seed_state','successor_state','endpoint_mask'])
def test_adapter_rebinds_certificate_to_real_inputs(kind):
    data=inputs();f,m,t,c,_,g,cv=data
    admission=original.review_original_anchors(f,m,t,c,evaluate=False)
    ledger=seed.validate_seed_pairs(f,g,m,t,admission,cv,camera_result=c)
    if kind=='image_hash':f[0]['sha256']='c'*64
    elif kind=='mask_hash':f[0]['mask_binding']['sha256']='c'*64
    elif kind=='camera':c['unsupported_models'][0]['unsupported_cameras'][11]['t'][0]+=.01
    elif kind=='raw_xy':t['tracks'][0]['observations'][0]['xy'][0]+=.01
    elif kind=='seed_state':t['tracks'][0]['observations'][0]['strong']=True
    elif kind=='successor_state':t['tracks'][0]['observations'][1]['strong']=False
    else:m[0][:]=0
    with pytest.raises((ValueError,KeyError)):
        original.review_original_anchors(f,m,t,c,seed_validation=ledger)


@pytest.mark.parametrize('kind',['numeric_output','source_state','successor_state','file_hash','camera','identity','drop_certificate'])
def test_gate_does_not_trust_seed_label_or_saved_derived_errors(kind):
    ledger,d=pipeline(inputs());group=d['groups'][0]
    source=group['observations'][0];admitted=group['admitted_observations'][0]
    if kind=='numeric_output':
        for row in (source,admitted):row['seed_certificate']['measurement']['backward_xy'][0][0][0]+=4.
    elif kind=='source_state':source['original_tracking']['strong']=True
    elif kind=='successor_state':group['observations'][1]['original_tracking']['strong']=False
    elif kind=='file_hash':source['image_binding']['sha256']='c'*64
    elif kind=='camera':
        for row in (source,admitted):row['seed_certificate']['source_camera']['t'][0]+=.1
    elif kind=='identity':
        for row in (source,admitted):row['seed_certificate']['track_id']=99
    else:admitted.pop('seed_certificate')
    assert classify_original_candidates(d)['rows'][0]['tier']=='REJECTED'


def test_three_view_priority_uses_no_residual_selection_and_gate_agrees():
    frames,masks,tracking,cameras,_=scene(3,tracks=34)
    # Only the last track sees all three images; its larger ID must not hide it.
    for track in tracking['tracks'][:-1]:track['observations']=track['observations'][:2]
    got=original.review_original_anchors(frames,masks,tracking,cameras)
    assert got['selected_groups'][0]=='0:33:0'
    assert got['selected_groups'][1:]==[f'0:{i}:0' for i in range(33)]
    assert got['counts']['selected_multiview_groups']==1
    result=classify_original_candidates(got)
    assert result['selected_groups']==got['selected_groups']
    tampered=deepcopy(got)
    tampered['selected_groups'][0],tampered['selected_groups'][1]=tampered['selected_groups'][1],tampered['selected_groups'][0]
    with pytest.raises(ValueError,match='PARTITION'):classify_original_candidates(tampered)


def test_failed_pair_never_upgrades_the_seed_or_fills_it_with_a_prediction():
    data=inputs();f,m,t,c,_,grays,cv=data
    original_call=cv.calcOpticalFlowPyrLK
    def failed(*args,**kw):
        p,s,e=original_call(*args,**kw);e[:]=26.
        return p,s,e
    cv.calcOpticalFlowPyrLK=failed
    admission=original.review_original_anchors(f,m,t,c,evaluate=False)
    ledger=seed.validate_seed_pairs(f,grays,m,t,admission,cv,camera_result=c)
    assert ledger['counts']['verified']==0
    d=original.review_original_anchors(f,m,t,c,seed_validation=ledger)
    assert len(d['groups'][0]['admitted_observations'])==2
    assert d['groups'][0]['observations'][0]['status']=='EXCLUDED'
    assert d['counts']['verified_seed_observations']==0
