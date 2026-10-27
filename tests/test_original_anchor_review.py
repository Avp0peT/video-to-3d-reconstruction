"""Synthetic numerical diagnostics only; no file/model/flow/native I/O."""
from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.original_anchor_review import review_original_anchors


def scene(n=4,k=0.,tracks=1,models=1):
    xyz=np.array([.1,.05,5.]);frames=[];masks=[];cameras={};observations=[]
    for i in range(n):
        index=11+i*7;f=dict(input_index=index,frame_id=f'f{index}',original_mask_status='OK',width=100,height=100,mask_binding={'path':'synthetic://original-mask','sha256':'a'*64})
        frames.append(f);masks.append(np.full((100,100),255,np.uint8))
        t=np.array([-.5+i*.1,0.,0.]);uv=(xyz+t)[:2]/(xyz+t)[2]
        xy=(100*uv*(1+k*np.dot(uv,uv))+[50,50]).tolist()
        cameras[index]=dict(input_index=index,frame_id=f['frame_id'],image_id=i+1,width=100,height=100,
            camera_model='SIMPLE_RADIAL',params=[100.,50.,50.,k],R=np.eye(3).tolist(),t=t.tolist())
        observations.append(dict(frame_index=index,sequence_position=i,identity_segment=0,xy=xy,
            label='foreground',target_verified=True,strong=True,fb_error_px=.1,lk_error=1.))
    tracking=dict(tracks=[dict(track_id=j,observations=deepcopy(observations)) for j in range(tracks)])
    camera_result=dict(models=[],unsupported_models=[dict(model_key=str(j),cameras={},unsupported_cameras=deepcopy(cameras)) for j in range(models)])
    return frames,masks,tracking,camera_result,xyz


@pytest.mark.parametrize('n,k',[(2,0.),(3,0.),(4,0.),(4,.2),(4,-.2)])
def test_truth_radial_and_every_leave_one_out(n,k):
    frames,masks,tracking,cameras,xyz=scene(n,k)
    frozen=deepcopy((frames,tracking,cameras));old_masks=[m.copy() for m in masks]
    result=review_original_anchors(frames,masks,tracking,cameras)
    group=result['groups'][0]
    assert group['status']=='CONDITIONAL_DIAGNOSTIC_EVALUATED'
    np.testing.assert_allclose(group['fit']['xyz'],xyz,atol=1e-6)
    assert len(group['leave_one_view_out'])==(n if n>=3 else 0)
    for fold in group['leave_one_view_out']:
        assert fold['heldout_input_index'] not in fold['training_input_indices']
        assert len(fold['training_input_indices'])==n-1
        assert fold['prediction']['max']<1e-5
        np.testing.assert_allclose(fold['predicted_raw_pixels'],fold['measured_raw_pixels'],atol=1e-5)
    assert [o['input_index'] for o in group['admitted_observations']]==[11+i*7 for i in range(n)]
    assert all(group[key] is False and result[key] is False for key in ('geometry_acceptance','native_injected','identity_verified'))
    assert group['admitted_observations'][0]['params'][3]==k
    assert (frames,tracking,cameras)==frozen
    for a,b in zip(masks,old_masks):np.testing.assert_array_equal(a,b)
    json.dumps(result,allow_nan=False)


def test_disappearance_identity_segments_never_merge():
    frames,masks,tracking,cameras,_=scene(4)
    for obs in tracking['tracks'][0]['observations'][2:]:obs['identity_segment']=1
    result=review_original_anchors(frames,masks,tracking,cameras)
    assert len(result['groups'])==2
    assert [len(g['admitted_observations']) for g in result['groups']]==[2,2]
    assert all(not g['leave_one_view_out'] for g in result['groups'])


def test_identity_segment_cannot_return_to_old_identity():
    frames,masks,tracking,cameras,_=scene(3)
    tracking['tracks'][0]['observations'][1]['identity_segment']=1
    with pytest.raises(ValueError,match='nonregressing'):
        review_original_anchors(frames,masks,tracking,cameras)


def test_boolean_camera_rejected_before_admission():
    frames,masks,tracking,cameras,_=scene(3)
    cameras['unsupported_models'][0]['unsupported_cameras'][11]['params'][0]=True
    result=review_original_anchors(frames,masks,tracking,cameras)
    assert result['groups'][0]['observations'][0]['status']=='EXCLUDED'
    assert len(result['groups'][0]['admitted_observations'])==2


def test_refused_training_fold_is_not_called_evaluated(monkeypatch):
    from integrations.demo_pipeline import original_anchor_review as module
    frames,masks,tracking,cameras,_=scene(3)
    real_fit=module.fixed.fit
    def fit(rows,**kwargs):
        if len(rows)==2:return dict(status='REFUSED',xyz=None,reason='TEST_DATA training refusal')
        return real_fit(rows,**kwargs)
    monkeypatch.setattr(module.fixed,'fit',fit)
    group=review_original_anchors(frames,masks,tracking,cameras)['groups'][0]
    assert group['status']=='CONDITIONAL_DIAGNOSTIC_EVALUATED'
    assert group['loo_status']=='ALL_VIEWS_REFUSED'
    assert len(group['leave_one_view_out'])==3
    assert all(f['prediction'] is None and f['fit']['status']=='REFUSED' for f in group['leave_one_view_out'])


@pytest.mark.parametrize('kind,reason',[('unknown','UNKNOWN_SOURCE'),('weak','NON_STRONG'),
    ('outside','OUTSIDE_OR_UNAVAILABLE_ORIGINAL_MASK'),('fb','INVALID_FB_ERROR_PX'),
    ('bad_camera','CAMERA_FRAME_IDENTITY_MISMATCH'),('bad_rotation','INVALID_ROTATION')])
def test_reject_observation_without_relabeling(kind,reason):
    frames,masks,tracking,cameras,_=scene(3)
    observation=tracking['tracks'][0]['observations'][1]
    if kind=='unknown':observation['label']='unknown'
    elif kind=='weak':observation['strong']=False
    elif kind=='outside':masks[1][:]=0
    elif kind=='fb':observation['fb_error_px']=2.
    elif kind=='bad_camera':cameras['unsupported_models'][0]['unsupported_cameras'][18]['frame_id']='other'
    else:cameras['unsupported_models'][0]['unsupported_cameras'][18]['R'][0][0]=2.
    original=deepcopy(tracking)
    result=review_original_anchors(frames,masks,tracking,cameras)
    group=result['groups'][0]
    assert reason in group['observations'][1]['reasons']
    assert len(group['admitted_observations'])==2 and tracking==original


def test_complete_over_budget_observations_kept():
    frames,masks,tracking,cameras,_=scene(13)
    result=review_original_anchors(frames,masks,tracking,cameras)
    group=result['groups'][0]
    assert group['status']=='NOT_EVALUATED_OBSERVATION_BUDGET'
    assert len(group['admitted_observations'])==len(group['observations'])==13
    assert result['selected_groups']==[] and group['fit'] is None


def test_round_robin_models_and_all_omissions_explicit():
    frames,masks,tracking,cameras,_=scene(2,tracks=20,models=2)
    result=review_original_anchors(frames,masks,tracking,cameras)
    assert len(result['groups'])==40 and len(result['selected_groups'])==40 and len(result['omitted_groups'])==0
    assert result['selected_groups'][:4]==['0:0:0','1:0:0','0:1:0','1:1:0']
    assert sum(s.startswith('0:') for s in result['selected_groups'])==20
    assert [len(b['group_ids']) for b in result['batches']]==[32,8]


def test_timeout_is_not_swallowed():
    frames,masks,tracking,cameras,_=scene()
    def budget():raise TimeoutError('budget')
    with pytest.raises(TimeoutError):review_original_anchors(frames,masks,tracking,cameras,budget)


def test_no_saved_cameras_has_explicit_zero_work():
    frames,masks,tracking,_,_=scene()
    result=review_original_anchors(frames,masks,tracking,dict(models=[],unsupported_models=[]))
    assert result['camera_count']==0 and result['groups']==[] and result['counts']['selected_groups']==0


def test_legacy_refusal_separate_from_central_hypothesis():
    frames,masks,tracking,cameras,_=scene(3,k=-.2)
    result=review_original_anchors(frames,masks,tracking,cameras)
    group=result['groups'][0]
    assert all(r['status']=='RADIAL_DOMAIN_REFUSED' for r in group['strict_radial_domain_check'])
    assert group['status']=='CONDITIONAL_DIAGNOSTIC_EVALUATED'
    assert group['geometry_acceptance'] is False


def test_original_mask_binding_required_without_file_reads():
    frames,masks,tracking,cameras,_=scene(3)
    frames[1].pop('mask_binding')
    result=review_original_anchors(frames,masks,tracking,cameras)
    assert 'NO_ORIGINAL_MASK_BINDING' in result['groups'][0]['observations'][1]['reasons']
    assert len(result['groups'][0]['admitted_observations'])==2


def test_three_batches_frozen_plan_progress_and_96_global_limit(monkeypatch):
    from integrations.demo_pipeline import original_anchor_review as module
    data=scene(2,tracks=50,models=2);events=[];calls=[]
    def numeric(group,budget):
        assert events[0]['kind']=='PLAN'
        calls.append(group['group_id']);group['status']='NUMERIC_REFUSED'
    monkeypatch.setattr(module,'_numeric',numeric)
    result=module.review_original_anchors(*data[:4],progress=events.append)
    assert len(calls)==len(result['selected_groups'])==96
    assert [len(b['group_ids']) for b in result['batches']]==[32,32,32]
    assert result['counts']['batches_completed']==3 and len(result['omitted_groups'])==4
    assert [e['kind'] for e in events]==['PLAN','BATCH_BEGIN','BATCH_END','BATCH_BEGIN','BATCH_END','BATCH_BEGIN','BATCH_END']
    assert all(b['status']=='NOT_STARTED' for b in events[0]['batches'])
    assert [i for b in events[0]['batches'] for i in b['group_ids']]==calls
    assert all(e['status']=='COMPLETE' and len(e['group_results'])==e['evaluated_groups']==32 for e in events if e['kind']=='BATCH_END')
    assert sum(i.startswith('0:') for i in calls)==sum(i.startswith('1:') for i in calls)==48


def test_shared_timeout_stops_next_batch_and_retains_completed_records(monkeypatch):
    from integrations.demo_pipeline import original_anchor_review as module
    data=scene(2,tracks=40,models=2);events=[];calls=[];stop=[False]
    def progress(event):
        events.append(event)
        if event['kind']=='BATCH_BEGIN' and event['batch_id']==1:stop[0]=True
    def budget():
        if stop[0]:raise TimeoutError('one shared deadline')
    def numeric(group,budget):calls.append(group['group_id']);group['status']='NUMERIC_REFUSED'
    monkeypatch.setattr(module,'_numeric',numeric)
    with pytest.raises(TimeoutError,match='shared deadline'):
        module.review_original_anchors(*data[:4],budget=budget,progress=progress)
    assert len(calls)==32
    assert events[-1]['kind']=='BATCH_FAILED' and events[-1]['batch_id']==1
    assert events[-1]['evaluated_groups']==0 and events[-1]['group_results']==[]
    assert sum(e['kind']=='BATCH_END' for e in events)==1
    assert not any(e.get('batch_id')==2 for e in events)


def test_progress_failure_propagates_without_numeric_calls(monkeypatch):
    from integrations.demo_pipeline import original_anchor_review as module
    events=[]
    def progress(event):
        events.append(event)
        if event['kind']=='BATCH_BEGIN':raise RuntimeError('checkpoint unavailable')
    monkeypatch.setattr(module,'_numeric',lambda *args:pytest.fail('no numeric call'))
    with pytest.raises(RuntimeError,match='checkpoint unavailable'):
        module.review_original_anchors(*scene(2)[:4],progress=progress)
    assert [e['kind'] for e in events]==['PLAN','BATCH_BEGIN','BATCH_FAILED']


def test_admission_only_has_no_numeric_batches_or_progress(monkeypatch):
    from integrations.demo_pipeline import original_anchor_review as module
    events=[]
    monkeypatch.setattr(module,'_numeric',lambda *args:pytest.fail('no numeric call'))
    result=module.review_original_anchors(*scene(2)[:4],evaluate=False,progress=events.append)
    assert result['numeric_calls']==0 and events==[] and 'batches' not in result


def test_multiview_priority_then_model_round_robin_preserved(monkeypatch):
    from integrations.demo_pipeline import original_anchor_review as module
    data=scene(3,tracks=4,models=2)
    for track in data[2]['tracks']:
        if track['track_id']%2==0:track['observations']=track['observations'][:2]
    calls=[]
    def numeric(group,budget):calls.append(group['group_id']);group['status']='NUMERIC_REFUSED'
    monkeypatch.setattr(module,'_numeric',numeric)
    got=module.review_original_anchors(*data[:4])
    assert calls==['0:1:0','1:1:0','0:3:0','1:3:0','0:0:0','1:0:0','0:2:0','1:2:0']
    assert got['counts']['selected_multiview_groups']==4


@pytest.mark.parametrize('error_type',[TimeoutError,RuntimeError])
def test_mid_batch_exception_reports_partial_group_without_running_next(monkeypatch,error_type):
    from integrations.demo_pipeline import original_anchor_review as module
    events=[];calls=[]
    def numeric(group,budget):
        calls.append(group['group_id'])
        if len(calls)==2:
            group['fit']={'status':'FITTED','xyz':[1.,2.,3.]}
            raise error_type('during group')
        group['status']='NUMERIC_REFUSED'
    monkeypatch.setattr(module,'_numeric',numeric)
    with pytest.raises(error_type,match='during group'):
        module.review_original_anchors(*scene(2,tracks=4)[:4],progress=events.append)
    assert len(calls)==2 and events[-1]['kind']=='BATCH_FAILED'
    assert events[-1]['evaluated_groups']==1 and len(events[-1]['group_results'])==1
    assert events[-1]['failed_group_result']['status']=='NOT_EVALUATED_INTERRUPTED'
    assert events[-1]['failed_group_result']['fit']['xyz']==[1.,2.,3.]
    assert not any(e['kind']=='BATCH_END' for e in events)


def test_later_fold_valueerror_retains_prior_numeric_evidence_but_refuses_export(monkeypatch):
    from integrations.demo_pipeline import original_anchor_review as module
    from integrations.demo_pipeline.anchor_candidate_gate import classify_original_candidates
    events=[];calls=[]
    def numeric(group,budget):
        calls.append(group['group_id'])
        group.update(fit=dict(status='FITTED',xyz=np.array([1.,2.,3.]),converged=True),
            evaluation=dict(status='EVALUATED',max=.1),observability=dict(status='DIAGNOSTIC_EVALUATED'),
            leave_one_view_out=[dict(heldout_input_index=11,prediction=dict(status='EVALUATED',max=.2))],
            raw_predictions=[dict(input_index=11,predicted_raw_pixels=np.array([20.,21.]))],
            radial_domains=[dict(input_index=11,max_ideal_radius=1.)],
            strict_radial_domain_check=[dict(input_index=11,status='RADIAL_DOMAIN_REFUSED')],
            loo_status='ALL_VIEWS_EVALUATED')
        raise ValueError('later fold failed')
    monkeypatch.setattr(module,'_numeric',numeric)
    result=module.review_original_anchors(*scene(3)[:4],progress=events.append)
    group=result['groups'][0];partial=group['partial_numeric_evidence']
    assert group['status']=='NUMERIC_REFUSED' and group['fit'] is None
    assert group['evaluation'] is None and group['observability'] is None
    assert group['leave_one_view_out']==group['raw_predictions']==[]
    assert partial['fit']['xyz']==[1.,2.,3.] and len(partial['leave_one_view_out'])==1
    assert partial['evaluation']['max']==.1 and partial['observability']['status']=='DIAGNOSTIC_EVALUATED'
    assert partial['raw_predictions'][0]['predicted_raw_pixels']==[20.,21.]
    assert partial['radial_domains'] and partial['strict_radial_domain_check']
    assert events[-1]['kind']=='BATCH_END' and events[-1]['group_results'][0]['partial_numeric_evidence']==partial
    classified=classify_original_candidates(result)
    assert classified['counts']['exportable']==0 and classified['rows'][0]['tier']=='REJECTED'
    json.dumps(result,allow_nan=False)


def test_failed_callback_preserves_original_exception_and_notes_second_failure(monkeypatch):
    from integrations.demo_pipeline import original_anchor_review as module
    def numeric(group,budget):raise TimeoutError('original deadline')
    def progress(event):
        if event['kind']=='BATCH_FAILED':raise RuntimeError('checkpoint failed too')
    monkeypatch.setattr(module,'_numeric',numeric)
    with pytest.raises(TimeoutError,match='original deadline') as caught:
        module.review_original_anchors(*scene(2)[:4],progress=progress)
    assert any('checkpoint failed too' in note for note in caught.value.__notes__)
