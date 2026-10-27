"""Stub LK contracts only; no native, GPU or actual optical flow."""
from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.seed_pair_validation import validate_seed_pairs,verify_seed_certificate


class StubCV:
    TERM_CRITERIA_COUNT=1;TERM_CRITERIA_EPS=2
    def __init__(self,mutation=None):self.calls=[];self.mutation=mutation
    def calcOpticalFlowPyrLK(self,a,b,p,n,**kw):
        self.calls.append((a.copy(),b.copy(),p.copy(),n,kw))
        i=len(self.calls);q=np.array([[[11.,10.]]],np.float32) if i%2 else np.array([[[10.,10.]]],np.float32)
        status=np.ones((1,1),np.uint8);error=np.array([[2.]],np.float32)
        return self.mutation(i,q,status,error) if self.mutation else (q,status,error)


def fixture():
    frames=[dict(input_index=i,frame_id=f'f{i}',original_mask_status='OK',width=32,height=32,path=f'fixture://image{i}',sha256='a'*64,mask_binding=dict(path=f'fixture://mask{i}',sha256='b'*64)) for i in (11,25,60)]
    grays=[np.full((32,32),i,np.uint8) for i in (10,20,30)];masks=[np.full((32,32),255,np.uint8) for _ in frames]
    observations=[];cameras={};admitted=[];rows=[]
    for pos,f in enumerate(frames):
        xy=[10.5+pos,10.5]
        obs=dict(frame_index=f['input_index'],sequence_position=pos,identity_segment=0,xy=xy,
            label='foreground',target_verified=True,strong=pos!=0,fb_error_px=.1 if pos else None,
            lk_error=1. if pos else None,tracking_status='tracked' if pos else 'seed')
        observations.append(obs)
        c=dict(input_index=f['input_index'],frame_id=f['frame_id'],image_id=pos+1,camera_model='SIMPLE_RADIAL',
            params=[32.,16.,16.,0.],R=np.eye(3).tolist(),t=[float(pos),0.,0.],width=32,height=32)
        cameras[f['input_index']]=c
        rows.append(dict(input_index=f['input_index'],frame_id=f['frame_id'],track_id=7,identity_segment=0,
            source_xy=xy.copy(),status='ADMITTED_ORIGINAL_ANCHOR' if pos else 'EXCLUDED',reasons=[] if pos else ['NON_STRONG']))
        if pos:admitted.append(dict(c,raw_pixels=xy.copy()))
    group=dict(group_id='0:7:0',model_key='0',track_id=7,identity_segment=0,observations=rows,admitted_observations=admitted)
    return frames,grays,masks,dict(tracks=[dict(track_id=7,observations=observations)]),dict(groups=[group]),dict(models=[dict(model_key='0',cameras=cameras)],unsupported_models=[])


def run(data,cv=None,budget=lambda:None):
    return validate_seed_pairs(*data[:5],cv or StubCV(),budget,camera_result=data[5])


def test_success_exact_args_noncontiguous_ids_no_mutation_and_certificate():
    data=fixture();cv=StubCV();frozen=deepcopy((data[0],data[3],data[4],data[5]));arrays=[a.copy() for a in data[1]+data[2]]
    got=run(data,cv);assert got['flow_calls']==2 and got['counts']['verified']==1
    first,second=cv.calls
    np.testing.assert_array_equal(first[2],np.array([[[10.,10.]]],np.float32))
    np.testing.assert_array_equal(second[2],np.array([[[11.,10.]]],np.float32))
    assert first[2].dtype==np.float32 and first[3] is None
    assert first[4]==dict(winSize=(21,21),maxLevel=3,criteria=(3,30,.01))
    np.testing.assert_array_equal(first[0],data[1][0]);np.testing.assert_array_equal(second[0],data[1][1])
    c=got['certificates'][0]
    assert c['source_input_index']==11 and c['successor_input_index']==25
    assert c['source_image_id']==1 and c['successor_image_id']==2 and c['strong_anchor_image_ids']==[2,3]
    assert (data[0],data[3],data[4],data[5])==frozen
    for a,b in zip(arrays,data[1]+data[2]):np.testing.assert_array_equal(a,b)
    assert data[3]['tracks'][0]['observations'][0]['strong'] is False
    assert data[3]['tracks'][0]['observations'][0]['fb_error_px'] is None
    json.dumps(got,allow_nan=False)


@pytest.mark.parametrize('kind',['forward_status','backward_status','nan','bool_status','bool_error','next_mismatch','fb_mismatch','forward_error','backward_error','shape'])
def test_bad_lk_fails_closed(kind):
    def mutation(i,q,s,e):
        if kind=='forward_status' and i==1:s[:]=0
        elif kind=='backward_status' and i==2:s[:]=0
        elif kind=='nan' and i==1:q[:]=np.nan
        elif kind=='bool_status':s=s.astype(bool)
        elif kind=='bool_error':e=e.astype(bool)
        elif kind=='next_mismatch' and i==1:q[0,0,0]=13.
        elif kind=='fb_mismatch' and i==2:q[0,0,0]=12.
        elif kind=='forward_error' and i==1:e[:]=26.
        elif kind=='backward_error' and i==2:e[:]=26.
        elif kind=='shape':q=q.reshape(1,2)
        return q,s,e
    got=run(fixture(),StubCV(mutation))
    assert got['counts']['verified']==0 and got['rows'][0]['status']=='REJECTED'
    json.dumps(got,allow_nan=False)


@pytest.mark.parametrize('kind',['outside_mask','gray_dtype','gap','segment','not_seed','seed_nonnull','seed_strong','missing_camera','bad_camera'])
def test_invalid_sources_do_not_run_flow(kind):
    data=fixture();obs=data[3]['tracks'][0]['observations']
    if kind=='outside_mask':data[2][0][:]=0
    elif kind=='gray_dtype':data[1][0]=data[1][0].astype(float)
    elif kind=='gap':obs[1]['sequence_position']=2
    elif kind=='segment':obs[1]['identity_segment']=1
    elif kind=='not_seed':obs[0]['tracking_status']='tracked'
    elif kind=='seed_nonnull':obs[0]['fb_error_px']=0.
    elif kind=='seed_strong':obs[0]['strong']=True
    elif kind=='missing_camera':data[5]['models'][0]['cameras'].pop(11)
    elif kind=='bad_camera':data[5]['models'][0]['cameras'][11]['R'][0][0]=2.
    cv=StubCV();got=run(data,cv)
    assert got['flow_calls']==0 and cv.calls==[] and got['rows'][0]['status']=='NOT_ELIGIBLE'


def test_budget_before_each_flow_and_unexpected_exception_propagates():
    cv=StubCV()
    def budget():
        if len(cv.calls)==1:raise TimeoutError('budget')
    with pytest.raises(TimeoutError):run(fixture(),cv,budget)
    assert len(cv.calls)==1
    class Broken(StubCV):
        def calcOpticalFlowPyrLK(self,*a,**k):raise RuntimeError('unexpected')
    with pytest.raises(RuntimeError):run(fixture(),Broken())


def test_shared_pair_deduplicated_and_group_budget_fair():
    data=fixture();g=data[4]['groups'][0];camera=data[5]['models'][0]
    data[5]['models']=[dict(deepcopy(camera),model_key=str(i)) for i in range(98)]
    data[4]['groups']=[dict(deepcopy(g),model_key=str(i),group_id=f'{i}:7:0') for i in range(98)]
    got=run(data)
    assert len(got['selected_groups'])==96 and len(got['omitted_groups'])==2
    assert got['unique_pair_measurements']==1 and got['flow_calls']==2 and len(got['certificates'])==96
    assert got['selected_groups'][:3]==['0:7:0','1:7:0','10:7:0']


def test_certificate_recomputes_outputs_not_reported_result():
    c=run(fixture())['certificates'][0]
    kw=dict(group_id='0:7:0',track_id=7,identity_segment=0,source_frame_id='f11',successor_frame_id='f25',source_raw_xy=[10.5,10.5],successor_raw_xy=[11.5,10.5])
    c['recomputed']['fb_error_px']=999
    assert verify_seed_certificate(c,**kw)['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY'
    c['measurement']['backward_xy'][0][0][0]=20.
    assert verify_seed_certificate(c,**kw)['status']=='REJECTED'


def test_pure_helper_validates_unique_certificates_and_rejects_failed_rows():
    from integrations.demo_pipeline.seed_pair_validation import validated_certificates,validate_certificate
    ledger=run(fixture())
    assert list(validated_certificates(ledger))==['0:7:0']
    cert=ledger['certificates'][0]
    assert validate_certificate(cert)['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY'
    cert['source_image_id']=2
    assert validate_certificate(cert)['status']=='REJECTED'
    ledger=run(fixture());ledger['groups'][0]['status']='REJECTED'
    with pytest.raises(ValueError):validated_certificates(ledger)


@pytest.mark.parametrize('kind',['mixed_bool_xy','string_xy','int_xy','mixed_bool_error','bool_anchor_id','mask_coordinate','mask_flag','binding_missing','binding_hash','original_seed'])
def test_certificate_leaf_binding_and_mask_check_tampering(kind):
    from integrations.demo_pipeline.seed_pair_validation import validate_certificate
    c=run(fixture())['certificates'][0]
    if kind=='mixed_bool_xy':c['measurement']['forward_xy'][0][0][0]=True
    elif kind=='string_xy':c['measurement']['forward_xy'][0][0][0]='11.0'
    elif kind=='int_xy':c['measurement']['forward_xy'][0][0][0]=11
    elif kind=='mixed_bool_error':c['measurement']['forward_error'][0][0]=False
    elif kind=='bool_anchor_id':c['strong_anchor_image_ids'][0]=True
    elif kind=='mask_coordinate':c['measurement']['mask_checks']['forward_measured_raw_xy'][0]+=1.
    elif kind=='mask_flag':c['measurement']['mask_checks']['forward_inside_successor']=False
    elif kind=='binding_missing':c.pop('source_image_binding')
    elif kind=='binding_hash':c['source_mask_binding']['sha256']='bad'
    else:c['original_seed_state']['strong']=True
    assert validate_certificate(c)['status']=='REJECTED'


@pytest.mark.parametrize('kind',['header','rights','aliases','duplicate_selected','omitted','counts','certificates','row_certificate','budget_promotion'])
def test_strict_ledger_partition_rejects_tampering(kind):
    from integrations.demo_pipeline.seed_pair_validation import validated_certificates
    ledger=run(fixture())
    if kind=='header':ledger['status']='fake'
    elif kind=='rights':ledger['identity_verified']=True
    elif kind=='aliases':ledger['groups']=[]
    elif kind=='duplicate_selected':ledger['selected_groups']*=2
    elif kind=='omitted':ledger['omitted_groups']=['0:7:0']
    elif kind=='counts':ledger['counts']['eligible']=2
    elif kind=='certificates':ledger['certificates']*=2
    elif kind=='row_certificate':ledger['certificates']=[deepcopy(ledger['certificates'][0])];ledger['certificates'][0]['source_frame_id']='changed'
    else:
        ledger['rows'][0]['status']='NOT_EVALUATED_GROUP_BUDGET'
        ledger['selected_groups']=[];ledger['omitted_groups']=['0:7:0'];ledger['certificates']=[]
        ledger['counts'].update(verified=0,statuses={'NOT_EVALUATED_GROUP_BUDGET':1})
    with pytest.raises(ValueError):validated_certificates(ledger)


def many_tracks(n):
    data=fixture();template_track=data[3]['tracks'][0];template_group=data[4]['groups'][0]
    data[3]['tracks']=[];data[4]['groups']=[]
    for tid in range(n):
        track=deepcopy(template_track);track['track_id']=tid;data[3]['tracks'].append(track)
        group=deepcopy(template_group);group.update(track_id=tid,group_id=f'0:{tid}:0')
        for row in group['observations']:row['track_id']=tid
        data[4]['groups'].append(group)
    return data


def run_progress(data,cv,events,budget=lambda:None):
    return validate_seed_pairs(*data[:5],cv,budget,camera_result=data[5],progress=lambda event:events.append(event))


def test_three_sequential_batches_and_remainder_exact_accounting():
    from integrations.demo_pipeline.seed_pair_validation import validated_certificates
    events=[];cv=StubCV();ledger=run_progress(many_tracks(70),cv,events)
    assert [len(b['group_ids']) for b in ledger['batches']]==[32,32,6]
    assert [b['flow_calls'] for b in ledger['batches']]==[64,64,12]
    assert ledger['flow_calls']==140 and len(cv.calls)==140
    assert [e['kind'] for e in events]==['PLAN','BATCH_BEGIN','BATCH_END','BATCH_BEGIN','BATCH_END','BATCH_BEGIN','BATCH_END']
    assert all(b['status']=='NOT_STARTED' for b in events[0]['batches'])
    assert all(b['status']=='COMPLETE' for b in ledger['batches'])
    assert len(events[2]['rows'])==len(events[2]['certificates'])==32
    assert len(validated_certificates(ledger))==70


def test_global96_ceiling_and_failed_selected_never_replaced():
    def mutation(i,q,s,e):
        if i==1:s[:]=0
        return q,s,e
    events=[];ledger=run_progress(many_tracks(98),StubCV(mutation),events)
    assert len(ledger['selected_groups'])==96 and len(ledger['omitted_groups'])==2
    assert ledger['rows'][0]['status']=='REJECTED'
    assert ledger['selected_groups']==[f'0:{i}:0' for i in range(96)]
    assert ledger['omitted_groups']==['0:96:0','0:97:0']
    assert len(ledger['batches'])==3


def test_timeout_after_completed_first_batch_preserves_events_without_future_flows():
    events=[];stop=[False];cv=StubCV()
    def progress(event):
        events.append(event)
        if event['kind']=='BATCH_END' and event['batch_id']==0:stop[0]=True
    def budget():
        if stop[0]:raise TimeoutError('shared deadline')
    data=many_tracks(70)
    with pytest.raises(TimeoutError):validate_seed_pairs(*data[:5],cv,budget,camera_result=data[5],progress=progress)
    assert len(cv.calls)==64
    assert [e['kind'] for e in events]==['PLAN','BATCH_BEGIN','BATCH_END','BATCH_BEGIN','BATCH_FAILED']
    assert events[-1]['batch_id']==1 and events[-1]['failure_type']=='TimeoutError' and events[-1]['flow_calls']==0
    assert len(events[2]['certificates'])==32 and events[2]['status']=='COMPLETE'
    assert not any(e.get('batch_id')==2 for e in events[1:])


def test_cache_across_batches_and_strict_completion_validation():
    from integrations.demo_pipeline.seed_pair_validation import validated_certificates
    data=fixture();group=data[4]['groups'][0];camera=data[5]['models'][0]
    data[5]['models']=[dict(deepcopy(camera),model_key=str(i)) for i in range(70)]
    data[4]['groups']=[dict(deepcopy(group),model_key=str(i),group_id=f'{i}:7:0') for i in range(70)]
    events=[];ledger=run_progress(data,StubCV(),events)
    assert [b['flow_calls'] for b in ledger['batches']]==[2,0,0]
    assert ledger['unique_pair_measurements']==1 and len(validated_certificates(ledger))==70
    for tamper in ('batch_ids','state','flow','partition','row_flow','global_flow'):
        changed=deepcopy(ledger)
        if tamper=='batch_ids':changed['batches'][1]['batch_id']=0
        elif tamper=='state':changed['batches'][1]['status']='NOT_STARTED'
        elif tamper=='flow':changed['batches'][1]['flow_calls']=2
        elif tamper=='partition':changed['batches'][1]['group_ids']=changed['batches'][0]['group_ids']
        elif tamper=='row_flow':changed['rows'][33]['flow_calls']=2
        else:changed['flow_calls']=3
        with pytest.raises(ValueError):validated_certificates(changed)


@pytest.mark.parametrize('mode',['budget_before_reverse','cv_reverse_error'])
def test_interrupted_forward_measurement_preserved_in_failed_event(mode):
    events=[]
    class ReverseError(StubCV):
        def calcOpticalFlowPyrLK(self,*args,**kwargs):
            if len(self.calls)==1:
                self.calls.append(('reverse_attempt',))
                raise ValueError('native reverse failed unexpectedly')
            return super().calcOpticalFlowPyrLK(*args,**kwargs)
    cv=ReverseError() if mode=='cv_reverse_error' else StubCV()
    def budget():
        if mode=='budget_before_reverse' and len(cv.calls)==1:raise TimeoutError('deadline after forward')
    data=many_tracks(70)
    expected=ValueError if mode=='cv_reverse_error' else TimeoutError
    with pytest.raises(expected):run_progress(data,cv,events,budget)
    assert [e['kind'] for e in events]==['PLAN','BATCH_BEGIN','BATCH_FAILED']
    failed=events[-1];row=failed['failed_group_result']
    assert failed['failed_group_id']=='0:0:0' and row['status']=='NOT_EVALUATED_INTERRUPTED'
    assert row['certificate'] is None and row['partial_measurement']['forward_xy']==[[[11.,10.]]]
    assert row['partial_measurement']['forward_status']==[[1]] and row['partial_measurement']['forward_error']==[[2.]]
    assert 'backward_xy' not in row['partial_measurement']
    assert failed['verified']==failed['rejected']==0 and failed['rows']==[] and failed['certificates']==[]
    assert failed['unique_completed_pair_measurements']==0
    assert failed['flow_calls']==row['flow_calls']==(2 if mode=='cv_reverse_error' else 1)
    assert len(cv.calls)==(2 if mode=='cv_reverse_error' else 1)


def test_failure_callback_does_not_replace_first_exception():
    observed=[];cause=RuntimeError('original optical flow error')
    class Broken(StubCV):
        def calcOpticalFlowPyrLK(self,*args,**kwargs):raise cause
    def progress(event):
        observed.append(event)
        if event['kind']=='BATCH_FAILED':raise OSError('recorder unavailable')
    data=fixture()
    with pytest.raises(RuntimeError) as caught:validate_seed_pairs(*data[:5],Broken(),camera_result=data[5],progress=progress)
    assert caught.value is cause
    assert observed[-1]['failed_group_result']['status']=='NOT_EVALUATED_INTERRUPTED'
    assert observed[-1]['failed_group_result']['flow_calls']==1
    assert any('progress callback' in note for note in getattr(caught.value,'__notes__',[]))
