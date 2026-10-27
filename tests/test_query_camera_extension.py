from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.query_camera_extension import assess_query_extensions


def fixture(tid=7,legs=2,lenses=('SIMPLE_PINHOLE','SIMPLE_RADIAL')):
    world=np.array([.1,.05,5.]);pid=f'track:{tid}:segment:2:source:27';rid=pid+':edge:0'
    frames=[dict(input_index=27,decode_index=10,frame_id='source',width=100,height=100,original_mask_status='OK')]
    cam=dict(input_index=27,image_id=501,frame_id='source',width=100,height=100,
        camera_model='SIMPLE_RADIAL',params=[100.,50.,50.,0.],R=np.eye(3).tolist(),t=[0.,0.,0.])
    cameras=dict(models=[dict(model_key='m',cameras={27:cam})]);obs=[];records=[]
    for i in range(legs+1):
        translation=np.array([.6*i,0.,0.]);local=world+translation
        raw=(100*local[:2]/local[2]+[50.,50.]).tolist()
        obs.append(dict(decode_index=10+i,raw_xy=raw,width=100,height=100,
                        original_mask_status='OK' if i==0 else 'UNKNOWN_NOT_SEGMENTED'))
        if not i:continue
        anchors=[dict(point_id=f'm:{100+j}:2',xyz=[j*.1,0.,5.],raw_xy=[50.+j,50.]) for j in range(9)]
        fits=[dict(model=lens,status='CONDITIONAL_CAMERA_CANDIDATE',f=100.,k=0.,
            K=[[100.,0.,50.],[0.,100.,50.],[0.,0.,1.]],R=np.eye(3).tolist(),t=translation.tolist()) for lens in lenses]
        records.append(dict(model_key='m',query_decode_index=10+i,cohort='STRICT',width=100,height=100,
            points=deepcopy(anchors),assessment=dict(points=deepcopy(anchors),train_ids=[a['point_id'] for j,a in enumerate(anchors) if j not in (3,7)],
                holdout_ids=[anchors[j]['point_id'] for j in (3,7)],models=fits)))
    point=dict(point_id=pid,track_id=tid,identity_segment=2,source_input_index=27,source_decode_index=10,
        source_raw_xy=deepcopy(obs[0]['raw_xy']),original_tracking=dict(strong=False,tracking_status='seed',fb_error_px=None,lk_error=None))
    track=dict(tracklet_id=rid,point_id=pid,edge_ordinal=0,track_id=tid,identity_segment=2,source_input_index=27,
               source_point=point,status='CANDIDATE_2D_TRACKLET',verified_legs=legs,observations=obs)
    return dict(records=[track]),frames,[np.full((100,100),255,np.uint8)],cameras,dict(records=records),world


@pytest.mark.parametrize('legs',[2,3])
def test_real_xyz_each_lens_all_view_loo_and_no_native_identity_reuse(legs):
    ledger,frames,masks,cameras,queries,world=fixture(legs=legs)
    before=deepcopy((ledger,frames,cameras,queries))
    result=assess_query_extensions(ledger,frames,masks,cameras,queries)
    assert result['counts']['conditional_points']==2 and result['numeric_groups']==2
    assert {c['lens_model'] for c in result['classifications']}=={'SIMPLE_PINHOLE','SIMPLE_RADIAL'}
    ids=[]
    for item in result['classifications']:
        row=item['classification']['rows'][0];ids.append(row['group_id'])
        assert row['tier']=='STRICT_CONDITIONAL' and row['source_point']['original_tracking']['strong'] is False
        np.testing.assert_allclose(row['xyz'],world,atol=1e-6)
        assert len(row['metrics']['loo'])==legs+1
        assert row['query_foreground_status']=='UNKNOWN_NOT_SEGMENTED' and row['geometry_accepted'] is False
    assert len(set(ids))==2
    for group in result['groups']:
        assert group['admitted_observations'][0]['image_id']==1
        assert group['admitted_observations'][0]['camera_ref']['image_id']==501
        assert all(f['heldout_input_index'] not in f['training_input_indices'] for f in group['leave_one_view_out'])
    assert (ledger,frames,cameras,queries)==before
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('tid',[100,103,107])
def test_every_training_and_holdout_camera_anchor_is_excluded(tid):
    args=fixture(tid=tid)
    result=assess_query_extensions(*args[:5])
    assert result['numeric_groups']==0 and result['counts']['conditional_points']==0
    assert all(g['status']=='EXCLUDED_CAMERA_FITTING_ANCHOR' for g in result['groups'])


@pytest.mark.parametrize('mutation,reason',[
    ('split','ANCHOR_SPLIT_COVERAGE'),('assessment','ASSESSMENT_POINT_BINDING'),
    ('identity','SOURCE_POINT_IDENTITY'),('source','SOURCE_COORDINATE_BINDING'),
    ('dimension','QUERY_OBSERVATION_DIMENSION_MISMATCH'),('K','FIXED_PRINCIPAL_POINT_OR_K_MISMATCH'),
    ('foreground','SOURCE_NOT_ORIGINAL_FOREGROUND'),('binary','ORIGINAL_SOURCE_MASK_REQUIRED')])
def test_identity_source_mask_and_camera_assumption_fail_closed(mutation,reason):
    ledger,frames,masks,cameras,query,_=fixture()
    if mutation=='split':query['records'][0]['assessment']['holdout_ids'].pop()
    if mutation=='assessment':query['records'][0]['assessment']['points'][0]['xyz'][0]=999
    if mutation=='identity':ledger['records'][0]['source_point']['track_id']=True
    if mutation=='source':ledger['records'][0]['observations'][0]['raw_xy'][0]+=.1
    if mutation=='dimension':ledger['records'][0]['observations'][1]['width']=101
    if mutation=='K':query['records'][0]['assessment']['models'][0]['K'][0][2]=51.
    if mutation=='foreground':masks[0][:]=0
    if mutation=='binary':masks[0][0,0]=1
    with pytest.raises(ValueError,match=reason):assess_query_extensions(ledger,frames,masks,cameras,query)


def test_rejected_query_camera_is_never_used_or_replaced():
    args=fixture()
    for fit in args[4]['records'][0]['assessment']['models']:fit['status']='REJECTED'
    result=assess_query_extensions(*args[:5])
    assert result['numeric_groups']==0 and all(g['status']=='BLOCKED_INSUFFICIENT_QUERY_CAMERAS' for g in result['groups'])


def test_models_remain_in_their_own_world_coordinates():
    ledger,frames,masks,cameras,query,world=fixture()
    other=deepcopy(cameras['models'][0]);other['model_key']='other'
    other['cameras'][27]['t'][0]-=1.
    cameras['models'].append(other)
    additional=deepcopy(query['records'])
    for record in additional:
        record['model_key']='other'
        for fit in record['assessment']['models']:fit['t'][0]-=1.
    query['records'].extend(additional)
    result=assess_query_extensions(ledger,frames,masks,cameras,query)
    assert len(result['classifications'])==4
    for item in result['classifications']:
        expected=world+([1.,0.,0.] if item['model_key']=='other' else [0.,0.,0.])
        row=item['classification']['rows'][0]
        np.testing.assert_allclose(row['xyz'],expected,atol=1e-6)
        assert all(s['camera_ref']['model_key']==item['model_key'] for s in row['sources'])


def test_ninety_six_groups_frozen_round_robin_no_refill(monkeypatch):
    import integrations.demo_pipeline.query_camera_extension as module
    args=fixture();tracks=[]
    for tid in range(200,298):tracks.extend(fixture(tid=tid)[0]['records'])
    args[0]['records']=tracks
    def refused(group,budget):budget();raise ValueError('forced numeric refusal')
    monkeypatch.setattr(module,'_numeric',refused)
    result=assess_query_extensions(*args[:5])
    assert result['numeric_groups']==96 and len(result['selection'])==96
    selected=[g for g in result['groups'] if g['group_id'] in result['selection']]
    assert {lens:sum(g['lens_hypothesis']==lens for g in selected) for lens in ('SIMPLE_PINHOLE','SIMPLE_RADIAL')}==dict(SIMPLE_PINHOLE=48,SIMPLE_RADIAL=48)
    assert sum(g['status']=='NOT_EVALUATED_GROUP_BUDGET' for g in result['groups'])==100
    assert all(g['status']=='CONDITIONAL_XYZ_REFUSED' for g in selected)


def test_budget_valueerror_propagates_from_numeric_helper(monkeypatch):
    import integrations.demo_pipeline.query_camera_extension as module
    active=[False]
    def budget():
        if active[0]:raise ValueError('shared clock')
    def stop(group,checked):active[0]=True;checked()
    monkeypatch.setattr(module,'_numeric',stop)
    with pytest.raises(ValueError,match='shared clock'):assess_query_extensions(*fixture()[:5],budget=budget)


def test_empty_does_not_require_legacy_unused_decode_metadata():
    result=assess_query_extensions(dict(records=[]),[dict(input_index=1)],[None],{},dict(records=[]))
    assert result['counts']['conditional_points']==0 and result['classifications']==[]


def known_query_fixture():
    args=list(fixture())
    args[1].append(dict(input_index=99,decode_index=11,frame_id='known-query',width=100,height=100,original_mask_status='OK'))
    args[2].append(np.full((100,100),255,np.uint8))
    args[0]['records'][0]['observations'][1].update(original_mask_status='OK',input_index=99,frame_id='known-query',label='foreground')
    return args


def test_existing_query_mask_is_verified_and_compact_camera_dependencies():
    args=known_query_fixture()
    # A large source assessment remains owned by CAMERA_HYPOTHESES; it must
    # never be repeated once per triangulated group/camera.
    for record in args[4]['records']:
        for model in record['assessment']['models']:model['large_unused_tree']={'DLT':[1]*10000}
    result=assess_query_extensions(*args[:5])
    assert all(g['query_foreground_status']=='PARTIALLY_VERIFIED_OR_UNKNOWN' for g in result['groups'])
    assert all(item['classification']['rows'][0]['query_foreground_status']=='PARTIALLY_VERIFIED_OR_UNKNOWN' for item in result['classifications'])
    for group in result['groups']:
        for camera in group['query_cameras']:
            assert 'source_assessment' not in camera and 'large_unused_tree' not in camera
            assert set(camera['train_group_ids'])|set(camera['holdout_group_ids'])==set(camera['anchor_group_ids'])
            assert camera['camera_ref']['kind']=='query'


@pytest.mark.parametrize('mutation,reason',[
    ('outside','KNOWN_QUERY_NOT_FOREGROUND'),('binary','KNOWN_QUERY_MASK_REQUIRED'),
    ('status','KNOWN_FRAME_DOMAIN_MISMATCH'),('frame','KNOWN_FRAME_IDENTITY_MISMATCH'),
    ('input','KNOWN_INPUT_IDENTITY_MISMATCH'),('dimensions','KNOWN_FRAME_DOMAIN_MISMATCH'),
    ('label','KNOWN_QUERY_LABEL_MISMATCH')])
def test_known_query_mask_and_frame_binding_cannot_be_hidden_as_unknown(mutation,reason):
    args=known_query_fixture();ob=args[0]['records'][0]['observations'][1]
    if mutation=='outside':args[2][1][:]=0
    if mutation=='binary':args[2][1][0,0]=1
    if mutation=='status':ob['original_mask_status']='UNKNOWN_NOT_SEGMENTED'
    if mutation=='frame':ob['frame_id']='wrong'
    if mutation=='input':ob['input_index']=100
    if mutation=='dimensions':ob['width']=101
    if mutation=='label':ob['label']='background'
    with pytest.raises(ValueError,match=reason):assess_query_extensions(*args[:5])


def test_no_known_frame_cannot_claim_original_query_mask():
    args=fixture();args[0]['records'][0]['observations'][1]['original_mask_status']='OK'
    with pytest.raises(ValueError,match='UNBOUND_ORIGINAL_QUERY_MASK_CLAIM'):assess_query_extensions(*args[:5])
