from copy import deepcopy
import json
import numpy as np
import pytest
from integrations.demo_pipeline.anchor_route_geometry import assess_anchor_routes


def fixture(k=0.):
    xyz=np.array([.1,.05,5.]);frames=[];masks=[];cameras={};observations=[]
    for i,index in enumerate((27,53,81)):
        frame=dict(input_index=index,decode_index=10+i*2,frame_id=f'f{index}',
                   width=100,height=100,original_mask_status='OK')
        translation=np.array([-.6+i*.6,0.,0.]);uv=(xyz+translation)[:2]/(xyz+translation)[2]
        xy=(100*uv*(1+k*np.dot(uv,uv))+[50.,50.]).tolist()
        camera=dict(input_index=index,frame_id=frame['frame_id'],image_id=i+1,width=100,height=100,
            camera_model='SIMPLE_RADIAL',params=[100.,50.,50.,k],R=np.eye(3).tolist(),t=translation.tolist())
        observations.append(dict(input_index=index,decode_index=frame['decode_index'],raw_xy=xy,
                                  width=100,height=100,frame_id=frame['frame_id'],image_id=i+1))
        frames.append(frame);cameras[index]=camera;masks.append(np.full((100,100),255,np.uint8))
    pid='track:7:segment:2:source:27'
    point=dict(point_id=pid,track_id=7,identity_segment=2,source_input_index=27,source_decode_index=10,
        source_raw_xy=deepcopy(observations[0]['raw_xy']),original_tracking=dict(strong=False,tracking_status='seed',fb_error_px=None,lk_error=None))
    row=dict(tracklet_id=pid+':edge:0',point_id=pid,model_key='m',track_id=7,identity_segment=2,
        source_input_index=27,source_point=point,observations=observations,status='MEASURED_THREE_VIEW_ANCHOR')
    return dict(records=[row]),frames,masks,dict(models=[dict(model_key='m',cameras=cameras)]),xyz


@pytest.mark.parametrize('k',[0.,.2,-.2])
def test_exact_dense_route_xyz_loo_and_original_seed_unchanged(k):
    measured,frames,masks,cameras,xyz=fixture(k);before=deepcopy((measured,frames,cameras))
    result=assess_anchor_routes(measured,frames,masks,cameras);group=result['groups'][0];row=result['classification']['rows'][0]
    assert result['numeric_groups']==1 and row['tier']=='STRICT_CONDITIONAL'
    np.testing.assert_allclose(row['xyz'],xyz,atol=1e-6)
    assert len(group['leave_one_view_out'])==3
    assert row['metrics']['max_loo_error_px']<1e-5
    assert all(o['evidence_kind']=='MEASURED_DENSE_ROUTE' for o in row['sources'])
    assert all(o['status']=='MEASURED_DENSE_ROUTE_FOREGROUND' for o in group['observations'])
    assert group['source_measurement']['source_point']['original_tracking']['strong'] is False
    assert (measured,frames,cameras)==before
    assert result['geometry_accepted'] is False and row['native_injected'] is False
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('mutation,reason',[
    ('source','SOURCE_COORDINATE_BINDING'),('pointid','SOURCE_POINT_IDENTITY'),
    ('image','OBSERVATION_CAMERA_BINDING'),('decode','OBSERVATION_CAMERA_BINDING'),
    ('dimensions','OBSERVATION_DIMENSIONS'),('binary','NONBINARY_ORIGINAL_MASK'),
    ('foreground','OUTSIDE_ORIGINAL_FOREGROUND'),('unknown','ORIGINAL_MASK_REQUIRED'),
    ('camera','CAMERA_FRAME_IDENTITY_MISMATCH'),('bool','SOURCE_POINT_IDENTITY')])
def test_identity_and_foreground_rejections_happen_before_numeric(mutation,reason):
    measured,frames,masks,cameras,_=fixture();record=measured['records'][0]
    if mutation=='source':record['observations'][0]['raw_xy'][0]+=.1
    if mutation=='pointid':record['source_point']['point_id']='injected'
    if mutation=='image':record['observations'][1]['image_id']=3
    if mutation=='decode':record['observations'][1]['decode_index']=999
    if mutation=='dimensions':record['observations'][1]['width']=101
    if mutation=='binary':masks[1][0,0]=1
    if mutation=='foreground':masks[1][:]=0
    if mutation=='unknown':frames[1]['original_mask_status']='UNKNOWN_NOT_SEGMENTED'
    if mutation=='camera':cameras['models'][0]['cameras'][53]['frame_id']='injected'
    if mutation=='bool':record['source_point']['track_id']=True
    result=assess_anchor_routes(measured,frames,masks,cameras)
    assert result['numeric_groups']==0 and result['classification']['rows'][0]['reasons']==[reason]


def test_no_cross_model_camera_fallback_or_duplicate_groups():
    measured,frames,masks,cameras,_=fixture()
    missing=cameras['models'][0]['cameras'].pop(53)
    cameras['models'].append(dict(model_key='other',cameras={53:missing}))
    assert assess_anchor_routes(measured,frames,masks,cameras)['classification']['rows'][0]['tier']=='REJECTED'
    measured,frames,masks,cameras,_=fixture();measured['records']*=2
    with pytest.raises(ValueError,match='DUPLICATE_ROUTE_GROUP'):assess_anchor_routes(measured,frames,masks,cameras)


def test_numeric_gate_does_not_trust_summary_or_bad_leaveout(monkeypatch):
    import integrations.demo_pipeline.anchor_route_geometry as module
    original=module._numeric
    def tamper(group,budget):
        original(group,budget)
        group['leave_one_view_out'][0]['fit']['converged']=False
        group['evaluation']={'status':'EVALUATED','max':0.}
    monkeypatch.setattr(module,'_numeric',tamper)
    measured,frames,masks,cameras,_=fixture()
    row=assess_anchor_routes(measured,frames,masks,cameras)['classification']['rows'][0]
    assert row['tier']=='REJECTED' and row['reasons']==['FIT_NOT_CONVERGED']


def test_budget_valueerror_propagates_and_blocked_record_not_refilled(monkeypatch):
    import integrations.demo_pipeline.anchor_route_geometry as module
    marker=ValueError('shared clock')
    # A budget callback error raised inside a numerical helper must propagate.
    def budget():
        if active[0]:raise marker
    original=module._numeric;active=[False]
    def run(group,checked):
        active[0]=True;checked()
    monkeypatch.setattr(module,'_numeric',run)
    measured,frames,masks,cameras,_=fixture()
    with pytest.raises(ValueError,match='shared clock'):assess_anchor_routes(measured,frames,masks,cameras,budget=budget)
    monkeypatch.setattr(module,'_numeric',original)
    measured['records'][0]['status']='BLOCKED_MEASUREMENT'
    result=assess_anchor_routes(measured,frames,masks,cameras)
    assert result['numeric_groups']==0 and len(result['groups'])==1


def test_empty_and_hard_record_bound():
    assert assess_anchor_routes(dict(records=[]),[],[],{})['groups']==[]
    with pytest.raises(ValueError,match='ROUTE_RECORD_BOUND'):
        assess_anchor_routes(dict(records=[{}]*97),[],[],{})


def test_empty_routes_do_not_require_unused_decode_or_camera_metadata():
    result=assess_anchor_routes(dict(records=[]),[dict(input_index=3,frame_id='legacy')],[None],{})
    assert result['status']=='ANCHOR_ROUTE_GEOMETRY_COMPLETE'
    assert result['groups']==[] and result['classification']['rows']==[]
    assert result['counts']==dict(records=0,groups=0,tier_counts={}) and result['numeric_groups']==0
    with pytest.raises(ValueError,match='FRAME_MASK_COUNT'):
        assess_anchor_routes(dict(records=[]),[dict(input_index=3)],[],{})
