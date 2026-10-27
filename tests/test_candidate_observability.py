import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from integrations.demo_pipeline import candidate_observability as review


def observations(same_center=False):
    xyz=np.array([.2,-.1,5.]);result=[]
    for i,center in enumerate(([0,0,0],[1,0,0],[0,1,0])):
        o=dict(image_id=i+1,params=[800.,320.,240.,-.05],R=np.eye(3).tolist(),
            t=(-np.array([0,0,0] if same_center else center,dtype=float)).tolist(),max_ideal_radius=1.,raw_pixels=[0.,0.])
        o['raw_pixels']=review.fixed.project(xyz,o)['pixels'].tolist();result.append(o)
    return xyz,result


def test_known_three_cameras_truth_and_leave_one_view_out():
    xyz,obs=observations();result=review._geometry(obs)
    assert result['status']=='CONDITIONAL_DIAGNOSTIC_EVALUATED'
    np.testing.assert_allclose(result['xyz'],xyz,atol=1e-8,rtol=0)
    assert result['evaluation']['max']<1e-8 and result['observability']['rank']==3
    assert all(d['positive'] for d in result['positive_depths'])
    assert len(result['leave_one_view_out'])==3
    for row in result['leave_one_view_out']:
        assert row['heldout_image_id'] not in row['training_image_ids']
        assert row['prediction']['max']<1e-8
    assert not result['geometry_acceptance'] and not result['identity_verified']


def test_coincident_centers_refuse_and_negative_nonmonotonic_domain_refuse():
    _,obs=observations(True);result=review._geometry(obs)
    assert result['xyz'] is None and 'coincident' in result['reason']
    _,obs=observations();obs[0]['params'][3]=-.5
    refused=review._geometry(obs)
    assert refused['status']=='RADIAL_DOMAIN_REFUSED' and refused['xyz'] is None
    assert refused['radial_domains'][0]['radial_derivative_at_bound']<0
    assert refused['radial_domains'][0]['declared_ideal_radius']==1.


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'native';root.mkdir();binary=root/'model';binary.mkdir();photos=root/'images';photos.mkdir()
    for name in ('images.bin','points3D.bin','cameras.bin'):(binary/name).write_bytes(b'synthetic native placeholder')
    _,obs=observations();frames=[];features=[];mapped=[];native_images={};poses={}
    for i,o in enumerate(obs):
        ident=i+1;name='images/'+str(i)+'.png';path=root/name;path.write_bytes(('synthetic image'+str(i)).encode())
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        frames.append(dict(input_index=i,frame_id='frame'+str(i),sha256=sha,path=str(path)))
        features.append(dict(frame_index=i,input_index=i,db_image_id=ident,image_name=name,
            points=[dict(id=str(ident)+':0',feature_row=0,xy=o['raw_pixels'])]))
        mapped.append(dict(input_index=i,upstream_io_run_id='synthetic-io',frame_id='frame'+str(i),canonical_image_sha256=sha,database_image_id=ident,
            staged_image_name=name,registrations=[dict(model_id='0',image_id=ident,camera_id=1)]))
        native_images[ident]=dict(camera_id=1,name=name,observations=[tuple(o['raw_pixels'])+(-1,)])
        poses[ident]=dict(camera_id=1,name=name,R=np.array(o['R']),t=np.array(o['t']))
    fmap=root/'frames_map.jsonl';fmap.write_text('\n'.join(json.dumps(m) for m in mapped),encoding='utf-8')
    run=dict(contract_version='sfm-native-object-v1',upstream_io_run_id='synthetic-io',status='PARTIAL',models=[dict(model_id='0',binary_path='model',usable=True,
        roundtrip_verified=True,registered_image_ids=[1,2,3],num_points3D=0)])
    run_path=root/'run.json';run_path.write_text(json.dumps(run),encoding='utf-8')
    pipeline=dict(stages=dict(sfm=dict(run_json=str(run_path))))
    camera=dict(params=[800.,320.,240.,-.05],radius=1.,width=640,height=480)
    monkeypatch.setattr(review.native,'read_native',lambda *a:(native_images,{}))
    monkeypatch.setattr(review.bridge,'read_calibration',lambda *a:({1:camera},poses))
    c=dict(wedge_index=0,endpoint_a=dict(frame=0,feature_id='1:0'),center=dict(frame=1,feature_id='2:0'),
        endpoint_b=dict(frame=2,feature_id='3:0'),descriptor_cycle_closed=True,
        direct_geometry=dict(status='EXISTING_F_SUPPORT_CANDIDATE'),two_edge_ids=['a','b'],
        observed_descriptor_competition=True,competition=[dict(alternatives=['synthetic alternative'])])
    multi=dict(closures=[c],seed_graph=dict(components=[dict(edge_ids=['a','b'])]))
    return pipeline,dict(frames=features),frames,multi,mapped,fmap,native_images,run_path


def test_wrapper_native_identity_bindings_and_competition_retained(tmp_path,monkeypatch):
    pipeline,features,frames,multi,*_=fixture(tmp_path,monkeypatch)
    refs=[];out=tmp_path/'report'
    result=review.review_candidate_observability(pipeline,features,frames,multi,out,refs)
    assert result['cases'][0]['status']=='CONDITIONAL_DIAGNOSTIC_EVALUATED'
    assert result['cases'][0]['observed_descriptor_competition'] is True
    assert not result['camera_independence'] and not result['geometry_acceptance']
    assert (out/'OBSERVABILITY.json').is_file() and (out/'index.html').is_file() and len(refs)>5


@pytest.mark.parametrize('mode',['missing','cross_model'])
def test_missing_or_cross_model_does_not_downgrade_to_two_views(tmp_path,monkeypatch,mode):
    pipeline,features,frames,multi,mapped,fmap,*_=fixture(tmp_path,monkeypatch)
    mapped[0]['registrations']=[] if mode=='missing' else [dict(model_id='1',image_id=1,camera_id=1)]
    fmap.write_text('\n'.join(json.dumps(m) for m in mapped),encoding='utf-8')
    r=review.review_candidate_observability(pipeline,features,frames,multi,tmp_path/'report',[])
    assert r['cases'][0]['status']=='MISSING_COMMON_MODEL_POSES'
    assert 'xyz' not in r['cases'][0]


def test_native_row_xy_mismatch_rejected(tmp_path,monkeypatch):
    pipeline,features,frames,multi,mapped,fmap,images,_=fixture(tmp_path,monkeypatch)
    images[1]['observations'][0]=(999.,999.,-1)
    with pytest.raises(ValueError,match='row/xy'):review.review_candidate_observability(pipeline,features,frames,multi,tmp_path/'report',[])


def test_source_changed_during_fit_refuses_publication(tmp_path,monkeypatch):
    pipeline,features,frames,multi,*tail=fixture(tmp_path,monkeypatch);run_path=tail[-1]
    original=review.fixed.fit
    def changed(*a,**k):
        result=original(*a,**k)
        run_path.write_text(run_path.read_text()+' ',encoding='utf-8')
        return result
    monkeypatch.setattr(review.fixed,'fit',changed)
    with pytest.raises(ValueError,match='drift'):review.review_candidate_observability(pipeline,features,frames,multi,tmp_path/'report',[])
    assert not (tmp_path/'report'/'OBSERVABILITY.json').exists()


def test_parent_large_binding_keeps_its_own_verification_policy(tmp_path,monkeypatch):
    pipeline,features,frames,multi,*_=fixture(tmp_path,monkeypatch)
    database=tmp_path/'parent_only.db'
    with database.open('wb') as stream:
        stream.seek((65<<20)-1);stream.write(b'\0')
    with database.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    caller_ref=dict(path=str(database),bytes=65<<20,sha256=digest)
    with pytest.raises(ValueError,match='64 MiB'):review.native.binding(database)
    refs=[caller_ref.copy()]
    result=review.review_candidate_observability(pipeline,features,frames,multi,tmp_path/'report',refs)
    assert result['cases'][0]['status']=='CONDITIONAL_DIAGNOSTIC_EVALUATED'
    assert refs[0]==caller_ref
    assert str(database) not in [r['path'] for r in result['source_bindings']]


def test_central_branch_is_separate_and_cannot_mutate_original_observations():
    import copy
    xyz,obs=observations()
    for o in obs:o.update(max_ideal_radius=3.,image_width=640,image_height=480)
    before=copy.deepcopy(obs)
    strict=review._geometry(obs)
    assert strict['status']=='RADIAL_DOMAIN_REFUSED' and strict['xyz'] is None
    branch=review._central_branch_hypothesis(obs)
    assert obs==before and branch['policy_id']=='central_branch_hypothesis-v1'
    assert branch['status']=='CONDITIONAL_DIAGNOSTIC_EVALUATED'
    np.testing.assert_allclose(branch['geometry']['xyz'],xyz,atol=1e-8,rtol=0)
    assert all(d['full_image_coverage']['covered'] for d in branch['domains'])
    assert all(r['prediction']['max']<1e-8 for r in branch['geometry']['leave_one_view_out'])
    assert not branch['physical_branch_verified'] and not branch['geometry_acceptance']
    assert review._geometry(obs)['status']=='RADIAL_DOMAIN_REFUSED'


def test_central_branch_input_refusal_never_fits(monkeypatch):
    _,obs=observations()
    for o in obs:o.update(image_width=640,image_height=480)
    obs[0]['params']=[100.,320.,240.,-1.]
    obs[0]['raw_pixels']=[370.,240.]
    def forbidden(*a,**k):raise AssertionError('refused branch must not fit')
    monkeypatch.setattr(review.fixed,'fit',forbidden)
    result=review._central_branch_hypothesis(obs)
    assert result['status']=='CENTRAL_BRANCH_INPUT_REFUSED' and result['geometry'] is None


def test_wrapper_preserves_strict_refusal_and_adds_only_conditional_branch(tmp_path,monkeypatch):
    pipeline,features,frames,multi,*_=fixture(tmp_path,monkeypatch)
    old=review.bridge.read_calibration
    def wider(*args):
        cameras,poses=old(*args)
        for camera in cameras.values():camera['radius']=3.
        return cameras,poses
    monkeypatch.setattr(review.bridge,'read_calibration',wider)
    result=review.review_candidate_observability(pipeline,features,frames,multi,tmp_path/'report',[])
    case=result['cases'][0]
    assert case['status']=='RADIAL_DOMAIN_REFUSED' and case['xyz'] is None
    assert case['central_branch_hypothesis']['status']=='CONDITIONAL_DIAGNOSTIC_EVALUATED'
    assert result['counts']['statuses']=={'RADIAL_DOMAIN_REFUSED':1}
    assert result['additional_branch_counts']=={'CONDITIONAL_DIAGNOSTIC_EVALUATED':1}
    assert case['observed_descriptor_competition'] is True


def test_missing_pose_uses_readonly_audit_once_without_native_registration(tmp_path,monkeypatch):
    import copy
    pipeline,features,frames,multi,mapped,fmap,*_=fixture(tmp_path,monkeypatch)
    mapped[0]['registrations']=[]
    fmap.write_text('\n'.join(json.dumps(m) for m in mapped),encoding='utf-8')
    multi['closures'].append(copy.deepcopy(multi['closures'][0]))
    calls=[]
    def audit(run_json,index,output,refs):
        calls.append((run_json,index,output))
        return dict(status='READ_ONLY_AUDIT_COMPLETE',counts={'observed_registration_attempts':4},registration_failure_cause='UNKNOWN')
    monkeypatch.setattr(review,'audit_registration',audit)
    result=review.review_candidate_observability(pipeline,features,frames,multi,tmp_path/'report',[])
    assert len(calls)==1 and calls[0][1]==0
    assert result['registration_audits'][0]['counts']['observed_registration_attempts']==4
    assert result['registration_audit_profile']['native_registration_executions']==0
    assert all(c['status']=='MISSING_COMMON_MODEL_POSES' for c in result['cases'])
