from fractions import Fraction as F
import numpy as np
import pytest
from kozlov_lab.stereo.baseline import rectified_parallel as rp
from kozlov_lab.stereo.baseline import radial_diagnostics as rd
from kozlov_lab.stereo.official import given_parallel_correspondence as g

K=np.array(((800.,0.,320.),(0.,800.,240.),(0.,0.,1.)))

def yaw(theta):
    c,s=np.cos(theta),np.sin(theta)
    return np.array(((c,0,s),(0,1,0),(-s,0,c)))

def project(world,K,R,C):
    q=(world-C)@R.T
    h=q@K.T
    return h[:,:2]/h[:,2,None]

def test_rotated_offset_camera_rectification_and_inverse_world_identity():
    c1=np.array((1.,2.,3.));c2=np.array((2.,2.2,3.1));r1=yaw(.1);r2=yaw(-.08)
    k2=K.copy();k2[0,0]=k2[1,1]=1000.;k2[0,2]=310.
    rect=rp.calibrated_rectification(K,r1,-r1@c1,k2,r2,-r2@c2)
    local=np.array(((.1,.2,4.),(.8,-.4,5.),(-.7,.7,6.),(1.1,.9,7.),(-1.2,-.6,8.),(.4,1.3,4.5)))
    world=local@rect['rotation_rectified']+c1
    p1=project(world,K,r1,c1);p2=project(world,k2,r2,c2)
    a=rp.apply_homography(p1,rect['homography1']);b=rp.apply_homography(p2,rect['homography2'])
    np.testing.assert_allclose(a[:,1],b[:,1],atol=1e-10,rtol=0)
    restored=rp.rectified_to_world(a,b,rect)['world_xyz']
    np.testing.assert_allclose(restored,world,atol=1e-10,rtol=0)


def test_algt_gauge_must_reproject_both_exact_views_before_world_conversion():
    rect=rp.calibrated_rectification(K,np.eye(3),np.zeros(3),K,np.eye(3),np.array((-1.,0.,0.)))
    # Rational inverse-depth body chosen independently: (X/Z,Y/Z,1/Z).
    inverse=[(F(0),F(0),F(1,4)),(F(1,4),F(0),F(1,5)),(F(0),F(1,4),F(1,6)),
             (F(1,4),F(1,4),F(1,7)),(F(1,2),F(3,4),F(1,8)),(F(3,4),F(1,4),F(2,9))]
    p1=np.array([[float(x)*800+320,float(y)*800+240] for x,y,z in inverse])
    p2=np.array([[float(x-z)*800+320,float(y)*800+240] for x,y,z in inverse])
    q=rp.quantize_rectified_six(p1,p2,K,scale=1.)
    a=dict(enumerate(q['first']));b=dict(enumerate(q['second']));m={i:i for i in a}
    result=g.assess_given_parallel_input(a,b,m)
    assert result['status']=='CONSTRUCTED'
    assert g.assess_given_parallel_certificate(a,b,m,result['certificate'])['status']=='VERIFIED'
    c=result['certificate']['candidate']['certificate'];body=dict(c['body'])
    reconstructed=[]
    for rows,want in zip(c['projections'],(a,b)):
        views=[tuple(sum(row[k]*body[i][k] for k in range(3))+row[3] for row in rows) for i in sorted(body)]
        assert dict(enumerate(views))==want
        reconstructed.append(rp.surrogate_to_rectified_pixels(views,K,scale=1.))
    xyz=rp.rectified_to_world(*reconstructed,rect)['world_xyz']
    # Independent positive-depth ray formulas, in inherited baseline units.
    for i,((u,v),(u2,v2)) in enumerate(zip(*reconstructed)):
        expected=np.array(((u-320)/(u-u2),(v-240)/(u-u2),800/(u-u2)))
        np.testing.assert_allclose(xyz[i],expected,atol=1e-12,rtol=0)


def test_radial_inverse_against_independent_scalar_forward_expression():
    params=(800.,320.,240.,-.1);ideal=np.array(((480.,320.),(160.,80.),(320.,240.)))
    normalized=(ideal-np.array((320.,240.)))/800
    raw=800*normalized*(1-.1*np.sum(normalized**2,axis=1)[:,None])+np.array((320.,240.))
    restored=rd.undistort_simple_radial(raw,params,max_ideal_radius=1.)
    np.testing.assert_allclose(restored['ideal_pixels'],ideal,atol=2e-8,rtol=0)
    with pytest.raises(ValueError,match='nonunique'):rd.undistort_simple_radial(raw,params)


def test_sampson_pose_convention_matches_independent_world_projections():
    c1=np.array((1.,2.,3.));c2=np.array((2.,2.2,3.1));r1=yaw(.1);r2=yaw(-.08)
    world=np.array(((1.,2.,7.),(1.8,1.6,8.),(.3,2.7,9.),(2.1,2.9,10.)))
    a=project(world,K,r1,c1);b=project(world,K,r2,c2)
    f=rd.fundamental_from_poses(K,r1,-r1@c1,K,r2,-r2@c2)
    residual=rd.sampson_distances(a,b,f)
    assert residual['distance'].max()<1e-10
    changed=b.copy();changed[0,1]+=2.
    assert rd.sampson_distances(a,changed,f)['distance'][0]>.1

@pytest.mark.parametrize('case',['disparity','y','collapse'])
def test_proxy_refuses_invalid_positive_depth_or_collapsed_points(case):
    rect=rp.calibrated_rectification(K,np.eye(3),np.zeros(3),K,np.eye(3),np.array((-1.,0.,0.)))
    a=np.array(((400.,300.),));b=np.array(((350.,300.),))
    if case=='disparity':b[0,0]=401.
    elif case=='y':b[0,1]+=1.
    else:
        with pytest.raises(ValueError,match='collapsed'):rp.quantize_rectified_six(np.repeat(a,6,axis=0),np.repeat(b,6,axis=0),K)
        return
    with pytest.raises(ValueError):rp.rectified_to_world(a,b,rect)

@pytest.mark.parametrize('attack',['truncated_frozen_observations','duplicate_native_binding'])
def test_integrator_prior_ledger_complete_admission(attack,tmp_path,monkeypatch):
    from integrations.demo_pipeline import approximate_photo_bridge as bridge
    frozen={
        'schema':'photo-parallel-selected-observations-v1',
        'source_bindings':[{'path':str(tmp_path/'native'/name)} for name in ('run.json','images.bin','points3D.bin','cameras.bin')],
        'image_pair':[1,2],'point_ids':list(range(6)),'common_tracks':16,
        'observations':[
            {'point3D_id':i,'observations':[{'image_id':j,'point2D_index':i,'xy':[float(i),float(i*i)],
                'binary64_hex':[float(i).hex(),float(i*i).hex()]} for j in (1,2)]} for i in range(6)]}
    if attack=='truncated_frozen_observations':frozen['observations']=[]
    else:frozen['source_bindings'].append({'path':str(tmp_path/'other'/'images.bin')})
    for ref in frozen['source_bindings']:
        path=__import__('pathlib').Path(ref['path']);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture')
    prior_input=tmp_path/'prior'/'INPUT.json';prior_input.parent.mkdir();prior_input.write_text('{}')
    images={j:{'observations':[(float(i),float(i*i),i) for i in range(16)]} for j in (1,2)}
    points={i:{'track':{1:i,2:i}} for i in range(16)}
    monkeypatch.setattr(bridge.prior,'read_json',lambda p:frozen)
    monkeypatch.setattr(bridge,'verify_binding',lambda ref:None)
    monkeypatch.setattr(bridge.prior,'read_native',lambda *a:(images,points))
    monkeypatch.setattr(bridge.prior,'choose_pair',lambda *a:((1,2),list(range(16)),[]))
    def calibration_must_not_run(*args):raise RuntimeError('CALIBRATION_REACHED_BEFORE_LEDGER_REJECTION')
    monkeypatch.setattr(bridge,'read_calibration',calibration_must_not_run)
    with pytest.raises(ValueError):bridge.experiment(prior_input,tmp_path/'separate'/'output')
