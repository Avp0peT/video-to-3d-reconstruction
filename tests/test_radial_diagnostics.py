import math
import numpy as np
import pytest

from kozlov_lab.stereo.baseline.radial_diagnostics import (
    distort_simple_radial as distort,undistort_simple_radial as undistort,
    simple_radial_intrinsics as intrinsics,quaternion_wxyz_to_rotation as rotation,
    fundamental_from_poses as fundamental,sampson_distances as sampson,
)


@pytest.mark.parametrize('k',(0,0.23,2.0,-0.1))
def test_roundtrip_ideal_pixels_keep_original_intrinsics(k):
    params=(900,512,341.5,k);points=np.array(((512,341.5),(600,400),(300,100),(1100,200)))
    kwargs=dict(max_ideal_radius=1) if k<0 else {}
    raw=distort(points,params,**kwargs);result=undistort(raw,params,**kwargs)
    assert result['coordinate_domain']=='ideal_pixels_original_intrinsics'
    np.testing.assert_allclose(result['ideal_pixels'],points,atol=2e-8,rtol=0)
    assert np.max(result['forward_residual_pixels'])<=1e-8
    assert all(n<=128 for n in result['iterations'])
    assert np.array_equal(points,np.array(((512,341.5),(600,400),(300,100),(1100,200))))


def test_hand_formula_and_fixed_origin():
    # normalized (0.3,0.4), r2=.25, scale=1.05
    params=(100,20,30,.2)
    np.testing.assert_allclose(distort([[50,70]],params),[[51.5,72]],atol=1e-12)
    assert np.array_equal(undistort([[20,30]],params)['ideal_pixels'],[[20,30]])


def test_negative_k_global_ambiguity_and_monotone_domain_refusal():
    p=(100,0,0,-1)
    with pytest.raises(ValueError,match='nonunique'):undistort([[10,0]],p)
    with pytest.raises(ValueError,match='nonunique'):distort([[10,0]],p)
    for limit in (1,1/math.sqrt(3),.8):
        with pytest.raises(ValueError,match='nonmonotone'):undistort([[10,0]],p,max_ideal_radius=limit)
    with pytest.raises(ValueError,match='outside'):undistort([[40,0]],p,max_ideal_radius=.5)
    with pytest.raises(ValueError,match='outside'):distort([[60,0]],p,max_ideal_radius=.5)
    # Explicit positive monotone branch: r=.2 -> rd=.192, rejecting outer roots.
    np.testing.assert_allclose(undistort([[19.2,0]],p,max_ideal_radius=.5)['ideal_pixels'],[[20,0]],atol=1e-8)


@pytest.mark.parametrize('params',[(0,0,0,0),(-1,0,0,0),(1,0,0,float('nan')),
                                   (1,0,0,float('inf')),(True,0,0,0),(1,0,0),(1+2j,0,0,0)])
def test_invalid_camera(params):
    with pytest.raises(ValueError):undistort([[0,0]],params)


@pytest.mark.parametrize('kwargs',[dict(model_id=3),dict(model_id=True),dict(max_iterations=0),
                                   dict(max_iterations=True),dict(max_iterations=257),
                                   dict(residual_tolerance_pixels=0),dict(residual_tolerance_pixels=float('nan')),
                                   dict(max_ideal_radius=0)])
def test_invalid_limits_and_models(kwargs):
    with pytest.raises(ValueError):undistort([[0,0]],(1,0,0,.1),**kwargs)


def test_inverse_iteration_exhaustion_and_input_shape():
    with pytest.raises(ArithmeticError,match='bounded'):undistort([[1,1]],(1,0,0,.5),max_iterations=1)
    for points in ([],[1,2],[[True,0]],[[float('inf'),0]],[[1,2,3]]):
        with pytest.raises(ValueError):undistort(points,(1,0,0,.1))


def test_hamilton_rotation_sign_and_validation():
    q=(math.sqrt(.5),0,0,math.sqrt(.5));r=rotation(q)
    np.testing.assert_allclose(r@np.array((1,0,0)),(0,1,0),atol=1e-15)
    np.testing.assert_allclose(rotation(tuple(-v for v in q)),r,atol=1e-15)
    for q in ((0,0,0,0),(2,0,0,0),(1,0,0),(True,0,0,0),(float('nan'),0,0,0)):
        with pytest.raises(ValueError):rotation(q)


def test_known_rectified_motion_sampson_squared_units_and_scale():
    k=intrinsics((500,320,240,.1));eye=np.eye(3)
    f=fundamental(k,eye,[0,0,0],k,eye,[-1,0,0])
    p1=np.array(((420,290),(370,340)));p2=np.array(((320,290),(270,340)))
    assert np.linalg.matrix_rank(f)==2
    assert np.max(sampson(p1,p2,f)['squared'])<1e-25
    off=p2.copy();off[:,1]+=2
    result=sampson(p1,off,f)
    np.testing.assert_allclose(result['squared'],[2,2],atol=1e-12)
    np.testing.assert_allclose(result['distance'],[math.sqrt(2)]*2,atol=1e-12)
    assert result['squared_unit']=='pixel^2' and result['distance_unit']=='pixel'
    np.testing.assert_allclose(sampson(p1,off,-7*f)['squared'],result['squared'],atol=1e-12)


def test_nontrivial_world_to_camera_relative_pose():
    k1=intrinsics((900,400,300,.1));k2=intrinsics((750,300,250,.2))
    r1=rotation((math.cos(.15),0,math.sin(.15),0));r2=rotation((math.cos(.1),math.sin(.1),0,0))
    t1=np.array((.1,-.2,.5));t2=np.array((-.8,.1,.7))
    world=np.array(((1,1,8),(-1,1,10),(0,-1,6),(2,0,12)))
    def project(k,r,t):
        cam=world@r.T+t;pixels=cam@k.T
        return pixels[:,:2]/pixels[:,2,None]
    f=fundamental(k1,r1,t1,k2,r2,t2)
    assert np.max(sampson(project(k1,r1,t1),project(k2,r2,t2),f)['distance'])<1e-10


def test_raw_distortion_has_no_pinhole_epipolar_guarantee():
    k=intrinsics((500,320,240,.4));eye=np.eye(3)
    f=fundamental(k,eye,[0,0,0],k,eye,[-1,0,0])
    p1=np.array(((420,290),));p2=np.array(((320,290),))
    raw1=distort(p1,(500,320,240,.4));raw2=distort(p2,(500,320,240,.4))
    assert sampson(raw1,raw2,f)['distance'][0]>.1
    ideal1=undistort(raw1,(500,320,240,.4))['ideal_pixels']
    ideal2=undistort(raw2,(500,320,240,.4))['ideal_pixels']
    assert sampson(ideal1,ideal2,f)['distance'][0]<1e-8


def test_epipolar_degenerate_inputs_rejected():
    k=np.eye(3);r=np.eye(3)
    with pytest.raises(ValueError,match='nonzero'):fundamental(k,r,[0,0,0],k,r,[0,0,0])
    with pytest.raises(ValueError,match='orthogonal'):fundamental(k,2*r,[0,0,0],k,r,[1,0,0])
    with pytest.raises(ValueError):fundamental(np.zeros((3,3)),r,[0,0,0],k,r,[1,0,0])
    with pytest.raises(ValueError):sampson([[0,0]],[[0,0]],np.zeros((3,3)))
    with pytest.raises(ValueError,match='denominator'):sampson([[0,0]],[[0,0]],np.diag([0,0,1]))
