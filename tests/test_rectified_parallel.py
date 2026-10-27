import math
from fractions import Fraction as F
import numpy as np
import pytest
from kozlov_lab.stereo.baseline.rectified_parallel import (
    calibrated_rectification as rectify,apply_homography as apply,
    quantize_rectified_six as quantize,surrogate_to_rectified_pixels as pixels,
    rectified_to_world as world,
)
from kozlov_lab.stereo.baseline.radial_diagnostics import quaternion_wxyz_to_rotation as rotation


def fixture():
    k1=np.array(((700,0,320),(0,700,240),(0,0,1)),float)
    k2=np.array(((800,0,300),(0,800,200),(0,0,1)),float)
    r1=rotation((math.cos(.06),0,math.sin(.06),0))
    r2=rotation((math.cos(-.04),0,math.sin(-.04),0))
    c1=np.array((.4,-.2,.1));c2=c1+np.array((1,0,0))
    t1=-r1@c1;t2=-r2@c2
    xyz=np.array(((-1,-1,6),(1,-1,6),(0,1,6),(.2,.3,4),(1,1,8),(-1,.5,5)))
    def project(k,r,t):
        p=(xyz@r.T+t)@k.T
        return p[:,:2]/p[:,2,None]
    rec=rectify(k1,r1,t1,k2,r2,t2)
    return xyz,rec,project(k1,r1,t1),project(k2,r2,t2)


def test_nonparallel_camera_common_rotation_metric_roundtrip():
    xyz,rec,p1,p2=fixture()
    a,b=apply(p1,rec['homography1']),apply(p2,rec['homography2'])
    np.testing.assert_allclose(a[:,1],b[:,1],atol=1e-12)
    recovered=world(a,b,rec)
    np.testing.assert_allclose(recovered['world_xyz'],xyz,atol=2e-13)
    np.testing.assert_allclose(rec['rotation_rectified']@rec['rotation_rectified'].T,np.eye(3),atol=1e-14)
    assert rec['baseline']==pytest.approx(1)


def test_modified_surrogate_actual_algt_reprojection_and_world_readback():
    from kozlov_lab.stereo.official import parallel_construction as pc
    xyz,rec,p1,p2=fixture();a,b=apply(p1,rec['homography1']),apply(p2,rec['homography2'])
    b[:,1]+=[.01,-.02,.03,-.01,.02,-.03]
    surrogate=quantize(a,b,rec['intrinsics_rectified'],scale=2)
    assert len(surrogate['ledger'])==6
    assert all(p[1]==q[1] for p,q in zip(surrogate['first'],surrogate['second']))
    assert any(abs(row['pixel_change2'][1])>.001 for row in surrogate['ledger'])
    assert all(abs(v.numerator)<=10**12 and v.denominator<=10**12
               for row in surrogate['first']+surrogate['second'] for v in row)
    names=tuple('abcdef');first=dict(zip(names,surrogate['first']));second=dict(zip(names,surrogate['second']))
    prepared=pc.prepare_views(first,second,'a','b','c','d')
    result=pc.construct_representative(prepared)
    assert result.ok,(result.status,result.failure_reason)
    def undo(matrix,p):
        (a,b,c),(d,e,f)=matrix;det=a*e-b*d
        u,v=p[0]-c,p[1]-f
        return ((e*u-b*v)/det,(-d*u+a*v)/det)
    reproj1=[];reproj2=[]
    for name in names:
        x,y,z=result.points[name]
        # Recorded representative gauge: u and u+z*v are the parallel projections.
        left=undo(prepared.normalisation_first,(x,y))
        right=undo(prepared.normalisation_second,(x+z*result.v[0],y+z*result.v[1]))
        assert left==first[name] and right==second[name]
        reproj1.append(left);reproj2.append(right)
    pixel1=pixels(reproj1,rec['intrinsics_rectified'],scale=surrogate['scale'])
    pixel2=pixels(reproj2,rec['intrinsics_rectified'],scale=surrogate['scale'])
    corrected=world(pixel1,pixel2,rec)['world_xyz']
    assert np.max(np.linalg.norm(corrected-xyz,axis=1))<.001
    # Approximate corrected physical coordinates are distinct from AlgT's affine gauge.
    assert not np.allclose(np.array(list(result.points.values()),float),corrected)


def test_horizon_nonpositive_and_depth_refusal():
    h=np.eye(3)
    for bottom in ((0,0,0),(0,0,-1)):
        bad=h.copy();bad[2]=bottom
        with pytest.raises(ValueError,match='horizon'):apply([[1,2]],bad)
    xyz,rec,p1,p2=fixture();a=apply(p1,rec['homography1']);b=apply(p2,rec['homography2'])
    for other in (a,a+np.array((1,0))):
        with pytest.raises(ValueError,match='disparity'):world(a,other,rec)
    b[:,1]+=1
    with pytest.raises(ValueError,match='y coordinates'):world(a,b,rec)


def test_degenerate_geometry_and_camera_inputs():
    k=np.eye(3);r=np.eye(3)
    with pytest.raises(ValueError,match='baseline'):rectify(k,r,[0,0,0],k,r,[0,0,0])
    with pytest.raises(ValueError,match='forward'):rectify(k,r,[0,0,0],k,r,[0,0,-1])
    for bad in (np.array(((1,1,0),(0,1,0),(0,0,1))),np.diag([1,2,1]),np.zeros((3,3))):
        with pytest.raises(ValueError):rectify(bad,r,[0,0,0],k,r,[-1,0,0])


def test_quantization_complete_input_and_fixed_grid():
    xyz,rec,p1,p2=fixture();a=apply(p1,rec['homography1']);b=apply(p2,rec['homography2'])
    for kwargs in (dict(scale=0),dict(scale=True),dict(grid_bits=17),dict(grid_bits=True)):
        with pytest.raises(ValueError):quantize(a,b,rec['intrinsics_rectified'],**kwargs)
    with pytest.raises(ValueError,match='six'):quantize(a[:5],b[:5],rec['intrinsics_rectified'])
    with pytest.raises(ValueError,match='collapsed'):quantize(np.tile(a[0],(6,1)),b,rec['intrinsics_rectified'])
    with pytest.raises(ValueError,match='bound'):quantize(a,b,rec['intrinsics_rectified'],scale=1e-20)


def test_ledger_pixel_change_and_scale_inverse():
    xyz,rec,p1,p2=fixture();a=apply(p1,rec['homography1']);b=apply(p2,rec['homography2'])
    s=quantize(a,b,rec['intrinsics_rectified'],scale=3)
    left=pixels(s['first'],rec['intrinsics_rectified'],scale=3)
    right=pixels(s['second'],rec['intrinsics_rectified'],scale=3)
    for i,row in enumerate(s['ledger']):
        np.testing.assert_array_equal(left[i]-a[i],row['pixel_change1'])
        np.testing.assert_array_equal(right[i]-b[i],row['pixel_change2'])
        assert row['normalized1_hex']==tuple(v.hex() for v in row['normalized1'])
