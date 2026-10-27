"""Calibrated numeric rectification and an explicitly modified rational surrogate.

Engineering extension: rectified perspective observations are parallel views of
the projective inverse-depth body (X/Z,Y/Z,1/Z). AlgT's affine representative of
that body is not a claim of affine equivalence to physical world XYZ. Cameras,
poses, epipolar averaging, quantization and the world conversion are additional
engineering assumptions, not the author's direct perspective construction.
"""
from fractions import Fraction as F
import math
import numpy as np
from .radial_diagnostics import _array,_rotation,_pixels


def _intrinsics(value):
    k=_array(value,(3,3))
    if (k[0,0]<=0 or k[0,0]!=k[1,1] or k[0,1]!=0 or k[1,0]!=0
            or not np.array_equal(k[2],(0,0,1))):
        raise ValueError('square-pixel positive-focal pinhole K without skew required')
    return k


def calibrated_rectification(k1,r1,t1,k2,r2,t2):
    """Common rotation, x axis along C2-C1; Krect is a copy of K1."""
    k1,k2=_intrinsics(k1),_intrinsics(k2)
    r1,r2=_rotation(r1),_rotation(r2)
    t1,t2=_array(t1,(3,)),_array(t2,(3,))
    c1,c2=-r1.T@t1,-r2.T@t2
    axis=c2-c1;baseline=float(np.linalg.norm(axis))
    if not math.isfinite(baseline) or baseline<=1e-12:raise ValueError('degenerate stereo baseline')
    x=axis/baseline
    forward=r1[2]+r2[2]
    y=np.cross(forward,x);length=float(np.linalg.norm(y))
    if not math.isfinite(length) or length<=1e-12:
        raise ValueError('average forward direction cannot define rectification axes')
    y=y/length;z=np.cross(x,y);rrect=np.array((x,y,z))
    _rotation(rrect)
    try:
        h1=k1@rrect@r1.T@np.linalg.inv(k1)
        h2=k1@rrect@r2.T@np.linalg.inv(k2)
    except np.linalg.LinAlgError as exc:raise ValueError('invertible K required') from exc
    if not np.all(np.isfinite(h1)) or not np.all(np.isfinite(h2)):raise ArithmeticError('homography overflow')
    return dict(homography1=h1,homography2=h2,rotation_rectified=rrect,center1=c1,center2=c2,
                baseline=baseline,intrinsics_rectified=k1.copy(),
                scope='frozen calibrated common-rotation engineering rectification')


def apply_homography(ideal_pixels,homography,*,horizon_epsilon=1e-12):
    p=_pixels(ideal_pixels);h=_array(homography,(3,3))
    if type(horizon_epsilon) not in (int,float) or not math.isfinite(horizon_epsilon) or horizon_epsilon<=0:
        raise ValueError('positive finite horizon threshold required')
    projected=np.column_stack((p,np.ones(len(p))))@h.T
    if not np.all(np.isfinite(projected)) or np.any(projected[:,2]<=horizon_epsilon):
        raise ValueError('rectified horizon or nonpositive projective depth')
    result=projected[:,:2]/projected[:,2,None]
    if not np.all(np.isfinite(result)):raise ArithmeticError('homography result overflow')
    return result


def quantize_rectified_six(points1,points2,intrinsics_rectified,*,scale=1.0,grid_bits=18):
    """Exactly six fixed correspondences; common y averaged then quantized once.

    Surrogate coordinates are ((rectified_pixel-principal_point)/f)/scale.
    At scale=1 the quantization step in pixel units is f/2**18; in general
    it is f*scale/2**18. Scale is fixed by the caller, not fitted here.
    Grid is exactly 2**18, rounding ties to even (Python round). Every change,
    unrounded binary64 value, and final rational coordinate is recorded.
    """
    p1,p2=_pixels(points1),_pixels(points2);k=_intrinsics(intrinsics_rectified)
    if p1.shape!=(6,2) or p2.shape!=(6,2):raise ValueError('exactly six complete paired observations required')
    if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0:raise ValueError('positive finite coordinate scale required')
    if type(grid_bits) is not int or grid_bits!=18:raise ValueError('frozen grid_bits=18 required')
    f,cx,cy=k[0,0],k[0,2],k[1,2];principal=np.array((cx,cy))
    normalized1=(p1-principal)/(f*scale);normalized2=(p2-principal)/(f*scale)
    if not np.all(np.isfinite(normalized1)) or not np.all(np.isfinite(normalized2)):
        raise ValueError('surrogate normalization overflow')
    grid=2**18;left=[];right=[];ledger=[]
    for index,(a,b) in enumerate(zip(normalized1,normalized2)):
        common_y=float((a[1]+b[1])/2)
        qa=(F(round(float(a[0])*grid),grid),F(round(common_y*grid),grid))
        qb=(F(round(float(b[0])*grid),grid),qa[1])
        if any(abs(v.numerator)>10**12 or v.denominator>10**12 for v in qa+qb):
            raise ValueError('surrogate rational numerator/denominator bound 1e12 exceeded')
        left.append(qa);right.append(qb)
        reconstructed1=np.array([float(v) for v in qa])*f*scale+principal
        reconstructed2=np.array([float(v) for v in qb])*f*scale+principal
        ledger.append(dict(index=index,raw_rectified1=tuple(float(v) for v in p1[index]),
            raw_rectified2=tuple(float(v) for v in p2[index]),normalized1=tuple(float(v) for v in a),
            normalized2=tuple(float(v) for v in b),normalized1_hex=tuple(float(v).hex() for v in a),
            normalized2_hex=tuple(float(v).hex() for v in b),averaged_y=common_y,
            surrogate1=qa,surrogate2=qb,pixel_change1=tuple(float(v) for v in reconstructed1-p1[index]),
            pixel_change2=tuple(float(v) for v in reconstructed2-p2[index])))
    if len(set(left))!=6 or len(set(right))!=6:raise ValueError('quantization collapsed distinct view points')
    return dict(schema='rectified-six-rational-surrogate-v1',first=tuple(left),second=tuple(right),ledger=tuple(ledger),
                scale=float(scale),grid_bits=18,intrinsics_rectified=k.copy(),
                scope='modified approximate observations; inverse-depth parallel surrogate, not physical affine body')


def surrogate_to_rectified_pixels(points,intrinsics_rectified,*,scale=1.0):
    """Undo normalized/scale coordinates; remains numeric, not exact pixel evidence."""
    k=_intrinsics(intrinsics_rectified)
    if type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0:raise ValueError('positive finite scale required')
    # Fraction coordinates intentionally converted to binary64 for metric diagnosis.
    p=np.asarray(points,dtype=np.float64)
    p=_pixels(p)
    result=p*(k[0,0]*scale)+np.array((k[0,2],k[1,2]))
    if not np.all(np.isfinite(result)):raise ArithmeticError('pixel restoration overflow')
    return result


def rectified_to_world(points1,points2,rectification,*,disparity_epsilon=1e-9,y_tolerance_pixels=1e-8):
    """Numeric inverse stereo-depth mapping after recovering rectified projections.

    Do not apply this directly to an arbitrary AlgT affine 3D gauge. First
    reproject AlgT's representative into both recorded rectified surrogate views.
    Recovered XYZ uses the frozen baseline's unknown units, not measured units.
    """
    p1,p2=_pixels(points1),_pixels(points2)
    if p1.shape!=p2.shape:raise ValueError('equally sized paired views required')
    for value in (disparity_epsilon,y_tolerance_pixels):
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('positive finite diagnostic thresholds required')
    k=_intrinsics(rectification['intrinsics_rectified']);r=_rotation(rectification['rotation_rectified'])
    c=_array(rectification['center1'],(3,));baseline=rectification['baseline']
    if type(baseline) not in (int,float) or not math.isfinite(baseline) or baseline<=1e-12:raise ValueError('positive finite baseline required')
    if np.any(np.abs(p1[:,1]-p2[:,1])>y_tolerance_pixels):raise ValueError('rectified y coordinates inconsistent')
    disparity=p1[:,0]-p2[:,0]
    if np.any(disparity<=disparity_epsilon):raise ValueError('zero/negative/too-small disparity; no positive depth')
    factor=baseline/disparity
    xyz=np.column_stack(((p1[:,0]-k[0,2])*factor,(p1[:,1]-k[1,2])*factor,k[0,0]*factor))
    world=xyz@r+c
    if not np.all(np.isfinite(world)):raise ArithmeticError('stereo world conversion overflow')
    return dict(world_xyz=world,rectified_xyz=xyz,disparity_pixels=disparity,
                scope='frozen-camera stereo engineering conversion; units inherited unknown; not independent ground truth')
