"""Frozen-camera numeric diagnostics; no pose refinement or ground-truth claim.

Primary sources checked 2026-10-14:
https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/sensor/models/radial.h
https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/sensor/models/base.h
https://colmap.github.io/format.html
https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/geometry/essential_matrix.cc
SIMPLE_RADIAL is model id 2, parameters f,cx,cy,k. COLMAP applies
(u,v)*(1+k*(u*u+v*v)) before focal/principal-point pixel conversion.
Hamilton wxyz poses map world to camera, X_cam=R X_world+t.
The essential_matrix source implements [t]_cross R, K2^-T E K1^-1,
and the squared Sampson formula used below. Its zero-denominator sentinel is
replaced here by explicit refusal, not a numeric success value.

Inverse is this project's bounded scalar bisection, not COLMAP's iterative
implementation. For negative k a declared normalized ideal-radius bound is
required, strictly inside the radial turning point: without a bound the
inverse is ambiguous. Returned ideal pixels retain the original f,cx,cy.
These floating diagnostics condition on existing intrinsics and poses; neither
agreement nor residual improvement is independent geometry ground truth.
"""
import math
import numpy as np

SOURCE_URLS=(
    'https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/sensor/models/radial.h',
    'https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/sensor/models/base.h',
    'https://colmap.github.io/format.html',
    'https://raw.githubusercontent.com/colmap/colmap/main/src/colmap/geometry/essential_matrix.cc',
)


def _array(value,shape=None):
    if any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat):
        raise ValueError('bool is not a numeric diagnostic input')
    raw=np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.dtype.kind=='b':
        raise ValueError('real numeric values required; bool/object/complex rejected')
    a=np.asarray(value,dtype=np.float64)
    if (shape is not None and a.shape!=shape) or not np.all(np.isfinite(a)):
        raise ValueError('finite values with the declared shape required')
    return a


def _camera(params,model_id):
    if type(model_id) is not int or model_id!=2:
        raise ValueError('only COLMAP SIMPLE_RADIAL model_id 2 supported')
    p=_array(params,(4,))
    if p[0]<=0:raise ValueError('positive focal length required')
    return p


def _pixels(points):
    p=_array(points)
    if p.ndim!=2 or p.shape[1]!=2 or not 1<=len(p)<=4096:
        raise ValueError('complete 1..4096 by 2 pixel array required')
    return p


def _domain(k,max_ideal_radius):
    if max_ideal_radius is None:
        if k<0:raise ValueError('negative k has nonunique global inverse; explicit monotone radius domain required')
        return None
    if type(max_ideal_radius) not in (int,float) or not math.isfinite(max_ideal_radius) or max_ideal_radius<=0:
        raise ValueError('positive finite normalized ideal-radius bound required')
    radius=float(max_ideal_radius)
    derivative=1+3*k*radius*radius
    if not math.isfinite(derivative) or derivative<=0:
        raise ValueError('declared distortion domain is nonmonotone or reaches the turning point')
    return radius


def simple_radial_intrinsics(params,*,model_id=2):
    f,cx,cy,k=_camera(params,model_id)
    return np.array(((f,0,cx),(0,f,cy),(0,0,1)),dtype=np.float64)


def distort_simple_radial(ideal_pixels,params,*,model_id=2,max_ideal_radius=None):
    """Forward ideal pixels -> measured raw pixels with unchanged K."""
    p=_pixels(ideal_pixels);f,cx,cy,k=_camera(params,model_id)
    limit=_domain(k,max_ideal_radius)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        uv=(p-np.array((cx,cy)))/f
        r2=np.sum(uv*uv,axis=1)
        if not np.all(np.isfinite(r2)):raise ValueError('normalized radius overflow')
        if limit is not None and np.any(np.sqrt(r2)>limit):raise ValueError('point outside declared ideal-radius domain')
        raw=f*uv*(1+k*r2[:,None])+np.array((cx,cy))
    if not np.all(np.isfinite(raw)):raise ValueError('distortion overflow')
    return raw


def undistort_simple_radial(raw_pixels,params,*,model_id=2,max_ideal_radius=None,
                           max_iterations=128,residual_tolerance_pixels=1e-8):
    """Bounded unique-domain inverse; returns ideal pixels and measured residuals.

    Radius bound is in normalized ideal camera coordinates, not pixel units.
    Failure/ambiguous domains raise ValueError; exhausted or unverified inverses
    raise ArithmeticError. No unchecked solution or partial array is returned.
    """
    p=_pixels(raw_pixels);f,cx,cy,k=_camera(params,model_id)
    limit=_domain(k,max_ideal_radius)
    if type(max_iterations) is not int or not 1<=max_iterations<=256:
        raise ValueError('integer iteration bound 1..256 required')
    if (type(residual_tolerance_pixels) not in (int,float)
            or not math.isfinite(residual_tolerance_pixels) or not 0<residual_tolerance_pixels<=1e-3):
        raise ValueError('finite positive pixel residual tolerance <=1e-3 required')
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        measured=(p-np.array((cx,cy)))/f
        radii=np.hypot(measured[:,0],measured[:,1])
    if not np.all(np.isfinite(radii)):raise ValueError('measured normalized radius overflow')
    ideal=np.empty_like(p);iterations=[]
    for idx,(direction,rd) in enumerate(zip(measured,radii)):
        if k==0:
            if limit is not None and rd>limit:raise ValueError('point outside declared radius domain')
            ideal[idx]=p[idx];iterations.append(0);continue
        if rd==0:
            ideal[idx]=(cx,cy);iterations.append(0);continue
        lo=0.0;hi=float(rd) if limit is None else limit
        with np.errstate(over='ignore',invalid='ignore'):
            upper=hi*(1+k*hi*hi)
        if not math.isfinite(upper):raise ValueError('radial domain overflow')
        if rd>upper:raise ValueError('raw radius outside image of declared monotone domain')
        converged=False
        for n in range(1,max_iterations+1):
            radius=(lo+hi)/2
            value=radius*(1+k*radius*radius)
            if not math.isfinite(value):raise ArithmeticError('inverse arithmetic overflow')
            if abs(value-rd)*f<=residual_tolerance_pixels:
                converged=True;break
            if value<rd:lo=radius
            else:hi=radius
        if not converged:raise ArithmeticError('bounded radial inverse failed residual check')
        ideal[idx]=f*direction*(radius/rd)+np.array((cx,cy));iterations.append(n)
    restored=distort_simple_radial(ideal,params,model_id=model_id,max_ideal_radius=limit)
    residuals=np.hypot(*(restored-p).T)
    if not np.all(np.isfinite(ideal)) or np.any(residuals>residual_tolerance_pixels):
        raise ArithmeticError('full 2D inverse roundtrip residual check failed')
    return dict(ideal_pixels=ideal,forward_residual_pixels=residuals,iterations=tuple(iterations),
                coordinate_domain='ideal_pixels_original_intrinsics',
                normalized_ideal_radius_bound=limit,
                scope='frozen SIMPLE_RADIAL numeric inverse; not independent camera calibration')


def quaternion_wxyz_to_rotation(quaternion):
    """Hamilton world-to-camera rotation; normalize only unit-roundoff (1e-6)."""
    q=_array(quaternion,(4,));norm=float(np.linalg.norm(q))
    if not math.isfinite(norm) or abs(norm-1)>1e-6:
        raise ValueError('unit Hamilton wxyz quaternion required')
    w,x,y,z=q/norm
    return np.array(((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
                     (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
                     (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y))),dtype=np.float64)


def _rotation(value):
    r=_array(value,(3,3))
    if not np.allclose(r.T@r,np.eye(3),atol=1e-10,rtol=0) or not np.isclose(np.linalg.det(r),1,atol=1e-10,rtol=0):
        raise ValueError('proper orthogonal rotation required')
    return r


def fundamental_from_poses(intrinsics1,rotation1,translation1,intrinsics2,rotation2,translation2):
    """F21 for x2.T F21 x1=0 from frozen world-to-camera poses.

    Derived identities R21=R2 R1.T, t21=t2-R21 t1;
    F21=K2^-T [t21]_cross R21 K1^-1. F is Frobenius-normalized.
    Ideal pixel coordinates are required; raw radial pixels violate this model.
    """
    k1,k2=(_array(k,(3,3)) for k in (intrinsics1,intrinsics2))
    for k in (k1,k2):
        if (k[0,0]<=0 or k[1,1]<=0 or not np.array_equal(k[2],(0,0,1))):
            raise ValueError('finite positive pinhole intrinsics with homogeneous final row required')
    r1,r2=_rotation(rotation1),_rotation(rotation2)
    t1,t2=_array(translation1,(3,)),_array(translation2,(3,))
    relative_r=r2@r1.T;relative_t=t2-relative_r@t1
    scale=float(np.linalg.norm(relative_t))
    if not math.isfinite(scale) or scale<=0:raise ValueError('nonzero finite relative translation required')
    # F is scale-free; normalize translation before products to avoid underflow.
    x,y,z=relative_t/scale
    cross=np.array(((0,-z,y),(z,0,-x),(-y,x,0)))
    try:f=np.linalg.inv(k2).T@cross@relative_r@np.linalg.inv(k1)
    except np.linalg.LinAlgError as exc:raise ValueError('invertible intrinsics required') from exc
    norm=float(np.linalg.norm(f))
    if not np.all(np.isfinite(f)) or not math.isfinite(norm) or norm<=0:
        raise ArithmeticError('fundamental matrix numeric degeneracy')
    return f/norm


def sampson_distances(points1,points2,fundamental):
    """Return squared Sampson approximation (pixel^2) and its root (pixels).

    This is a first-order geometric residual, not exact reprojection error.
    Undefined zero denominators are rejected rather than reported as zero.
    """
    a,b=_pixels(points1),_pixels(points2);f=_array(fundamental,(3,3))
    if a.shape!=b.shape:raise ValueError('complete equally sized corresponding point arrays required')
    norm=float(np.linalg.norm(f))
    if not math.isfinite(norm) or norm<=0:raise ValueError('nonzero finite F required')
    f=f/norm
    ah=np.column_stack((a,np.ones(len(a))));bh=np.column_stack((b,np.ones(len(b))))
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        lines2=ah@f.T;lines1=bh@f
        numerator=np.sum(bh*lines2,axis=1)
        denominator=np.sum(lines2[:,:2]**2,axis=1)+np.sum(lines1[:,:2]**2,axis=1)
        if np.any(denominator<=0) or not np.all(np.isfinite(denominator)):
            raise ValueError('undefined or nonfinite Sampson denominator')
        squared=numerator*numerator/denominator
    if not np.all(np.isfinite(squared)):raise ArithmeticError('Sampson overflow')
    return dict(squared=squared,distance=np.sqrt(squared),squared_unit='pixel^2',distance_unit='pixel',
                scope='frozen-pose first-order Sampson diagnostic; not independent ground truth')
