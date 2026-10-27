"""Edge-local eligibility and positive interior veto for silhouette candidates.

This module never proposes deletions. It can only narrow separately qualified
v1/v2 candidates. A projected enclosing disk bounds the entire triangle, not
just its three corners. Foreground-depth observations protect interior faces.
"""
from dataclasses import dataclass
import math
import numpy as np
from .face_mask_gate import _mesh, _camera, _view_id, _cv


@dataclass(frozen=True)
class EdgePolicy:
    band_extent_fraction: float = .02
    minimum_band_px: float = 6.
    maximum_band_px: float = 32.
    core_band_multiplier: float = 2.
    depth_relative_tolerance: float = .02

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('finite numeric edge policy required: '+name)
        if (not 0 < self.band_extent_fraction <= .1
                or not 0 < self.minimum_band_px <= self.maximum_band_px <= 100
                or not 1 < self.core_band_multiplier <= 10
                or not 0 <= self.depth_relative_tolerance < .1):
            raise ValueError('invalid edge policy')


def _project(points, k, r, c, shape):
    camera = (points-c)@r.T
    hom = camera@k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = hom[:, :2]/hom[:, 2:]
    h,w = shape
    good = ((camera[:, 2] > 1e-12)&np.isfinite(uv).all(axis=1)
            &(uv[:,0]>=-.5)&(uv[:,0]<w-.5)&(uv[:,1]>=-.5)&(uv[:,1]<h-.5))
    pixels = np.zeros((len(points),2),np.int64)
    pixels[good] = np.floor(uv[good]+.5).astype(np.int64)
    return camera[:,2],uv,pixels,good


def evaluate_edge_view(vertices, faces, *, K, R, C, mask, validity, depth, view_id,
                       policy=EdgePolicy(), check=lambda:None):
    if not isinstance(policy,EdgePolicy): raise ValueError('EdgePolicy required')
    _view_id(view_id)
    v,f = _mesh(vertices,faces); k,r,c = _camera(K,R,C)
    mask,valid = np.asarray(mask),np.asarray(validity)
    if (mask.dtype!=np.bool_ or valid.dtype!=np.bool_ or mask.ndim!=2
            or mask.shape!=valid.shape or not all(mask.shape) or mask.size>4_194_304
            or np.any(mask&~valid)):
        raise ValueError('bounded same-domain bool mask/validity required')
    if depth is not None:
        depth=np.asarray(depth)
        if depth.shape!=mask.shape or not np.issubdtype(depth.dtype,np.number):
            raise ValueError('depth in exact mask domain required')
    check(); n=len(f)
    yy,xx=np.nonzero(mask)
    extent=int(max(xx.max()-xx.min()+1,yy.max()-yy.min()+1)) if len(xx) else 0
    band=float(np.clip(extent*policy.band_extent_fraction,policy.minimum_band_px,policy.maximum_band_px))
    out=dict(view_id=view_id,edge_eligible=np.zeros(n,bool),core_protected=np.zeros(n,bool),
             edge_band_pixels=band,core_margin_pixels=band*policy.core_band_multiplier,
             enclosing_radius_pixels=np.zeros(n,float),enclosing_contour_bound_pixels=np.zeros(n,float),
             known_domain_disk=np.zeros(n,bool),protected_sample_count=np.zeros(n,np.uint8),
             geometry_accepted=False,source_mask_changed=False,
             method='whole projected triangle disk bound; any depth-supported interior vertex/centroid veto')
    if not n or not mask.any(): return out
    cv=_cv()
    # The image border and unknown labels are not artificial object contours.
    # Distance zero sites are exclusively known pixels of the opposite class.
    to_fg=cv.distanceTransform((~mask).astype(np.uint8),cv.DIST_L2,cv.DIST_MASK_PRECISE)
    to_bg=(cv.distanceTransform((mask|~valid).astype(np.uint8),cv.DIST_L2,cv.DIST_MASK_PRECISE)
           if np.any(~mask&valid) else np.full(mask.shape,np.inf))
    contour=np.where(mask,to_bg,to_fg)
    domain=cv.distanceTransform(np.pad(valid.astype(np.uint8),1),cv.DIST_L2,cv.DIST_MASK_PRECISE)[1:-1,1:-1]
    z,uv,pixels,inframe=_project(v,k,r,c,mask.shape)
    good=inframe&valid[pixels[:,1],pixels[:,0]]
    ids=np.flatnonzero(good[f].all(axis=1))
    if len(ids):
        tri=uv[f[ids]]; center=tri.mean(axis=1)
        p=np.floor(center+.5).astype(np.int64)
        radius=np.max(np.linalg.norm(tri-center[:,None,:],axis=2),axis=1)
        # EDT is sampled at the nearest pixel centre; add its exact offset.
        # A further half-pixel-diagonal safety margin covers the pixel cell.
        envelope=radius+np.linalg.norm(center-p,axis=1)+math.sqrt(.5)
        bound=contour[p[:,1],p[:,0]]+envelope
        known=domain[p[:,1],p[:,0]]>envelope
        out['enclosing_radius_pixels'][ids]=radius
        out['enclosing_contour_bound_pixels'][ids]=bound
        out['known_domain_disk'][ids]=known
        out['edge_eligible'][ids]=known&np.isfinite(bound)&(bound<=band)
    # No depth -> no positive geometry support. This is never a negative vote.
    if depth is not None:
        def supported(points):
            zz,_,p,inside=_project(points,k,r,c,mask.shape)
            x,y=p[:,0],p[:,1]; d=depth[y,x]
            return (inside&valid[y,x]&mask[y,x]&(to_bg[y,x]>out['core_margin_pixels'])
                    &np.isfinite(d)&(d>0)&(np.abs(zz-d)<=policy.depth_relative_tolerance*d))
        vertex_support=supported(v)
        centroid_support=supported(v[f].mean(axis=1))
        count=vertex_support[f].sum(axis=1,dtype=np.uint8)+centroid_support.astype(np.uint8)
        out['protected_sample_count']=count
        out['core_protected']=count>0
    check()
    return out
