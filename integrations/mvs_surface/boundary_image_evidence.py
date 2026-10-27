"""Read-only image-edge agreement for sampled mask boundaries."""
import cv2
import numpy as np

REASONS = {0:'UNKNOWN', 1:'NO_STRONG_PEAK', 2:'AMBIGUOUS', 3:'DISTINCT_IMAGE_EDGE'}


def inspect_mask_boundary(imageRGBuint8, maskbool, validitybool, region_hintbool=None, check=lambda:None):
    """Inspect outward normal image profiles without changing masks or geometry.

    Signed offsets are positive outward from the mask. UNKNOWN includes missing
    complete profiles or peaks at the search range edge. Weak profiles have no
    measured edge (NO_STRONG_PEAK), never a zero-offset agreement.
    """
    check()
    image,mask,valid=imageRGBuint8,maskbool,validitybool
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<1:
        raise ValueError('expected nonempty RGB uint8 image')
    for value in [mask,valid]:
        if not isinstance(value,np.ndarray) or value.dtype!=bool or value.shape!=image.shape[:2]: raise ValueError('expected matching bool mask/validity')
    if region_hintbool is None: region=np.zeros(mask.shape,bool)
    else:
        region=region_hintbool
        if not isinstance(region,np.ndarray) or region.dtype!=bool or region.shape!=mask.shape: raise ValueError('expected matching bool region hint')
    inner=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    y,x=np.nonzero(mask & ~inner & valid)
    if len(x)>2048:
        indices=np.linspace(0,len(x)-1,2048,dtype=np.int64); x,y=x[indices],y[indices]
    xy=np.column_stack([x,y]).astype(np.float64); n=len(x)
    signed=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)-cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    # Central finite differences remain defined for tiny images.
    padded=np.pad(signed,1,mode='edge')
    gx=(padded[1:-1,2:]-padded[1:-1,:-2])/2
    gy=(padded[2:,1:-1]-padded[:-2,1:-1])/2
    normals=np.column_stack([gx[y,x],gy[y,x]]).astype(np.float64)
    norm=np.linalg.norm(normals,axis=1)
    normals/=np.maximum(norm[:,None],1e-12)
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray=cv2.GaussianBlur(gray,(0,0),sigmaX=1.,sigmaY=1.,borderType=cv2.BORDER_REPLICATE)
    peak_xy=np.full((n,2),np.nan); peak_offset=np.full(n,np.nan); strength=np.full(n,np.nan)
    reason=np.zeros(n,np.uint8); accepted=np.zeros(n,bool)
    offsets=np.arange(-13,14,dtype=np.float64)
    h,w=mask.shape
    for start in range(0,n,256):
        check(); end=min(start+256,n)
        points=xy[start:end,None,:]+normals[start:end,None,:]*offsets[None,:,None]
        px,py=points[:,:,0],points[:,:,1]
        inside=(px>=0)&(px<=w-1)&(py>=0)&(py<=h-1)
        ix=np.rint(np.clip(px,0,w-1)).astype(int); iy=np.rint(np.clip(py,0,h-1)).astype(int)
        complete=(inside & valid[iy,ix]).all(axis=1)&(norm[start:end]>1e-8)
        profiles=cv2.remap(gray,px.astype(np.float32),py.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        derivative=np.abs(profiles[:,2:]-profiles[:,:-2])/2
        for j in np.flatnonzero(complete):
            index=start+j; values=derivative[j]; best=int(np.argmax(values)); value=float(values[best]); strength[index]=value
            if value<6:
                reason[index]=1; continue
            if best in (0,24): continue
            alternative=values[np.abs(np.arange(25)-best)>2]
            if value<1.25*float(alternative.max()):
                reason[index]=2; continue
            offset=float(best-12)
            peak_offset[index]=offset; peak_xy[index]=xy[index]+normals[index]*offset
            reason[index]=3; accepted[index]=True
    selected=region[y,x]
    def stats(cohort):
        measured=cohort & accepted; count=int(measured.sum())
        return dict(sampled=int(cohort.sum()),unknown=int((cohort & (reason==0)).sum()),
                    no_peak=int((cohort & (reason==1)).sum()),ambiguous=int((cohort & (reason==2)).sum()),
                    accepted_distinct=count,
                    fraction_abs_peak_offset_le_2=float((np.abs(peak_offset[measured])<=2).mean()) if count else None,
                    median_abs_peak_offset=float(np.median(np.abs(peak_offset[measured]))) if count else None,
                    median_signed_peak_offset=float(np.median(peak_offset[measured])) if count else None)
    return dict(boundary_xy=xy,peak_xy=peak_xy,peak_offset=peak_offset,strength=strength,
                accepted_distinct=accepted,region_hint_selected=selected,reason=reason,reason_legend=dict(REASONS),
                stats=dict(global_stats=stats(np.ones(n,bool)),region=stats(selected)),
                policy=dict(max_boundary_samples=2048,offset_min=-12,offset_max=12,derivative_step=1,
                            gray_gaussian_sigma=1.,minimum_strength=6.,distinct_ratio=1.25,alternative_exclusion_pixels=2),
                semantics='Image edge agreement does not prove mask truth or a 3D cause. Positive offsets point outward. Region hint selects reporting cohorts only. Missing, weak, ambiguous, and range-edge profiles do not establish agreement.')
