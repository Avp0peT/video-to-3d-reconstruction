"""Native RGB consistency with source-mask gap hypotheses; never segmentation.

The cached book motivates checking additional actual observations and consistent
labels. It does not supply this grayscale/gradient heuristic, its thresholds or
an occlusion solver. Nearest-centre mask transfer adds no measured segmentation
detail. Frame identity, crop/orientation and camera-domain binding remain the
adapter's responsibility; equal aspect ratio alone cannot prove those facts.
"""
import hashlib
import math
import numpy as np
from .narrow_gap import _cv_module


MAX_NATIVE_PIXELS=16_777_216
MAX_CANONICAL_PIXELS=4_194_304
MAX_BANK_PAIRS=4096
RULES=dict(min_native_open_span_px=6.,minimum_contrast_gray_levels=12.,contrast_noise_multiplier=3.,
           minimum_directional_gradient=8.,gradient_noise_multiplier=1.5,
           patch_center_offset_native_px=2.,patch_side_pixels=3,max_bank_pairs=MAX_BANK_PAIRS,
           sobel_scale=.125,noise_estimator='1.4826 * maximum FG/BG patch median absolute deviation')
EVIDENCE_NAMES={0:'NOT_TARGETED',1:'SUPPORTED_RGB_CONSISTENCY',2:'WEAK_RGB_EVIDENCE',3:'UNKNOWN'}
REASON_NAMES={0:'NOT_TARGETED',1:'BOTH_BANKS_SUPPORTED',2:'WEAK_BANK_CONTRAST_OR_GRADIENT',
              3:'CANONICAL_SUBRESOLUTION',4:'NATIVE_PATCH_SPACE_TOO_NARROW',5:'NATIVE_IMAGE_BORDER',
              6:'UNKNOWN_OR_INVALID_SAMPLING_DOMAIN',7:'SOURCE_LABEL_PATCH_MIXED',8:'UNSAMPLED_BUDGET'}


def _digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def _json_array(array):
    if array.dtype.kind=='f':
        value=array.astype(object);value[~np.isfinite(array)]=None
        return value.tolist()
    return array.tolist()


def _bool(value,shape=None):
    a=np.asarray(value)
    if a.dtype!=np.bool_ or a.ndim!=2 or not a.size or (shape is not None and a.shape!=shape):
        raise ValueError('nonempty same-grid bool canonical masks required')
    return a


def _nearest_transfer(mask,height,width):
    """Inverse centre mapping, including noninteger proportional scale factors."""
    h,w=mask.shape
    y=np.floor((np.arange(height,dtype=float)+.5)*h/height).astype(np.int64)
    x=np.floor((np.arange(width,dtype=float)+.5)*w/width).astype(np.int64)
    return mask[y[:,None],x[None,:]]


def assess_native_gap(rgb_native,base_canonical,literal_union,protected_candidate,canonical_diag,cv=None):
    rgb=np.asarray(rgb_native)
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or not rgb.size:
        raise ValueError('finite uint8 HxWx3 native RGB required')
    nh,nw=rgb.shape[:2]
    if nh*nw>MAX_NATIVE_PIXELS:raise ValueError('native RGB pixel budget exceeded')
    base=_bool(base_canonical);literal=_bool(literal_union,base.shape);candidate=_bool(protected_candidate,base.shape)
    h,w=base.shape
    if base.size>MAX_CANONICAL_PIXELS:raise ValueError('canonical mask pixel budget exceeded')
    if nh*w!=nw*h or nh<h or nw<w:
        raise ValueError('exact equal-axis proportional native up-transfer required; no crop/warp/downsample inferred')
    if np.any(base&~literal) or np.any(base&~candidate) or np.any(candidate&~literal):
        raise ValueError('base <= protected candidate <= literal union required')
    diag=canonical_diag
    if diag.get('source_mask_sha256')!=_digest(base):raise ValueError('diagnostics must bind the exact base mask')
    labels=np.asarray(diag['labels'])
    if labels.shape!=base.shape or not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0):
        raise ValueError('same-grid nonnegative background labels required')
    if np.any(base&(labels>0)):raise ValueError('foreground cannot have a background component label')
    valid=base|(labels>0)
    if diag.get('validity_sha256')!=_digest(valid):raise ValueError('diagnostic validity cannot be recovered from bound labels')
    target=_bool(diag['narrow_external'],base.shape)
    unresolved=_bool(diag['unresolved'],base.shape);subresolution=_bool(diag['subresolution'],base.shape)
    if np.any(target&(base|~valid|unresolved)):raise ValueError('narrow external target must be resolved valid source background')
    a,b=(np.asarray(diag[key]) for key in ('bank_a_xy','bank_b_xy'))
    directions=np.asarray(diag['direction']);width=np.asarray(diag['width_upper_bound']);opening=np.asarray(diag['open_span'])
    if (any(x.shape!=(*base.shape,2) or not np.issubdtype(x.dtype,np.integer) for x in (a,b))
            or directions.shape!=base.shape or not np.issubdtype(directions.dtype,np.integer)
            or width.shape!=base.shape or opening.shape!=base.shape
            or width.dtype.kind not in 'fiu' or opening.dtype.kind not in 'fiu'):
        raise ValueError('canonical bank/direction/span arrays required')
    positions=np.flatnonzero(target)
    all_a=a.reshape(-1,2)[positions];all_b=b.reshape(-1,2)[positions];all_d=directions.ravel()[positions]
    if len(positions):
        if (not np.isin(all_d,[0,1,2,3]).all() or (all_a<0).any() or (all_b<0).any()
                or (all_a[:,0]>=w).any() or (all_b[:,0]>=w).any()
                or (all_a[:,1]>=h).any() or (all_b[:,1]>=h).any()):
            raise ValueError('source bank coordinates or direction invalid')
        span=np.linalg.norm(all_b-all_a,axis=1)
        step=np.where(all_d<2,1.,math.sqrt(2))
        if (not base[all_a[:,1],all_a[:,0]].all() or not base[all_b[:,1],all_b[:,0]].all()
                or not np.isfinite(width.ravel()[positions]).all() or not np.isfinite(opening.ravel()[positions]).all()
                or not np.allclose(span,width.ravel()[positions],rtol=0,atol=1e-8)
                or not np.allclose(span-step,opening.ravel()[positions],rtol=0,atol=1e-8)
                or not (span>step).all()):
            raise ValueError('source bank span/foreground certificate inconsistent')
    keys=np.column_stack((all_a,all_b,all_d))
    unique,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    groups=len(unique);sample_count=min(groups,MAX_BANK_PAIRS)
    chosen=(np.arange(sample_count,dtype=np.int64)*groups//sample_count if sample_count else np.empty(0,np.int64))
    sample_positions=positions[first[chosen]]
    group_code=np.full(groups,3,np.uint8);group_reason=np.full(groups,8,np.uint8)
    group_contrast=np.full(groups,np.nan,np.float32);group_gradient=np.full(groups,np.nan,np.float32);group_noise=np.full(groups,np.nan,np.float32)
    s=nw/w
    selected_a=all_a[first[chosen]].astype(float);selected_b=all_b[first[chosen]].astype(float)
    selected_d=all_d[first[chosen]]
    delta=selected_b-selected_a
    lengths=np.linalg.norm(delta,axis=1)
    normals=delta/lengths[:,None] if sample_count else np.empty((0,2),float)
    steps=np.where(selected_d<2,1.,math.sqrt(2))
    # Predicted boundaries are halfway from each foreground pixel centre toward
    # the adjacent background lattice sample. They are hypotheses, not fitted edges.
    edge_a=(selected_a+normals*(steps*.5)[:,None]+.5)*s-.5
    edge_b=(selected_b-normals*(steps*.5)[:,None]+.5)*s-.5
    edges=np.stack((edge_a,edge_b),axis=1)
    into_gap=np.stack((normals,-normals),axis=1)
    tangents=np.stack((-into_gap[:,:,1],into_gap[:,:,0]),axis=2)
    code=np.full(sample_count,3,np.uint8);reason=np.zeros(sample_count,np.uint8)
    contrast=np.full((sample_count,2),np.nan,np.float32);gradient=np.full_like(contrast,np.nan);noise=np.full_like(contrast,np.nan)
    pending=np.ones(sample_count,bool)
    def reject(condition,why):
        rejected=pending&condition;reason[rejected]=why;pending[rejected]=False
    reject(subresolution.ravel()[sample_positions],3)
    reject(opening.ravel()[sample_positions]*s<RULES['min_native_open_span_px'],4)
    if sample_count and pending.any():
        cv=_cv_module(cv)
        gray=cv.cvtColor(rgb,cv.COLOR_RGB2GRAY).astype(np.float32)
        gx=cv.Sobel(gray,cv.CV_32F,1,0,ksize=3,scale=RULES['sobel_scale'],borderType=cv.BORDER_REFLECT101)
        gy=cv.Sobel(gray,cv.CV_32F,0,1,ksize=3,scale=RULES['sobel_scale'],borderType=cv.BORDER_REFLECT101)
        native_base=_nearest_transfer(base,nh,nw)
        known=_nearest_transfer(valid&~unresolved,nh,nw)
        # Sobel's full 3x3 footprint must be known, not just its centre.
        gradient_known=cv.erode(known.astype(np.uint8),np.ones((3,3),np.uint8),
                                borderType=cv.BORDER_CONSTANT,borderValue=0)!=0
        dn,dt=np.meshgrid(np.arange(-1,2,dtype=float),np.arange(-1,2,dtype=float))
        def patch(side):
            normal_offset=(side*RULES['patch_center_offset_native_px']+dn.ravel())[None,None,:,None]
            return edges[:,:,None,:]+into_gap[:,:,None,:]*normal_offset+tangents[:,:,None,:]*dt.ravel()[None,None,:,None]
        fg_points=patch(-1);bg_points=patch(1)
        radius=min(2.5,max(1.,s*.5))
        en,et=np.meshgrid(np.linspace(-radius,radius,5),np.arange(-1,2,dtype=float))
        edge_points=edges[:,:,None,:]+into_gap[:,:,None,:]*en.ravel()[None,None,:,None]+tangents[:,:,None,:]*et.ravel()[None,None,:,None]
        def pixelize(points):
            p=np.floor(points+.5).astype(np.int64)
            inside=(p[:,:,:,0]>=0)&(p[:,:,:,0]<nw)&(p[:,:,:,1]>=0)&(p[:,:,:,1]<nh)
            safe=p.copy();safe[:,:,:,0]=np.clip(safe[:,:,:,0],0,nw-1);safe[:,:,:,1]=np.clip(safe[:,:,:,1],0,nh-1)
            return safe,inside
        fp,fi=pixelize(fg_points);bp,bi=pixelize(bg_points);ep,ei=pixelize(edge_points)
        reject(~(fi.all(axis=(1,2))&bi.all(axis=(1,2))&ei.all(axis=(1,2))),5)
        fx,fy=fp[:,:,:,0],fp[:,:,:,1];bx,by=bp[:,:,:,0],bp[:,:,:,1];ex,ey=ep[:,:,:,0],ep[:,:,:,1]
        reject(~(known[fy,fx].all(axis=(1,2))&known[by,bx].all(axis=(1,2))&gradient_known[ey,ex].all(axis=(1,2))),6)
        reject(~(native_base[fy,fx].all(axis=(1,2))&(~native_base[by,bx]).all(axis=(1,2))),7)
        ids=np.flatnonzero(pending)
        if len(ids):
            fg=gray[fy[ids],fx[ids]];bg=gray[by[ids],bx[ids]]
            fm=np.median(fg,axis=2);bm=np.median(bg,axis=2)
            mad=np.maximum(np.median(np.abs(fg-fm[:,:,None]),axis=2),np.median(np.abs(bg-bm[:,:,None]),axis=2))*1.4826
            differences=np.abs(fm-bm)
            gradients=np.abs(gx[ey[ids],ex[ids]]*into_gap[ids,:,None,0]+gy[ey[ids],ex[ids]]*into_gap[ids,:,None,1])
            # Robust across three tangential positions; search only a narrow normal band.
            g=np.median(gradients.reshape(len(ids),2,3,5).max(axis=3),axis=2)
            strong=(differences>=np.maximum(RULES['minimum_contrast_gray_levels'],RULES['contrast_noise_multiplier']*mad))
            strong&=g>=np.maximum(RULES['minimum_directional_gradient'],RULES['gradient_noise_multiplier']*mad)
            supported=strong.all(axis=1)
            code[ids]=np.where(supported,1,2);reason[ids]=np.where(supported,1,2)
            contrast[ids]=differences;gradient[ids]=g;noise[ids]=mad
    group_code[chosen]=code;group_reason[chosen]=reason
    evaluated=(code==1)|(code==2)
    if evaluated.any():
        group_contrast[chosen[evaluated]]=contrast[evaluated].min(axis=1)
        group_gradient[chosen[evaluated]]=gradient[evaluated].min(axis=1)
        group_noise[chosen[evaluated]]=noise[evaluated].max(axis=1)
    maps=dict(target=target.copy(),evidence_code=np.zeros(base.shape,np.uint8),reason_code=np.zeros(base.shape,np.uint8),
              min_bank_contrast=np.full(base.shape,np.nan,np.float32),min_bank_gradient=np.full(base.shape,np.nan,np.float32),
              max_patch_noise=np.full(base.shape,np.nan,np.float32))
    for name,values in (('evidence_code',group_code),('reason_code',group_reason),('min_bank_contrast',group_contrast),
                        ('min_bank_gradient',group_gradient),('max_patch_noise',group_noise)):
        maps[name].ravel()[positions]=values[inverse]
    counts={EVIDENCE_NAMES[k]:int(np.count_nonzero(maps['evidence_code'][target]==k)) for k in (1,2,3)}
    stats=dict(canonical_height=h,canonical_width=w,native_height=nh,native_width=nw,scale=float(s),
        source_target_pixels=len(positions),unique_bank_pairs=groups,sampled_bank_pairs=sample_count,
        evaluated_bank_pairs=int(evaluated.sum()),supported_bank_pairs=int((code==1).sum()),weak_bank_pairs=int((code==2).sum()),
        unknown_bank_pairs=int((group_code==3).sum()),target_pixel_counts=counts,
        unknown_reasons={REASON_NAMES[k]:int(np.count_nonzero(maps['reason_code'][target]==k)) for k in range(3,9)},
        median_min_bank_contrast=float(np.median(contrast[evaluated].min(axis=1))) if evaluated.any() else None,
        median_min_bank_gradient=float(np.median(gradient[evaluated].min(axis=1))) if evaluated.any() else None,
        literal_union_added_foreground=int((literal&~base).sum()),candidate_added_foreground=int((candidate&~base).sum()),
        protected_candidate_rejected_union_pixels=int((literal&~candidate).sum()),
        literal_filled_source_target_pixels=int((literal&target).sum()),candidate_filled_source_target_pixels=int((candidate&target).sum()),
        mask_changed=False,geometry_background_claim=False,measured_new_segmentation_pixels=0)
    sample_xy=np.column_stack((sample_positions%w,sample_positions//w)).astype(np.int32)
    return dict(schema='native-gap-rgb-evidence-v1',status='NATIVE_RGB_EVIDENCE_NOT_ACCEPTED',quality_status='NOT_ACCEPTED',
        geometry_accepted=False,mask_changed=False,stats=stats,maps=maps,
        samples={key:_json_array(value) for key,value in dict(canonical_xy=sample_xy,bank_edges_native_xy=edges,
                     evidence_code=code,reason_code=reason,bank_contrast=contrast,bank_gradient=gradient,bank_noise=noise).items()},
        evidence_names=EVIDENCE_NAMES.copy(),reason_names=REASON_NAMES.copy(),rules=RULES.copy(),
        input_sha256=dict(rgb_native=_digest(rgb),base_canonical=_digest(base),literal_union=_digest(literal),protected_candidate=_digest(candidate)),
        coordinate_transfer='native_center=(canonical_center+0.5)*scale-0.5; nearest-centre labels only',
        method_origin='ENGINEERING_HEURISTIC_NOT_BOOK_ALGORITHM',
        limitations=['Frame identity, crop, orientation and distortion domain must be bound by the caller; matching aspect alone cannot prove them.',
            'Nearest-centre mask transfer adds no observed mask detail or more precise gap width.',
            'Grayscale contrast/gradient can come from texture, shadow, highlights or an unlabelled occluder. Unlabelled occlusion is not detected.',
            'Weak color evidence never authorizes filling a gap; strong edge evidence never establishes 3D free space.',
            'Only source-mask labels choose patches. Union/candidate pixels do not select evidence. No automatic mask promotion.'])
