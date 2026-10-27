"""Bounded CPU inverse-depth proposals. Bounds are experimental, not metric scale."""
import math
import cv2
import numpy as np

_DEFAULT=dict(planes=96,patch_size=5,min_std=3.0,min_ncc=0.75,prior_weight=0.03,
              prior_relative_scale=0.1,alternative_relative_separation=0.02,
              min_uniqueness_gap=0.01,max_photo_cost=0.25,
              geometry_relative_tolerance=0.02,max_reprojection_error=1.2)
REASONS={0:'OUTSIDE_ROI',1:'NO_PHOTOMETRIC_SUPPORT',2:'NON_UNIQUE',3:'INSUFFICIENT_GEOMETRIC_SUPPORT',4:'GEOMETRIC_CONFLICT',5:'ACCEPTED',6:'INSUFFICIENT_JOINT_SOURCE_SUPPORT'}


def _policy(policy):
    p=dict(_DEFAULT)
    if policy is not None:
        if not isinstance(policy,dict) or set(policy)-set(p): raise ValueError('unknown depth policy')
        p.update(policy)
    for k,v in p.items():
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0: raise ValueError('invalid depth policy '+k)
    if int(p['planes'])!=p['planes'] or not 2<=p['planes']<=128: raise ValueError('plane count must be 2..128')
    p['planes']=int(p['planes'])
    # Keep the experiment gates fixed; plane count may be reduced for pure tests.
    if any(p[k]!=_DEFAULT[k] for k in p if k!='planes'): raise ValueError('frozen depth gates cannot be overridden')
    return p


def _view(v):
    if not isinstance(v,dict) or type(v.get('view_id')) is not int or v['view_id']<0: raise ValueError('invalid view ID')
    im=v.get('image'); mask=v.get('mask'); depth=v.get('depth')
    if not isinstance(im,np.ndarray) or im.dtype!=np.uint8 or im.ndim!=3 or im.shape[2]!=3 or min(im.shape[:2])<5: raise ValueError('expected RGB uint8 image at least 5x5')
    shape=im.shape[:2]
    if not isinstance(mask,np.ndarray) or mask.dtype!=bool or mask.shape!=shape: raise ValueError('invalid view mask')
    if not isinstance(depth,np.ndarray) or depth.shape!=shape or depth.dtype.kind!='f' or not np.isfinite(depth).all() or (depth<0).any(): raise ValueError('invalid axial depth')
    vals=[]
    for key,expected in [('K',(3,3)),('R',(3,3)),('C',(3,))]:
        raw=v.get(key)
        arr=np.asarray(raw)
        if any(isinstance(value,(bool,np.bool_)) for value in np.asarray(raw,dtype=object).flat) or arr.shape!=expected or arr.dtype.kind not in 'fiu' or not np.isfinite(arr).all(): raise ValueError('invalid camera '+key)
        vals.append(arr.astype(np.float64))
    k,r,c=vals
    if not np.allclose(k[2],[0,0,1],atol=1e-8,rtol=0) or k[0,0]<=0 or k[1,1]<=0 or abs(np.linalg.det(k))<1e-10: raise ValueError('invalid calibration')
    if not np.allclose(r@r.T,np.eye(3),atol=1e-6,rtol=0) or not np.isclose(np.linalg.det(r),1,atol=1e-6,rtol=0): raise ValueError('R must be a rotation')
    return dict(v,K=k,R=r,C=c)


def _inputs(reference,sources,roi,check):
    check()
    ref=_view(reference)
    if not isinstance(sources,list) or not 2<=len(sources)<=5: raise ValueError('two to five source views required (six total views maximum)')
    src=[]
    for s in sources:
        check(); src.append(_view(s))
    ids=[s['view_id'] for s in src]
    if len(set(ids))!=len(ids) or ref['view_id'] in ids: raise ValueError('source IDs must be distinct and non-reference')
    if not isinstance(roi,np.ndarray) or roi.dtype!=bool or roi.shape!=ref['mask'].shape or (roi&~ref['mask']).any(): raise ValueError('ROI must lie inside reference mask')
    y,x=np.nonzero(roi)
    if len(x):
        x0=max(0,int(x.min())-2); x1=min(roi.shape[1],int(x.max())+3)
        y0=max(0,int(y.min())-2); y1=min(roi.shape[0],int(y.max())+3)
        if (x1-x0)*(y1-y0)>200000: raise ValueError('ROI bounding context exceeds 200000 pixels')
    else: x0=y0=x1=y1=0
    return ref,src,(x0,y0,x1,y1)


def _rays(view,x,y):
    pixels=np.column_stack([np.asarray(x).ravel(),np.asarray(y).ravel(),np.ones(np.size(x))])
    return pixels@np.linalg.inv(view['K']).T


def _project(ref,source,rays,z):
    world=(rays*np.asarray(z).reshape(-1,1))@ref['R']+ref['C']
    camera=(world-source['C'])@source['R'].T
    uv=camera@source['K'].T
    positive=camera[:,2]>1e-9
    uv=uv[:,:2]/np.where(positive,camera[:,2],1)[:,None]
    return uv,camera[:,2],positive


def _validate(ref,sources,depth,roi,p,check):
    shape=roi.shape
    support=np.zeros(shape,np.uint8); conflict=np.zeros(shape,np.uint8); unknown=np.zeros(shape,np.uint8)
    support_bits=np.zeros(shape,np.uint8); conflict_bits=np.zeros(shape,np.uint8)
    y,x=np.nonzero(roi); rays=_rays(ref,x,y); z=depth[y,x]
    valid_ref=z>0
    stats={}
    for source_index,s in enumerate(sources):
        check()
        uv,zs,pos=_project(ref,s,rays,z)
        h,w=s['mask'].shape
        inimage=pos&valid_ref&(uv[:,0]>=0)&(uv[:,0]<=w-1)&(uv[:,1]>=0)&(uv[:,1]<=h-1)
        ix=np.rint(np.clip(uv[:,0],0,w-1)).astype(int); iy=np.rint(np.clip(uv[:,1],0,h-1)).astype(int)
        inside=inimage&s['mask'][iy,ix]
        ds=s['depth'][iy,ix]
        known=inside&(ds>0)
        delta=np.abs(ds-zs)/np.maximum(zs,1e-12)
        backworld=(_rays(s,ix,iy)*ds[:,None])@s['R']+s['C']
        backcam=(backworld-ref['C'])@ref['R'].T
        backpix=backcam@ref['K'].T
        backpos=backcam[:,2]>1e-9
        backuv=backpix[:,:2]/np.where(backpos,backcam[:,2],1)[:,None]
        error=np.linalg.norm(backuv-np.column_stack([x,y]),axis=1)
        good=known&backpos&(delta<=p['geometry_relative_tolerance'])&(error<=p['max_reprojection_error'])
        bad=known&~good
        support[y,x]+=good; conflict[y,x]+=bad; unknown[y,x]+=~known
        support_bits[y,x]|=good.astype(np.uint8)<<source_index
        conflict_bits[y,x]|=bad.astype(np.uint8)<<source_index
        stats[str(s['view_id'])]=dict(roi_pixels=len(x),support=int(good.sum()),conflict=int(bad.sum()),unknown=int((~known).sum()),reasons=dict(reference_depth_missing=int((~valid_ref).sum()),projection_outside=int((valid_ref&~inimage).sum()),source_mask_outside=int((inimage&~inside).sum()),source_depth_missing=int((inside&(ds<=0)).sum()),relative_depth_conflict=int((known&(delta>p['geometry_relative_tolerance'])).sum()),reprojection_conflict=int((known&(~backpos|(error>p['max_reprojection_error']))).sum())))
    return dict(support_count=support,conflict_count=conflict,unknown_count=unknown,support_bits=support_bits,conflict_bits=conflict_bits,source_ids=[s['view_id'] for s in sources],per_source=stats,geometry_accepted=False)


def validate_depth(view,sources,depth,roi,*,policy=None,check=lambda:None):
    """Count existing source-depth agreement; missing geometry is unknown."""
    p=_policy(policy); ref,src,_=_inputs(view,sources,roi,check)
    if not isinstance(depth,np.ndarray) or depth.shape!=roi.shape or depth.dtype.kind!='f' or not np.isfinite(depth).all() or (depth<0).any(): raise ValueError('invalid candidate depth')
    return _validate(ref,src,depth,roi,p,check)


def propose_local_depth(reference,sources,roi,depth_bounds,*,policy=None,check=lambda:None):
    p=_policy(policy); ref,src,bbox=_inputs(reference,sources,roi,check)
    if not isinstance(depth_bounds,(list,tuple)) or len(depth_bounds)!=2 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in depth_bounds) or depth_bounds[0]>=depth_bounds[1]: raise ValueError('finite positive increasing axial depth bounds required')
    low,high=map(float,depth_bounds)
    if low<np.finfo(np.float32).tiny or high>np.finfo(np.float32).max: raise ValueError('bounds exceed supported float32 depth range')
    planes=1/np.linspace(1/low,1/high,p['planes'],dtype=np.float64)
    shape=roi.shape; x0,y0,x1,y1=bbox
    proposal=np.zeros(shape,np.float32); raw=proposal.copy(); bestphoto=np.full(shape,np.inf,np.float32); gap=np.full(shape,-np.inf,np.float32); priorused=np.zeros(shape,bool); accepted=np.zeros(shape,bool); reason=np.zeros(shape,np.uint8)
    oldcheck=_validate(ref,src,ref['depth'],roi,p,check)
    priorok=roi&(ref['depth']>0)&(oldcheck['support_count']>=2)&(oldcheck['conflict_count']<2)
    priorused[roi]=priorok[roi]
    checks=0
    photo_bits=np.zeros(shape,np.uint8)
    if roi.any():
        gray=cv2.cvtColor(ref['image'],cv2.COLOR_RGB2GRAY).astype(np.float32)[y0:y1,x0:x1]
        localroi=roi[y0:y1,x0:x1]
        # Every contribution is inside both images and full foreground patches.
        def box(a): return cv2.boxFilter(a,cv2.CV_32F,(5,5),normalize=True,borderType=cv2.BORDER_CONSTANT)
        refvalid=box(ref['mask'][y0:y1,x0:x1].astype(np.float32))>=1-1e-6
        mean=box(gray); variance=np.maximum(box(gray*gray)-mean*mean,0)
        refvalid &= np.sqrt(variance)>=p['min_std']
        yy,xx=np.mgrid[y0:y1,x0:x1]; rays=_rays(ref,xx,yy)
        photovol=np.full((len(planes),int(localroi.sum())),np.inf,np.float32)
        photobitsvol=np.zeros(photovol.shape,np.uint8)
        bestscore=np.full(int(localroi.sum()),np.inf,np.float32); bestindex=np.zeros(int(localroi.sum()),np.int32)
        source_data=[]
        for s in src:
            check()
            sg=cv2.cvtColor(s['image'],cv2.COLOR_RGB2GRAY).astype(np.float32)
            sm=cv2.erode(s['mask'].astype(np.uint8),np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(np.float32)
            source_data.append((s,sg,sm))
        old=ref['depth'][roi]; use=priorok[roi]
        for index,z in enumerate(planes):
            check()
            first=np.full(gray.shape,np.inf,np.float32); second=first.copy()
            plane_bits=np.zeros(gray.shape,np.uint8)
            for source_index,(s,sg,sm) in enumerate(source_data):
                check(); checks+=1
                uv,_,positive=_project(ref,s,rays,np.full(len(rays),z))
                mx=uv[:,0].reshape(gray.shape).astype(np.float32); my=uv[:,1].reshape(gray.shape).astype(np.float32)
                warped=cv2.remap(sg,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                valid=cv2.remap(sm,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>=1-1e-6
                valid &= positive.reshape(gray.shape)
                full=box(valid.astype(np.float32))>=1-1e-6
                wm=box(warped); wv=np.maximum(box(warped*warped)-wm*wm,0)
                covariance=box(gray*warped)-mean*wm
                ncc=np.clip(covariance/np.sqrt(np.maximum(variance*wv,1e-12)),-1,1)
                good=refvalid&full&(np.sqrt(wv)>=p['min_std'])
                plane_bits|=(good&(ncc>=p['min_ncc'])).astype(np.uint8)<<source_index
                cost=np.where(good,1-ncc,np.inf).astype(np.float32)
                newfirst=np.minimum(first,cost); second=np.minimum(second,np.maximum(first,cost)); first=newfirst
            photo=(first+second)*.5
            pv=photo[localroi]; photovol[index]=pv; photobitsvol[index]=plane_bits[localroi]
            penalty=np.zeros(len(old),np.float32)
            penalty[use]=p['prior_weight']*np.clip(np.abs(z-old[use])/(p['prior_relative_scale']*old[use]),0,1)
            # Low correlation is measured evidence, not missing evidence.
            # Both chosen source views must independently meet the NCC gate.
            eligible=second[localroi]<=1-p['min_ncc']
            score=np.where(eligible,pv+penalty,np.inf)
            take=score<bestscore; bestscore[take]=score[take]; bestindex[take]=index
        bestz=planes[bestindex]; selected=photovol[bestindex,np.arange(len(bestindex))].copy()
        selected[~np.isfinite(bestscore)]=np.inf
        chosen_bits=photobitsvol[bestindex,np.arange(len(bestindex))].copy()
        chosen_bits[~np.isfinite(bestscore)]=0
        photo_bits[roi]=chosen_bits
        alternative=np.full(len(bestindex),np.inf,np.float32)
        for index,z in enumerate(planes):
            check()
            separated=np.abs(z-bestz)/bestz>=p['alternative_relative_separation']
            alternative=np.minimum(alternative,np.where(separated,photovol[index],np.inf))
        unique=np.full(len(selected),-np.inf,np.float32)
        np.subtract(alternative,selected,out=unique,where=np.isfinite(selected))
        # Missing alternative support is insufficient uniqueness evidence.
        unique[~np.isfinite(selected)|~np.isfinite(alternative)]=-np.inf
        raw[roi]=np.where(np.isfinite(selected),bestz,0); bestphoto[roi]=selected; gap[roi]=unique
    candidatecheck=_validate(ref,src,raw,roi,p,check)
    photo_ok=roi&(raw>0)&(bestphoto<=p['max_photo_cost'])
    unique_ok=photo_ok&(gap>=p['min_uniqueness_gap'])
    geom_ok=candidatecheck['support_count']>=2
    no_conflict=candidatecheck['conflict_count']<2
    popcount=np.asarray([int(i).bit_count() for i in range(256)],np.uint8)
    photo_count=popcount[photo_bits]
    joint_count=popcount[photo_bits&candidatecheck['support_bits']]
    joint_ok=joint_count>=2
    accepted=unique_ok&geom_ok&no_conflict&joint_ok
    reason[roi]=1; reason[photo_ok]=2; reason[unique_ok]=3; reason[unique_ok&geom_ok&no_conflict&~joint_ok]=6; reason[unique_ok&~no_conflict]=4; reason[accepted]=5
    proposal[accepted]=raw[accepted]
    counters=dict(roi_pixels=int(roi.sum()),accepted_pixels=int(accepted.sum()),planes=len(planes),source_plane_checks=checks,photo_volume_elements=int(roi.sum())*len(planes),photo_bits_volume_bytes=int(roi.sum())*len(planes),roi_bbox=list(bbox),prior_pixels=int(priorused.sum()),reasons={name:int(np.count_nonzero(reason==key)) for key,name in REASONS.items()})
    return dict(source_ids=[s['view_id'] for s in src],photo_support_bits=photo_bits,photo_support_count=photo_count,joint_support_count=joint_count,proposal_depth=proposal,raw_best_depth=raw,accepted=accepted,best_photo_cost=bestphoto,uniqueness_gap=gap,support_count=candidatecheck['support_count'],conflict_count=candidatecheck['conflict_count'],unknown_count=candidatecheck['unknown_count'],prior_used=priorused,reason=reason,reason_legend=dict(REASONS),policy=p,policy_version='local-inverse-depth-v1',counters=counters,validation=candidatecheck,prior_validation=oldcheck,geometry_accepted=False,bounds_semantics='Experimental axial-depth range; not verified physical scale')
