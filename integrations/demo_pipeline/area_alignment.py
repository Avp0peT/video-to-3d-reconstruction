"""One whole-area affine registration with frozen spatial holdout pixels.

OpenCV4.13 findTransformECCWithMask maps template coordinates into input coordinates.
No optical flow, fitted points, or inferred geometry/physical identity.
"""
from copy import deepcopy
import math
import numpy as np
PROFILE=dict(id='whole-area-affine-v1',template_size=33,max_search_size=129,holdout_block_size=8,holdout_modulus=4,holdout_remainder=3,
    min_train_pixels=128,min_holdout_pixels=64,min_source_mask_coverage=.75,min_std=3.,min_valid_train_coverage=.85,min_valid_holdout_coverage=.85,
    min_train_ncc=.85,min_holdout_ncc=.80,max_holdout_normalized_rmse=.7,min_singular_value=.65,max_singular_value=1.5,max_affine_condition=1.8,
    max_iterations=40,epsilon=1e-4,gauss_size=1)
RIGHTS=dict(geometry_accepted=False,identity_verified=False,new_cameras=0,new_model_points=0)
def _need(ok,reason):
    if not ok:raise ValueError(reason)
def _finite(value):
    if isinstance(value,np.ndarray):return _finite(value.tolist())
    if isinstance(value,np.generic):return _finite(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,(list,tuple)):return [_finite(v) for v in value]
    if isinstance(value,dict):return {k:_finite(v) for k,v in value.items()}
    return value

def map_cv(matrix,points):
    m=np.asarray(matrix,float);p=np.asarray(points,float)
    _need(m.shape==(2,3) and p.ndim==2 and p.shape[1]==2 and np.isfinite(m).all() and np.isfinite(p).all(),'INVALID_AFFINE_MAP_INPUT')
    return (np.column_stack((p,np.ones(len(p))))@m.T).tolist()
def compose_cv(after,before):
    a=np.asarray(after,float);b=np.asarray(before,float)
    _need(a.shape==b.shape==(2,3) and np.isfinite(a).all() and np.isfinite(b).all(),'INVALID_AFFINE_COMPOSITION')
    return (np.vstack((a,[0.,0.,1.]))@np.vstack((b,[0.,0.,1.])))[:2].tolist()
def raw_to_cv(points):return (np.asarray(points,float)-.5).tolist()
def cv_to_raw(points):return (np.asarray(points,float)+.5).tolist()

def align_area(template,search,template_mask,search_mask,cv,budget=lambda:None):
    budget()
    for image,mask in ((template,template_mask),(search,search_mask)):
        _need(isinstance(image,np.ndarray) and image.dtype==np.uint8 and image.ndim==2,'INVALID_UINT8_GRAY')
        _need(isinstance(mask,np.ndarray) and mask.dtype==np.uint8 and mask.shape==image.shape,'INVALID_UINT8_MASK')
    _need(template.shape==(33,33) and all(33<=d<=129 for d in search.shape),'AREA_IMAGE_BOUND')
    templatef=template.astype(float)
    yy,xx=np.indices(template.shape);hold=((xx//8+yy//8)%4)==3;source=template_mask.astype(float)/255.;trainweights=source*(~hold);holdweights=source*hold
    trainmask=np.where(~hold,template_mask,0).astype(np.uint8);heldmask=np.where(hold,template_mask,0).astype(np.uint8)
    result=dict(status='BLOCKED',reasons=[],profile=deepcopy(PROFILE),matrix_cv2x3=None,initialization=dict(status='NOT_EVALUATED'),optimizer=dict(status='NOT_EVALUATED'),
        train=None,holdout=None,photometric=None,pixel_evidence=dict(template_uint8=template.tolist(),template_mask_uint8=template_mask.tolist(),train_mask_uint8=trainmask.tolist(),holdout_mask_uint8=heldmask.tolist(),warped_search_gray=None,valid_mask_uint8=None,search_foreground_valid_mask_uint8=None,residual_after_train_photometric=None),
        native_calls=dict(matchTemplate=0,findTransformECCWithMask=0),mask_semantics='weighted evaluation from uint8/255; ECC masks indicate nonzero validity; no residual-based pixel removal',**RIGHTS)
    def blocked(reason):result['reasons'].append(reason);return result
    def weighted_stats(x,y,w):
        z=w>0;weight=w[z];a=x[z];b=y[z]
        if not len(a):return None
        am=float(np.average(a,weights=weight));bm=float(np.average(b,weights=weight));ac=a-am;bc=b-bm;av=float(np.average(ac*ac,weights=weight));bv=float(np.average(bc*bc,weights=weight))
        den=math.sqrt(max(0.,av*bv));ncc=float(np.average(ac*bc,weights=weight)/den) if den>1e-12 else None
        return dict(mean_template=am,mean_search=bm,std_template=math.sqrt(max(0.,av)),std_search=math.sqrt(max(0.,bv)),ncc=ncc)
    sourcecoverage=float(np.mean(source));ntrain=int(np.count_nonzero(trainweights));nheld=int(np.count_nonzero(holdweights))
    result['source_support']=dict(mask_coverage=sourcecoverage,train_pixels=ntrain,holdout_pixels=nheld)
    if sourcecoverage<.75:return blocked('SOURCE_MASK_COVERAGE')
    if ntrain<128 or nheld<64:return blocked('INSUFFICIENT_FROZEN_PIXEL_SUPPORT')
    initialstats=weighted_stats(templatef,templatef,trainweights)
    if initialstats['std_template']<3:return blocked('FLAT_TEMPLATE_TRAINING_SUPPORT')
    if not callable(getattr(cv,'findTransformECCWithMask',None)):return blocked('MISSING_FIND_TRANSFORM_ECC_WITH_MASK')
    criteria=(cv.TERM_CRITERIA_COUNT|cv.TERM_CRITERIA_EPS,40,1e-4);errorclass=getattr(cv,'error',None)
    try:
        budget();result['native_calls']['matchTemplate']+=1
        response=cv.matchTemplate(search.copy(),template.copy(),cv.TM_CCOEFF_NORMED,mask=trainweights.astype(np.float32))
        result['initialization']['native_response_shape']=list(response.shape) if isinstance(response,np.ndarray) else None
        budget();_need(isinstance(response,np.ndarray) and response.shape==(search.shape[0]-32,search.shape[1]-32) and response.dtype.kind=='f','INVALID_MATCH_TEMPLATE_OUTPUT')
        best=None;eligible=0;finitecandidates=int(np.count_nonzero(np.isfinite(response)))
        fast=bool(np.all(search_mask==255) and np.isin(trainmask,(0,255)).all())
        if fast:
            finite=np.isfinite(response)
            if finite.any():
                masked=np.where(finite,response,-np.inf);index=int(np.argmax(masked));y,x=np.unravel_index(index,response.shape)
                native_score=float(response[y,x]);target=search[y:y+33,x:x+33].astype(float)
                stats=weighted_stats(templatef,target,trainweights)
                if stats['std_search']>=3 and stats['ncc'] is not None:
                    eligible=finitecandidates;best=(stats['ncc'],int(y),int(x),native_score)
            result['initialization']['selection_path']='FINITE_NATIVE_MAX_FULL_SEARCH_BINARY_TRAIN_MASK'
        else:
            finitecandidates=0
            for y in range(response.shape[0]):
                budget()
                for x in range(response.shape[1]):
                    score=float(response[y,x])
                    if not math.isfinite(score):continue
                    finitecandidates+=1;target=search[y:y+33,x:x+33].astype(float);w=trainweights*search_mask[y:y+33,x:x+33].astype(float)/255.
                    if np.count_nonzero(w)<128 or w.sum()/trainweights.sum()<.85:continue
                    stats=weighted_stats(templatef,target,w)
                    if stats['std_search']<3 or stats['ncc'] is None:continue
                    eligible+=1
                    # Search validity changes the actual objective support. Evaluate
                    # precisely this training intersection; native score is retained.
                    candidate=(stats['ncc'],y,x,score)
                    if best is None or candidate[0]>best[0]:best=candidate
        result['initialization']['selection_path']=result['initialization'].get('selection_path','SUPPORTED_MASK_INTERSECTION_SEARCH')
        result['initialization'].update(eligible_candidates=eligible,finite_native_candidates=finitecandidates,selection='single maximum training NCC on supported template/search-mask intersection; stable row-major ties')
        if best is None:return blocked('NO_VALID_TRAINING_TRANSLATION')
        score,y,x,nativescore=best;initial=np.array([[1.,0.,x],[0.,1.,y]],np.float32)
        result['initialization'].update(status='EVALUATED',translation_cv_xy=[x,y],match_score=nativescore,supported_train_ncc=score,matrix_cv2x3=initial.tolist())
        result['optimizer']=dict(status='RUNNING',correlation=None,initial_matrix_cv2x3=initial.tolist(),criteria=list(criteria),gauss_size=1)
        budget();result['native_calls']['findTransformECCWithMask']+=1
        try:
            correlation,matrix=cv.findTransformECCWithMask(template.copy(),search.copy(),trainmask.copy(),search_mask.copy(),initial.copy(),cv.MOTION_AFFINE,criteria,1)
        except Exception as exc:
            if errorclass is not None and isinstance(exc,errorclass) and getattr(exc,'code',None)==-7:
                result['optimizer'].update(status='NATIVE_NONCONVERGENCE',reason=str(exc));budget();result['status']='REJECTED';result['reasons'].append('NATIVE_NONCONVERGENCE');return result
            raise
        result['optimizer'].update(status='RETURNED',correlation=_finite(correlation),returned_matrix_cv2x3=_finite(matrix));budget()
        matrix=np.asarray(matrix)
        if matrix.shape!=(2,3) or matrix.dtype.kind!='f' or not np.isfinite(matrix).all() or type(correlation) not in (int,float,np.float32,np.float64) or not math.isfinite(float(correlation)):
            result['status']='REJECTED';return blocked('INVALID_NATIVE_AFFINE_OR_CORRELATION')
        matrix=matrix.astype(float);result['matrix_cv2x3']=matrix.tolist();result['status']='REJECTED'
        det=float(np.linalg.det(matrix[:,:2]));singular=np.linalg.svd(matrix[:,:2],compute_uv=False);condition=float(singular[0]/singular[-1]) if singular[-1]>0 else None
        result['affine_shape']=dict(determinant=det,singular_values=singular.tolist(),condition=condition)
        if not math.isfinite(det) or det<=0 or singular[-1]<.65 or singular[0]>1.5 or condition is None or not math.isfinite(condition) or condition>1.8:
            result['reasons'].append('AFFINE_SHAPE_GATE');return _finite(result)
        grid=np.column_stack((xx.ravel(),yy.ravel(),np.ones(template.size)));target=(grid@matrix.T).reshape(33,33,2);tx=target[:,:,0];ty=target[:,:,1]
        inside=(tx>=0)&(ty>=0)&(tx<=search.shape[1]-1)&(ty<=search.shape[0]-1)
        if not inside.all():result['reasons'].append('WARP_OUTSIDE_SEARCH_IMAGE')
        xc=np.clip(tx,0,search.shape[1]-1);yc=np.clip(ty,0,search.shape[0]-1);x0=np.floor(xc).astype(int);y0=np.floor(yc).astype(int);x1=np.minimum(x0+1,search.shape[1]-1);y1=np.minimum(y0+1,search.shape[0]-1);wx=xc-x0;wy=yc-y0
        def sample(image):
            return (1-wy)*((1-wx)*image[y0,x0].astype(float)+wx*image[y0,x1].astype(float))+wy*((1-wx)*image[y1,x0].astype(float)+wx*image[y1,x1].astype(float))
        warped=sample(search);support=sample(search_mask)/255.;support*=inside
        train=trainweights*support;held=holdweights*support
        if np.count_nonzero(train)<128 or np.count_nonzero(held)<64:result['reasons'].append('INSUFFICIENT_WARPED_PIXEL_SUPPORT')
        trainstats=weighted_stats(templatef,warped,train)
        if trainstats is None or trainstats['std_search']<3 or trainstats['std_template']<3:return blocked('FLAT_OR_EMPTY_WARPED_TRAINING_SUPPORT')
        z=train>0;a=warped[z];b=template[z].astype(float);w=train[z];mean_a=np.average(a,weights=w);mean_b=np.average(b,weights=w);variance=np.average((a-mean_a)**2,weights=w)
        if variance<=1e-12:return blocked('PHOTOMETRIC_TRAINING_SINGULAR')
        scale=float(np.average((a-mean_a)*(b-mean_b),weights=w)/variance);offset=float(mean_b-scale*mean_a);residual=scale*warped+offset-templatef
        result['photometric']=dict(scale=scale,offset=offset,fit_scope='TRAIN_ONLY')
        for name,weights,sourceweights,count,mincoverage,minncc in [('train',train,trainweights,ntrain,.85,.85),('holdout',held,holdweights,nheld,.85,.80)]:
            stats=weighted_stats(templatef,warped,weights);coverage=float(weights.sum()/sourceweights.sum());valid_count=int(np.count_nonzero(weights))
            rmse=math.sqrt(float(np.average(residual[weights>0]**2,weights=weights[weights>0]))) if valid_count else None
            std=stats['std_template'] if stats else 0.;normalized=rmse/std if rmse is not None and std>1e-12 else None
            result[name]=dict(count=count,valid_count=valid_count,coverage=coverage,ncc=stats['ncc'] if stats else None,rmse=rmse,normalized_rmse=normalized,std_template=std,std_search=stats['std_search'] if stats else None)
            if coverage<mincoverage:result['reasons'].append(name.upper()+'_VALID_COVERAGE_GATE')
            if stats is None or std<3 or stats['std_search']<3:result['reasons'].append(name.upper()+'_FLAT_SUPPORT')
            if stats is None or stats['ncc'] is None or stats['ncc']<minncc:result['reasons'].append(name.upper()+'_NCC_GATE')
        if result['holdout']['normalized_rmse'] is None or result['holdout']['normalized_rmse']>.7:result['reasons'].append('HOLDOUT_NORMALIZED_RMSE_GATE')
        result['pixel_evidence'].update(warped_search_gray=warped.tolist(),valid_mask_uint8=(inside.astype(np.uint8)*255).tolist(),search_foreground_valid_mask_uint8=np.round(support*255).astype(np.uint8).tolist(),residual_after_train_photometric=residual.tolist())
        if not result['reasons']:result['status']='ACCEPTED'
        budget();return _finite(result)
    except Exception as exc:
        try:exc.partial_result=_finite(deepcopy(result))
        except (TypeError,AttributeError):pass
        raise
