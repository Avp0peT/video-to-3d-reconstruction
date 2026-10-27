"""Whole-area registration against the fixed original template; no point LK."""
from copy import deepcopy
import math
import numpy as np
from .area_alignment import align_area

RIGHTS=dict(geometry_accepted=False,identity_verified=False,new_cameras=0,new_model_points=0,
            segmentation_calls=0,independent_point_matches=0)


def _need(value,reason):
    if not value:raise ValueError(reason)


class _Stop(Exception):
    def __init__(self,reason,status='BLOCKED'):
        self.reason=reason;self.status=status


def _xy(value):
    _need(isinstance(value,(list,tuple)) and len(value)==2 and
          all(type(v) in (int,float) and math.isfinite(v) for v in value),'INVALID_RAW_CENTER')
    return np.asarray(value,float)


def _roi(center,domain,size,*,full=False):
    cx,cy=[int(round(float(v))) for v in center];half=size//2
    left=max(0,cx-half);top=max(0,cy-half);right=min(domain['width'],cx+half+1);bottom=min(domain['height'],cy+half+1)
    if (full and (right-left!=size or bottom-top!=size)) or right-left<33 or bottom-top<33:
        raise _Stop('INSUFFICIENT_REAL_CROP_BOUNDS')
    return dict(left=left,top=top,width=right-left,height=bottom-top)


def _mask(roi,domain):
    if domain['original_mask_status']=='UNKNOWN_NOT_SEGMENTED':
        return np.full((roi['height'],roi['width']),255,np.uint8)
    _need(domain['original_mask_status']=='OK' and isinstance(domain.get('mask_rle'),list) and
          len(domain['mask_rle'])==domain['height'],'INVALID_ORIGINAL_MASK_RLE')
    mask=np.zeros((roi['height'],roi['width']),np.uint8)
    for yy in range(roi['height']):
        previous=0
        for a,b in domain['mask_rle'][roi['top']+yy]:
            _need(type(a) is int and type(b) is int and previous<=a<b<=domain['width'],'INVALID_MASK_RUN')
            previous=b;lo=max(a,roi['left']);hi=min(b,roi['left']+roi['width'])
            if hi>lo:mask[yy,lo-roi['left']:hi-roi['left']]=255
    return mask


def _crop(gray,roi):
    return gray[roi['top']:roi['top']+roi['height'],roi['left']:roi['left']+roi['width']].copy()


def _translation(x,y):
    return np.array([[1.,0.,x],[0.,1.,y],[0.,0.,1.]])


def _global_raw(matrix,template_roi,search_roi):
    a=np.asarray(matrix)
    _need(a.shape==(2,3) and a.dtype.kind in 'iuf' and np.isfinite(a).all(),'INVALID_AREA_MATRIX')
    cv=np.vstack((a.astype(float),[0.,0.,1.]))
    whole=_translation(search_roi['left'],search_roi['top'])@cv@_translation(-template_roi['left'],-template_roi['top'])
    return _translation(.5,.5)@whole@_translation(-.5,-.5)


def _polygon(roi):
    x,y=roi['left']+.5,roi['top']+.5
    return np.array([[x,y],[x+roi['width']-1,y],[x+roi['width']-1,y+roi['height']-1],[x,y+roi['height']-1]])


def _map(matrix,points):
    transformed=np.column_stack((points,np.ones(len(points))))@matrix.T
    _need(np.isfinite(transformed).all() and np.all(np.abs(transformed[:,2])>1e-12),'INVALID_TRANSFORMED_AREA')
    return transformed[:,:2]/transformed[:,2,None]


def _corners(raw_points,domain):
    for raw in raw_points:
        cv=raw-.5;x,y=np.floor(cv).astype(int)
        if x<0 or y<0 or x+1>=domain['width'] or y+1>=domain['height']:
            raise _Stop('FORWARD_POLYGON_OUTSIDE_BILINEAR_IMAGE','REJECTED')
        if domain['original_mask_status']=='OK':
            for yy in (y,y+1):
                if not any(a<=x and b>x+1 for a,b in domain['mask_rle'][yy]):
                    raise _Stop('FORWARD_POLYGON_OUTSIDE_ORIGINAL_FOREGROUND','REJECTED')


def _finalize(result):
    regions=result['regions'];legs=[leg for r in regions for leg in r['region_legs']]
    result['counts']=dict(regions=len(regions),region_tracklets_2d=sum(r['verified_targets']>=2 for r in regions),
        incomplete_regions=sum(r['verified_targets']<2 for r in regions),targets_attempted=len(legs),
        targets_accepted=sum(l['status']=='ACCEPTED' for l in legs),
        targets_rejected=sum(l['status']=='REJECTED' for l in legs),targets_blocked=sum(l['status']=='BLOCKED' for l in legs),
        targets_interrupted=sum(l['status']=='IN_PROGRESS' for l in legs),
        verified_targets=sum(r['verified_targets'] for r in regions))


def track_areas(plan,decoded,cv,*,budget=lambda:None):
    """Register the 33px source area directly to at most four actual targets.

    UNKNOWN mask arrays supplied to registration are sampling permissions only,
    never SAM foreground claims. No old point measurements are read or reused.
    """
    budget();roots=plan.get('regions')
    _need(isinstance(roots,list) and len(roots)<=8,'ROOT_AREA_BOUND_EIGHT')
    result=dict(status='RUNNING',regions=[],counts={},native_calls=dict(matchTemplate=0,findTransformECCWithMask=0),
        native_calls_complete=True,lk_calls=0,template_policy='fixed original source template for every target',
        coordinate_policy='raw pixel centers = full-image CV coordinates + 0.5',**RIGHTS)
    if not roots:
        result['status']='AREA_TRACKING_COMPLETE';_finalize(result);return result
    domains=plan['frame_domains'];seen=set()
    def domain(index):
        d=domains[str(index)];f=decoded[index]
        _need(all(type(d.get(k)) is int and d[k]>0 and f.get(k)==d[k] for k in ('width','height')),'AREA_FRAME_DIMENSIONS')
        _need(f.get('path')==d.get('path') and isinstance(f.get('gray'),np.ndarray) and f['gray'].dtype==np.uint8 and
              f['gray'].shape==(d['height'],d['width']),'AREA_DECODED_DOMAIN_CHANGED')
        _need(d.get('original_mask_status') in ('OK','UNKNOWN_NOT_SEGMENTED'),'AREA_MASK_PROVENANCE')
        return d
    def align(template,search,tm,sm):
        budget()
        try:answer=align_area(template.copy(),search.copy(),tm.copy(),sm.copy(),cv,budget=budget)
        except Exception as exc:
            partial=getattr(exc,'partial_result',None)
            if isinstance(partial,dict) and isinstance(partial.get('native_calls'),dict):
                exc.area_alignment_partial=deepcopy(partial)
            raise
        _need(answer.get('status') in ('ACCEPTED','REJECTED','BLOCKED'),'AREA_ALIGNMENT_STATUS')
        for name,count in answer['native_calls'].items():
            _need(name in result['native_calls'] and type(count) is int and count>=0,'AREA_NATIVE_COUNT')
            result['native_calls'][name]+=count
        try:budget()
        except Exception as exc:
            exc.area_alignment_partial=deepcopy(answer);exc.area_native_counts_recorded=True;raise
        return deepcopy(answer)
    try:
        for root in roots:
            budget();rid=root.get('region_id');_need(type(rid) is str and rid and rid not in seen,'AREA_IDENTITY');seen.add(rid)
            _need(type(root.get('depth',0)) is int and root.get('depth',0)==0,'ONLY_ROOT_AREAS_ALLOWED')
            path=root['decode_path'];_need(isinstance(path,list) and 3<=len(path)<=5 and all(type(i) is int and i>=0 for i in path),'AREA_PATH_BOUND_FOUR_TARGETS')
            direction=1 if path[-1]>path[0] else -1
            _need(all(b-a==direction for a,b in zip(path,path[1:])) and path[0]==root['source_decode_index'],'AREA_SOURCE_PATH_IDENTITY')
            _need(type(root.get('source_input_index')) is int and root['source_input_index']>=0,'AREA_SOURCE_INPUT_IDENTITY')
            center=_xy(root['center_raw_xy']);area=dict(region_id=rid,depth=0,tracklet_id=root.get('tracklet_id'),
                source_input_index=root['source_input_index'],source_decode_index=path[0],decode_path=list(path),
                source_center_raw_xy=center.tolist(),template_roi=None,source_polygon_raw4x2=None,
                region_legs=[],verified_targets=0,status='INCOMPLETE',accepted_polygons=[],stop_reason=None,**RIGHTS)
            result['regions'].append(area)
            try:
                source=domain(path[0]);_need(source['original_mask_status']=='OK','ORIGINAL_SOURCE_MASK_REQUIRED')
                template_roi=_roi(center-.5,source,33,full=True);area['template_roi']=template_roi
                polygon=_polygon(template_roi);area['source_polygon_raw4x2']=polygon.tolist()
                actual_center=polygon.mean(0);area['template_center_raw_xy']=actual_center.tolist()
                template_mask=_mask(template_roi,source);coverage=float(np.mean(template_mask>0))
                area['source_mask_coverage']=coverage;area['source_mask_status']='OK'
                if coverage<.75:raise _Stop('SOURCE_TEMPLATE_FOREGROUND_COVERAGE_BELOW_0_75')
                template=_crop(decoded[path[0]]['gray'],template_roi);prediction=actual_center.copy()
                for target_index in path[1:]:
                    budget();leg=dict(source_decode_index=path[0],target_decode_index=target_index,status='IN_PROGRESS',reasons=[],
                        forward=None,backward=None,cycle_points=[],cycle_error_px=None)
                    area['region_legs'].append(leg)
                    try:
                        target=domain(target_index);search_roi=_roi(prediction-.5,target,97)
                        forward=dict(alignment=None,template_roi=deepcopy(template_roi),search_roi=search_roi,
                            template_mask_status='OK',search_mask_status=target['original_mask_status'],
                            unknown_search_sampling_mask=target['original_mask_status']=='UNKNOWN_NOT_SEGMENTED',
                            matrix_global_raw3x3=None,polygon_raw4x2=None)
                        leg['forward']=forward
                        answer=align(template,_crop(decoded[target_index]['gray'],search_roi),template_mask,_mask(search_roi,target))
                        forward['alignment']=answer
                        if answer['status']!='ACCEPTED':raise _Stop('FORWARD_'+answer['status'],answer['status'])
                        F=_global_raw(answer['matrix_cv2x3'],template_roi,search_roi);quad=_map(F,polygon)
                        forward.update(matrix_global_raw3x3=F.tolist(),polygon_raw4x2=quad.tolist())
                        _corners(quad,target)
                        mapped_center=_map(F,actual_center.reshape(1,2))[0]
                        reverse_roi=_roi(mapped_center-.5,target,33,full=True)
                        source_search_roi=_roi(actual_center-.5,source,97)
                        backward=dict(alignment=None,template_roi=reverse_roi,search_roi=source_search_roi,
                            template_mask_status=target['original_mask_status'],search_mask_status='OK',
                            unknown_template_sampling_mask=target['original_mask_status']=='UNKNOWN_NOT_SEGMENTED',
                            matrix_global_raw3x3=None,polygon_raw4x2=None)
                        leg['backward']=backward
                        reverse=align(_crop(decoded[target_index]['gray'],reverse_roi),_crop(decoded[path[0]]['gray'],source_search_roi),
                            _mask(reverse_roi,target),_mask(source_search_roi,source))
                        backward['alignment']=reverse
                        if reverse['status']!='ACCEPTED':raise _Stop('BACKWARD_'+reverse['status'],reverse['status'])
                        B=_global_raw(reverse['matrix_cv2x3'],reverse_roi,source_search_roi)
                        backward.update(matrix_global_raw3x3=B.tolist(),polygon_raw4x2=_map(B,_polygon(reverse_roi)).tolist())
                        controls=np.vstack((polygon,actual_center));returned=_map(B@F,controls)
                        errors=np.linalg.norm(returned-controls,axis=1);cycle=float(errors.max())
                        leg['cycle_points']=[dict(source_raw_xy=p.tolist(),returned_raw_xy=q.tolist(),error_px=float(e)) for p,q,e in zip(controls,returned,errors)]
                        leg['cycle_error_px']=cycle
                        if cycle>2:raise _Stop('AREA_FIVE_POINT_CYCLE_THRESHOLD','REJECTED')
                        leg['status']='ACCEPTED';area['verified_targets']+=1;prediction=mapped_center
                        area['accepted_polygons'].append(dict(decode_index=target_index,polygon_raw4x2=quad.tolist()))
                    except _Stop as exc:
                        leg['status']=exc.status;leg['reasons'].append(exc.reason)
                        if leg['forward'] and leg['forward'].get('alignment'):
                            leg['reasons'].extend('FORWARD: '+str(r) for r in leg['forward']['alignment'].get('reasons',[]))
                        if leg['backward'] and leg['backward'].get('alignment'):
                            leg['reasons'].extend('BACKWARD: '+str(r) for r in leg['backward']['alignment'].get('reasons',[]))
                        area['stop_reason']=exc.reason;break
                if area['stop_reason'] is None:area['stop_reason']='ROUTE_END'
            except _Stop as exc:area['stop_reason']=exc.reason
            area['status']='REGION_TRACKLET_2D' if area['verified_targets']>=2 else 'INCOMPLETE'
        result['status']='AREA_TRACKING_COMPLETE';_finalize(result);return result
    except Exception as exc:
        alignment_partial=getattr(exc,'area_alignment_partial',None)
        if alignment_partial is not None and result['regions'] and result['regions'][-1]['region_legs']:
            leg=result['regions'][-1]['region_legs'][-1]
            owner=leg['backward'] if leg['backward'] is not None else leg['forward']
            if owner is not None:owner['alignment']=deepcopy(alignment_partial)
            if not getattr(exc,'area_native_counts_recorded',False):
                for name,count in alignment_partial['native_calls'].items():
                    if name in result['native_calls'] and type(count) is int and count>=0:result['native_calls'][name]+=count
        result.update(status='INTERRUPTED',native_calls_complete=alignment_partial is not None)
        for area in result['regions']:
            area['status']='REGION_TRACKLET_2D' if area['verified_targets']>=2 else 'INCOMPLETE'
        _finalize(result)
        try:exc.partial_result=deepcopy(result)
        except (AttributeError,TypeError):pass
        raise
