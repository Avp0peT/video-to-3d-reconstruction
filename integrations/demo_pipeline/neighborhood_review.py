"""Image-backed provisional correspondence; strict measurements remain immutable."""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import time
import numpy as np
from .area_alignment import align_area
from .area_tracking import _roi, _mask, _crop, _polygon, _map, _global_raw, _corners, _Stop
from .neighbor_consensus import assess_motion
from .neighborhood_report import render_neighborhood

PROFILE=dict(id='neighborhood-candidates-v1', radius_px=96, max_neighbors=8,
    neighbor_inlier_px=2, candidate_residual_px=4, minimum_neighbors=4,
    max_continuations=8, max_extra_targets=2, provisional_seed_cycle_limit_px=4, new_target_cycle_limit_px=2,
    prediction_is_measurement=False, geometry_accepted=False, identity_verified=False)


def _need(ok,reason):
    if not ok:raise ValueError(reason)


def _number(value):
    return type(value) in (int,float) and math.isfinite(value)


def _photo_ok(alignment):
    if not isinstance(alignment,dict) or alignment.get('status')!='ACCEPTED':return False
    if alignment.get('reasons'):return False
    for name,threshold in [('train',.85),('holdout',.8)]:
        metrics=alignment.get(name) or {}
        if not all(_number(metrics.get(k)) for k in ('coverage','ncc','std_template','std_search')):return False
        if metrics['coverage']<.85 or metrics['ncc']<threshold or min(metrics['std_template'],metrics['std_search'])<3:return False
    error=alignment['holdout'].get('normalized_rmse')
    if not _number(error) or error>.7:return False
    pixels=alignment.get('pixel_evidence') or {}
    for key in ('template_uint8','warped_search_gray','valid_mask_uint8'):
        value=np.asarray(pixels.get(key))
        if value.shape!=(33,33) or value.dtype.kind not in 'iuf' or not np.isfinite(value).all():return False
    return True


def collect_measurements(plan,regions,*,budget=lambda:None):
    records=[]
    try:return _collect_measurements(plan,regions,records,budget=budget)
    except Exception as exc:
        exc.neighborhood_collected=records
        raise


def _collect_measurements(plan,regions,records,*,budget):
    """Pool forward image fits before strict cycle rejection; never refill by fit results."""
    frozen={r['region_id']:r for r in plan['regions']}
    _need(len(frozen)==len(plan['regions'])<=256,'NEIGHBORHOOD_PLAN_BOUND')
    seen=set()
    for area in regions:
        budget();rid=area['region_id'];_need(rid in frozen and rid not in seen,'NEIGHBORHOOD_REGION_IDENTITY');seen.add(rid)
        root=frozen[rid]
        _need(all(area[k]==root[k] for k in ('source_input_index','source_decode_index','decode_path')),'NEIGHBORHOOD_SOURCE_IDENTITY')
        if not area['region_legs']:continue
        source=plan['frame_domains'][str(root['source_decode_index'])]
        center=np.asarray(root['center_raw_xy'],float)
        roi=_roi(center-.5,source,33,full=True);polygon=_polygon(roi);center=polygon.mean(0)
        _need(area.get('template_roi')==roi,'NEIGHBORHOOD_SOURCE_ROI_CHANGED')
        stopped=False
        for ordinal,leg in enumerate(area['region_legs']):
            budget();target=leg['target_decode_index']
            _need(not stopped and ordinal+1<len(root['decode_path']) and target==root['decode_path'][ordinal+1]
                and leg['source_decode_index']==root['source_decode_index'],'NEIGHBORHOOD_TARGET_PREFIX')
            stopped=leg['status']!='ACCEPTED'
            record=dict(region_id=rid,source_input_index=root['source_input_index'],source_decode_index=root['source_decode_index'],
                target_decode_index=target,strict_status=leg['status'],strict_reasons=deepcopy(leg.get('reasons',[])),
                cycle_error_px=leg.get('cycle_error_px'),eligibility_reason='NOT_EVALUATED',measurement=None,neighborhood=None,
                continuation=None,continuation_eligible=False,geometry_accepted=False,identity_verified=False)
            records.append(record)
            forward=leg.get('forward') or {};alignment=forward.get('alignment') or {}
            if not _photo_ok(alignment):
                record['eligibility_reason']='NO_ACCEPTED_FORWARD_PIXEL_EVIDENCE';continue
            _need(forward.get('template_roi')==roi,'NEIGHBORHOOD_FORWARD_ROI_CHANGED')
            F=_global_raw(alignment['matrix_cv2x3'],roi,forward['search_roi'])
            saved=np.asarray(forward.get('matrix_global_raw3x3'))
            _need(saved.shape==(3,3) and saved.dtype.kind in 'iuf' and np.isfinite(saved).all()
                and np.allclose(F,saved,rtol=0,atol=1e-9),'NEIGHBORHOOD_FORWARD_MATRIX_CHANGED')
            linear=F[:2,:2];singular=np.linalg.svd(linear,compute_uv=False)
            _need(np.linalg.det(linear)>0 and singular[-1]>=.65 and singular[0]<=1.5
                and singular[0]/singular[-1]<=1.8,'NEIGHBORHOOD_FORWARD_SHAPE_CHANGED')
            quad=_map(F,polygon)
            _need(np.allclose(quad,np.asarray(forward['polygon_raw4x2']),rtol=0,atol=1e-9),'NEIGHBORHOOD_FORWARD_QUAD_CHANGED')
            try:_corners(quad,plan['frame_domains'][str(target)])
            except _Stop as exc:record['eligibility_reason']=exc.reason;continue
            actual=_map(F,center[None,:])[0]
            record['measurement']=dict(source_center_raw_xy=center.tolist(),source_polygon_raw4x2=polygon.tolist(),
                target_center_raw_xy=actual.tolist(),target_polygon_raw4x2=quad.tolist(),matrix_global_raw3x3=F.tolist(),
                train=deepcopy(alignment['train']),holdout=deepcopy(alignment['holdout']),
                target_mask_status=plan['frame_domains'][str(target)]['original_mask_status'],
                coordinate_semantics='center and corners derived from independently image-fitted area warp; not independent point matches')
            if leg['status']=='ACCEPTED':
                _need(_number(record['cycle_error_px']) and record['cycle_error_px']<=2
                    and _photo_ok((leg.get('backward') or {}).get('alignment')),'NEIGHBORHOOD_STRICT_REFERENCE_INVALID')
                record['eligibility_reason']='STRICT_REFERENCE'
            else:
                record['eligibility_reason']='RETAINED_FORWARD_IMAGE_CANDIDATE'
                cycle=record['cycle_error_px']
                record['continuation_eligible']=bool(_number(cycle) and 2<cycle<=4
                    and 'AREA_FIVE_POINT_CYCLE_THRESHOLD' in record['strict_reasons']
                    and _photo_ok((leg.get('backward') or {}).get('alignment')))
    return records


def classify_candidates(records,*,budget=lambda:None):
    """Freeze all image fits before decisions; no candidate is promoted into its own pool."""
    pools={}
    for row in records:
        budget()
        if row['measurement'] is None:continue
        key=(row['source_input_index'],row['source_decode_index'],row['target_decode_index'])
        m=row['measurement']
        pools.setdefault(key,[]).append(dict(region_id=row['region_id'],source_raw_xy=m['source_center_raw_xy'],
            target_raw_xy=m['target_center_raw_xy'],strict_accepted=row['strict_status']=='ACCEPTED'))
    out=deepcopy(records)
    for row in out:
        budget()
        if row['measurement'] is None or row['strict_status']=='ACCEPTED':continue
        m=row['measurement'];key=(row['source_input_index'],row['source_decode_index'],row['target_decode_index'])
        controls=m['source_polygon_raw4x2']+[m['source_center_raw_xy']]
        row['neighborhood']=assess_motion(row['region_id'],m['source_center_raw_xy'],controls,
            m['matrix_global_raw3x3'],pools[key],budget=budget)
    return out


def _continue(plan,root,seed,decoded,cv,*,budget):
    """New actual fits after one provisional seed; never replace source appearance."""
    start=root['source_decode_index'];source=plan['frame_domains'][str(start)]
    roi=_roi(np.asarray(root['center_raw_xy'])-.5,source,33,full=True);polygon=_polygon(roi);center=polygon.mean(0)
    def frame(index):
        d=plan['frame_domains'][str(index)];f=decoded[index]
        _need(all(f.get(k)==d[k] for k in ('path','width','height')) and isinstance(f.get('gray'),np.ndarray)
            and f['gray'].dtype==np.uint8 and f['gray'].shape==(d['height'],d['width']),'NEIGHBORHOOD_DECODED_IDENTITY')
        return d,f['gray']
    _,source_gray=frame(start);template=_crop(source_gray,roi);tm=_mask(roi,source)
    position=np.asarray(seed['measurement']['target_center_raw_xy'])
    path=root['decode_path'];offset=path.index(seed['target_decode_index'])
    targets=path[offset+1:offset+1+PROFILE['max_extra_targets']]
    out=dict(status='RUNNING',seed_target_decode_index=seed['target_decode_index'],seed_is_provisional=True,
        targets=list(targets),legs=[],native_calls=dict(matchTemplate=0,findTransformECCWithMask=0),
        actual_extra_targets_accepted=0,geometry_accepted=False,identity_verified=False)
    def align(t,s,mask,sm):
        try:a=align_area(t,s,mask,sm,cv,budget=budget)
        except Exception as exc:
            partial=getattr(exc,'partial_result',None)
            if partial:
                out['interrupted_alignment']=partial
                for k,n in partial['native_calls'].items():out['native_calls'][k]+=n
            raise
        for k,n in a['native_calls'].items():out['native_calls'][k]+=n
        return a
    try:
        for index in targets:
            budget();target,target_gray=frame(index)
            leg=dict(source_decode_index=start,target_decode_index=index,status='IN_PROGRESS',reasons=[],
                forward=None,backward=None,cycle_error_px=None,search_anchor_raw_xy=position.tolist())
            out['legs'].append(leg)
            try:
                search=_roi(position-.5,target,97)
                a=align(template,_crop(target_gray,search),tm,_mask(search,target))
                leg['forward']=dict(alignment=a,template_roi=roi,search_roi=search,matrix_global_raw3x3=None,polygon_raw4x2=None)
                if a['status']!='ACCEPTED':raise _Stop('FORWARD_'+a['status'],a['status'])
                F=_global_raw(a['matrix_cv2x3'],roi,search);quad=_map(F,polygon)
                leg['forward'].update(matrix_global_raw3x3=F.tolist(),polygon_raw4x2=quad.tolist())
                _corners(quad,target);mapped=_map(F,center[None,:])[0]
                reverse_roi=_roi(mapped-.5,target,33,full=True);reverse_search=_roi(center-.5,source,97)
                b=align(_crop(target_gray,reverse_roi),_crop(source_gray,reverse_search),_mask(reverse_roi,target),_mask(reverse_search,source))
                leg['backward']=dict(alignment=b,template_roi=reverse_roi,search_roi=reverse_search,matrix_global_raw3x3=None)
                if b['status']!='ACCEPTED':raise _Stop('BACKWARD_'+b['status'],b['status'])
                B=_global_raw(b['matrix_cv2x3'],reverse_roi,reverse_search);leg['backward']['matrix_global_raw3x3']=B.tolist()
                controls=np.vstack((polygon,center));returned=_map(B@F,controls)
                errors=np.linalg.norm(returned-controls,axis=1);cycle=float(errors.max())
                leg['cycle_points']=[dict(source_raw_xy=p.tolist(),returned_raw_xy=q.tolist(),error_px=float(e)) for p,q,e in zip(controls,returned,errors)]
                leg['cycle_error_px']=cycle
                if cycle>2:raise _Stop('AREA_FIVE_POINT_CYCLE_THRESHOLD','REJECTED')
                leg['status']='ACCEPTED';position=mapped;out['actual_extra_targets_accepted']+=1
            except _Stop as exc:
                leg['status']=exc.status;leg['reasons'].append(exc.reason)
                for name in ('forward','backward'):
                    leg['reasons'].extend(name.upper()+': '+s for s in ((leg.get(name) or {}).get('alignment') or {}).get('reasons',[]))
                break
        out['status']='NO_FUTURE_FRAME' if not targets else 'PROVISIONAL_CONTINUED' if out['actual_extra_targets_accepted'] else 'CONTINUATION_STOPPED'
        return out
    except Exception as exc:
        out['status']='INTERRUPTED';exc.neighborhood_partial=out;raise


def review_neighborhood(plan,regions,decoded,output,cv,*,budget=lambda:None):
    budget();output=Path(output);output.mkdir(parents=True,exist_ok=False);start=time.monotonic();artifacts=[]
    def write(name,value):
        path=output/name
        with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
        artifacts.append(path)
    result=dict(status='RUNNING',profile=deepcopy(PROFILE),records=[],counts={},native_calls=dict(matchTemplate=0,findTransformECCWithMask=0),
        geometry_accepted=False,identity_verified=False,new_model_points=0)
    try:
        collected=collect_measurements(plan,regions,budget=budget);result['records']=collected
        # The complete pool is fixed before any neighbor verdict or continuation.
        write('CANDIDATE_POOL.json',collected)
        records=classify_candidates(collected,budget=budget);result['records']=records
        eligible=[r for r in records if (r.get('neighborhood') or {}).get('status')=='NEIGHBOR_CONSISTENT_CANDIDATE' and r['continuation_eligible']]
        eligible.sort(key=lambda r:(r['source_input_index'],r['region_id'],r['target_decode_index']))
        selected=eligible[:PROFILE['max_continuations']]
        write('PLAN.json',dict(profile=PROFILE,candidate_region_targets=[dict(region_id=r['region_id'],target_decode_index=r['target_decode_index']) for r in selected],
            budget_not_selected=[dict(region_id=r['region_id'],target_decode_index=r['target_decode_index']) for r in eligible[PROFILE['max_continuations']:]],
            selection='fixed source/id/target order before continuation; not refilled after failure'))
        roots={r['region_id']:r for r in plan['regions']}
        for row in eligible:
            budget()
            if not any(row is r for r in selected):row['continuation']=dict(status='NOT_RUN_BUDGET');continue
            try:out=_continue(plan,roots[row['region_id']],row,decoded,cv,budget=budget)
            except Exception as exc:
                row['continuation']=getattr(exc,'neighborhood_partial',dict(status='INTERRUPTED'))
                for k,n in row['continuation'].get('native_calls',{}).items():result['native_calls'][k]+=n
                raise
            row['continuation']=out
            for k,n in out['native_calls'].items():result['native_calls'][k]+=n
        result['counts']=dict(measured_observations=len(records),strict_reference_observations=sum(r['eligibility_reason']=='STRICT_REFERENCE' for r in records),
            retained_image_candidates=sum(r['eligibility_reason']=='RETAINED_FORWARD_IMAGE_CANDIDATE' for r in records),
            no_image_support=sum(r['measurement'] is None for r in records),
            neighborhood_statuses=dict(Counter(r['neighborhood']['status'] for r in records if r['neighborhood'])),
            continuation_selected=len(selected),continuation_extra_accepted=sum((r['continuation'] or {}).get('actual_extra_targets_accepted',0) for r in records))
        result['status']='NEIGHBORHOOD_REVIEW_COMPLETE';result['wall_seconds']=time.monotonic()-start
        artifacts+=render_neighborhood(output,result,decoded,budget=budget)
        write('RESULT.json',result)
        return result,artifacts
    except Exception as exc:
        if hasattr(exc,'neighborhood_collected'):result['records']=exc.neighborhood_collected
        result.update(status='RESOURCE_LIMIT' if isinstance(exc,(TimeoutError,MemoryError)) else 'FAILED',reason=f'{type(exc).__name__}: {exc}')
        exc.neighborhood_result=result
        try:write('FAILURE.json',result)
        except Exception as write_error:
            if hasattr(exc,'add_note'):exc.add_note('neighborhood evidence failure: '+str(write_error))
        raise
