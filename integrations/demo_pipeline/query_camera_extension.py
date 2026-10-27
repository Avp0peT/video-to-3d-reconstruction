"""Conditional XYZ from disjoint short tracks and fixed query-camera hypotheses."""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import math
import numpy as np
from .original_anchor_review import _camera, _numeric, _json
from .anchor_candidate_gate import _fit_point, _measure, _tier
from kozlov_lab.stereo.baseline import multiview_fixed as fixed
from kozlov_lab.stereo.baseline.radial_branch import propose_central_domain

RIGHTS=dict(geometry_accepted=False,geometry_acceptance=False,camera_pose_established=False,
            identity_verified=False,native_injected=False,new_native_points=0,new_model_points=0,new_cameras=0)
MAX_GROUPS=96


def _need(value,reason):
    if not value:raise ValueError(reason)


def _array(value,shape):
    _need(not any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat),'BOOLEAN_GEOMETRY')
    raw=np.asarray(value)
    _need(raw.shape==shape and raw.dtype.kind in 'iuf' and np.isfinite(raw).all(),'INVALID_NUMERIC_GEOMETRY')
    return raw.astype(float)


def _classify(group,budget):
    observations=[]
    for ob in group['admitted_observations']:
        budget();domain=propose_central_domain(ob['params'],ob['width'],ob['height'],[ob['raw_pixels']])
        _need(domain['all_observations_invertible'],'CENTRAL_BRANCH_INVERSE_REFUSED')
        observations.append(dict(ob,max_ideal_radius=domain['max_ideal_radius']))
    xyz=_fit_point(group['fit'],[o['image_id'] for o in observations]);metrics=_measure(xyz,observations,budget)
    folds=group['leave_one_view_out'];_need(len(folds)==len(observations),'LOO_FOLD_COUNT')
    seen=set();loo=[]
    for fold in folds:
        budget();idx=fold['heldout_input_index'];held=next((o for o in observations if o['input_index']==idx),None)
        _need(held is not None and idx not in seen,'LOO_IDENTITY')
        train=[o for o in observations if o['input_index']!=idx]
        _need(fold['heldout_image_id']==held['image_id'] and fold['training_input_indices']==[o['input_index'] for o in train],'LOO_TRAIN_BINDING')
        _need(all(fold.get(k) is False for k in ('geometry_acceptance','native_injected','identity_verified')),'LOO_RIGHTS')
        point=_fit_point(fold['fit'],[o['image_id'] for o in train])
        _measure(point,train,budget);diagnostic=_measure(point,observations,budget)
        prediction=fixed.evaluate(point,[held]);_need(prediction['status']=='EVALUATED','LOO_PROJECTION_REFUSED')
        loo.append(dict(heldout_input_index=idx,error_px=prediction['residual_norms'][0],metrics=diagnostic));seen.add(idx)
    metrics['loo']=loo;metrics['max_loo_error_px']=max(r['error_px'] for r in loo)
    tier=_tier(len(observations),metrics['max_raw_error_px'],metrics['max_loo_error_px'],metrics['measurement_ray_angle_deg'])
    return tier,xyz.tolist(),metrics


def assess_query_extensions(ledger,frames,masks,camera_result,query_result,*,budget=lambda:None):
    """Never reuse any camera-fitting anchor as a new point, including holdouts.

    Numerical camera/view IDs are synthetic, local to each hypothesis group.
    Their camera_ref fields retain actual saved identities and query decode IDs.
    Query foreground is unknown, even when conditional numerical gates pass.
    """
    _need(len(frames)==len(masks),'FRAME_MASK_COUNT')
    records=query_result.get('records')
    _need(isinstance(records,list) and len(records)<=96,'QUERY_RECORD_BOUND')
    tracks=ledger.get('records');_need(isinstance(tracks,list) and len(tracks)<=1536,'TRACKLET_BOUND')
    result=dict(status='QUERY_CAMERA_EXTENSION_COMPLETE',groups=[],classifications=[],counts={},
        numeric_groups=0,max_numeric_groups=MAX_GROUPS,selection=[],
        query_foreground_status='PER_GROUP_ORIGINAL_MASK_STATUS',
        scope='fixed conditional query cameras and disjoint measured short tracks; internal numerical consistency only',**RIGHTS)
    if not tracks or not records:
        result['counts']=dict(groups=0,conditional_points=0,statuses={});return result
    registry={};bydecode={};maskmap={};models={}
    for frame,mask in zip(frames,masks):
        budget();idx=frame.get('input_index');dec=frame.get('decode_index')
        _need(type(idx) is int and idx>=0 and idx not in registry and type(dec) is int and dec>=0 and dec not in bydecode,'FRAME_IDENTITY')
        registry[idx]=frame;bydecode[dec]=frame;maskmap[idx]=mask
    for model in camera_result.get('models',[])+camera_result.get('unsupported_models',[]):
        key=model['model_key'];_need(type(key) is str and key and key not in models,'MODEL_IDENTITY');cams={}
        for name in ('cameras','unsupported_cameras'):
            for idx,cam in model.get(name,{}).items():
                idx=int(idx) if type(idx) is str and idx.isdigit() else idx
                _need(type(idx) is int and idx in registry and idx not in cams,'SAVED_CAMERA_IDENTITY');cams[idx]=cam
        models[key]=cams
    queries=defaultdict(dict);seen_queries=set()
    for record in records:
        budget();key=record.get('model_key');dec=record.get('query_decode_index');cohort=record.get('cohort')
        _need(key in models and type(dec) is int and dec>=0 and type(cohort) is str and cohort,'QUERY_IDENTITY')
        _need((key,cohort,dec) not in seen_queries,'DUPLICATE_QUERY_RECORD');seen_queries.add((key,cohort,dec))
        _need(all(type(record.get(k)) is int and 1<=record[k]<=16384 for k in ('width','height')),'QUERY_DIMENSIONS')
        points=record.get('points');assessment=record.get('assessment',{})
        _need(isinstance(points,list) and len(points)<=768 and assessment.get('points')==points,'ASSESSMENT_POINT_BINDING')
        ids=[p.get('point_id') for p in points]
        _need(all(type(i) is str and i for i in ids) and len(set(ids))==len(ids),'ANCHOR_IDENTITIES')
        train=assessment.get('train_ids');hold=assessment.get('holdout_ids')
        _need(isinstance(train,list) and isinstance(hold,list) and all(type(i) is str for i in train+hold) and
              len(set(train))==len(train) and len(set(hold))==len(hold) and not set(train)&set(hold) and set(train)|set(hold)==set(ids),'ANCHOR_SPLIT_COVERAGE')
        _need(not (dec in bydecode and bydecode[dec]['input_index'] in models[key]),'QUERY_ALREADY_SAVED_CAMERA')
        for fit in assessment.get('models',[]):
            budget();lens=fit.get('model');_need(lens in ('SIMPLE_PINHOLE','SIMPLE_RADIAL'),'UNSUPPORTED_LENS_HYPOTHESIS')
            if fit.get('status')!='CONDITIONAL_CAMERA_CANDIDATE':continue
            bucket=queries[(key,cohort,lens)];_need(dec not in bucket,'DUPLICATE_LENS_QUERY')
            K=_array(fit.get('K'),(3,3));R=_array(fit.get('R'),(3,3));t=_array(fit.get('t'),(3,))
            f=fit.get('f');k=fit.get('k')
            _need(type(f) in (int,float) and math.isfinite(f) and f>0 and type(k) in (int,float) and math.isfinite(k),'INVALID_QUERY_FOCAL_RADIAL')
            _need(lens!='SIMPLE_PINHOLE' or k==0,'PINHOLE_NONZERO_RADIAL')
            expected=np.array([[f,0.,record['width']/2],[0.,f,record['height']/2],[0.,0.,1.]])
            _need(np.allclose(K,expected,rtol=0,atol=1e-8),'FIXED_PRINCIPAL_POINT_OR_K_MISMATCH')
            _need(np.allclose(R@R.T,np.eye(3),rtol=0,atol=1e-8) and np.isclose(np.linalg.det(R),1,rtol=0,atol=1e-8),'INVALID_QUERY_ROTATION')
            bucket[dec]=dict(decode_index=dec,width=record['width'],height=record['height'],
                params=[float(f),float(K[0,2]),float(K[1,2]),float(k)],R=R.tolist(),t=t.tolist(),
                anchor_group_ids=sorted(ids),train_group_ids=list(train),holdout_group_ids=list(hold),
                camera_ref=dict(kind='query',model_key=key,decode_index=dec,cohort=cohort,lens_hypothesis=lens),
                principal_point_assumption='fixed image center; square pixels; zero skew')
    candidates=[];seen_tracks=set()
    for track in tracks:
        budget();rid=track.get('tracklet_id');_need(type(rid) is str and rid and rid not in seen_tracks,'TRACKLET_IDENTITY');seen_tracks.add(rid)
        if track.get('status')!='CANDIDATE_2D_TRACKLET':continue
        tid,seg,source=(track.get(k) for k in ('track_id','identity_segment','source_input_index'))
        _need(all(type(v) is int and v>=0 for v in (tid,seg,source)) and source in registry,'TRACKLET_SOURCE_IDENTITY')
        pid=f'track:{tid}:segment:{seg}:source:{source}';point=track.get('source_point',{});edge=track.get('edge_ordinal')
        _need(type(edge) is int and edge>=0 and track.get('point_id')==pid and point.get('point_id')==pid and rid==f'{pid}:edge:{edge}','TRACKLET_POINT_IDENTITY')
        _need(all(type(point.get(k)) is int and point[k]==v for k,v in (('track_id',tid),('identity_segment',seg),('source_input_index',source))),'SOURCE_POINT_IDENTITY')
        obs=track.get('observations');legs=track.get('verified_legs')
        _need(type(legs) is int and 2<=legs<=8 and isinstance(obs,list) and len(obs)==legs+1,'VERIFIED_PREFIX_LENGTH')
        indices=[o.get('decode_index') for o in obs]
        _need(all(type(i) is int and i>=0 for i in indices) and len(set(indices))==len(indices),'PREFIX_DECODE_IDENTITY')
        direction=1 if indices[-1]>indices[0] else -1
        _need(all(b-a==direction for a,b in zip(indices,indices[1:])),'NONCONSECUTIVE_PREFIX')
        frame=registry[source]
        _need(indices[0]==point.get('source_decode_index')==frame['decode_index'] and np.array_equal(_array(point.get('source_raw_xy'),(2,)),_array(obs[0].get('raw_xy'),(2,))),'SOURCE_COORDINATE_BINDING')
        for ob in obs:
            raw=_array(ob.get('raw_xy'),(2,));_need(all(type(ob.get(k)) is int and ob[k]>0 for k in ('width','height')),'OBSERVATION_DIMENSIONS')
            _need(.5<=raw[0]<=ob['width']-.5 and .5<=raw[1]<=ob['height']-.5,'OBSERVATION_OUTSIDE_IMAGE')
            actual=bydecode.get(ob['decode_index'])
            if actual is not None:
                _need((ob['width'],ob['height'],ob.get('original_mask_status'))==
                      (actual['width'],actual['height'],actual.get('original_mask_status')),'KNOWN_FRAME_DOMAIN_MISMATCH')
                if 'frame_id' in ob:_need(ob['frame_id']==actual['frame_id'],'KNOWN_FRAME_IDENTITY_MISMATCH')
                if 'input_index' in ob:_need(type(ob['input_index']) is int and ob['input_index']==actual['input_index'],'KNOWN_INPUT_IDENTITY_MISMATCH')
                if actual.get('original_mask_status')=='OK':
                    mask=maskmap[actual['input_index']]
                    _need(isinstance(mask,np.ndarray) and mask.dtype==np.uint8 and mask.shape==(actual['height'],actual['width']) and np.isin(mask,(0,255)).all(),
                          'ORIGINAL_SOURCE_MASK_REQUIRED' if ob is obs[0] else 'KNOWN_QUERY_MASK_REQUIRED')
                    x,y=np.floor(raw).astype(int)
                    _need(mask[y,x]==255,'SOURCE_NOT_ORIGINAL_FOREGROUND' if ob is obs[0] else 'KNOWN_QUERY_NOT_FOREGROUND')
                    if 'label' in ob:_need(ob['label']=='foreground','KNOWN_QUERY_LABEL_MISMATCH')
            else:
                _need(ob.get('original_mask_status')!='OK','UNBOUND_ORIGINAL_QUERY_MASK_CLAIM')
        for (key,cohort,lens),camera_map in sorted(queries.items()):
            budget()
            if source not in models[key]:continue
            base_gid=f'{key}:{tid}:{seg}';selected=[(o,camera_map[o['decode_index']]) for o in obs[1:] if o['decode_index'] in camera_map]
            gid=f'{key}:{cohort}:{lens}:{rid}'
            group=dict(group_id=gid,base_group_id=base_gid,model_key=key,cohort=cohort,lens_hypothesis=lens,
                tracklet_id=rid,track_id=tid,identity_segment=seg,source_input_index=source,
                source_point=deepcopy(point),status='BLOCKED_INSUFFICIENT_QUERY_CAMERAS',reasons=[],
                admitted_observations=[],observations=[],fit=None,evaluation=None,observability=None,
                leave_one_view_out=[],raw_predictions=[],radial_domains=[],strict_radial_domain_check=[],
                loo_status='NOT_EVALUATED',query_foreground_status='UNKNOWN_NOT_SEGMENTED',**RIGHTS)
            candidates.append(group)
            if len(selected)<2:
                group['reasons']=['AT_LEAST_TWO_SUPPORTED_QUERY_CAMERAS_REQUIRED'];continue
            excluded=sorted({i for _,c in selected for i in c['anchor_group_ids']})
            group['excluded_camera_anchor_group_ids']=excluded
            if base_gid in excluded:
                group.update(status='EXCLUDED_CAMERA_FITTING_ANCHOR',reasons=['ALL_TRAIN_AND_HOLDOUT_ANCHORS_EXCLUDED']);continue
            source_camera=models[key][source];_camera(source_camera,frame,source)
            mask=maskmap[source]
            _need(frame.get('original_mask_status')=='OK' and isinstance(mask,np.ndarray) and mask.dtype==np.uint8 and mask.shape==(frame['height'],frame['width']) and np.isin(mask,(0,255)).all(),'ORIGINAL_SOURCE_MASK_REQUIRED')
            raw=_array(point['source_raw_xy'],(2,));x,y=np.floor(raw).astype(int)
            _need((obs[0]['width'],obs[0]['height'])==(frame['width'],frame['height']) and
                  .5<=raw[0]<=frame['width']-.5 and .5<=raw[1]<=frame['height']-.5 and mask[y,x]==255,'SOURCE_NOT_ORIGINAL_FOREGROUND')
            source_ob=dict(input_index=0,image_id=1,frame_id=f'synthetic:saved:{key}:{source}',raw_pixels=raw.tolist(),
                width=frame['width'],height=frame['height'],camera_model='SIMPLE_RADIAL',
                camera_ref=dict(kind='saved',model_key=key,input_index=source,decode_index=frame['decode_index'],
                                image_id=source_camera['image_id'],frame_id=frame['frame_id']),
                evidence_kind='ORIGINAL_SOURCE_FOREGROUND',**{k:deepcopy(source_camera[k]) for k in ('params','R','t')})
            group['admitted_observations'].append(source_ob)
            for ordinal,(ob,cam) in enumerate(selected,1):
                _need((ob['width'],ob['height'])==(cam['width'],cam['height']),'QUERY_OBSERVATION_DIMENSION_MISMATCH')
                group['admitted_observations'].append(dict(input_index=ordinal,image_id=ordinal+1,
                    frame_id=f"synthetic:query:{key}:{cohort}:{lens}:{ob['decode_index']}",
                    raw_pixels=deepcopy(ob['raw_xy']),width=ob['width'],height=ob['height'],camera_model='SIMPLE_RADIAL',
                    params=deepcopy(cam['params']),R=deepcopy(cam['R']),t=deepcopy(cam['t']),
                    camera_ref=dict(kind='query',model_key=key,decode_index=ob['decode_index'],cohort=cohort,lens_hypothesis=lens),
                    evidence_kind='MEASURED_SHORT_PREFIX_WITH_CONDITIONAL_QUERY_CAMERA',
                    original_mask_status=ob.get('original_mask_status','UNKNOWN_NOT_PROVIDED')))
            mask_statuses=[ob.get('original_mask_status','UNKNOWN_NOT_PROVIDED') for ob,_ in selected]
            group.update(status='READY_FOR_CONDITIONAL_XYZ',query_cameras=[deepcopy(c) for _,c in selected],
                query_foreground_status='VERIFIED_ORIGINAL_MASKS' if all(s=='OK' for s in mask_statuses) else 'PARTIALLY_VERIFIED_OR_UNKNOWN' if any(s=='OK' for s in mask_statuses) else 'UNKNOWN_NOT_SEGMENTED')
    buckets=defaultdict(deque)
    for group in sorted(candidates,key=lambda g:(g['model_key'],g['cohort'],g['lens_hypothesis'],g['tracklet_id'])):
        if group['status']=='READY_FOR_CONDITIONAL_XYZ':buckets[(group['model_key'],group['cohort'],group['lens_hypothesis'])].append(group)
    queue=[]
    while any(buckets.values()):
        budget()
        for key in sorted(buckets):
            if buckets[key]:queue.append(buckets[key].popleft())
    result['selection']=[g['group_id'] for g in queue[:MAX_GROUPS]]
    for group in queue[MAX_GROUPS:]:group.update(status='NOT_EVALUATED_GROUP_BUDGET',reasons=['MAX_96_FROZEN_GROUPS'])
    budget_error=None
    def checked_budget():
        nonlocal budget_error
        try:budget()
        except Exception as exc:budget_error=exc;raise
    rowmap={}
    for group in candidates:
        rowmap[group['group_id']]=dict(group_id=group['group_id'],base_group_id=group['base_group_id'],
            model_key=group['model_key'],cohort=group['cohort'],lens_hypothesis=group['lens_hypothesis'],
            tracklet_id=group['tracklet_id'],track_id=group['track_id'],identity_segment=group['identity_segment'],
            source_input_index=group['source_input_index'],source_point=deepcopy(group['source_point']),
            tier='REJECTED',xyz=None,metrics={},reasons=list(group['reasons']),sources=[],
            provenance='conditional_query_camera_extension',query_foreground_status=group['query_foreground_status'],
            conditional_camera_dependency=True,independent_geometry_validation=False,**RIGHTS)
    for group in queue[:MAX_GROUPS]:
        checked_budget();row=rowmap[group['group_id']];result['numeric_groups']+=1
        try:
            _numeric(group,checked_budget)
            for fold in group['leave_one_view_out']:fold['scope']='conditional saved/query cameras; disjoint camera-fitting anchors; shared upstream data, not independent truth'
            tier,xyz,metrics=_classify(group,checked_budget)
            row.update(tier=tier,xyz=xyz if tier!='REJECTED' else None,metrics=metrics,
                branch_scope='CENTRAL_BRANCH_CONDITIONAL' if any(o['params'][3]<0 for o in group['admitted_observations']) else 'MONOTONIC_RADIAL_CONDITIONAL',
                sources=[dict(raw_pixels=deepcopy(o['raw_pixels']),camera_ref=deepcopy(o['camera_ref']),
                              evidence_kind=o['evidence_kind']) for o in group['admitted_observations']])
            group['status']='CONDITIONAL_XYZ_EVALUATED'
            if tier=='REJECTED':row['reasons']=['PIXEL_ERROR_OR_MEASURED_ANGLE_THRESHOLD']
        except (ValueError,KeyError,TypeError,ArithmeticError,np.linalg.LinAlgError) as exc:
            if exc is budget_error:raise
            group.update(status='CONDITIONAL_XYZ_REFUSED',reasons=[str(exc)]);row['reasons']=[str(exc)]
    grouped=defaultdict(list)
    for group in candidates:grouped[(group['model_key'],group['cohort'],group['lens_hypothesis'])].append(rowmap[group['group_id']])
    for (key,cohort,lens),rows in sorted(grouped.items()):
        result['classifications'].append(dict(model_key=key,cohort=cohort,lens_model=lens,
            classification=dict(status='ANCHOR_CANDIDATES_CLASSIFIED',rows=rows,
                counts=dict(groups=len(rows),exportable=sum(r['tier'] in ('STRICT_CONDITIONAL','RELAXED_CONDITIONAL') for r in rows)),**RIGHTS),**RIGHTS))
    result['groups']=candidates
    result['counts']=dict(groups=len(candidates),conditional_points=sum(r['xyz'] is not None for r in rowmap.values()),
        statuses=dict(Counter(g['status'] for g in candidates)),tier_counts=dict(Counter(r['tier'] for r in rowmap.values())))
    return _json(result)
