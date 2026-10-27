"""Separate dense-route XYZ evidence under the existing saved-camera gates."""
from collections import Counter
from copy import deepcopy
import math
import numpy as np
from .original_anchor_review import _camera, _numeric, _json
from .anchor_candidate_gate import _fit_point, _measure, _tier
from kozlov_lab.stereo.baseline import multiview_fixed as fixed
from kozlov_lab.stereo.baseline.radial_branch import propose_central_domain

RIGHTS = dict(geometry_accepted=False, geometry_acceptance=False, native_injected=False,
              identity_verified=False, new_cameras=0, new_model_points=0)


def _need(value, reason):
    if not value: raise ValueError(reason)


def _xy(value):
    _need(isinstance(value,(list,tuple)) and len(value)==2 and
          all(type(v) in (int,float) and math.isfinite(v) for v in value), 'INVALID_RAW_COORDINATES')
    return list(value)


def _classify_route(group, budget):
    """Reuse numerical gates without masquerading as original strong tracking."""
    observations=[]
    for observation in group['admitted_observations']:
        budget()
        domain=propose_central_domain(observation['params'],observation['width'],observation['height'],[observation['raw_pixels']])
        _need(domain['all_observations_invertible'], 'CENTRAL_BRANCH_INVERSE_REFUSED')
        observations.append(dict(deepcopy(observation),max_ideal_radius=domain['max_ideal_radius']))
    xyz=_fit_point(group['fit'],[o['image_id'] for o in observations])
    metrics=_measure(xyz,observations,budget)
    folds=group['leave_one_view_out']; _need(len(folds)==3,'LOO_FOLD_COUNT_MISMATCH')
    seen=set();loo=[]
    for fold in folds:
        budget();index=fold['heldout_input_index']
        held=next((o for o in observations if o['input_index']==index),None)
        _need(type(index) is int and held is not None and index not in seen,'LOO_HELDOUT_IDENTITY')
        _need(all(fold.get(k) is False for k in ('geometry_acceptance','native_injected','identity_verified')),'LOO_RIGHTS')
        train=[o for o in observations if o['input_index']!=index]
        _need(fold['heldout_image_id']==held['image_id'] and
              fold['training_input_indices']==[o['input_index'] for o in train],'LOO_CAMERA_BINDING')
        point=_fit_point(fold['fit'],[o['image_id'] for o in train])
        _measure(point,train,budget); all_metrics=_measure(point,observations,budget)
        prediction=fixed.evaluate(point,[held]);_need(prediction['status']=='EVALUATED','LOO_PROJECTION_REFUSED')
        loo.append(dict(heldout_input_index=index,error_px=prediction['residual_norms'][0],
                        all_depths_positive=True,metrics=all_metrics));seen.add(index)
    metrics['loo']=loo;metrics['max_loo_error_px']=max(f['error_px'] for f in loo)
    tier=_tier(3,metrics['max_raw_error_px'],metrics['max_loo_error_px'],metrics['measurement_ray_angle_deg'])
    return tier,xyz.tolist(),metrics


def assess_anchor_routes(measured, frames, masks, camera_result, *, budget=lambda:None):
    """At most 96 frozen records; no replacement after numerical rejection.

    LK evidence belongs to the caller's measured ledger. Here exact saved camera,
    source identity, foreground masks and all fixed-camera numerical gates are
    rechecked. No original strong label or old XYZ is inherited.
    """
    _need(len(frames)==len(masks),'FRAME_MASK_COUNT')
    records=measured.get('records');_need(isinstance(records,list) and len(records)<=96,'ROUTE_RECORD_BOUND')
    if not records:
        return dict(status='ANCHOR_ROUTE_GEOMETRY_COMPLETE',groups=[],
            classification=dict(status='ANCHOR_CANDIDATES_CLASSIFIED',rows=[],provenance='dense_anchor_route',**RIGHTS),
            counts=dict(records=0,groups=0,tier_counts={}),numeric_groups=0,
            provenance='dense_anchor_route',max_groups=96,**RIGHTS)
    registry={};mask_map={};decodes=set();models={}
    for frame,mask in zip(frames,masks):
        budget();idx=frame['input_index'];dec=frame['decode_index']
        _need(type(idx) is int and idx>=0 and idx not in registry and type(dec) is int and dec>=0 and dec not in decodes,'FRAME_IDENTITY')
        registry[idx]=frame;mask_map[idx]=mask;decodes.add(dec)
    for model in camera_result.get('models',[])+camera_result.get('unsupported_models',[]):
        budget();key=model['model_key'];_need(type(key) is str and key and key not in models,'MODEL_IDENTITY')
        cameras={}
        for table in ('cameras','unsupported_cameras'):
            for idx,camera in model.get(table,{}).items():
                idx=int(idx) if type(idx) is str and idx.isdigit() else idx
                _need(type(idx) is int and idx in registry and idx not in cameras,'CAMERA_INPUT_IDENTITY')
                cameras[idx]=camera
        models[key]=cameras
    groups=[];rows=[];seen=set();numeric_groups=0;last_budget_error=None
    def checked_budget():
        nonlocal last_budget_error
        try: budget()
        except Exception as exc:
            last_budget_error=exc;raise
    for record in records:
        checked_budget()
        key,tid,seg=(record.get(k) for k in ('model_key','track_id','identity_segment'))
        _need(type(key) is str and key in models and all(type(v) is int and v>=0 for v in (tid,seg)),'ROUTE_IDENTITY')
        gid=f'{key}:{tid}:{seg}';_need(gid not in seen,'DUPLICATE_ROUTE_GROUP');seen.add(gid)
        group=dict(group_id=gid,model_key=key,track_id=tid,identity_segment=seg,
            tracklet_id=record.get('tracklet_id'),point_id=record.get('point_id'),
            provenance='dense_anchor_route',source_kind='measured_dense_route',source_measurement=deepcopy(record),status='BLOCKED_ROUTE_MEASUREMENT',
            observations=[],admitted_observations=[],fit=None,evaluation=None,observability=None,
            leave_one_view_out=[],raw_predictions=[],radial_domains=[],strict_radial_domain_check=[],
            loo_status='NOT_EVALUATED',**RIGHTS)
        candidate=dict(group_id=gid,model_key=key,track_id=tid,identity_segment=seg,tier='REJECTED',
            xyz=None,metrics={},sources=[],source_indices=[],frame_ids=[],reasons=[],
            provenance='dense_anchor_route',source_kind='measured_dense_route',tracklet_id=record.get('tracklet_id'),scale_status='unknown',**RIGHTS)
        groups.append(group);rows.append(candidate)
        if record.get('status')!='MEASURED_THREE_VIEW_ANCHOR':
            candidate['reasons']=['ROUTE_NOT_MEASURED_THREE_VIEW_ANCHOR'];continue
        try:
            source=record.get('source_input_index');point=record.get('source_point',{});pid=record.get('point_id')
            _need(type(source) is int and source in registry,'SOURCE_INPUT_IDENTITY')
            _need(pid==f'track:{tid}:segment:{seg}:source:{source}' and point.get('point_id')==pid,'SOURCE_POINT_IDENTITY')
            rid=record.get('tracklet_id')
            _need(type(rid) is str and rid.startswith(pid+':edge:') and rid[len(pid+':edge:'):].isdigit(),'TRACKLET_POINT_IDENTITY')
            _need(all(type(point.get(k)) is int and point[k]==v for k,v in
                      (('track_id',tid),('identity_segment',seg),('source_input_index',source))), 'SOURCE_POINT_IDENTITY')
            _need(point.get('source_decode_index')==registry[source]['decode_index'],'SOURCE_DECODE_IDENTITY')
            obs=record.get('observations');_need(isinstance(obs,list) and len(obs)==3,'THREE_OBSERVATIONS_REQUIRED')
            _need(obs[0].get('input_index')==source and _xy(obs[0]['raw_xy'])==_xy(point['source_raw_xy']),'SOURCE_COORDINATE_BINDING')
            input_ids=set();image_ids=set();frame_ids=set()
            for ob in obs:
                checked_budget();idx=ob.get('input_index');_need(type(idx) is int and idx in registry and idx in models[key] and idx not in input_ids,'OBSERVATION_INPUT_IDENTITY')
                frame=registry[idx];camera=models[key][idx];_camera(camera,frame,idx)
                _need(type(ob.get('decode_index')) is int and ob['decode_index']==frame['decode_index'] and
                      ob.get('frame_id')==frame['frame_id'] and type(ob.get('image_id')) is int and ob['image_id']==camera['image_id'],'OBSERVATION_CAMERA_BINDING')
                _need(ob['frame_id'] not in frame_ids and ob['image_id'] not in image_ids,'THREE_DISTINCT_CAMERAS_REQUIRED')
                _need(all(type(ob.get(k)) is int and ob[k]==frame[k] for k in ('width','height')),'OBSERVATION_DIMENSIONS')
                raw=_xy(ob['raw_xy']);mask=mask_map[idx]
                _need(frame.get('original_mask_status')=='OK' and isinstance(mask,np.ndarray) and mask.dtype==np.uint8 and mask.shape==(frame['height'],frame['width']),'ORIGINAL_MASK_REQUIRED')
                _need(np.isin(mask,(0,255)).all(),'NONBINARY_ORIGINAL_MASK')
                _need(.5<=raw[0]<=frame['width']-.5 and .5<=raw[1]<=frame['height']-.5,'RAW_PIXEL_OUTSIDE_IMAGE')
                x,y=map(math.floor,raw);_need(mask[y,x]==255,'OUTSIDE_ORIGINAL_FOREGROUND')
                observation=dict(input_index=idx,frame_id=frame['frame_id'],image_id=camera['image_id'],
                    raw_pixels=raw,width=frame['width'],height=frame['height'],camera_model='SIMPLE_RADIAL',
                    evidence_kind='MEASURED_DENSE_ROUTE',source_kind='measured_dense_route',**{k:deepcopy(camera[k]) for k in ('params','R','t')})
                group['observations'].append(dict(input_index=idx,frame_id=frame['frame_id'],source_xy=raw,
                    status='MEASURED_DENSE_ROUTE_FOREGROUND',evidence_kind='MEASURED_DENSE_ROUTE',reasons=[]))
                group['admitted_observations'].append(observation)
                input_ids.add(idx);image_ids.add(ob['image_id']);frame_ids.add(ob['frame_id'])
            candidate['sources']=[dict(input_index=o['input_index'],frame_id=o['frame_id'],image_id=o['image_id'],
                raw_pixels=deepcopy(o['raw_pixels']),evidence_kind='MEASURED_DENSE_ROUTE',source_kind='measured_dense_route') for o in group['admitted_observations']]
            candidate['source_indices']=[o['input_index'] for o in group['admitted_observations']]
            candidate['frame_ids']=[o['frame_id'] for o in group['admitted_observations']]
            numeric_groups+=1;_numeric(group,checked_budget)
            # Replace legacy seed-specific prose from the reused numerical helper.
            for fold in group['leave_one_view_out']:
                fold['scope']='fixed saved cameras; dense-route selected pixels; internal consistency, not independent identity truth'
            tier,xyz,metrics=_classify_route(group,checked_budget)
            candidate.update(tier=tier,xyz=xyz if tier!='REJECTED' else None,metrics=metrics,
                branch_scope='CENTRAL_BRANCH_CONDITIONAL' if any(o['params'][3]<0 for o in group['admitted_observations']) else 'MONOTONIC_RADIAL_CONDITIONAL')
            if tier=='REJECTED':candidate['reasons']=['PIXEL_ERROR_OR_MEASURED_ANGLE_THRESHOLD']
        except (ValueError,KeyError,TypeError,ArithmeticError,np.linalg.LinAlgError) as exc:
            if exc is last_budget_error:raise
            group.update(status='ROUTE_GEOMETRY_REFUSED',reason=str(exc))
            candidate.update(tier='REJECTED',xyz=None,metrics={},reasons=[str(exc)])
    return _json(dict(status='ANCHOR_ROUTE_GEOMETRY_COMPLETE',groups=groups,
        classification=dict(status='ANCHOR_CANDIDATES_CLASSIFIED',rows=rows,provenance='dense_anchor_route',**RIGHTS),
        counts=dict(records=len(records),groups=len(groups),tier_counts=dict(Counter(r['tier'] for r in rows))),
        numeric_groups=numeric_groups,provenance='dense_anchor_route',max_groups=96,**RIGHTS))
