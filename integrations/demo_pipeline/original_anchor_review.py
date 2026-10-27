"""Pure original-foreground saved-camera diagnostics; never geometry acceptance."""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import math
from kozlov_lab.stereo.baseline import radial_diagnostics as radial
import numpy as np
from kozlov_lab.stereo.baseline.radial_branch import propose_central_domain
from kozlov_lab.stereo.baseline import multiview_fixed as fixed, observability

PROFILE=dict(profile_id='original-anchor-review-v3',max_groups=96,batch_size=32,max_batches=3,max_observations=12,max_iterations=20,
    selection='3+ admitted views before 2 views; model_key round robin within each tier; track_id then identity_segment; no residual selection',
    seed_policy='explicit repeated seed-pair verification; original strong and null errors unchanged',
    strict_policy='legacy twice normalized raw corner radius declared as ideal domain; refusal retained',
    central_policy='central_branch_hypothesis-v1; physical branch unverified',
    geometry_acceptance=False,native_injected=False,identity_verified=False)

RIGHTS=dict(geometry_acceptance=False,native_injected=False,identity_verified=False)


def _json(value):
    if isinstance(value,np.ndarray):return _json(value.tolist())
    if isinstance(value,np.generic):return _json(value.item())
    if isinstance(value,dict):return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite diagnostic value')
    if value is None or type(value) in (str,bool,int,float):return value
    raise ValueError('unsupported diagnostic value')


def _number(value):return type(value) in (int,float) and math.isfinite(value)


def _camera(camera,frame,index):
    if not isinstance(camera,dict) or camera.get('input_index')!=index or camera.get('frame_id')!=frame['frame_id']:
        raise ValueError('CAMERA_FRAME_IDENTITY_MISMATCH')
    if type(camera.get('image_id')) is not int or camera['image_id']<=0:raise ValueError('INVALID_IMAGE_ID')
    if camera.get('camera_model')!='SIMPLE_RADIAL':raise ValueError('UNSUPPORTED_CAMERA_MODEL')
    if (camera.get('width'),camera.get('height'))!=(frame['width'],frame['height']):raise ValueError('CAMERA_DIMENSIONS_MISMATCH')
    params=radial._array(camera['params'],(4,));rotation=radial._array(camera['R'],(3,3));translation=radial._array(camera['t'],(3,))
    if params.shape!=(4,) or rotation.shape!=(3,3) or translation.shape!=(3,) or not all(np.isfinite(v).all() for v in (params,rotation,translation)) or params[0]<=0:
        raise ValueError('INVALID_CAMERA_NUMERICS')
    if not np.allclose(rotation@rotation.T,np.eye(3),rtol=0,atol=1e-8) or not np.isclose(np.linalg.det(rotation),1.,rtol=0,atol=1e-8):raise ValueError('INVALID_ROTATION')


def _numeric(group,budget):
    observations=[];domains=[];strict=[]
    for row in group['admitted_observations']:
        budget()
        f,cx,cy,k=map(float,row['params'])
        old_radius=(2*max(math.hypot((x-cx)/f,(y-cy)/f) for x,y in
                    [(0.,0.),(row['width'],0.),(0.,row['height']),(row['width'],row['height'])])) if k<0 else None
        strict_row=dict(input_index=row['input_index'],status='STRICT_DOMAIN_EVALUATED',
                        max_ideal_radius=old_radius,policy='legacy twice normalized raw corner radius declared as ideal domain; not an inverse-derived bound')
        try:radial.undistort_simple_radial([row['raw_pixels']],row['params'],max_ideal_radius=old_radius)
        except (ValueError,ArithmeticError) as exc:strict_row.update(status='RADIAL_DOMAIN_REFUSED',reason=str(exc))
        strict.append(strict_row)
        domain=propose_central_domain(row['params'],row['width'],row['height'],[row['raw_pixels']])
        domains.append(dict(input_index=row['input_index'],image_id=row['image_id'],**domain))
        observations.append(dict(row,max_ideal_radius=domain['max_ideal_radius']))
    group['strict_radial_domain_check']=strict
    group['radial_domains']=domains
    if not all(d['all_observations_invertible'] for d in domains):
        group.update(status='RADIAL_DOMAIN_REFUSED');return
    budget();fit=fixed.fit(observations,max_iterations=PROFILE['max_iterations']);group['fit']=fit
    group['status']='CONDITIONAL_DIAGNOSTIC_EVALUATED' if fit.get('xyz') is not None else 'NUMERIC_REFUSED'
    if fit.get('xyz') is not None:
        group['evaluation']=fixed.evaluate(fit['xyz'],observations)
        group['observability']=observability.analyze(fit['xyz'],observations)
        group['raw_predictions']=[dict(input_index=o['input_index'],measured_raw_pixels=o['raw_pixels'],
            predicted_raw_pixels=fixed.project(fit['xyz'],o)['pixels']) for o in observations]
    if len(observations)>=3:
        group['loo_status']='ALL_VIEWS_ATTEMPTED'
        for i,held in enumerate(observations):
            train=observations[:i]+observations[i+1:]
            budget();training_fit=fixed.fit(train,max_iterations=PROFILE['max_iterations'])
            prediction=None;projected=None
            if training_fit.get('xyz') is not None:
                prediction=fixed.evaluate(training_fit['xyz'],[held])
                try:projected=fixed.project(training_fit['xyz'],held)['pixels']
                except (ValueError,ArithmeticError,KeyError,TypeError,np.linalg.LinAlgError):pass
            group['leave_one_view_out'].append(dict(heldout_input_index=held['input_index'],heldout_image_id=held['image_id'],
                training_input_indices=[o['input_index'] for o in train],fit=training_fit,
                measured_raw_pixels=held['raw_pixels'],predicted_raw_pixels=projected,prediction=prediction,
                scope='fixed saved cameras; admission conditioned on pair verification, including heldout successor; internal consistency, not independent validation',**RIGHTS))
        predicted=sum(fold['prediction'] is not None and fold['prediction']['status']=='EVALUATED'
                      and fold['predicted_raw_pixels'] is not None for fold in group['leave_one_view_out'])
        group['loo_status']=('ALL_VIEWS_PREDICTED' if predicted==len(observations) else
                             'PARTIALLY_REFUSED' if predicted else 'ALL_VIEWS_REFUSED')


def review_original_anchors(frames,masks,mapped_tracking,camera_result,budget=lambda:None,*,seed_validation=None,evaluate=True,progress=lambda event:None):
    """No files, native reads, fitting of cameras, new correspondences or points.

    Three-view groups precede two-view groups, with model round robin in each tier. All source
    observations retain explicit eligibility reasons, including budget omissions.
    """
    if not isinstance(frames,list) or len(frames)!=len(masks):raise ValueError('complete aligned frames/masks required')
    registry={};mask_map={};frame_ids=set()
    for frame,mask in zip(frames,masks):
        index=frame['input_index']
        if type(index) is not int or index<0 or index in registry or frame['frame_id'] in frame_ids:raise ValueError('unique frame identities required')
        registry[index]=frame;mask_map[index]=mask;frame_ids.add(frame['frame_id'])
    models={};camera_count=0
    for model in camera_result.get('models',[])+camera_result.get('unsupported_models',[]):
        key=model['model_key']
        if type(key) is not str or not key or key in models:raise ValueError('unique model keys required')
        cameras={}
        for table in ('cameras','unsupported_cameras'):
            for index,camera in model.get(table,{}).items():
                if type(index) is str and index.isdigit():index=int(index)
                if type(index) is not int or index in cameras:raise ValueError('unique integer camera map required')
                cameras[index]=camera
        models[key]=cameras;camera_count+=len(cameras)
    certificates={}
    if seed_validation is not None:
        from .seed_pair_validation import validated_certificates
        certificates=validated_certificates(seed_validation)
    groups=[];seen_tracks=set();used_certificates=set()
    for track in mapped_tracking['tracks']:
        tid=track['track_id']
        if type(tid) is not int or tid<0 or tid in seen_tracks:raise ValueError('unique track IDs required')
        seen_tracks.add(tid);segments=defaultdict(list)
        previous=-1;previous_segment=-1
        for observation in track['observations']:
            index=observation['frame_index'];segment=observation['identity_segment']
            if type(index) is not int or index<=previous or index not in registry or type(segment) is not int or segment<0 or segment<previous_segment:raise ValueError('valid ordered mapped observations and nonregressing identity segments required')
            previous=index;previous_segment=segment;segments[segment].append(observation)
        for key,cameras in sorted(models.items()):
            for segment,observations in sorted(segments.items()):
                if not any(o['frame_index'] in cameras for o in observations):continue
                group=dict(group_id=f'{key}:{tid}:{segment}',model_key=key,track_id=tid,identity_segment=segment,
                    status='INSUFFICIENT_ORIGINAL_ANCHORS',observations=[],admitted_observations=[],radial_domains=[],
                    fit=None,evaluation=None,observability=None,leave_one_view_out=[],raw_predictions=[],
                    loo_status='NOT_EVALUATED_INSUFFICIENT_VIEWS',strict_radial_domain_check=[],
                    branch_scope='independent central-branch hypothesis; does not replace strict-domain refusal; physical root unverified',
                    new_native_model_points=0,**RIGHTS)
                for obs in observations:
                    budget();index=obs['frame_index'];frame=registry[index];reasons=[];camera=cameras.get(index)
                    if frame.get('original_mask_status')!='OK' or obs.get('label')!='foreground' or obs.get('target_verified') is not True:reasons.append('UNKNOWN_SOURCE')
                    if obs.get('strong') is not True:reasons.append('NON_STRONG')
                    for field,limit in [('fb_error_px',1.),('lk_error',25.)]:
                        if not _number(obs.get(field)) or not 0<=obs[field]<=limit:reasons.append('INVALID_'+field.upper())
                    xy=obs.get('xy');valid_xy=isinstance(xy,(list,tuple)) and len(xy)==2 and all(_number(v) for v in xy)
                    mask=mask_map[index];inside=False
                    if valid_xy and isinstance(mask,np.ndarray) and mask.dtype==np.uint8 and mask.shape==(frame['height'],frame['width']):
                        x,y=map(math.floor,xy);inside=bool(0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x]==255)
                    if not inside:reasons.append('OUTSIDE_OR_UNAVAILABLE_ORIGINAL_MASK')
                    binding=frame.get('mask_binding')
                    if not (isinstance(binding,dict) and isinstance(binding.get('path'),str) and binding['path'] and isinstance(binding.get('sha256'),str) and len(binding['sha256'])==64 and all(c in '0123456789abcdefABCDEF' for c in binding['sha256'])):reasons.append('NO_ORIGINAL_MASK_BINDING')
                    try:_camera(camera,frame,index)
                    except (ValueError,KeyError,TypeError,OverflowError) as exc:reasons.append(str(exc))
                    certificate=certificates.get(group['group_id'])
                    seed_evidence=None
                    if certificate and certificate['source_input_index']==index:
                        from .seed_pair_validation import verify_seed_certificate
                        successor=next((o for o in observations if o['frame_index']==certificate['successor_input_index']),None)
                        successor_camera=cameras.get(certificate['successor_input_index'])
                        successor_frame=registry.get(certificate['successor_input_index'])
                        if (successor is None or successor_frame is None or successor_camera is None
                                or obs.get('tracking_status')!='seed' or obs.get('strong') is not False
                                or obs.get('fb_error_px') is not None or obs.get('lk_error') is not None
                                or certificate['model_key']!=key
                                or certificate['source_image_id']!=camera['image_id']
                                or certificate['successor_image_id']!=successor_camera['image_id']
                                or certificate['source_position']!=obs.get('sequence_position')
                                or certificate['successor_position']!=successor.get('sequence_position')
                                or successor.get('identity_segment')!=segment):
                            raise ValueError('SEED_CERTIFICATE_SOURCE_BINDING_MISMATCH')
                        checked=verify_seed_certificate(certificate,group_id=group['group_id'],track_id=tid,
                            identity_segment=segment,source_frame_id=frame['frame_id'],
                            successor_frame_id=successor_frame['frame_id'],source_raw_xy=xy,successor_raw_xy=successor['xy'])
                        if checked['status']!='VERIFIED_REPEATED_PAIR_CONSISTENCY':raise ValueError('INVALID_SEED_CERTIFICATE')
                        for role,f,c in [('source',frame,camera),('successor',successor_frame,successor_camera)]:
                            if (certificate[role+'_camera']!=c
                                    or certificate[role+'_image_binding']!={'path':f['path'],'sha256':f['sha256']}
                                    or certificate[role+'_mask_binding']!=f['mask_binding']):
                                raise ValueError('SEED_CERTIFICATE_INPUT_BINDING_MISMATCH')
                        for role,field in [('source','backward_xy'),('successor','forward_xy')]:
                            endpoint=np.asarray(certificate['measurement'][field],dtype=float)[0,0]+.5
                            endpoint_index=certificate[role+'_input_index'];endpoint_mask=mask_map[endpoint_index]
                            x,y=map(math.floor,endpoint)
                            if not (isinstance(endpoint_mask,np.ndarray) and 0<=x<endpoint_mask.shape[1]
                                    and 0<=y<endpoint_mask.shape[0] and endpoint_mask[y,x]==255):
                                raise ValueError('SEED_CERTIFICATE_ENDPOINT_OUTSIDE_ORIGINAL_MASK')
                        # Only these missing original flow fields may be replaced by a separately measured pair.
                        if set(reasons)!= {'NON_STRONG','INVALID_FB_ERROR_PX','INVALID_LK_ERROR'}:
                            raise ValueError('SEED_CERTIFICATE_CANNOT_OVERRIDE_SOURCE_REJECTION')
                        reasons=[];seed_evidence=deepcopy(certificate);used_certificates.add(group['group_id'])
                    row=dict(input_index=index,frame_id=frame['frame_id'],track_id=tid,identity_segment=segment,
                        source_xy=deepcopy(xy),status='EXCLUDED' if reasons else 'ADMITTED_ORIGINAL_ANCHOR',reasons=reasons,
                        original_tracking={k:deepcopy(obs.get(k)) for k in ('tracking_status','strong','fb_error_px','lk_error','sequence_position')})
                    row['image_binding']={'path':frame.get('path'),'sha256':frame.get('sha256')}
                    row['mask_binding']=deepcopy(frame.get('mask_binding'))
                    if seed_evidence is not None:row.update(evidence_kind='VERIFIED_SEED_PAIR',seed_certificate=seed_evidence)
                    group['observations'].append(row)
                    if not reasons:group['admitted_observations'].append(dict(input_index=index,frame_id=frame['frame_id'],
                        image_id=camera['image_id'],raw_pixels=deepcopy(xy),width=frame['width'],height=frame['height'],
                        camera_model='SIMPLE_RADIAL',**({'evidence_kind':'VERIFIED_SEED_PAIR','seed_certificate':deepcopy(seed_evidence)} if seed_evidence else {}),
                        **{k:deepcopy(camera[k]) for k in ('params','R','t')}))
                if group['group_id'] in used_certificates:
                    cert=certificates[group['group_id']]
                    strong=[o for o in group['admitted_observations'] if o.get('evidence_kind')!='VERIFIED_SEED_PAIR']
                    if (len(strong)!=2 or cert['successor_input_index'] not in [o['input_index'] for o in strong]
                            or cert['strong_anchor_image_ids']!=[o['image_id'] for o in strong]
                            or len({o['image_id'] for o in group['admitted_observations']})!=3):
                        raise ValueError('SEED_CERTIFICATE_ANCHOR_SET_MISMATCH')
                count=len(group['admitted_observations'])
                if count>PROFILE['max_observations']:group['status']='NOT_EVALUATED_OBSERVATION_BUDGET'
                elif count>=2:group['status']='READY_FOR_NUMERIC_DIAGNOSTIC'
                groups.append(group)
    if used_certificates!=set(certificates):raise ValueError('UNBOUND_SEED_CERTIFICATES')
    if not evaluate:
        return _json(dict(status='ORIGINAL_ANCHOR_ADMISSION_ONLY',profile=PROFILE,groups=groups,
            camera_count=camera_count,numeric_calls=0,**RIGHTS))
    queue=[]
    for multiview in (True,False):
        buckets=defaultdict(deque)
        for group in sorted(groups,key=lambda g:(g['model_key'],g['track_id'],g['identity_segment'])):
            if group['status']=='READY_FOR_NUMERIC_DIAGNOSTIC' and (len(group['admitted_observations'])>=3)==multiview:
                buckets[group['model_key']].append(group)
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:queue.append(buckets[key].popleft())
    for group in queue[PROFILE['max_groups']:]:group['status']='NOT_EVALUATED_GROUP_BUDGET'
    selected=queue[:PROFILE['max_groups']]
    selected_ids={g['group_id'] for g in selected}
    omitted=[g for g in groups if g['group_id'] not in selected_ids]
    planned=[dict(batch_id=i//PROFILE['batch_size'],group_ids=[g['group_id'] for g in selected[i:i+PROFILE['batch_size']]],status='NOT_STARTED')
             for i in range(0,len(selected),PROFILE['batch_size'])]
    selection=dict(policy=PROFILE['selection'],eligible_groups=len(queue),batch_size=PROFILE['batch_size'],
                   max_batches=PROFILE['max_batches'],max_groups=PROFILE['max_groups'])
    progress(deepcopy(dict(kind='PLAN',profile=PROFILE,batches=planned,selection=selection,
        selected_groups=[g['group_id'] for g in selected],
        omitted_groups=[dict(group_id=g['group_id'],status=g['status']) for g in omitted])))
    batches=[];by_id={g['group_id']:g for g in selected}
    for plan in planned:
        completed=[];current=None
        record=dict(plan,status='RUNNING',evaluated_groups=0,group_statuses={},loo_checks=0)
        try:
            progress(deepcopy(dict(kind='BATCH_BEGIN',**record,group_results=[])))
            for ident in plan['group_ids']:
                current=ident;group=by_id[ident];budget()
                try:
                    _numeric(group,budget)
                    converted=_json(group);group.clear();group.update(converted)
                except (ValueError,ArithmeticError,KeyError,TypeError,np.linalg.LinAlgError) as exc:
                    partial=deepcopy({key:group[key] for key in
                        ('fit','evaluation','observability','leave_one_view_out','raw_predictions','radial_domains',
                         'strict_radial_domain_check','loo_status') if key in group})
                    group.update(status='NUMERIC_REFUSED',reason=str(exc),partial_numeric_evidence=_json(partial),
                                 fit=None,evaluation=None,observability=None,leave_one_view_out=[],raw_predictions=[],
                                 loo_status='NOT_EVALUATED_NUMERIC_REFUSAL')
                completed.append(group)
                record.update(evaluated_groups=len(completed),group_statuses=dict(Counter(g['status'] for g in completed)),
                              loo_checks=sum(len(g['leave_one_view_out']) for g in completed))
            record['status']='COMPLETE'
            progress(deepcopy(dict(kind='BATCH_END',**record,group_results=completed)))
            batches.append(deepcopy(record))
        except Exception as exc:
            record.update(status='FAILED',failed_group_id=current,reason=f'{type(exc).__name__}: {exc}')
            partial=(dict(deepcopy(by_id[current]),status='NOT_EVALUATED_INTERRUPTED',
                          status_before_interruption=by_id[current]['status'])
                     if current is not None and all(g['group_id']!=current for g in completed) else None)
            try:progress(deepcopy(dict(kind='BATCH_FAILED',**record,group_results=completed,failed_group_result=partial)))
            except Exception as callback_error:
                exc.add_note(f'BATCH_FAILED callback also failed: {type(callback_error).__name__}: {callback_error}')
            raise
    return _json(dict(status='ORIGINAL_ANCHOR_DIAGNOSTICS_COMPLETE',profile=PROFILE,groups=groups,
        batches=batches,selection=selection,
        selected_groups=[g['group_id'] for g in selected],omitted_groups=[dict(group_id=g['group_id'],status=g['status']) for g in omitted],
        camera_count=camera_count,new_native_model_points=0,source_unchanged=True,binding_scope='Caller authenticates path/hash; pure adapter checks binding fields only',
        counts=dict(models=len(models),cameras=camera_count,groups=len(groups),selected_groups=len(selected),omitted_groups=len(omitted),batches_completed=len(batches),
            verified_seed_observations=len(used_certificates),
            selected_multiview_groups=sum(len(g['admitted_observations'])>=3 for g in selected),
            statuses=dict(Counter(g['status'] for g in groups)),admitted_observations=sum(len(g['admitted_observations']) for g in groups),
            loo_checks=sum(len(g['leave_one_view_out']) for g in groups)),**RIGHTS))
