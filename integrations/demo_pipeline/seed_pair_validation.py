"""Bounded repeated seed-pair consistency; never independent physical identity."""
from collections import Counter,defaultdict,deque
from copy import deepcopy
import math
import numpy as np
from .original_anchor_review import _camera

PROFILE=dict(profile_id='seed-pair-validation-v2',max_groups=96,batch_size=32,max_batches=3,win_size=21,max_level=3,
    max_iterations=30,epsilon=.01,max_fb_px=1.,max_forward_lk=25.,max_backward_lk=25.,max_next_error_px=1.)
RIGHTS=dict(geometry_accepted=False,native_injected=False,identity_verified=False,
            original_observations_modified=False)


def _need(value,reason):
    if not value:raise ValueError(reason)


def _xy(value):
    _need(isinstance(value,(list,tuple)) and len(value)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in value),'INVALID_SOURCE_XY')
    return np.asarray(value,np.float64)


def _inside(raw,frame,mask):
    p=_xy(raw)-.5
    _need(0<=p[0]<=frame['width']-1 and 0<=p[1]<=frame['height']-1,'POINT_OUTSIDE_RAW_IMAGE')
    _need(isinstance(mask,np.ndarray) and mask.dtype==np.uint8 and mask.shape==(frame['height'],frame['width']),'INVALID_MASK_ARRAY')
    x,y=np.floor(_xy(raw)).astype(int)
    _need(0<=x<mask.shape[1] and 0<=y<mask.shape[0] and mask[y,x]==255,'POINT_OUTSIDE_ORIGINAL_MASK')


def _output(value,shape,label,status=False):
    _need(isinstance(value,np.ndarray) and value.shape==shape,label+'_SHAPE')
    _need(value.dtype.kind in ('u','i') if status else value.dtype.kind=='f',label+'_DTYPE')
    _need(np.isfinite(value).all(),label+'_NONFINITE')
    if status:_need(int(value[0,0])==1,label+'_FAILED')
    elif 'ERROR' in label:_need(float(value[0,0])>=0,label+'_NEGATIVE')
    return value.tolist()


def _safe(value):
    if isinstance(value,np.ndarray):return _safe(value.tolist())
    if isinstance(value,np.generic):return _safe(value.item())
    if isinstance(value,(tuple,list)):return [_safe(v) for v in value]
    if isinstance(value,dict):return {k:_safe(v) for k,v in value.items()}
    if isinstance(value,float) and not math.isfinite(value):return None
    if value is None or type(value) in (int,float,str,bool):return value
    return str(type(value).__name__)


def verify_seed_certificate(certificate, *, group_id,track_id,identity_segment,source_frame_id,
                            successor_frame_id,source_raw_xy,successor_raw_xy):
    """Recompute pair numbers from outputs; caller authenticates frames/masks/cameras."""
    try:
        for key,expected in [('group_id',group_id),('track_id',track_id),('identity_segment',identity_segment),
                ('source_frame_id',source_frame_id),('successor_frame_id',successor_frame_id)]:
            _need(certificate.get(key)==expected and type(certificate.get(key)) is type(expected),'CERTIFICATE_IDENTITY_MISMATCH')
        _need(certificate.get('profile')==PROFILE,'CERTIFICATE_PROFILE_MISMATCH')
        _need(certificate['source_raw_xy']==source_raw_xy and certificate['successor_raw_xy']==successor_raw_xy,'CERTIFICATE_COORDINATE_MISMATCH')
        _need(isinstance(certificate.get('strong_anchor_image_ids'),list) and len(certificate['strong_anchor_image_ids'])==2 and all(type(i) is int and i>0 for i in certificate['strong_anchor_image_ids']),'INVALID_STRONG_ANCHOR_IMAGE_IDS')
        measured=certificate['measurement']
        def array(key,shape,status=False):
            data=measured[key]
            def leaves(value):
                if isinstance(value,list):
                    for item in value:yield from leaves(item)
                else:yield value
            values=list(leaves(data))
            _need(bool(values) and all(type(v) is int if status else type(v) is float for v in values),key+'_LEAF_TYPE')
            a=np.asarray(data)
            _need(a.shape==shape and a.dtype.kind!='b',key+'_STRUCTURE')
            _need(np.isfinite(a.astype(float)).all(),key+'_NONFINITE')
            if not status:_need(a.dtype.kind=='f',key+'_DTYPE')
            if status:_need(a.dtype.kind in ('i','u') and int(a[0,0])==1,key+'_FAILED')
            return a.astype(float)
        forward=array('forward_xy',(1,1,2))[0,0];back=array('backward_xy',(1,1,2))[0,0]
        array('forward_status',(1,1),True);array('backward_status',(1,1),True)
        ferr=float(array('forward_error',(1,1))[0,0]);berr=float(array('backward_error',(1,1))[0,0])
        fb=float(np.linalg.norm(back-(_xy(source_raw_xy)-.5)));next_error=float(np.linalg.norm(forward-(_xy(successor_raw_xy)-.5)))
        _need(0<=ferr<=25 and 0<=berr<=25 and fb<=1 and next_error<=1,'PAIR_NUMERIC_THRESHOLD')
        _need(all(certificate.get(k) is False for k in ('geometry_accepted','native_injected','identity_verified','original_observations_modified')),'CERTIFICATE_RIGHTS_INVALID')
        return dict(status='VERIFIED_REPEATED_PAIR_CONSISTENCY',fb_error_px=fb,next_error_px=next_error,forward_lk_error=ferr,backward_lk_error=berr,**RIGHTS)
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        return dict(status='REJECTED',reason=str(exc),**RIGHTS)


def validate_seed_pairs(frames,grays,masks,mapped_tracking,original_diagnostics,cv,budget=lambda:None, *,camera_result,progress=lambda event:None):
    _need(isinstance(frames,list) and len(frames)==len(grays)==len(masks),'COMPLETE_FRAME_ARRAYS_REQUIRED')
    registry={};positions={};frame_ids=set()
    for position,frame in enumerate(frames):
        index=frame['input_index'];_need(type(index) is int and index not in registry,'DUPLICATE_FRAME_INDEX')
        _need(isinstance(frame.get('frame_id'),str) and frame['frame_id'] and frame['frame_id'] not in frame_ids,'DUPLICATE_FRAME_ID')
        registry[index]=frame;positions[index]=position;frame_ids.add(frame['frame_id'])
    tracks={t['track_id']:t for t in mapped_tracking['tracks']}
    _need(len(tracks)==len(mapped_tracking['tracks']),'DUPLICATE_TRACK_ID')
    models={}
    for model in camera_result.get('models',[])+camera_result.get('unsupported_models',[]):
        _need(model['model_key'] not in models,'DUPLICATE_MODEL_KEY')
        cameras={}
        for table in ('cameras','unsupported_cameras'):
            for key,value in model.get(table,{}).items():
                index=int(key) if type(key) is str and key.isdigit() else key
                _need(type(index) is int and index not in cameras,'DUPLICATE_CAMERA_INPUT_INDEX')
                cameras[index]=value
        models[model['model_key']]=cameras
    rows=[];candidates=[]
    for group in original_diagnostics['groups']:
        row=dict(group_id=group['group_id'],model_key=group['model_key'],track_id=group['track_id'],identity_segment=group['identity_segment'],
            status='NOT_ELIGIBLE',reasons=[],certificate=None,**RIGHTS);rows.append(row)
        try:
            admitted=group['admitted_observations'];_need(len(admitted)==2,'REQUIRE_EXACTLY_TWO_STRONG_ANCHORS')
            _need(group['group_id']==f"{group['model_key']}:{group['track_id']}:{group['identity_segment']}",'GROUP_ID_MISMATCH')
            observations=tracks[group['track_id']]['observations'];by_index={o['frame_index']:o for o in observations}
            _need(len(by_index)==len(observations),'DUPLICATE_TRACK_FRAME')
            _need(all(o['frame_index'] in positions and o.get('sequence_position',positions[o['frame_index']])==positions[o['frame_index']] for o in observations),'TRACK_POSITION_MAP_MISMATCH')
            seeds=[o for o in observations if o['identity_segment']==group['identity_segment'] and o.get('tracking_status')=='seed' and o.get('strong') is False and o.get('fb_error_px') is None and o.get('lk_error') is None]
            excluded={o['input_index']:o for o in group['observations'] if o['status']=='EXCLUDED'}
            seeds=[o for o in seeds if o['frame_index'] in excluded]
            _need(len(seeds)==1,'REQUIRE_EXACTLY_ONE_EXCLUDED_NULL_FLOW_SEED')
            seed=seeds[0];index=seed['frame_index'];position=positions[index]
            successors=[o for o in observations if o.get('sequence_position',positions.get(o['frame_index']))==position+1 and o['identity_segment']==group['identity_segment']]
            _need(len(successors)==1,'NO_IMMEDIATE_SAME_SEGMENT_SUCCESSOR')
            successor=successors[0];next_index=successor['frame_index'];next_position=positions[next_index]
            _need(next_position==position+1,'SEQUENCE_GAP')
            _need(next_index in {a['input_index'] for a in admitted},'SUCCESSOR_NOT_ORIGINAL_STRONG_ANCHOR')
            source_row=excluded[index];camera=models[group['model_key']].get(index)
            _camera(camera,registry[index],index)
            all_cameras=[camera]+admitted
            _need(len({c['image_id'] for c in all_cameras})==3 and len({c['input_index'] for c in all_cameras})==3 and len({c['frame_id'] for c in all_cameras})==3,'THREE_DISTINCT_SOURCE_CAMERAS_REQUIRED')
            _need(source_row.get('track_id')==group['track_id'] and source_row.get('identity_segment')==group['identity_segment'],'SEED_ROW_TRACK_SEGMENT_MISMATCH')
            _need(source_row['source_xy']==seed['xy'] and source_row['frame_id']==registry[index]['frame_id'],'SEED_SOURCE_IDENTITY_MISMATCH')
            for obs in [seed,successor]:
                f=registry[obs['frame_index']];p=positions[obs['frame_index']]
                _need(f['original_mask_status']=='OK' and obs.get('label')=='foreground' and obs.get('target_verified') is True,'NOT_ORIGINAL_FOREGROUND')
                _need(isinstance(f.get('path'),str) and bool(f['path']) and isinstance(f.get('sha256'),str) and len(f['sha256'])==64 and all(c in '0123456789abcdefABCDEF' for c in f['sha256']),'SOURCE_IMAGE_BINDING_REQUIRED')
                binding=f.get('mask_binding')
                _need(isinstance(binding,dict) and isinstance(binding.get('path'),str) and bool(binding['path']) and isinstance(binding.get('sha256'),str) and len(binding['sha256'])==64 and all(c in '0123456789abcdefABCDEF' for c in binding['sha256']),'SOURCE_MASK_BINDING_REQUIRED')
                _inside(obs['xy'],f,masks[p])
                _need(isinstance(grays[p],np.ndarray) and grays[p].dtype==np.uint8 and grays[p].shape==(f['height'],f['width']),'INVALID_GRAY_ARRAY')
            _need(grays[position].shape==grays[next_position].shape,'GRAY_SHAPE_MISMATCH')
            for anchor in admitted:
                anchor_index=anchor['input_index'];actual=models[group['model_key']].get(anchor_index)
                _camera(actual,registry[anchor_index],anchor_index)
                _need(all(anchor.get(k)==actual.get(k) for k in ('input_index','frame_id','image_id','camera_model','params','R','t','width','height')),'ANCHOR_CAMERA_MISMATCH')
                _inside(anchor['raw_pixels'],registry[anchor_index],masks[positions[anchor_index]])
                obs=by_index[anchor['input_index']]
                _need(registry[anchor_index]['original_mask_status']=='OK' and obs.get('label')=='foreground' and obs.get('target_verified') is True,'ANCHOR_NOT_ORIGINAL_FOREGROUND')
                _need(obs['identity_segment']==group['identity_segment'] and obs.get('strong') is True and obs['xy']==anchor['raw_pixels'],'ANCHOR_SOURCE_MISMATCH')
                _need(type(obs.get('fb_error_px')) in (int,float) and 0<=obs['fb_error_px']<=1 and type(obs.get('lk_error')) in (int,float) and 0<=obs['lk_error']<=25,'ANCHOR_NOT_STRONG')
            row.update(status='READY',source_image_id=camera['image_id'],successor_image_id=models[group['model_key']][next_index]['image_id'],strong_anchor_image_ids=[a['image_id'] for a in admitted],source_input_index=index,successor_input_index=next_index,source_position=position,successor_position=next_position)
            candidates.append((row,seed,successor))
        except (ValueError,KeyError,TypeError,OverflowError) as exc:row['reasons']=[str(exc)]
    buckets=defaultdict(deque)
    for item in sorted(candidates,key=lambda v:(v[0]['model_key'],v[0]['track_id'],v[0]['identity_segment'])):buckets[item[0]['model_key']].append(item)
    ordered=[]
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:ordered.append(buckets[key].popleft())
    for row,_,_ in ordered[PROFILE['max_groups']:]:row.update(status='NOT_EVALUATED_GROUP_BUDGET',reasons=['MAX_96_GROUPS'])
    cache={};flow_calls=0;certificates=[]
    lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv.TERM_CRITERIA_COUNT|cv.TERM_CRITERIA_EPS,30,.01))
    def process(row,seed,successor):
        nonlocal flow_calls
        key=(row['track_id'],row['identity_segment'],row['source_input_index'],row['successor_input_index'],tuple(seed['xy']),tuple(successor['xy']))
        before_calls=flow_calls;cached=key in cache
        row.update(status='RUNNING',flow_calls=0,measurement_cached=cached,source_raw_xy=deepcopy(seed['xy']),successor_raw_xy=deepcopy(successor['xy']))
        if key not in cache:
            measurement={};row['partial_measurement']=measurement;error=None;p=row['source_position'];q=row['successor_position']
            start=(_xy(seed['xy'])-.5).astype(np.float32).reshape(1,1,2)
            budget();flow_calls+=1;row['flow_calls']=flow_calls-before_calls;forward,status,ferr=cv.calcOpticalFlowPyrLK(grays[p].copy(),grays[q].copy(),start.copy(),None,**lk)
            measurement.update(forward_xy=_safe(forward),forward_status=_safe(status),forward_error=_safe(ferr))
            cv_running=False
            try:
                _output(forward,(1,1,2),'FORWARD');_output(status,(1,1),'FORWARD_STATUS',True);_output(ferr,(1,1),'FORWARD_ERROR')
                cv_running=True
                budget();flow_calls+=1;row['flow_calls']=flow_calls-before_calls;back,bstatus,berr=cv.calcOpticalFlowPyrLK(grays[q].copy(),grays[p].copy(),forward.astype(np.float32).copy(),None,**lk)
                cv_running=False
                measurement.update(backward_xy=_safe(back),backward_status=_safe(bstatus),backward_error=_safe(berr))
                _output(back,(1,1,2),'BACKWARD');_output(bstatus,(1,1),'BACKWARD_STATUS',True);_output(berr,(1,1),'BACKWARD_ERROR')
                _inside((forward[0,0]+.5).tolist(),registry[row['successor_input_index']],masks[q])
                _inside((back[0,0]+.5).tolist(),registry[row['source_input_index']],masks[p])
            except ValueError as exc:
                if cv_running:raise
                error=str(exc)
            if error is None:
                measurement['mask_checks']=dict(source_original_raw_xy=deepcopy(seed['xy']),successor_original_raw_xy=deepcopy(successor['xy']),
                    forward_measured_raw_xy=(forward[0,0]+.5).tolist(),backward_measured_raw_xy=(back[0,0]+.5).tolist(),
                    source_original_inside=True,successor_original_inside=True,forward_inside_successor=True,backward_inside_source=True,
                    sampling='floor(COLMAP raw xy); LK coordinates=raw-.5')
        else:
            measurement,error=cache[key]
            row['partial_measurement']=deepcopy(measurement)
        certificate=dict(group_id=row['group_id'],track_id=row['track_id'],identity_segment=row['identity_segment'],
            source_frame_id=registry[row['source_input_index']]['frame_id'],successor_frame_id=registry[row['successor_input_index']]['frame_id'],
            source_input_index=row['source_input_index'],successor_input_index=row['successor_input_index'],source_image_id=row['source_image_id'],successor_image_id=row['successor_image_id'],strong_anchor_image_ids=row['strong_anchor_image_ids'],
            source_image_binding=dict(path=registry[row['source_input_index']]['path'],sha256=registry[row['source_input_index']]['sha256']),successor_image_binding=dict(path=registry[row['successor_input_index']]['path'],sha256=registry[row['successor_input_index']]['sha256']),source_mask_binding=deepcopy(registry[row['source_input_index']]['mask_binding']),successor_mask_binding=deepcopy(registry[row['successor_input_index']]['mask_binding']),original_seed_state=deepcopy(seed),source_position=row['source_position'],successor_position=row['successor_position'],model_key=row['model_key'],source_camera=deepcopy(models[row['model_key']][row['source_input_index']]),successor_camera=deepcopy(models[row['model_key']][row['successor_input_index']]),source_raw_xy=deepcopy(seed['xy']),
            successor_raw_xy=deepcopy(successor['xy']),profile=deepcopy(PROFILE),measurement=deepcopy(measurement),
            evidence_scope='Repeated adjacent-pair consistency, not independent physical identity',**RIGHTS)
        checked=validate_certificate(certificate) if error is None else dict(status='REJECTED',reason=error)
        row.update(flow_calls=flow_calls-before_calls,measurement_cached=cached)
        row.update(status=checked['status'],reasons=[] if checked['status'].startswith('VERIFIED') else [checked['reason']],certificate=certificate)
        cache[key]=(measurement,error)
        row.pop('partial_measurement',None)
        certificate['recomputed']=checked
        if checked['status'].startswith('VERIFIED'):certificates.append(certificate)
    selected_items=ordered[:PROFILE['max_groups']]
    selected_ids=[r['group_id'] for r,_,_ in selected_items]
    omitted_ids=[r['group_id'] for r in rows if r['status'] in ('NOT_ELIGIBLE','NOT_EVALUATED_GROUP_BUDGET')]
    batches=[dict(batch_id=i//PROFILE['batch_size'],group_ids=[r['group_id'] for r,_,_ in selected_items[i:i+PROFILE['batch_size']]],status='NOT_STARTED')
             for i in range(0,len(selected_items),PROFILE['batch_size'])]
    progress(deepcopy(dict(kind='PLAN',profile=PROFILE,selected_groups=selected_ids,omitted_groups=omitted_ids,batches=batches)))
    for batch in batches:
        batch_start=flow_calls;begin=batch['batch_id']*PROFILE['batch_size']
        items=selected_items[begin:begin+PROFILE['batch_size']]
        active_row=None
        try:
            batch['status']='RUNNING'
            progress(deepcopy(dict(kind='BATCH_BEGIN',**batch)))
            for row,seed,successor in items:
                active_row=row
                row.update(flow_calls=0,partial_measurement={})
                budget();process(row,seed,successor)
                active_row=None
            outcomes=[row for row,_,_ in items]
            batch.update(status='COMPLETE',flow_calls=flow_calls-batch_start,
                verified=sum(r['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY' for r in outcomes),
                rejected=sum(r['status']=='REJECTED' for r in outcomes))
            progress(deepcopy(dict(kind='BATCH_END',**batch,rows=outcomes,
                certificates=[r['certificate'] for r in outcomes if r['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY'],
                cumulative_flow_calls=flow_calls)))
        except BaseException as exc:
            if active_row is not None:
                active_row.update(status='NOT_EVALUATED_INTERRUPTED',reasons=[type(exc).__name__],certificate=None)
            finished=[r for r,_,_ in items if r['status'] in ('VERIFIED_REPEATED_PAIR_CONSISTENCY','REJECTED')]
            batch.update(status='FAILED',flow_calls=flow_calls-batch_start,failure_type=type(exc).__name__,
                verified=sum(r['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY' for r in finished),
                rejected=sum(r['status']=='REJECTED' for r in finished))
            event=deepcopy(dict(kind='BATCH_FAILED',**batch,rows=finished,
                certificates=[r['certificate'] for r in finished if r['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY'],
                failed_group_id=active_row['group_id'] if active_row is not None else None,
                failed_group_result=active_row,cumulative_flow_calls=flow_calls,
                unique_completed_pair_measurements=len(cache)))
            try:progress(event)
            except BaseException as callback_exc:
                if hasattr(exc,'add_note'):exc.add_note('BATCH_FAILED progress callback also failed: '+type(callback_exc).__name__+': '+str(callback_exc))
            raise
    return dict(status='SEED_PAIRS_EVALUATED',profile=deepcopy(PROFILE),rows=rows,groups=rows,certificates=certificates,
        selected_groups=[r['group_id'] for r,_,_ in ordered[:PROFILE['max_groups']]],omitted_groups=[r['group_id'] for r in rows if r['status'] in ('NOT_ELIGIBLE','NOT_EVALUATED_GROUP_BUDGET')],
        batches=batches,flow_calls=flow_calls,unique_pair_measurements=len(cache),counts=dict(groups=len(rows),eligible=len(ordered),verified=len(certificates),batches_planned=len(batches),batches_completed=sum(b['status']=='COMPLETE' for b in batches),statuses=dict(Counter(r['status'] for r in rows))),
        evidence_scope='Repeated pair consistency only; seed remains strong=False and null original flow',**RIGHTS)


def validate_certificate(certificate):
    """Pure numeric/identity recheck; caller rebinds actual masks and camera data."""
    try:
        c=certificate
        _need(isinstance(c,dict) and isinstance(c.get('model_key'),str) and c['model_key'],'INVALID_CERTIFICATE')
        _need(c['group_id']==f"{c['model_key']}:{c['track_id']}:{c['identity_segment']}",'GROUP_ID_MISMATCH')
        _need(all(type(c.get(k)) is int and c[k]>=0 for k in ('track_id','identity_segment','source_input_index','successor_input_index','source_position','successor_position')),'INVALID_CERTIFICATE_INDICES')
        _need(c['successor_position']==c['source_position']+1 and c['source_input_index']!=c['successor_input_index'],'CERTIFICATE_NOT_ADJACENT')
        _need(c['source_frame_id']!=c['successor_frame_id'],'CERTIFICATE_FRAME_IDENTITIES')
        _need(type(c['source_image_id']) is int and type(c['successor_image_id']) is int and c['source_image_id']>0 and c['successor_image_id']>0 and c['source_image_id']!=c['successor_image_id'],'CERTIFICATE_CAMERA_IDENTITIES')
        _need(len(c['strong_anchor_image_ids'])==2 and all(type(i) is int and i>0 for i in c['strong_anchor_image_ids']) and len(set(c['strong_anchor_image_ids']+[c['source_image_id']]))==3 and c['successor_image_id'] in c['strong_anchor_image_ids'],'CERTIFICATE_STRONG_CAMERA_IDENTITIES')
        for role in ('source','successor'):
            camera=c[role+'_camera']
            frame=dict(input_index=c[role+'_input_index'],frame_id=c[role+'_frame_id'],width=camera['width'],height=camera['height'])
            _camera(camera,frame,frame['input_index'])
            _need(camera['image_id']==c[role+'_image_id'],'CERTIFICATE_CAMERA_BINDING')
            point=_xy(c[role+'_raw_xy'])-.5
            _need(0<=point[0]<=frame['width']-1 and 0<=point[1]<=frame['height']-1,'CERTIFICATE_SOURCE_OUTSIDE_IMAGE')
        for key in ('source_image_binding','successor_image_binding','source_mask_binding','successor_mask_binding'):
            binding=c[key]
            _need(isinstance(binding,dict) and isinstance(binding.get('path'),str) and bool(binding['path']) and isinstance(binding.get('sha256'),str) and len(binding['sha256'])==64 and all(v in '0123456789abcdefABCDEF' for v in binding['sha256']),'CERTIFICATE_FILE_BINDING')
            if 'bytes' in binding:_need(type(binding['bytes']) is int and binding['bytes']>0,'CERTIFICATE_BINDING_BYTES')
        original=c['original_seed_state']
        _need(original.get('tracking_status')=='seed' and original.get('strong') is False and original.get('fb_error_px') is None and original.get('lk_error') is None,'CERTIFICATE_ORIGINAL_SEED_STATE')
        _need(original.get('frame_index')==c['source_input_index'] and original.get('identity_segment')==c['identity_segment'] and original.get('xy')==c['source_raw_xy'] and original.get('sequence_position',c['source_position'])==c['source_position'],'CERTIFICATE_SEED_SOURCE_IDENTITY')
        _need(original.get('label')=='foreground' and original.get('target_verified') is True,'CERTIFICATE_SEED_FOREGROUND')
        result=verify_seed_certificate(c,group_id=c['group_id'],track_id=c['track_id'],identity_segment=c['identity_segment'],
            source_frame_id=c['source_frame_id'],successor_frame_id=c['successor_frame_id'],source_raw_xy=c['source_raw_xy'],successor_raw_xy=c['successor_raw_xy'])
        if result['status'].startswith('VERIFIED'):
            checks=c['measurement']['mask_checks']
            _need(isinstance(checks,dict) and all(checks.get(k) is True for k in ('source_original_inside','successor_original_inside','forward_inside_successor','backward_inside_source')),'CERTIFICATE_MASK_CHECK_FLAGS')
            _need(checks.get('sampling')=='floor(COLMAP raw xy); LK coordinates=raw-.5','CERTIFICATE_MASK_SAMPLING')
            expected_checks=dict(source_original_raw_xy=c['source_raw_xy'],successor_original_raw_xy=c['successor_raw_xy'],
                forward_measured_raw_xy=(np.asarray(c['measurement']['forward_xy'],np.float32)[0,0]+.5).tolist(),
                backward_measured_raw_xy=(np.asarray(c['measurement']['backward_xy'],np.float32)[0,0]+.5).tolist())
            _need(all(checks.get(k)==v and all(type(x) in (int,float) and type(x) is not bool for x in checks[k]) for k,v in expected_checks.items()),'CERTIFICATE_MASK_CHECK_COORDINATES')
            for role,field in [('successor','forward_xy'),('source','backward_xy')]:
                point=np.asarray(c['measurement'][field],float)[0,0];camera=c[role+'_camera']
                _need(0<=point[0]<=camera['width']-1 and 0<=point[1]<=camera['height']-1,'CERTIFICATE_MEASUREMENT_OUTSIDE_IMAGE')
        return result
    except (ValueError,KeyError,TypeError,OverflowError) as exc:return dict(status='REJECTED',reason=str(exc),**RIGHTS)


def _validated_certificates(ledger):
    """Fail closed on ledger structure; recheck every successful certificate."""
    _need(isinstance(ledger,dict) and ledger.get('profile')==PROFILE and ledger.get('status')=='SEED_PAIRS_EVALUATED','INVALID_SEED_LEDGER_HEADER')
    _need(all(ledger.get(k) is False for k in RIGHTS),'INVALID_LEDGER_RIGHTS')
    rows=ledger.get('rows');groups=ledger.get('groups');selected=ledger.get('selected_groups');omitted=ledger.get('omitted_groups');certificates=ledger.get('certificates');counts=ledger.get('counts')
    _need(isinstance(rows,list) and rows==groups and isinstance(selected,list) and isinstance(omitted,list) and isinstance(certificates,list) and isinstance(counts,dict),'INVALID_LEDGER_STRUCTURE')
    ids=[r['group_id'] for r in rows]
    _need(all(isinstance(i,str) and i for i in ids+selected+omitted),'INVALID_LEDGER_IDENTITIES')
    _need(len(set(ids))==len(ids) and len(set(selected))==len(selected) and len(set(omitted))==len(omitted) and len(selected)<=PROFILE['max_groups'],'DUPLICATE_LEDGER_IDENTITIES')
    _need(not set(selected)&set(omitted) and set(selected)|set(omitted)==set(ids),'LEDGER_PARTITION_MISMATCH')
    allowed={'VERIFIED_REPEATED_PAIR_CONSISTENCY','REJECTED','NOT_EVALUATED_GROUP_BUDGET','NOT_ELIGIBLE'}
    for row in rows:
        _need(isinstance(row.get('model_key'),str) and row['model_key'] and type(row.get('track_id')) is int and row['track_id']>=0 and type(row.get('identity_segment')) is int and row['identity_segment']>=0,'ROW_IDENTITIES_INVALID')
        _need(row['group_id']==f"{row['model_key']}:{row['track_id']}:{row['identity_segment']}" and row['status'] in allowed,'INVALID_ROW_IDENTITY_OR_STATUS')
        _need(all(row.get(k) is False for k in RIGHTS),'INVALID_ROW_RIGHTS')
    buckets=defaultdict(deque)
    for row in sorted(rows,key=lambda r:(r['model_key'],r['track_id'],r['identity_segment'])):
        if row['status']!='NOT_ELIGIBLE':buckets[row['model_key']].append(row['group_id'])
    ordered=[]
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:ordered.append(buckets[key].popleft())
    _need(selected==ordered[:PROFILE['max_groups']],'LEDGER_ROUND_ROBIN_SELECTION_MISMATCH')
    byid={r['group_id']:r for r in rows}
    _need(all(byid[i]['status'] in ('VERIFIED_REPEATED_PAIR_CONSISTENCY','REJECTED') for i in selected),'SELECTED_ROW_STATUS_INVALID')
    _need(all(byid[i]['status']=='NOT_EVALUATED_GROUP_BUDGET' for i in ordered[PROFILE['max_groups']:]),'BUDGET_OMISSION_STATUS_INVALID')
    _need(omitted==[r['group_id'] for r in rows if r['status'] in ('NOT_ELIGIBLE','NOT_EVALUATED_GROUP_BUDGET')],'OMITTED_PARTITION_MISMATCH')
    verified=[r for r in rows if r['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY']
    _need(all(type(counts.get(k)) is int for k in ('groups','eligible','verified')) and counts['groups']==len(rows) and counts['eligible']==len(ordered) and counts['verified']==len(verified) and counts.get('statuses')==dict(Counter(r['status'] for r in rows)),'LEDGER_COUNTS_MISMATCH')
    batches=ledger.get('batches')
    expected_chunks=[selected[i:i+PROFILE['batch_size']] for i in range(0,len(selected),PROFILE['batch_size'])]
    _need(isinstance(batches,list) and len(batches)==len(expected_chunks)<=PROFILE['max_batches'],'BATCH_PARTITION_COUNT')
    _need(type(counts.get('batches_planned')) is int and type(counts.get('batches_completed')) is int and counts['batches_planned']==counts['batches_completed']==len(batches),'BATCH_COMPLETION_COUNTS')
    pair_cache=set();total_calls=0
    for batch_id,(batch,chunk) in enumerate(zip(batches,expected_chunks)):
        _need(type(batch.get('batch_id')) is int and batch['batch_id']==batch_id and batch.get('group_ids')==chunk and batch.get('status')=='COMPLETE','BATCH_PARTITION_OR_COMPLETION')
        calls=0
        for group_id in chunk:
            row=byid[group_id];c=row.get('certificate');_need(isinstance(c,dict),'ATTEMPTED_CERTIFICATE_REQUIRED')
            key=(c['track_id'],c['identity_segment'],c['source_input_index'],c['successor_input_index'],tuple(c['source_raw_xy']),tuple(c['successor_raw_xy']))
            cached=key in pair_cache
            expected_calls=0 if cached else (2 if 'backward_xy' in c['measurement'] else 1)
            _need(row.get('measurement_cached') is cached and type(row.get('flow_calls')) is int and row['flow_calls']==expected_calls,'ROW_MEASUREMENT_ACCOUNTING')
            pair_cache.add(key);calls+=expected_calls
        _need(all(type(batch.get(k)) is int for k in ('flow_calls','verified','rejected')) and batch['flow_calls']==calls and batch['verified']==sum(byid[i]['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY' for i in chunk) and batch['rejected']==sum(byid[i]['status']=='REJECTED' for i in chunk),'BATCH_OUTCOME_ACCOUNTING')
        total_calls+=calls
    _need(type(ledger.get('flow_calls')) is int and ledger['flow_calls']==total_calls and type(ledger.get('unique_pair_measurements')) is int and ledger['unique_pair_measurements']==len(pair_cache),'GLOBAL_FLOW_ACCOUNTING')
    certids=[c['group_id'] for c in certificates]
    _need(len(set(certids))==len(certids) and set(certids)=={r['group_id'] for r in verified},'CERTIFICATE_LIST_IDENTITY_MISMATCH')
    certmap={c['group_id']:c for c in certificates};out={}
    for row in verified:
        c=row.get('certificate')
        _need(isinstance(c,dict) and c==certmap[row['group_id']] and all(c.get(k)==row.get(k) and type(c.get(k)) is type(row.get(k)) for k in ('group_id','model_key','track_id','identity_segment')),'CERTIFICATE_ROW_MISMATCH')
        _need(validate_certificate(c)['status']=='VERIFIED_REPEATED_PAIR_CONSISTENCY','CERTIFICATE_NUMERIC_OR_BINDING_REJECTED')
        out[row['group_id']]=deepcopy(c)
    return out


def validated_certificates(ledger):
    try:return _validated_certificates(ledger)
    except (KeyError,TypeError,IndexError,OverflowError) as exc:raise ValueError('Malformed seed ledger: '+str(exc)) from exc
