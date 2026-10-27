"""Frozen-camera conditional correspondences. No pose estimation or native writes."""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import math
import numpy as np
from .seed_pair_validation import _inside, _output, _safe, _xy, _need
from .original_anchor_review import _camera

PROFILE = dict(profile_id='view-extension-v1', max_frames=192, max_models=96, max_queries=8, max_points_per_query=64,
    max_total_points=256, max_source_distance_positions=3, max_source_time_seconds=.5,
    source_routes=2, win_size=21, max_level=3, max_iterations=30, epsilon=.01,
    max_fb_px=1., max_forward_lk=25., max_backward_lk=25., max_endpoint_agreement_px=1.,
    min_diagnostic_correspondences=30)
RIGHTS = dict(geometry_acceptance=False, geometry_accepted=False, native_injected=False,
    identity_verified=False, original_observations_modified=False)


def _rights(row):
    for key in ('geometry_accepted', 'native_injected', 'identity_verified'):
        _need(row.get(key) is False, 'SOURCE_RIGHTS_INVALID')
    for key in ('geometry_acceptance', 'original_observations_modified'):
        if key in row: _need(row[key] is False, 'SOURCE_RIGHTS_INVALID')


def _summary(row, frame, mask):
    accepted = [p for p in row['points'] if p['status']=='ACCEPTED_CONDITIONAL_CORRESPONDENCE']
    n=len(accepted)
    out=dict(correspondence_count=n, registration_reasons=['INTRINSICS_UNESTABLISHED'],
        engineering_precheck_passed=n>=30, engineering_precheck_scope='COUNT_ONLY_NOT_POSE_OR_GEOMETRY_ACCEPTANCE', pose_fits=0, new_cameras=0,
        evidence_scope='descriptive spread; shared route frames are not independent identity proof',
        image_bbox_fraction=None, object_mask_bbox_fraction=None, covariance=None, rank=None, **RIGHTS)
    if n<30:out['registration_reasons'].append('INSUFFICIENT_CORRESPONDENCES')
    if n:
        xy=np.asarray([p['query_raw_xy'] for p in accepted],float)
        extent=np.ptp(xy,axis=0);area=float(np.prod(extent))
        out['image_bbox_fraction']=area/(frame['width']*frame['height'])
        ys,xs=np.nonzero(mask==255)
        if len(xs):out['object_mask_bbox_fraction']=area/float((xs.max()-xs.min()+1)*(ys.max()-ys.min()+1))
        centered=xy-xy.mean(axis=0)
        out['covariance']=(centered.T@centered/max(1,n-1)).tolist()
        out['rank']=int(np.linalg.matrix_rank(centered))
        out['bbox_raw_xy']=[xy.min(axis=0).tolist(),xy.max(axis=0).tolist()]
    return out


def extend_views(frames,grays,masks,classification,camera_result,cv,budget=lambda:None,progress=lambda event:None):
    """Plan all omissions before LK; failures never refill the frozen selection.

    The caller authenticates and independently recomputes classification. All raw pixels
    use COLMAP coordinates; LK receives raw-.5 and masks sample floor(raw).
    """
    result=dict(status='PLANNING',profile=deepcopy(PROFILE),plan=[],query_rows=[],flow_calls=0,
        pose_fits=0,new_cameras=0,counts={},source_group_ledger=[],**RIGHTS)
    result['queries']=result['query_rows']
    try:
        _need(isinstance(frames,list) and len(frames)==len(grays)==len(masks),'COMPLETE_FRAME_ARRAYS_REQUIRED')
        _need(len(frames)<=PROFILE['max_frames'],'FRAME_SCHEMA_BOUND')
        positions={};ids=set();previous_index=-1;previous_time=None
        for pos,f in enumerate(frames):
            budget();index=f['input_index']
            _need(type(index) is int and index>=0 and index not in positions,'DUPLICATE_FRAME_INDEX')
            _need(isinstance(f.get('frame_id'),str) and f['frame_id'] and f['frame_id'] not in ids,'DUPLICATE_FRAME_ID')
            _need(type(f.get('width')) is int and type(f.get('height')) is int and f['width']>0 and f['height']>0,'INVALID_FRAME_DIMENSIONS')
            _need(index>previous_index,'FRAME_INPUT_ORDER_INVALID')
            timestamp=f.get('time_seconds')
            _need(type(timestamp) in (int,float) and math.isfinite(timestamp) and (previous_time is None or timestamp>previous_time),'FRAME_TIME_ORDER_INVALID')
            previous_index=index;previous_time=timestamp
            def binding(value):
                return isinstance(value,dict) and isinstance(value.get('path'),str) and bool(value['path'].strip()) and isinstance(value.get('sha256'),str) and len(value['sha256'])==64 and all(c in '0123456789abcdefABCDEF' for c in value['sha256'])
            _need(binding(dict(path=f.get('path'),sha256=f.get('sha256'))),'IMAGE_BINDING_REQUIRED')
            if f.get('original_mask_status')=='OK':_need(binding(f.get('mask_binding')),'ORIGINAL_MASK_BINDING_REQUIRED')
            positions[index]=pos;ids.add(f['frame_id'])
        models={}
        for model in camera_result.get('models',[])+camera_result.get('unsupported_models',[]):
            budget();key=model['model_key']
            _need(isinstance(key,str) and key and key not in models,'DUPLICATE_MODEL_KEY')
            cams={};image_ids=set()
            for table in ('cameras','unsupported_cameras'):
                for index,camera in model.get(table,{}).items():
                    budget();index=int(index) if isinstance(index,str) and index.isdigit() else index
                    _need(type(index) is int and index in positions and index not in cams,'CAMERA_INPUT_MAPPING_INVALID')
                    _need(camera.get('input_index')==index and camera.get('frame_id')==frames[positions[index]]['frame_id'],'CAMERA_FRAME_IDENTITY_MISMATCH')
                    image_id=camera.get('image_id');_need(type(image_id) is int and image_id>0 and image_id not in image_ids,'DUPLICATE_CAMERA_IMAGE_ID')
                    cams[index]=camera;image_ids.add(image_id)
            models[key]=cams
        _need(len(models)<=PROFILE['max_models'],'MODEL_SCHEMA_BOUND')
        _rights(classification)
        _need(classification.get('status')=='ANCHOR_CANDIDATES_CLASSIFIED','CLASSIFICATION_STATUS_INVALID')
        candidates=classification['rows'];_need(isinstance(candidates,list) and len(candidates)<=10000,'GROUP_SCHEMA_BOUND')
        seen=set();eligible=defaultdict(list)
        for c in candidates:
            budget();_rights(c);key=c['model_key'];tid=c['track_id'];seg=c['identity_segment']
            _need(key in models and type(tid) is int and tid>=0 and type(seg) is int and seg>=0,'GROUP_IDENTITY_INVALID')
            _need(c['group_id']==f'{key}:{tid}:{seg}' and c['group_id'] not in seen,'DUPLICATE_OR_INVALID_GROUP_ID');seen.add(c['group_id'])
            result['source_group_ledger'].append(dict(group_id=c['group_id'],model_key=key,track_id=tid,identity_segment=seg,tier=c['tier'],reasons=deepcopy(c.get('reasons',[])),eligible_tier=c['tier'] in ('STRICT_CONDITIONAL','RELAXED_CONDITIONAL'),**RIGHTS))
            if c['tier'] not in ('STRICT_CONDITIONAL','RELAXED_CONDITIONAL'):continue
            xyz=c['xyz'];_need(isinstance(xyz,list) and len(xyz)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in xyz),'INVALID_CONDITIONAL_XYZ')
            sources=c['sources'];_need(isinstance(sources,list) and 3<=len(sources)<=12,'THREE_SOURCE_VIEWS_REQUIRED')
            used=set()
            for s in sources:
                budget();index=s['input_index'];_need(type(index) is int and index in models[key] and index not in used,'SOURCE_CAMERA_MAPPING_INVALID');used.add(index)
                f=frames[positions[index]];cam=models[key][index];_camera(cam,f,index)
                _need(s['frame_id']==f['frame_id'] and s['image_id']==cam['image_id'],'SOURCE_IDENTITY_MISMATCH')
                _need(f.get('original_mask_status')=='OK','SOURCE_ORIGINAL_MASK_NOT_OK')
                _inside(s['raw_pixels'],f,masks[positions[index]])
                for field,expected in [('image_binding',dict(path=f['path'],sha256=f['sha256'])),('mask_binding',f['mask_binding'])]:
                    if field in s:_need(s[field]==expected,'SOURCE_BINDING_MISMATCH')
            if 'source_indices' in c:_need(c['source_indices']==[s['input_index'] for s in sources],'SOURCE_INDICES_MISMATCH')
            if 'frame_ids' in c:_need(c['frame_ids']==[s['frame_id'] for s in sources],'SOURCE_FRAME_IDS_MISMATCH')
            eligible[key].append(c)
        def route(s,q):
            p=positions[s['input_index']];distance=abs(q-p)
            if not 0<distance<=3:return None
            path=list(range(p,q+(1 if q>p else -1),1 if q>p else -1));elapsed=0.
            for a in path:
                budget();f=frames[a]
                if f.get('original_mask_status')!='OK' or (f['width'],f['height'])!=(frames[q]['width'],frames[q]['height']):return None
                if not (isinstance(grays[a],np.ndarray) and grays[a].dtype==np.uint8 and grays[a].shape==(f['height'],f['width'])):return None
                if not (isinstance(masks[a],np.ndarray) and masks[a].dtype==np.uint8 and masks[a].shape==grays[a].shape):return None
                t=f.get('time_seconds');
                if type(t) not in (int,float) or not math.isfinite(t):return None
            for a,b in zip(path,path[1:]):elapsed+=abs(frames[a]['time_seconds']-frames[b]['time_seconds'])
            if elapsed>.5:return None
            return dict(source_input_index=s['input_index'],source_raw_xy=deepcopy(s['raw_pixels']),path_positions=path,
                path_input_indices=[frames[a]['input_index'] for a in path],time_seconds=elapsed,status='PLANNED',steps=[],**RIGHTS)
        buckets=defaultdict(deque)
        for key in sorted(models):
            for q,f in enumerate(frames):
                budget();row=dict(model_key=key,query_input_index=f['input_index'],query_frame_id=f['frame_id'],query_position=q,
                    status='PLANNED',reasons=[],points=[],summary=None,committed=False,**RIGHTS)
                result['query_rows'].append(row)
                if f['input_index'] in models[key]:row.update(status='ALREADY_REGISTERED',reasons=['SAVED_MODEL_CAMERA']);continue
                if f.get('original_mask_status')!='OK':row.update(status=f.get('original_mask_status','UNKNOWN_MASK_STATUS'),reasons=['ORIGINAL_MASK_NOT_OK']);continue
                for c in sorted(eligible[key],key=lambda c:(c['track_id'],c['identity_segment'])):
                    budget();routes=[r for s in c['sources'] if (r:=route(s,q)) is not None]
                    routes.sort(key=lambda r:(abs(r['path_positions'][0]-q),r['path_positions'][0],r['source_input_index']))
                    point=dict(group_id=c['group_id'],track_id=c['track_id'],identity_segment=c['identity_segment'],xyz=deepcopy(c['xyz']),
                        tier=c['tier'],status='PLANNED' if len(routes)>=2 else 'NO_TWO_SOURCE_ROUTES',reasons=[] if len(routes)>=2 else ['NO_TWO_SOURCE_ROUTES'],
                        query_raw_xy=None,routes=routes[:2],source_origin_bound=True,
                        matched_identity_hypothesis='conditional inherited source group; tracked route identity unproven',
                        evidence_scope='two routes may share frames; not independent identity proof',**RIGHTS)
                    row['points'].append(point)
                if any(p['status']=='PLANNED' for p in row['points']):buckets[key].append(row)
                else:row.update(status='NO_ANCHORS',reasons=['NO_TWO_SOURCE_ROUTES'])
        selected=[]
        while any(buckets.values()):
            budget()
            for key in sorted(buckets):
                if buckets[key]:selected.append(buckets[key].popleft())
        total=0
        for qi,row in enumerate(selected):
            budget();attempts=0
            for point in row['points']:
                budget()
                if point['status']!='PLANNED':continue
                reason='QUERY_BUDGET' if qi>=8 else 'POINT_BUDGET' if attempts>=64 or total>=256 else None
                if reason:point.update(status='NOT_EVALUATED_'+reason,reasons=[reason])
                else:attempts+=1;total+=1
            row['planned_point_attempts']=attempts
            if not attempts:row.update(status='NOT_EVALUATED_QUERY_BUDGET' if qi>=8 else 'NOT_EVALUATED_POINT_BUDGET')
        result['plan']=deepcopy(result['query_rows']);result['status']='PLAN_FROZEN'
        progress(dict(event='PLAN',plan=deepcopy(result['plan']),profile=deepcopy(PROFILE),**RIGHTS))
        lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv.TERM_CRITERIA_COUNT|cv.TERM_CRITERIA_EPS,30,.01))
        def measure(r):
            raw=deepcopy(r['source_raw_xy']);path=r['path_positions'];r['status']='RUNNING'
            for a,b in zip(path,path[1:]):
                budget();step=dict(source_input_index=frames[a]['input_index'],target_input_index=frames[b]['input_index'],source_raw_xy=deepcopy(raw),status='RUNNING',reason=None);r['steps'].append(step)
                in_call=False
                try:
                    _inside(raw,frames[a],masks[a]);start=(_xy(raw)-.5).astype(np.float32).reshape(1,1,2)
                    budget();result['flow_calls']+=1
                    in_call=True
                    fw,st,err=cv.calcOpticalFlowPyrLK(grays[a].copy(),grays[b].copy(),start.copy(),None,**lk)
                    in_call=False
                    step.update(forward_xy=_safe(fw),forward_status=_safe(st),forward_error=_safe(err))
                    _output(fw,(1,1,2),'FORWARD');_output(st,(1,1),'FORWARD_STATUS',True);_output(err,(1,1),'FORWARD_ERROR')
                    _need(float(err[0,0])<=25,'FORWARD_ERROR_THRESHOLD');_inside((fw[0,0]+.5).tolist(),frames[b],masks[b])
                    budget();result['flow_calls']+=1
                    in_call=True
                    bw,bs,be=cv.calcOpticalFlowPyrLK(grays[b].copy(),grays[a].copy(),fw.astype(np.float32).copy(),None,**lk)
                    in_call=False
                    step.update(backward_xy=_safe(bw),backward_status=_safe(bs),backward_error=_safe(be))
                    _output(bw,(1,1,2),'BACKWARD');_output(bs,(1,1),'BACKWARD_STATUS',True);_output(be,(1,1),'BACKWARD_ERROR')
                    _inside((bw[0,0]+.5).tolist(),frames[a],masks[a]);fb=float(np.linalg.norm(bw[0,0]-start[0,0]));step['fb_error_px']=fb
                    _need(float(be[0,0])<=25 and fb<=1,'BACKWARD_ERROR_OR_FB_THRESHOLD')
                    raw=(fw[0,0]+.5).tolist();step.update(status='PASSED',endpoint_raw_xy=raw)
                except ValueError as exc:
                    if in_call:raise
                    step.update(status='REJECTED',reason=str(exc));r.update(status='REJECTED',reason=str(exc));return
            r.update(status='PASSED',query_raw_xy=raw)
        for row in selected:
            budget()
            if not row['planned_point_attempts']:continue
            row['status']='RUNNING';progress(dict(event='QUERY_BEGIN',query=deepcopy(row),**RIGHTS))
            for point in row['points']:
                budget()
                if point['status']!='PLANNED':continue
                point['status']='RUNNING'
                for r in point['routes']:budget();measure(r)
                point.update(status='REJECTED',reasons=[r.get('reason','ROUTE_FAILED') for r in point['routes'] if r['status']!='PASSED'])
                if all(r['status']=='PASSED' for r in point['routes']):
                    distance=float(np.linalg.norm(_xy(point['routes'][0]['query_raw_xy'])-_xy(point['routes'][1]['query_raw_xy'])))
                    point['endpoint_agreement_px']=distance
                    if distance<=1:point.update(status='PROVISIONAL_ROUTE_AGREEMENT',query_raw_xy=deepcopy(point['routes'][0]['query_raw_xy']),reasons=[])
                    else:point['reasons']=['ENDPOINT_DISAGREEMENT']
                progress(dict(event='POINT_END',model_key=row['model_key'],query_input_index=row['query_input_index'],point=deepcopy(point),**RIGHTS))
            conflicts=set()
            for field in ('xyz','query_raw_xy'):
                bins=defaultdict(list)
                for i,p in enumerate(row['points']):
                    budget()
                    if p['status']=='PROVISIONAL_ROUTE_AGREEMENT':bins[tuple(p[field])].append(i)
                for group in bins.values():
                    if len(group)>1:conflicts.update(group)
            # No budget/callback boundary may expose final acceptance before conflicts are known.
            for i,p in enumerate(row['points']):
                if i in conflicts:p.update(status='REJECTED',reasons=['AMBIGUOUS_EXACT_QUERY_COORDINATE_OR_XYZ'])
                elif p['status']=='PROVISIONAL_ROUTE_AGREEMENT':p['status']='ACCEPTED_CONDITIONAL_CORRESPONDENCE'
            row['status']='CORRESPONDENCES_MEASURED';q=row['query_position'];row['summary']=_summary(row,frames[q],masks[q])
            progress(dict(event='QUERY_END',query=deepcopy(row),**RIGHTS))
            row['committed']=True
        result['status']='VIEW_EXTENSION_LEDGER_COMPLETE'
        result['counts']=dict(queries=len(result['query_rows']),query_attempts=sum(bool(r.get('planned_point_attempts')) for r in selected),
            point_attempts=total,accepted_correspondences=sum(p['status']=='ACCEPTED_CONDITIONAL_CORRESPONDENCE' for r in result['query_rows'] for p in r['points']),
            point_statuses=dict(Counter(p['status'] for r in result['query_rows'] for p in r['points'])),query_statuses=dict(Counter(r['status'] for r in result['query_rows'])),pose_fits=0,new_cameras=0)
        return result
    except BaseException as exc:
        for query in result['query_rows']:
            if not query.get('committed'):
                query['summary']=None
                for point in query['points']:
                    if point['status']=='ACCEPTED_CONDITIONAL_CORRESPONDENCE':
                        point.update(status='PROVISIONAL_ROUTE_AGREEMENT',reasons=['QUERY_END_NOT_COMMITTED'])
        result.update(status='FAILED',error=dict(type=type(exc).__name__,message=str(exc)))
        try:progress(dict(event='FAILED',partial=deepcopy(result),**RIGHTS))
        except BaseException as logging_error:
            try:exc.add_note('FAILED progress callback also failed: '+type(logging_error).__name__+': '+str(logging_error))
            except BaseException:pass
        raise
