"""Frozen opposite-direction measured anchor routes; no pose or XYZ estimation."""
from copy import deepcopy
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
from .original_anchor_review import _camera
from .intermediate_tracking import compare_edge
from .short_tracklets import _flow, _xy, _Rejected
from .temporal_review import fingerprint

RIGHTS=dict(geometry_accepted=False,identity_verified=False,native_injected=False,new_cameras=0,new_model_points=0,original_observations_modified=False)
def need(ok,reason):
    if not ok:raise ValueError(reason)


def plan_anchor_routes(population_plan, frames, camera_result):
    need(isinstance(population_plan.get('edges'),list),'INVALID_POPULATION_EDGES')
    if not population_plan['edges']:
        return dict(status='ANCHOR_ROUTE_PLAN_FROZEN',routes=[],omitted_routes=[],
            decode_indices=[],extra_decode_indices=[],max_points=96,max_points_per_route=64,**RIGHTS)
    need(all(all(type(f.get(k)) is int and f[k]>=0 for k in ('input_index','decode_index')) for f in frames),'INVALID_FRAME_IDENTITY_TYPE')
    registry={f['input_index']:f for f in frames}
    need(len(registry)==len(frames),'DUPLICATE_FRAME_IDENTITY')
    models={}
    for m in camera_result.get('models',[])+camera_result.get('unsupported_models',[]):
        need(m['model_key'] not in models,'DUPLICATE_MODEL')
        cams={}
        for table in ('cameras','unsupported_cameras'):
            for index,c in m.get(table,{}).items():
                index=int(index);need(index not in cams,'DUPLICATE_CAMERA');cams[index]=c
        models[m['model_key']]=cams
    candidates=[];omitted=[];seen=set()
    for edge in population_plan['edges']:
        source=edge['source_input_index'];sf=registry[source];direction=-1 if edge['target_decode_index']>edge['source_decode_index'] else 1
        for model in edge.get('model_keys',[edge['model_key']]):
            if (source,model) in seen:continue
            seen.add((source,model));base=dict(model_key=model,source_input_index=source)
            cams=models.get(model,{})
            def eligible(index):
                f=registry.get(index);c=cams.get(index)
                if f is None or c is None or f.get('original_mask_status')!='OK' or (f['width'],f['height'])!=(sf['width'],sf['height']):return False
                try:_camera(c,f,index)
                except ValueError:return False
                return True
            if not eligible(source):omitted.append(dict(base,reason='NO_SUPPORTED_ORIGINAL_SOURCE_CAMERA'));continue
            choices=sorted((i for i in cams if eligible(i) and 0<direction*(registry[i]['decode_index']-sf['decode_index'])<=8),key=lambda i:(abs(registry[i]['decode_index']-sf['decode_index']),i))
            if len(choices)<2:omitted.append(dict(base,reason='NO_TWO_NEARBY_OPPOSITE_CAMERAS'));continue
            indices=[source]+choices[:2];decodes=[registry[i]['decode_index'] for i in indices]
            if len(set(cams[i]['image_id'] for i in indices))!=3:omitted.append(dict(base,reason='DUPLICATE_CAMERA_IMAGE_ID'));continue
            path=list(range(decodes[0],decodes[-1]+direction,direction))
            candidates.append(dict(base,route_id=f'{model}:source:{source}',source_decode_index=decodes[0],camera_input_indices=indices,camera_decode_indices=decodes,decode_path=path,
                point_ids=[p['point_id'] for p in edge['points']],points=deepcopy(edge['points']),cameras=[deepcopy(cams[i]) for i in indices],direction=direction))
    routes=[];extra=set();existing=set(population_plan.get('decode_indices',[]))
    for route in sorted(candidates,key=lambda r:(r['source_input_index'],r['model_key'])):
        added=set(route['decode_path'])-existing
        if len(routes)>=2:omitted.append(dict(route_id=route['route_id'],reason='MAX_TWO_ROUTES'));continue
        if len(extra|added)>18:omitted.append(dict(route_id=route['route_id'],reason='EXTRA_FRAME_BUDGET'));continue
        route['status']='SELECTED';routes.append(route);extra|=added
    return dict(status='ANCHOR_ROUTE_PLAN_FROZEN',routes=routes,omitted_routes=omitted,
                decode_indices=sorted({i for r in routes for i in r['decode_path']}),extra_decode_indices=sorted(extra),max_points=96,max_points_per_route=64,**RIGHTS)


def measure_anchor_routes(plan, ledger, frames, masks, decoded, output, refs, cv, *, budget=lambda:None):
    import numpy as np
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    source_refs=deepcopy(refs)
    def verify():
        for ref in source_refs:need(fingerprint(ref['path'])==ref,'ANCHOR_SOURCE_DRIFT')
    verify()
    def write(name,value):
        p=output/name
        with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
        return p
    original={f['input_index']:(f,m) for f,m in zip(frames,masks)}
    need(len(frames)==len(masks) and len(original)==len(frames),'FRAME_MASK_IDENTITY')
    candidates={}
    for r in ledger['records']:
        if r['status']!='CANDIDATE_2D_TRACKLET':continue
        need(all(type(r.get(k)) is int and r[k]>=0 for k in ('track_id','identity_segment','source_input_index')),'INVALID_CANDIDATE_IDENTITY_TYPE')
        key=(r['point_id'],r['source_input_index']);need(key not in candidates,'DUPLICATE_CANDIDATE_IDENTITY');candidates[key]=r
    queues={};omitted=[]
    for route in plan['routes']:
        need(type(route.get('source_input_index')) is int and all(type(i) is int and i>=0 for i in route['camera_input_indices']+route['camera_decode_indices']),'INVALID_ROUTE_IDENTITY_TYPE')
        queue=deque();route_point_ids=set()
        for p in route['points']:
            need(p['point_id'] not in route_point_ids,'DUPLICATE_ROUTE_POINT_ID');route_point_ids.add(p['point_id'])
            need(all(type(p.get(k)) is int and p[k]>=0 for k in ('track_id','identity_segment','source_input_index')),'INVALID_POINT_IDENTITY_TYPE')
            r=candidates.get((p['point_id'],route['source_input_index']))
            if r is None:continue
            need(p['point_id']==f"track:{p['track_id']}:segment:{p['identity_segment']}:source:{p['source_input_index']}",'INVALID_POINT_IDENTITY')
            need(r['source_point']==p and r['track_id']==p['track_id'] and r['identity_segment']==p['identity_segment'],'COHORT_SOURCE_IDENTITY_MISMATCH')
            queue.append(r)
        queues[route['route_id']]=queue
    selected=[];perroute=Counter();selected_groups=set()
    while any(queues.values()):
        for route in plan['routes']:
            q=queues[route['route_id']]
            if not q:continue
            r=q.popleft()
            group=(route['model_key'],r['track_id'],r['identity_segment'])
            if group in selected_groups:
                omitted.append(dict(route_id=route['route_id'],tracklet_id=r['tracklet_id'],reason='DUPLICATE_GROUP_ANOTHER_SOURCE'));continue
            if len(selected)>=96 or perroute[route['route_id']]>=64:
                omitted.append(dict(route_id=route['route_id'],tracklet_id=r['tracklet_id'],reason='FROZEN_POINT_BUDGET'));continue
            selected_groups.add(group);perroute[route['route_id']]+=1;selected.append((route,r))
    frozen=[dict(route_id=route['route_id'],tracklet_id=r['tracklet_id'],point_id=r['point_id'],source_raw_xy=deepcopy(r['source_point']['source_raw_xy'])) for route,r in selected]
    write('PLAN.json',dict(route_plan=plan,frozen_cohort=frozen,omitted=omitted))
    result=dict(status='RUNNING',records=[],frozen_cohort=frozen,omitted=omitted,counts={},flow_calls=0,source_bindings=[],**RIGHTS)
    eventpath=output/'EVENTS.jsonl';sequence=0
    def emit(kind,**payload):
        nonlocal sequence
        with eventpath.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(seq=sequence,event=kind,**payload),allow_nan=False)+'\n');f.flush()
        sequence+=1
    emit('PLAN',frozen_cohort=frozen)
    try:
        for route,seed in selected:
            budget();point=seed['source_point']
            row=dict(tracklet_id=seed['tracklet_id'],point_id=seed['point_id'],model_key=route['model_key'],track_id=seed['track_id'],identity_segment=seed['identity_segment'],source_input_index=seed['source_input_index'],source_point=deepcopy(point),route=deepcopy({k:v for k,v in route.items() if k not in ('points','point_ids')}),observations=[],segment_measurements=[],status='BLOCKED',reasons=[],**RIGHTS)
            result['records'].append(row);current=list(point['source_raw_xy']);indices=route['camera_input_indices'];decodes=route['camera_decode_indices']
            for index,cam in zip(indices,route['cameras']):
                f,m=original[index];_camera(cam,f,index)
                need(m is not None and f['original_mask_status']=='OK','ORIGINAL_CAMERA_MASK_REQUIRED')
                need(decoded[f['decode_index']].get('endpoint_verified') is True,'DECODED_CAMERA_ENDPOINT_UNVERIFIED')
            sf,_=original[indices[0]];cam=route['cameras'][0]
            row['observations'].append(dict(input_index=indices[0],decode_index=decodes[0],raw_xy=list(current),width=sf['width'],height=sf['height'],frame_id=sf['frame_id'],image_id=cam['image_id']))
            for si,(a,b) in enumerate(zip(decodes,decodes[1:])):
                direction=route['direction'];path=list(range(a,b+direction,direction));domain={i:dict(decoded[i],mask=None,original_mask_status='UNKNOWN_NOT_SEGMENTED') for i in path}
                for idx in (si,si+1):
                    f,m=original[indices[idx]];domain[decodes[idx]].update(mask=m,original_mask_status='OK')
                edge=dict(model_key=route['model_key'],source_input_index=indices[si],target_input_index=indices[si+1],source_decode_index=a,target_decode_index=b,decode_path=path,points=[dict(group_id=seed['point_id'],source_raw_xy=list(current))])
                try:
                    measured=compare_edge(edge,domain,cv,budget=budget,progress=lambda e:emit('MEASUREMENT_PROGRESS',tracklet_id=seed['tracklet_id'],segment=si,measurement=e))
                except Exception as exc:
                    partial=getattr(exc,'partial_result',None)
                    if partial is not None:row['segment_measurements'].append(partial);result['flow_calls']+=partial.get('flow_calls',0)
                    raise
                row['segment_measurements'].append(measured);result['flow_calls']+=measured['flow_calls'];arm=measured['rows'][0]['dense']
                if arm['status']!='PASSED':row['reasons']=list(arm['reasons']);break
                # Independently recompute every actual forward/backward leg.
                verified=list(current)
                need(len(arm['steps'])==len(path)-1,'ANCHOR_STEP_COUNT_MISMATCH')
                try:
                    for j,(u,v) in enumerate(zip(path,path[1:])):
                        step=arm['steps'][j];fw=_flow(step['forward'],u,v,_xy(verified),domain);bw=_flow(step['backward'],v,u,fw,domain)
                        fb=float(np.linalg.norm(bw-_xy(verified)))
                        stored=step.get('fb_error_px');need(type(stored) in (int,float) and np.isfinite(stored) and 0<=stored<=1 and abs(stored-fb)<=1e-6,'ANCHOR_FB_RECORD_MISMATCH')
                        need(fb<=1,'RECOMPUTED_FB_THRESHOLD');verified=fw.tolist()
                    cycle=arm['fullcycle'];need(cycle.get('status')=='PASSED' and len(cycle.get('steps',[]))==len(path)-1,'ANCHOR_CYCLE_STEP_COUNT_OR_STATUS')
                    reverse=list(verified)
                    for j,(u,v) in enumerate(zip(reversed(path),list(reversed(path))[1:])):
                        reverse=_flow(cycle['steps'][j],u,v,_xy(reverse),domain).tolist()
                    error=float(np.linalg.norm(_xy(reverse)-_xy(current)));stored=cycle.get('error_px')
                    need(type(stored) in (int,float) and np.isfinite(stored) and 0<=stored<=1 and abs(stored-error)<=1e-6 and error<=1,'ANCHOR_RECOMPUTED_CYCLE_THRESHOLD_OR_RECORD')
                except _Rejected as exc:raise ValueError('ANCHOR_RAW_MEASUREMENT_INVALID: '+str(exc)) from exc
                current=verified;f,_=original[indices[si+1]];cam=route['cameras'][si+1]
                row['observations'].append(dict(input_index=indices[si+1],decode_index=b,raw_xy=list(current),width=f['width'],height=f['height'],frame_id=f['frame_id'],image_id=cam['image_id']))
            if len(row['observations'])==3:row['status']='MEASURED_THREE_VIEW_ANCHOR'
            emit('POINT_END',record=row)
        result['status']='ANCHOR_ROUTE_MEASUREMENTS_COMPLETE';result['counts']=dict(Counter(r['status'] for r in result['records']));verify();budget();write('RESULT.json',result)
        emit('COMPLETE',counts=result['counts'],flow_calls=result['flow_calls'])
    except Exception as exc:
        result.update(status='INTERRUPTED',reason=f'{type(exc).__name__}: {exc}',counts=dict(Counter(r['status'] for r in result['records'])))
        try:exc.partial_result=deepcopy(result)
        except (AttributeError,TypeError):pass
        try:write('FAILURE.json',result)
        except Exception as delivery:
            if hasattr(exc,'add_note'):exc.add_note(str(delivery))
        raise
    bindings=[]
    for path in sorted(output.iterdir()):
        bindings.append(fingerprint(path))
    refs.extend(bindings);result['source_bindings']=bindings
    return result
