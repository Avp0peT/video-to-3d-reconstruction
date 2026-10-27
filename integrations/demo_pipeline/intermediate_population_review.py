"""Execute a frozen foreground population in bounded 64-point comparisons."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from . import temporal_review as inputs
from .intermediate_tracking import compare_edge
from .intermediate_population import summarize_edge
from .intermediate_population_report import render_population


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')


def review_population(plan, frames, masks, decoded, output, refs, cv, *, budget=lambda:None, camera_result=None, classification=None, anchor_plan=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write(output/'PLAN.json',plan)
    refs.append(inputs.fingerprint(output/'PLAN.json'))
    result=dict(status='RUNNING',plan=deepcopy(plan),edges=[],flow_calls=0,counts={},
        geometry_accepted=False,identity_verified=False,intermediate_masks_verified=False,
        new_model_points=0,new_cameras=0,new_corner_detection_calls=0,
        intermediate_new_seeds=0,source_observations_modified=False)
    originals={f['decode_index']:(f,m) for f,m in zip(frames,masks)} if plan['edges'] else {}
    event_path=output/'EVENTS.jsonl';digest=hashlib.sha256();seq=0;artifacts=[]
    def emit(kind,**payload):
        nonlocal seq
        data=(json.dumps(dict(seq=seq,event=kind,monotonic=time.monotonic(),**payload),
                ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
        with event_path.open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
        digest.update(data);seq+=1
    def verify():
        for ref in refs:
            if inputs.fingerprint(ref['path']) != ref:
                raise ValueError('foreground population source drift: '+ref['path'])
    emit('PLAN',plan=plan)
    try:
        verify();budget()
        for edge_number,edge in enumerate(plan['edges']):
            budget()
            merged=dict(model_key=edge['model_key'],source_input_index=edge['source_input_index'],
                target_input_index=edge['target_input_index'],decode_path=edge['decode_path'],
                status='RUNNING',rows=[],flow_calls=0,batches=[],geometry_accepted=False)
            result['edges'].append(merged)
            domain={i:dict(decoded[i],mask=None,original_mask_status='UNKNOWN_NOT_SEGMENTED') for i in edge['decode_path']}
            for i in (edge['source_decode_index'],edge['target_decode_index']):
                f,mask=originals[i]
                if not domain[i].get('endpoint_verified') or mask is None or f['original_mask_status']!='OK':
                    raise ValueError('verified original foreground endpoint required')
                domain[i].update(mask=mask,original_mask_status='OK')
            for batch_number,batch in enumerate(edge['batches']):
                budget()
                # The cohort and order were fixed before any new LK execution.
                requested=dict(edge,points=deepcopy(batch['points']))
                emit('BATCH_BEGIN',edge_number=edge_number,batch_number=batch_number,point_ids=batch['point_ids'])
                def progress(value):
                    # One actual last row per event, not a quadratic dump of
                    # every previously completed row in the batch.
                    emit('POINT_PROGRESS',edge_number=edge_number,batch_number=batch_number,
                         batch_status=value['status'],flow_calls=value['flow_calls'],
                         point=value['rows'][-1] if value['rows'] else None)
                try:
                    measured=compare_edge(requested,domain,cv,budget=budget,progress=progress)
                except Exception as exc:
                    partial=getattr(exc,'partial_result',None)
                    if partial is not None:
                        merged['rows'].extend(partial.get('rows',[]))
                        merged['flow_calls']+=partial.get('flow_calls',0)
                    raise
                if [r['group_id'] for r in measured['rows']] != batch['point_ids']:
                    raise ValueError('measured point cohort differs from frozen batch')
                batch_path=output/f'edge_{edge_number:03d}_batch_{batch_number:03d}.json'
                write(batch_path,measured);artifacts.append(batch_path)
                refs.append(inputs.fingerprint(batch_path))
                merged['batches'].append(dict(batch_number=batch_number,points=len(measured['rows']),
                    record=batch_path.name,flow_calls=measured['flow_calls']))
                merged['rows'].extend(measured['rows']);merged['flow_calls']+=measured['flow_calls']
                emit('BATCH_END',edge_number=edge_number,batch_number=batch_number,
                     counts=measured['counts'],flow_calls=measured['flow_calls'])
            merged['status']='COMPARED'
            merged['counts']=dict(Counter(r['outcome'] for r in merged['rows']))
            merged['summary']=summarize_edge(edge,merged,domain)
            emit('EDGE_END',edge_number=edge_number,summary=merged['summary'])
        result['flow_calls']=sum(e['flow_calls'] for e in result['edges'])
        result['counts']=dict(Counter(r['outcome'] for e in result['edges'] for r in e['rows']))
        result['status']='POPULATION_COMPARISON_COMPLETE' if plan['edges'] else 'NOT_APPLICABLE_NO_ELIGIBLE_INTERVALS'
        verify();budget()
        display={i:dict(value) for i,value in decoded.items()}
        for edge in plan['edges']:
            for i in (edge['source_decode_index'],edge['target_decode_index']):
                f,mask=originals[i];display[i].update(mask=mask,original_mask_status='OK')
        from .short_tracklets import extract_tracklets
        from .short_tracklet_geometry import assess_tracklets
        from .short_tracklet_report import render_tracklets
        short_dir=output/'short_tracks';short_dir.mkdir(exist_ok=False)
        ledger=extract_tracklets(result,display,budget=budget)
        write(short_dir/'TRACKLETS.json',ledger)
        geometry=assess_tracklets(ledger,frames,masks,camera_result or {},budget=budget)
        write(short_dir/'GEOMETRY.json',geometry)
        artifacts+=render_tracklets(short_dir,ledger,geometry,display,budget=budget)
        artifacts += [short_dir/'TRACKLETS.json',short_dir/'GEOMETRY.json']
        result['short_tracks']=dict(status=ledger['status'],counts=ledger['counts'],
            geometry_counts=geometry['counts'],numeric_groups=geometry['numeric_groups'],
            geometry_accepted=False,new_model_points=0,
            report='intermediate_frames/population/short_tracks/index.html')
        emit('SHORT_TRACKLETS',summary=result['short_tracks'])
        from .tracklet_camera_review import assess_cameras
        from .tracklet_epipolar import assess_epipolar
        from .tracklet_camera_report import render_camera_review
        diagnostic_dir=output/'camera_diagnostics';diagnostic_dir.mkdir(exist_ok=False)
        # Both consume the exact current in-memory ledger; neither imports old runs.
        projection=assess_cameras(ledger,classification,frames,camera_result or {},budget=budget)
        write(diagnostic_dir/'PROJECTION.json',projection)
        epipolar=assess_epipolar(ledger,budget=budget)
        write(diagnostic_dir/'EPIPOLAR.json',epipolar)
        artifacts+=render_camera_review(diagnostic_dir,projection,epipolar,display,budget=budget)
        artifacts += [diagnostic_dir/'PROJECTION.json',diagnostic_dir/'EPIPOLAR.json']
        result['camera_diagnostics']=dict(status=projection['status'],projection_counts=projection['counts'],
            numeric_projection_fits=projection['numeric_fits'],exact_xyz_links=len(projection['joins']),
            epipolar_counts=epipolar['counts'],geometry_accepted=False,camera_pose_established=False,
            new_cameras=0,new_model_points=0,report='intermediate_frames/population/camera_diagnostics/index.html')
        emit('CAMERA_DIAGNOSTICS',summary=result['camera_diagnostics'])
        if anchor_plan is not None:
            from .anchor_route_review import review_anchor_routes
            result['anchor_routes'] = review_anchor_routes(anchor_plan,ledger,frames,masks,decoded,
                output/'anchor_routes',refs,cv,camera_result or {},projection,budget=budget)
            artifacts += [Path(b['path']) for b in result['anchor_routes']['source_bindings']]
            emit('ANCHOR_ROUTES',summary=result['anchor_routes'])
        from .region_review import review_regions
        region_display = {i: dict(v) for i, v in display.items()}
        if region_display:
            for f, mask in zip(frames, masks):
                if f['decode_index'] in region_display:
                    region_display[f['decode_index']].update(mask=mask, original_mask_status=f['original_mask_status'])
        result['regions'] = review_regions(ledger, frames, masks, region_display,
            output/'regions', refs, cv, budget=budget)
        artifacts += [Path(b['path']) for b in result['regions']['source_bindings']]
        emit('REGION_GROWTH', summary=result['regions'])
        artifacts+=render_population(output,result,display,budget=budget)
        verify();budget()
        emit('COMPLETE',status=result['status'],counts=result['counts'],flow_calls=result['flow_calls'])
        if hashlib.sha256(event_path.read_bytes()).hexdigest()!=digest.hexdigest():
            raise ValueError('population event evidence drift')
        result['source_unchanged']=True;write(output/'RESULT.json',result)
        artifacts += [output/'PLAN.json',output/'RESULT.json',event_path]
        bindings=[inputs.fingerprint(p) for p in dict.fromkeys(artifacts)]
        return dict(status=result['status'],counts=result['counts'],flow_calls=result['flow_calls'],
            points=sum(len(e['rows']) for e in result['edges']),edges=len(result['edges']),
            summaries=[e['summary'] for e in result['edges']],
            short_tracks=result['short_tracks'],
            camera_diagnostics=result['camera_diagnostics'],
            anchor_routes=result.get('anchor_routes'),
            regions=result['regions'],
            geometry_accepted=False,new_model_points=0,intermediate_masks_verified=False,
            report='intermediate_frames/population/index.html',source_bindings=bindings)
    except Exception as exc:
        result.update(status='RESOURCE_LIMIT' if isinstance(exc,(TimeoutError,MemoryError)) else 'FAILED',
            reason=f'{type(exc).__name__}: {exc}',source_unchanged=None,
            flow_calls=sum(e.get('flow_calls',0) for e in result['edges']),
            counts=dict(Counter(r.get('outcome','INCOMPLETE') for e in result['edges'] for r in e.get('rows',[]))))
        try:write(output/'FAILURE.json',result)
        except Exception as delivery:
            if hasattr(exc,'add_note'):exc.add_note('Failure evidence delivery: '+str(delivery))
        raise
