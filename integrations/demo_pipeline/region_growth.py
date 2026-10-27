"""Frozen, bounded source-mask expansion of measured region neighborhoods."""
from collections import Counter
from copy import deepcopy
import math
import numpy as np
from .region_tracking import make_region_points, window, xy, RIGHTS

OFFSETS=((0,-32),(32,0),(0,32),(-32,0))
MAX_REGIONS=24
MAX_DEPTH=2


def _need(value,reason):
    if not value:raise ValueError(reason)


def _descriptor(region):
    depth=region.get('depth',0)
    _need(type(depth) is int and 0<=depth<=MAX_DEPTH,'INVALID_REGION_DEPTH')
    source=region.get('source_input_index');decode=region.get('source_decode_index')
    _need(type(source) is int and source>=0 and type(decode) is int and decode>=0,'REGION_SOURCE_IDENTITY')
    rid=region.get('region_id');root=region.get('root_tracklet_id',region.get('tracklet_id'))
    _need(type(rid) is str and rid and type(root) is str and root,'REGION_IDENTITY')
    path=region.get('decode_path');_need(isinstance(path,list) and 3<=len(path)<=5 and
        all(type(i) is int and i>=0 for i in path) and path[0]==decode,'REGION_PATH_BOUND')
    direction=1 if path[-1]>path[0] else -1
    _need(all(b-a==direction for a,b in zip(path,path[1:])),'NONCONSECUTIVE_REGION_PATH')
    parent=region.get('parent_region_id')
    _need(parent is None if depth==0 else type(parent) is str and bool(parent),'REGION_PARENT_IDENTITY')
    return dict(region_id=rid,depth=depth,parent_region_id=parent,root_tracklet_id=root,
        source_input_index=source,source_decode_index=decode,decode_path=list(path),
        center_raw_xy=xy(region['center_raw_xy']).tolist())


def _numeric_record(record):
    if not isinstance(record,dict):return None
    status=record.get('status');error=record.get('error')
    if type(status) is not int or status!=1 or type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=25:return None
    try:
        observed=xy(record.get('observed_raw_xy'));raw=xy(record.get('xy'))
    except (ValueError,TypeError):return None
    return observed if np.array_equal(raw+.5,observed) else None


def _prefix(point,region,domains):
    """Recompute the first two legs; final labels alone never qualify a parent."""
    path=region['decode_path'];source=xy(point['source_raw_xy'])
    if window(source.tolist(),domains[str(path[0])]):return []
    observations=point.get('observations',[])
    if not observations or observations[0].get('decode_index')!=path[0]:return []
    try:
        if not np.array_equal(xy(observations[0]['raw_xy']),source):return []
    except (ValueError,KeyError,TypeError):return []
    current=source;verified=[source.tolist()]
    for ordinal,(a,b) in enumerate(zip(path[:2],path[1:3])):
        steps=point.get('steps',[])
        if ordinal>=len(steps) or ordinal+1>=len(observations):break
        step=steps[ordinal]
        if step.get('status')!='PASSED' or type(step.get('from_decode_index')) is not int or type(step.get('to_decode_index')) is not int or (step['from_decode_index'],step['to_decode_index'])!=(a,b):break
        forward=_numeric_record(step.get('forward'));backward=_numeric_record(step.get('backward'))
        if forward is None or backward is None:break
        if window(forward.tolist(),domains[str(b)]) or window(backward.tolist(),domains[str(a)]):break
        fb=float(np.linalg.norm(backward-current));stored=step.get('fb_error_px')
        if type(stored) not in (int,float) or not math.isfinite(stored) or not 0<=stored<=1 or fb>1 or abs(stored-fb)>1e-6:break
        if step.get('destination_mask_status')!=domains[str(b)]['original_mask_status']:break
        ob=observations[ordinal+1]
        try:
            if ob.get('decode_index')!=b or not np.array_equal(xy(ob['raw_xy']),forward):break
        except (ValueError,KeyError,TypeError):break
        current=forward;verified.append(forward.tolist())
    return verified


def _parent_eligibility(region,domains,complete,budget):
    report=dict(region_id=region['region_id'],eligible=False,reasons=[],unique_two_leg_points=0,
                center_two_legs=False,affine_legs=[])
    if not complete:report['reasons']=['MEASUREMENT_NOT_COMPLETE'];return report
    points=region.get('points');_need(isinstance(points,list) and len(points)==81,'REGION_GRID_BOUND')
    expected=make_region_points(region['region_id'],region['center_raw_xy'],domains[str(region['source_decode_index'])],budget=budget)
    byordinal={p.get('grid_ordinal'):p for p in points}
    _need(len(byordinal)==81 and set(byordinal)==set(range(81)),'GRID_ORDINAL_IDENTITIES')
    prefixes={};unique=set();ids=set();center=None
    for original in expected:
        budget();p=byordinal[original['grid_ordinal']];pid=p.get('point_id')
        _need(pid==original['point_id'] and pid not in ids,'GRID_POINT_IDENTITY');ids.add(pid)
        _need(p['source_raw_xy']==original['source_raw_xy'] and p.get('is_center')==original['is_center'],'GRID_SOURCE_COORDINATE_CHANGED')
        coords=_prefix(p,region,domains);prefixes[pid]=coords
        if original['is_center']:center=p
        count=p.get('verified_legs')
        if len(coords)>=3 and p.get('status')=='TRACKLET_2D' and type(count) is int and 2<=count<=4:
            unique.add(tuple(p['source_raw_xy']))
    report['unique_two_leg_points']=len(unique)
    report['center_two_legs']=bool(center and len(prefixes[center['point_id']])>=3 and
        center.get('status')=='TRACKLET_2D' and type(center.get('verified_legs')) is int and 2<=center['verified_legs']<=4)
    if not report['center_two_legs']:report['reasons'].append('CENTER_WITHOUT_TWO_VERIFIED_LEGS')
    if len(unique)<12:report['reasons'].append('FEWER_THAN_TWELVE_UNIQUE_TWO_LEG_POINTS')
    diagnostics=region.get('affine_diagnostics',[])
    for leg,(a,b) in enumerate(zip(region['decode_path'][:2],region['decode_path'][1:3])):
        budget();item=dict(leg=leg,from_decode_index=a,to_decode_index=b,accepted=False,reason=None)
        report['affine_legs'].append(item)
        matches=[d for d in diagnostics if (d.get('from_decode_index'),d.get('to_decode_index'))==(a,b)]
        if len(matches)!=1 or matches[0].get('status')!='EVALUATED':item['reason']='AFFINE_NOT_EVALUATED';continue
        d=matches[0];passed=[p for p in points if len(prefixes[p['point_id']])>=leg+2]
        train=[p for p in passed if p['grid_ordinal']%4!=3];held=[p for p in passed if p['grid_ordinal']%4==3]
        if d.get('train_ids')!=[p['point_id'] for p in train] or d.get('holdout_ids')!=[p['point_id'] for p in held]:item['reason']='AFFINE_SPLIT_IDENTITY';continue
        if len(train)<6 or len(held)<3:item['reason']='AFFINE_SPLIT_COUNT';continue
        matrix=np.asarray(d.get('matrix'))
        if matrix.shape!=(2,3) or matrix.dtype.kind not in 'iuf' or not np.isfinite(matrix).all():item['reason']='AFFINE_MATRIX_INVALID';continue
        valid=True
        for name,subset in [('train',train),('holdout',held)]:
            budget();source=np.asarray([prefixes[p['point_id']][leg] for p in subset]);target=np.asarray([prefixes[p['point_id']][leg+1] for p in subset])
            support=[]
            for coords in (source,target):
                budget();singular=np.linalg.svd(coords-coords.mean(0),compute_uv=False);budget()
                support.append(bool(singular[0]>0 and singular[-1]>singular[0]*1e-8))
            item[name+'_rank_supported']=all(support)
            if not all(support):valid=False;item['reason']='AFFINE_SPLIT_RANK';break
            errors=np.linalg.norm(np.column_stack((source,np.ones(len(source))))@matrix.T-target,axis=1)
            stored=d.get(name) or {}
            if not np.isfinite(errors).all() or len(stored.get('residual_pixels',[]))!=len(errors) or not np.allclose(errors,stored['residual_pixels'],rtol=1e-9,atol=1e-7) or not np.isclose(errors.max(),stored.get('max_px',math.inf),rtol=1e-9,atol=1e-7) or not np.isclose(np.median(errors),stored.get('median_px',math.inf),rtol=1e-9,atol=1e-7):
                valid=False;item['reason']='AFFINE_RESIDUAL_RECORD_MISMATCH';break
            item[name+'_max_px']=float(errors.max());item[name+'_median_px']=float(np.median(errors))
        if valid and item['holdout_max_px']<=3 and item['holdout_median_px']<=2:item['accepted']=True
        elif valid:item['reason']='AFFINE_HELDOUT_THRESHOLD'
    if not all(d['accepted'] for d in report['affine_legs']):report['reasons'].append('FIRST_TWO_AFFINE_LEGS_NOT_SUPPORTED')
    report['eligible']=not report['reasons']
    return report


def next_region_plan(base_plan,prior_results,*,budget=lambda:None):
    """Use cumulative measured rounds and previous frozen plan, never new LK.

    Only the highest measured depth is a frontier. All four offsets are frozen
    before any child measurement; rejected children are not replaced later.
    """
    budget();domains=base_plan['frame_domains'];_need(isinstance(domains,dict),'FRAME_DOMAINS_REQUIRED')
    results=[prior_results] if isinstance(prior_results,dict) else prior_results
    _need(isinstance(results,list) and all(isinstance(r,dict) for r in results),'PRIOR_RESULTS_REQUIRED')
    planned={};state=base_plan.get('growth_state',{})
    def add(region):
        d=_descriptor(region);rid=d['region_id']
        if rid in planned:_need(planned[rid]==d,'PLANNED_REGION_IDENTITY_CHANGED')
        else:planned[rid]=d
    for region in state.get('planned_regions',[]):add(region)
    for region in base_plan['regions']:add(region)
    _need(len(planned)<=24,'TOTAL_REGION_BOUND')
    if not state:_need(len(planned)<=8 and all(d['depth']==0 for d in planned.values()),'INITIAL_ROOT_BOUND')
    measured={};complete={}
    for result in results:
        for region in result.get('regions',[]):
            budget();rid=region['region_id'];_need(rid not in measured,'DUPLICATE_MEASURED_REGION')
            _need(rid in planned,'UNPLANNED_MEASURED_REGION');_need(_descriptor(region)==planned[rid],'MEASURED_REGION_IDENTITY_CHANGED')
            measured[rid]=region;complete[rid]=result.get('status')=='REGION_MEASUREMENTS_COMPLETE'
    expanded=set(state.get('expanded_parent_ids',[]));_need(expanded<=set(planned),'UNKNOWN_EXPANDED_PARENT')
    for d in planned.values():
        for index in d['decode_path']:_need(str(index) in domains,'MISSING_INHERITED_FRAME_DOMAIN')
        source=domains[str(d['source_decode_index'])]
        _need(source.get('original_mask_status')=='OK' and isinstance(source.get('mask_rle'),list) and len(source['mask_rle'])==source['height'],'SOURCE_MASK_DOMAIN_REQUIRED')
        if d['depth']:
            parent=planned.get(d['parent_region_id'])
            _need(parent is not None and parent['depth']+1==d['depth'] and parent['root_tracklet_id']==d['root_tracklet_id'] and
                parent['source_input_index']==d['source_input_index'] and parent['decode_path']==d['decode_path'],'GROWTH_LINEAGE_MISMATCH')
    depth=max((d['depth'] for rid,d in planned.items() if rid in measured),default=None)
    frontier=sorted((r for rid,r in measured.items() if planned[rid]['depth']==depth and rid not in expanded),key=lambda r:r['region_id'])
    reports=[];proposed=[];omitted=[];stops=[];regions=[]
    if not frontier:stops.append('FRONTIER_EXHAUSTED')
    elif depth>=MAX_DEPTH:stops.append('MAX_DEPTH_REACHED')
    elif len(planned)>=MAX_REGIONS:stops.append('TOTAL_REGION_BUDGET_REACHED')
    else:
        for parent in frontier:
            budget();eligibility=_parent_eligibility(parent,domains,complete[parent['region_id']],budget)
            reports.append(eligibility);expanded.add(parent['region_id'])
            if not eligibility['eligible']:continue
            for ordinal,(dx,dy) in enumerate(OFFSETS):
                proposed.append(dict(parent=parent,offset_ordinal=ordinal,offset_xy=[dx,dy],
                    center=(xy(parent['center_raw_xy'])+[dx,dy]).tolist()))
        for candidate in proposed:
            budget();parent=candidate['parent'];center=candidate['center'];source=parent['source_input_index']
            omission=dict(parent_region_id=parent['region_id'],depth=depth+1,offset_xy=candidate['offset_xy'],center_raw_xy=center)
            if len(planned)>=MAX_REGIONS:omitted.append(dict(omission,reason='TOTAL_REGION_BUDGET_REACHED'));continue
            if any(d['source_input_index']==source and np.linalg.norm(xy(center)-xy(d['center_raw_xy']))<32 for d in planned.values()):
                omitted.append(dict(omission,reason='CENTER_SPACING_32PX'));continue
            reason=window(center,domains[str(parent['source_decode_index'])])
            if reason:omitted.append(dict(omission,reason='SOURCE_'+reason));continue
            rid=f"{parent['region_id']}:growth:{depth+1}:offset:{candidate['offset_ordinal']}"
            root=planned[parent['region_id']]['root_tracklet_id']
            region=dict(region_id=rid,depth=depth+1,parent_region_id=parent['region_id'],root_tracklet_id=root,
                tracklet_id=root,source_input_index=source,source_decode_index=parent['source_decode_index'],
                decode_path=list(parent['decode_path']),center_raw_xy=center,
                source_bbox_raw_xy=[center[0]-16,center[1]-16,center[0]+16,center[1]+16],
                points=make_region_points(rid,center,domains[str(parent['source_decode_index'])],budget=budget),
                selection_cell=parent.get('selection_cell'),growth_offset_xy=candidate['offset_xy'],
                physical_sample_independence=False,measurement_inherited=False)
            add(region);regions.append(region)
        if not regions:
            if not any(r['eligible'] for r in reports):stops.append('NO_ELIGIBLE_FRONTIER_PARENTS')
            if any(o['reason'].startswith('SOURCE_') for o in omitted):stops.append('SOURCE_MASK_OR_IMAGE_BOUNDARY')
            if any(o['reason']=='CENTER_SPACING_32PX' for o in omitted):stops.append('FRONTIER_SPACING_EXHAUSTED')
            if len(planned)>=MAX_REGIONS:stops.append('TOTAL_REGION_BUDGET_REACHED')
    profile=dict(deepcopy(base_plan['profile']),max_regions=24,max_total_regions=24,max_depth=2,max_legs=4,
        max_total_grid_points=1944,min_center_distance_px=32)
    statistics=dict(planned_total=len(planned),new_regions=len(regions),remaining_regions=24-len(planned),
        max_total_regions=24,max_depth=2,frontier_depth=depth,frontier_parents=len(frontier),
        eligible_parents=sum(r['eligible'] for r in reports),proposed_candidates=len(proposed),
        omitted_candidates=len(omitted),stop_reasons=stops,terminal=not regions,parent_eligibility=reports)
    return dict(status='REGION_PLAN_FROZEN',regions=regions,omitted_centers=omitted,frame_domains=deepcopy(domains),
        profile=profile,growth_state=dict(planned_regions=list(planned.values()),expanded_parent_ids=sorted(expanded)),
        growth_statistics=statistics,growth_basis='source-mask spatial expansion of correlated measured neighborhoods; no identity or geometry acceptance',**RIGHTS)
