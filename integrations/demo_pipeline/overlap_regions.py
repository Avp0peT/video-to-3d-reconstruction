"""Source-foreground area planning and fixed-edge warp agreement, without fitting."""
from collections import Counter
from copy import deepcopy
import math
import numpy as np

RIGHTS=dict(geometry_accepted=False,identity_verified=False,new_cameras=0,new_model_points=0,
            new_3d_points=0,independent_point_matches=0)
SIZE=33
STRIDE=16


def _need(ok,reason):
    if not ok:raise ValueError(reason)


def _sources(plan,budget):
    sources={}
    for r in plan['regions']:
        budget();source=r.get('source_input_index');decode=r.get('source_decode_index');path=r.get('decode_path')
        _need(type(source) is int and source>=0 and type(decode) is int and decode>=0,'SOURCE_IDENTITY')
        _need(isinstance(path,list) and 3<=len(path)<=5 and all(type(i) is int and i>=0 for i in path) and path[0]==decode,'SOURCE_ROUTE_BOUND')
        direction=1 if path[-1]>path[0] else -1
        _need(all(b-a==direction for a,b in zip(path,path[1:])),'SOURCE_ROUTE_NOT_CONSECUTIVE')
        if source not in sources:
            sources[source]=dict(source_input_index=source,source_decode_index=decode,decode_path=list(path),_declared=[])
        item=sources[source];previous=item['decode_path'];short=min(len(previous),len(path))
        _need(item['source_decode_index']==decode and previous[:short]==path[:short],
            'CONFLICTING_SAME_SOURCE_ROUTE')
        if len(path)>len(previous):item['decode_path']=list(path)
        item['_declared'].append(tuple(path))
    _need(len(sources)<=2 and len({s['source_decode_index'] for s in sources.values()})==len(sources),'SOURCE_BOUND_OR_DECODE_COLLISION')
    for item in sources.values():
        declarations=Counter(item.pop('_declared'))
        item['route_selection']=dict(policy='LONGEST_DECLARED_COMPATIBLE_PREFIX',
            selected_decode_path=list(item['decode_path']),
            declared_routes=[dict(decode_path=list(path),declaration_count=declarations[path])
                for path in sorted(declarations,key=lambda p:(len(p),p))],
            extended_beyond_declared_route=False)
    return [sources[i] for i in sorted(sources)]


def _mask(domain,budget):
    w=domain.get('width');h=domain.get('height')
    _need(type(w) is int and type(h) is int and 33<=w<=16384 and 33<=h<=16384,'SOURCE_IMAGE_BOUND')
    _need(domain.get('original_mask_status')=='OK' and isinstance(domain.get('mask_rle'),list) and len(domain['mask_rle'])==h,'KNOWN_SOURCE_MASK_REQUIRED')
    mask=np.zeros((h,w),np.bool_)
    for y,line in enumerate(domain['mask_rle']):
        budget();previous=0
        for a,b in line:
            _need(type(a) is int and type(b) is int and previous<=a<b<=w,'INVALID_SOURCE_MASK_RLE')
            mask[y,a:b]=True;previous=b
    _need(mask.any(),'EMPTY_SOURCE_FOREGROUND')
    return mask


def _bbox(domain,budget):
    _need(domain.get('original_mask_status')=='OK' and isinstance(domain.get('mask_rle'),list) and
        len(domain['mask_rle'])==domain.get('height'),'KNOWN_SOURCE_MASK_REQUIRED')
    left=domain['width'];right=0;top=domain['height'];bottom=0
    for y,line in enumerate(domain['mask_rle']):
        budget()
        for a,b in line:
            _need(type(a) is int and type(b) is int and 0<=a<b<=domain['width'],'INVALID_SOURCE_MASK_RLE')
            left=min(left,a);right=max(right,b);top=min(top,y);bottom=max(bottom,y+1)
    _need(right>left and bottom>top,'EMPTY_SOURCE_FOREGROUND')
    return [left,top,right,bottom]


def _indices(mask,roi):
    y,x=np.nonzero(mask[roi['top']:roi['top']+roi['height'],roi['left']:roi['left']+roi['width']])
    return ((y+roi['top'])*mask.shape[1]+x+roi['left']).astype(np.int32)


def _roi(region):
    center=region.get('center_raw_xy');_need(isinstance(center,list) and len(center)==2 and
        all(type(v) in (int,float) and math.isfinite(v) for v in center),'REGION_CENTER')
    x,y=[int(round(v-.5)) for v in center]
    return dict(left=x-16,top=y-16,width=33,height=33)


def _region_indices(mask,region):
    r=_roi(region);_need(r['left']>=0 and r['top']>=0 and r['left']+33<=mask.shape[1] and r['top']+33<=mask.shape[0],'REGION_ROI_OUTSIDE_SOURCE')
    return _indices(mask,r)


def build_overlap_plan(source_plan,*,budget=lambda:None):
    """Freeze greedy adjacent pairs before any registration; no result refill.

    ROI top-lefts begin at each foreground bbox minimum and advance 16px.
    Image-boundary and low-foreground candidates remain in the ledger. Only
    pairs of two unselected eight-neighbor grid cells sharing >=64 foreground
    pixels can be selected. Every gain is a source-mask union gain.
    """
    budget();sources=_sources(source_plan,budget);domains=source_plan.get('frame_domains',{})
    plan=dict(status='OVERLAP_PLAN_FROZEN',regions=[],frame_domains=deepcopy(domains),sources=[],candidates=[],edges=[],planned_connected_components=[],
        selection_counts=dict(sources=0,candidates=0,selected_regions=0,edges=0,remaining_region_slots=64,unselected_eligible_candidates=0),
        profile=dict(max_sources=2,max_candidates=4096,max_regions=64,region_size=33,stride=16,
            min_source_foreground_fraction=.75,min_edge_foreground_pixels=64,max_controls=64,
            selection='source round robin; largest novel foreground union of two unselected eight-neighbor grid ROIs; y/x ties; no measurement-based refill',
            control_selection='64 evenly spaced row-major true foreground pixel indices; derived area controls, not point matches'),**RIGHTS)
    if not sources:return plan
    expected=0
    for source in sources:
        _need(str(source['source_decode_index']) in domains,'MISSING_SOURCE_DOMAIN')
        box=_bbox(domains[str(source['source_decode_index'])],budget)
        expected+=len(range(box[0],box[2],16))*len(range(box[1],box[3],16))
        _need(expected<=4096,'CANDIDATE_BOUND_4096_EXCEEDED')
    cache={};total=0
    for source in sources:
        budget();idx=source['source_input_index'];decode=source['source_decode_index']
        _need(str(decode) in domains,'MISSING_SOURCE_DOMAIN')
        for d in source['decode_path']:_need(str(d) in domains,'MISSING_ROUTE_DOMAIN')
        mask=_mask(domains[str(decode)],budget);ys,xs=np.nonzero(mask)
        bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        items=[];indices=[];byposition={};eligible_union=np.zeros(mask.size,np.bool_)
        for top in range(bbox[1],bbox[3],16):
            for left in range(bbox[0],bbox[2],16):
                budget();total+=1;_need(total<=4096,'CANDIDATE_BOUND_4096_EXCEEDED')
                roi=dict(left=left,top=top,width=33,height=33);legal=left+33<=mask.shape[1] and top+33<=mask.shape[0]
                pixels=_indices(mask,roi) if legal else np.array([],np.int32)
                eligible=legal and len(pixels)/1089>=.75
                reason=None if eligible else 'SOURCE_FOREGROUND_BELOW_0_75' if legal else 'ROI_OUTSIDE_IMAGE'
                item=dict(candidate_id=f'candidate:source:{idx}:y:{top}:x:{left}',roi=roi,
                    foreground_pixels=len(pixels),eligible=eligible,selected=False,region_id=None,reason=reason)
                ordinal=len(items);items.append(item);indices.append(pixels)
                if eligible:byposition[(top,left)]=ordinal;eligible_union[pixels]=True
        pairs=[]
        for (top,left),a in sorted(byposition.items()):
            for dy,dx in ((0,16),(16,-16),(16,0),(16,16)):
                budget();b=byposition.get((top+dy,left+dx))
                if b is None:continue
                shared=np.intersect1d(indices[a],indices[b],assume_unique=True)
                if len(shared)>=64:pairs.append((a,b,shared))
        pairs.sort(key=lambda p:(items[p[0]]['roi']['top'],items[p[0]]['roi']['left'],items[p[1]]['roi']['top'],items[p[1]]['roi']['left']))
        summary=dict(source,foreground_pixels=int(mask.sum()),foreground_bbox=bbox,candidate_count=len(items),
            candidate_eligible_count=sum(i['eligible'] for i in items),selected_count=0,
            eligible_union_pixels=int(eligible_union.sum()),planned_union_pixels=0,
            uncovered_eligible_pixels=int(eligible_union.sum()),uncovered_foreground_pixels=int(mask.sum()))
        plan['sources'].append(summary);plan['candidates'].append(dict(source_input_index=idx,source_decode_index=decode,items=items))
        cache[idx]=dict(mask=mask,indices=indices,items=items,pairs=pairs,covered=np.zeros(mask.size,np.bool_),summary=summary)
    while len(plan['regions'])+2<=64:
        progress=False
        for source in sources:
            budget()
            if len(plan['regions'])+2>64:break
            idx=source['source_input_index'];c=cache[idx];items=c['items'];covered=c['covered']
            gains={i:int(np.count_nonzero(~covered[p])) for i,p in enumerate(c['indices']) if items[i]['eligible'] and not items[i]['selected']}
            best=None;best_gain=0
            for a,b,shared in c['pairs']:
                budget()
                if items[a]['selected'] or items[b]['selected']:continue
                gain=gains[a]+gains[b]-int(np.count_nonzero(~covered[shared]))
                if gain>best_gain:best_gain=gain;best=(a,b)
            if best is None:continue
            progress=True
            for candidate in best:
                item=items[candidate];item['selected']=True;r=item['roi'];rid=f"overlap:source:{idx}:y:{r['top']}:x:{r['left']}"
                item['region_id']=rid;covered[c['indices'][candidate]]=True
                plan['regions'].append(dict(region_id=rid,depth=0,tracklet_id=None,source_input_index=idx,
                    source_decode_index=source['source_decode_index'],decode_path=list(source['decode_path']),
                    center_raw_xy=[r['left']+16.5,r['top']+16.5]))
            c['summary']['selected_count']+=2
        if not progress:break
    for source in sources:
        budget();c=cache[source['source_input_index']];summary=c['summary'];summary['planned_union_pixels']=int(c['covered'].sum())
        summary['uncovered_eligible_pixels']=summary['eligible_union_pixels']-summary['planned_union_pixels']
        summary['uncovered_foreground_pixels']=summary['foreground_pixels']-summary['planned_union_pixels']
        selected=[r for r in plan['regions'] if r['source_input_index']==source['source_input_index']]
        for i,a in enumerate(selected):
            ai=_region_indices(c['mask'],a)
            for b in selected[i+1:]:
                budget();shared=np.intersect1d(ai,_region_indices(c['mask'],b),assume_unique=True)
                if len(shared)<64:continue
                chosen=shared[np.linspace(0,len(shared)-1,min(64,len(shared)),dtype=int)]
                controls=[[int(p%c['mask'].shape[1])+.5,int(p//c['mask'].shape[1])+.5] for p in chosen]
                plan['edges'].append(dict(edge_id=f"overlap-edge:{len(plan['edges'])}",region_a=a['region_id'],region_b=b['region_id'],
                    source_input_index=source['source_input_index'],source_decode_index=source['source_decode_index'],
                    shared_foreground_pixels=len(shared),control_raw_xy=controls,controls_kind='DERIVED_SOURCE_AREA_CONTROLS'))
    plan['selection_counts']=dict(sources=len(sources),candidates=total,selected_regions=len(plan['regions']),edges=len(plan['edges']),
        remaining_region_slots=64-len(plan['regions']),unselected_eligible_candidates=sum(i['eligible'] and not i['selected'] for c in plan['candidates'] for i in c['items']))
    components=[]
    for source in sources:
        ids=[r['region_id'] for r in plan['regions'] if r['source_input_index']==source['source_input_index']]
        neighbors={rid:set() for rid in ids}
        for edge in plan['edges']:
            if edge['source_input_index']==source['source_input_index']:
                neighbors[edge['region_a']].add(edge['region_b']);neighbors[edge['region_b']].add(edge['region_a'])
        found=[];remaining=set(ids)
        while remaining:
            budget();stack=[min(remaining)];component=[];remaining.remove(stack[0])
            while stack:
                rid=stack.pop();component.append(rid)
                for neighbor in sorted(neighbors[rid]):
                    if neighbor in remaining:remaining.remove(neighbor);stack.append(neighbor)
            found.append(sorted(component))
        components.append(dict(source_input_index=source['source_input_index'],source_decode_index=source['source_decode_index'],
            components=found,component_count=len(found),isolated_region_ids=sorted(rid for rid in ids if not neighbors[rid]),
            meaning='source-space planned overlap graph only; not verified cross-target connectivity'))
    plan['planned_connected_components']=components
    return plan


def _matrix(leg,expected_roi):
    if leg is None or leg.get('status')!='ACCEPTED':return None
    forward=leg.get('forward');_need(isinstance(forward,dict) and forward.get('template_roi')==expected_roi,'ACCEPTED_FORWARD_SOURCE_ROI_MISMATCH')
    value=forward.get('matrix_global_raw3x3')
    _need(not any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat),'BOOLEAN_AREA_MATRIX')
    raw=np.asarray(value)
    _need(raw.shape==(3,3) and raw.dtype.kind in 'iuf' and np.isfinite(raw).all() and np.array_equal(raw[2],[0.,0.,1.]),'INVALID_ACCEPTED_AREA_MATRIX')
    return raw.astype(float)


def _map(M,controls):
    with np.errstate(over='ignore',invalid='ignore'):
        values=(np.column_stack((controls,np.ones(len(controls))))@M.T)[:,:2]
    _need(np.isfinite(values).all(),'NONFINITE_AREA_CONTROL_PREDICTION')
    return values


def assess_overlap(plan,regions,*,budget=lambda:None):
    """Compare frozen controls under accepted area warps; never repair conflicts."""
    budget();_need(isinstance(regions,list),'REGION_MEASUREMENTS_LIST_REQUIRED')
    profile=plan.get('profile',{})
    limit=profile.get('max_regions',64)
    connected=profile.get('selection_policy')=='spatial-seeds-bridges-frontier-v1'
    _need(type(limit) is int and limit in ((64,256) if connected else (64,)),'PLAN_REGION_BOUND')
    frozen={r['region_id']:r for r in plan['regions']};_need(len(frozen)==len(plan['regions'])<=limit,'PLAN_REGION_IDENTITIES')
    sources=_sources(dict(regions=plan.get('sources',plan['regions'])),budget)
    masks={s['source_input_index']:_mask(plan['frame_domains'][str(s['source_decode_index'])],budget) for s in sources}
    measured={};matrices={};local={}
    for area in regions:
        budget();rid=area.get('region_id');_need(rid in frozen and rid not in measured,'FOREIGN_OR_DUPLICATE_MEASURED_REGION')
        r=frozen[rid];_need(all(area.get(k)==r[k] for k in ('source_input_index','source_decode_index','decode_path','depth')) and
            area.get('tracklet_id') is None and area.get('source_center_raw_xy')==r['center_raw_xy'] and area.get('template_roi')==_roi(r),'AREA_MEASUREMENT_IDENTITY_OR_ROI')
        legs=area.get('region_legs');_need(isinstance(legs,list) and len(legs)<=len(r['decode_path'])-1,'AREA_LEG_BOUND')
        previous_failed=False;local[rid]=[]
        for ordinal,leg in enumerate(legs):
            _need(not previous_failed and leg.get('source_decode_index')==r['source_decode_index'] and leg.get('target_decode_index')==r['decode_path'][ordinal+1],'AREA_TARGET_ROUTE_OR_PREFIX')
            M=_matrix(leg,_roi(r));matrices[(rid,leg['target_decode_index'])]=M
            if M is None:previous_failed=True
            else:local[rid].append(leg['target_decode_index'])
        _need(type(area.get('verified_targets')) is int and area['verified_targets']==len(local[rid]) and
            area.get('status')==('REGION_TRACKLET_2D' if len(local[rid])>=2 else 'INCOMPLETE'),'AREA_VERIFIED_TARGETS_OR_FINAL_STATUS')
        measured[rid]=area
    edges=[];consistent={};conflicts={};untested={};edge_ids=set();edge_pairs=set()
    for edge in plan['edges']:
        budget();_need(edge['edge_id'] not in edge_ids and edge['region_a'] in frozen and edge['region_b'] in frozen and edge['region_a']!=edge['region_b'],'EDGE_IDENTITY')
        pair=frozenset((edge['region_a'],edge['region_b']));_need(pair not in edge_pairs,'DUPLICATE_EDGE_PAIR');edge_pairs.add(pair)
        edge_ids.add(edge['edge_id']);a=frozen[edge['region_a']];b=frozen[edge['region_b']];source=edge['source_input_index'];mask=masks[source]
        _need(a['source_input_index']==b['source_input_index']==source and a['source_decode_index']==b['source_decode_index']==edge['source_decode_index'] and a['decode_path']==b['decode_path'],'EDGE_SOURCE_IDENTITY')
        shared=np.intersect1d(_region_indices(mask,a),_region_indices(mask,b),assume_unique=True)
        _need(len(shared)==edge['shared_foreground_pixels'] and len(shared)>=64,'EDGE_FOREGROUND_INTERSECTION')
        selected=shared[np.linspace(0,len(shared)-1,min(64,len(shared)),dtype=int)]
        expected=[[int(i%mask.shape[1])+.5,int(i//mask.shape[1])+.5] for i in selected]
        _need(edge['control_raw_xy']==expected,'EDGE_REAL_FOREGROUND_CONTROLS_CHANGED')
        out=dict(deepcopy(edge),targets=[])
        for target in a['decode_path'][1:]:
            budget();A=matrices.get((a['region_id'],target));B=matrices.get((b['region_id'],target));controls=np.asarray(expected,float)
            pa=_map(A,controls) if A is not None else None;pb=_map(B,controls) if B is not None else None
            errors=np.linalg.norm(pa-pb,axis=1) if pa is not None and pb is not None else None
            _need(errors is None or np.isfinite(errors).all(),'NONFINITE_OVERLAP_ERROR')
            status='UNTESTED' if errors is None else 'CONSISTENT' if float(errors.max())<=2 else 'CONFLICT'
            reasons=[] if errors is not None else ['MISSING_OR_UNACCEPTED_'+label for label,m in [('A',A),('B',B)] if m is None]
            row=dict(target_decode_index=target,status=status,reasons=reasons,
                max_error_px=float(errors.max()) if errors is not None else None,
                p50_error_px=float(np.percentile(errors,50)) if errors is not None else None,
                p95_error_px=float(np.percentile(errors,95)) if errors is not None else None,
                controls=[dict(source_raw_xy=p,prediction_a_raw_xy=pa[i].tolist() if pa is not None else None,
                    prediction_b_raw_xy=pb[i].tolist() if pb is not None else None,error_px=float(errors[i]) if errors is not None else None) for i,p in enumerate(expected)])
            out['targets'].append(row)
            table=consistent if status=='CONSISTENT' else conflicts if status=='CONFLICT' else untested
            for rid in (a['region_id'],b['region_id']):table.setdefault((rid,target),[]).append(edge['edge_id'])
        edges.append(out)
    records=[]
    for rid,r in frozen.items():
        budget();targets=local.get(rid,[]);supported=[t for t in targets if consistent.get((rid,t)) and not conflicts.get((rid,t))]
        def prefix(values):
            found=[]
            for target in r['decode_path'][1:]:
                if target not in values:break
                found.append(target)
            return found
        lp=prefix(targets);op=prefix(supported)
        records.append(dict(region_id=rid,source_input_index=r['source_input_index'],local_targets=targets,
            overlap_supported_targets=supported,local_prefix_targets=lp,overlap_prefix_targets=op,
            local_tracklet=len(lp)>=2,network_tracklet=len(op)>=2,
            conflict_edge_ids=sorted({e for t in r['decode_path'][1:] for e in conflicts.get((rid,t),[])}),
            untested_edge_ids=sorted({e for t in r['decode_path'][1:] for e in untested.get((rid,t),[])})))
    coverage=[];byid={r['region_id']:r for r in records}
    for source in sources:
        budget();idx=source['source_input_index'];mask=masks[idx];union={k:np.zeros(mask.size,np.bool_) for k in ('planned','local_two_target','overlap_two_target')}
        for rid,r in frozen.items():
            if r['source_input_index']!=idx:continue
            pixels=_region_indices(mask,r);union['planned'][pixels]=True
            if byid[rid]['local_tracklet']:union['local_two_target'][pixels]=True
            if byid[rid]['network_tracklet']:union['overlap_two_target'][pixels]=True
        denominator=int(mask.sum());row=dict(source_input_index=idx,source_decode_index=source['source_decode_index'],foreground_pixels=denominator)
        for key,value in union.items():row[key+'_union_pixels']=int(value.sum());row[key+'_fraction']=int(value.sum())/denominator
        coverage.append(row)
    return dict(status='OVERLAP_ASSESSMENT_COMPLETE',edges=edges,regions=records,coverage=coverage,
        counts=dict(regions=len(records),local_tracklets=sum(r['local_tracklet'] for r in records),network_tracklets=sum(r['network_tracklet'] for r in records),
            edge_targets=dict(Counter(t['status'] for e in edges for t in e['targets']))),
        control_semantics='derived coordinates of shared source area; not independently matched point identities',**RIGHTS)
