"""Frozen source-mask area selection; connectivity is a plan, not a match."""
from collections import deque
from copy import deepcopy
import numpy as np
from .overlap_regions import build_overlap_plan, _mask, _indices, _need

POLICY='spatial-seeds-bridges-frontier-v1'


def _components(nodes,graph,budget):
    remaining=set(nodes);output=[]
    while remaining:
        budget();first=min(remaining);remaining.remove(first);todo=[first];part=[]
        while todo:
            budget();node=todo.pop();part.append(node)
            for other in sorted(graph[node]):
                if other in remaining:remaining.remove(other);todo.append(other)
        output.append(sorted(part))
    return output


def _rid(source,item):
    r=item['roi']
    return f"connected:source:{source}:y:{r['top']}:x:{r['left']}"


def _cache(plan,budget):
    caches={}
    for source,catalogue in zip(plan['sources'],plan['candidates']):
        budget();idx=source['source_input_index'];mask=_mask(plan['frame_domains'][str(source['source_decode_index'])],budget)
        items=catalogue['items'];pixels={};positions={};graph={};strata={};box=source['foreground_bbox']
        for i,item in enumerate(items):
            budget();item['selected']=False;item['region_id']=None
            if not item['eligible']:continue
            pixels[i]=_indices(mask,item['roi']);positions[(item['roi']['top'],item['roi']['left'])]=i;graph[i]=set()
            x=item['roi']['left']+16.5;y=item['roi']['top']+16.5
            col=min(2,max(0,int(3*(x-box[0])/(box[2]-box[0]))));row=min(2,max(0,int(3*(y-box[1])/(box[3]-box[1]))))
            strata[i]=row*3+col;item['stratum']=row*3+col
        for (y,x),a in sorted(positions.items()):
            for dy,dx in ((0,16),(16,-16),(16,0),(16,16)):
                budget();b=positions.get((y+dy,x+dx))
                if b is not None and len(np.intersect1d(pixels[a],pixels[b],assume_unique=True))>=64:
                    graph[a].add(b);graph[b].add(a)
        domains={s:np.unique(np.concatenate([pixels[i] for i in pixels if strata[i]==s]))
            if any(strata[i]==s for i in pixels) else np.array([],dtype=np.int32) for s in range(9)}
        caches[idx]=dict(source=source,mask=mask,items=items,pixels=pixels,graph=graph,strata=strata,
            stratum_domains=domains,selected=set(),covered=np.zeros(mask.size,dtype=bool),steps=[],
            full_components=_components(graph,graph,budget))
    return caches


def _seed_order(nonempty):
    # Center first; next cell maximizes its minimum distance from prior cells.
    remaining=set(nonempty);order=[]
    while remaining:
        if not order:
            key=lambda s:((s//3-1)**2+(s%3-1)**2,s)
        else:
            key=lambda s:(-min((s//3-t//3)**2+(s%3-t%3)**2 for t in order),s)
        chosen=min(remaining,key=key);order.append(chosen);remaining.remove(chosen)
    return order


def _add(c,node,stage,*,parent=None,path=None):
    c['selected'].add(node);c['covered'][c['pixels'][node]]=True
    c['steps'].append(dict(stage=stage,source_input_index=c['source']['source_input_index'],
        candidate_id=c['items'][node]['candidate_id'],stratum=c['strata'][node],
        parent_candidate_id=None if parent is None else c['items'][parent]['candidate_id'],
        declared_path=[] if path is None else [c['items'][i]['candidate_id'] for i in path]))


def _bridge(c,budget):
    """Shortest graph path between selected components, with stable path ties."""
    components=_components(c['selected'],c['graph'],budget)
    if len(components)<2:return None
    best=None
    for component in components:
        budget();own=set(component);queue=deque(component);previous={i:None for i in component}
        while queue:
            budget();node=queue.popleft()
            if node in c['selected'] and node not in own:
                path=[];cursor=node
                while cursor is not None:path.append(cursor);cursor=previous[cursor]
                path.reverse();key=(len(path),tuple(path))
                if best is None or key<best[0]:best=(key,path)
                break
            for other in sorted(c['graph'][node]):
                if other not in previous:previous[other]=node;queue.append(other)
    return None if best is None else best[1]


def _sequence(c,limit,budget):
    items=c['items'];box=c['source']['foreground_bbox'];strata=c['strata']
    for cell in _seed_order(set(strata.values())):
        budget();cx=box[0]+(cell%3+.5)*(box[2]-box[0])/3;cy=box[1]+(cell//3+.5)*(box[3]-box[1])/3
        node=min((i for i in strata if strata[i]==cell),key=lambda i:
            ((items[i]['roi']['left']+16.5-cx)**2+(items[i]['roi']['top']+16.5-cy)**2,
             -items[i]['foreground_pixels'],items[i]['roi']['top'],items[i]['roi']['left']))
        _add(c,node,'SEED')
    while len(c['selected'])<limit:
        budget();path=_bridge(c,budget)
        if path is None:break
        for ordinal,node in enumerate(path):
            budget()
            if node not in c['selected']:
                if len(c['selected'])>=limit:return
                _add(c,node,'BRIDGE',parent=path[ordinal-1],path=path)
    while len(c['selected'])<limit:
        budget();frontier=sorted({n for i in c['selected'] for n in c['graph'][i]}-c['selected'])
        if not frontier:break
        fractions={s:float(np.count_nonzero(c['covered'][p]))/len(p) if len(p) else 0.
            for s,p in c['stratum_domains'].items()}
        gains={i:int(np.count_nonzero(~c['covered'][c['pixels'][i]])) for i in frontier}
        node=min(frontier,key=lambda i:(fractions[strata[i]],-gains[i],items[i]['roi']['top'],items[i]['roi']['left']))
        _add(c,node,'FRONTIER',parent=min(c['graph'][node]&c['selected']))


def _finalize(plan,budget):
    caches=_cache(plan,budget);byid={r['region_id']:r for r in plan['regions']};plan['edges']=[]
    for c in caches.values():
        idx=c['source']['source_input_index'];lookup={_rid(idx,item):i for i,item in enumerate(c['items']) if item['eligible']}
        for rid in byid:
            budget()
            if rid in lookup:
                i=lookup[rid];c['selected'].add(i);c['covered'][c['pixels'][i]]=True
                c['items'][i].update(selected=True,region_id=rid)
        selected=sorted(c['selected'])
        for ordinal,a in enumerate(selected):
            for b in selected[ordinal+1:]:
                budget();shared=np.intersect1d(c['pixels'][a],c['pixels'][b],assume_unique=True)
                if len(shared)<64:continue
                chosen=shared[np.linspace(0,len(shared)-1,min(64,len(shared)),dtype=int)]
                plan['edges'].append(dict(edge_id=f"connected-edge:{len(plan['edges'])}",region_a=_rid(idx,c['items'][a]),
                    region_b=_rid(idx,c['items'][b]),source_input_index=idx,source_decode_index=c['source']['source_decode_index'],
                    shared_foreground_pixels=len(shared),control_raw_xy=[[int(p%c['mask'].shape[1])+.5,int(p//c['mask'].shape[1])+.5] for p in chosen],
                    controls_kind='DERIVED_SOURCE_AREA_CONTROLS'))
        source=c['source'];source['selected_count']=len(selected);source['planned_union_pixels']=int(c['covered'].sum())
        source['uncovered_eligible_pixels']=source['eligible_union_pixels']-source['planned_union_pixels']
        source['uncovered_foreground_pixels']=source['foreground_pixels']-source['planned_union_pixels']
        components=_components(c['selected'],c['graph'],budget)
        source['selected_component_count']=len(components);source['candidate_component_count']=len(c['full_components'])
        seeded={i for part in c['full_components'] if any(n in c['selected'] for n in part) for i in part}
        for i,item in enumerate(c['items']):
            if item['eligible']:item['selection_reason']='SELECTED' if i in c['selected'] else 'UNREACHABLE_FROM_SELECTED_SEEDS' if i not in seeded else 'BUDGET_NOT_SELECTED'
        source['unreachable_eligible_candidates']=len(set(c['graph'])-seeded)
        source['budget_unselected_candidates']=len(seeded-c['selected'])
        source['selection_stop_reason']=('PREFIX_BUDGET_BOUND' if plan['profile']['prefix_only'] else
            'SOURCE_QUOTA_REACHED' if len(selected)>=plan['profile']['per_source_region_limit'] else
            'NO_REACHABLE_UNSELECTED_CANDIDATE')
        source['spatial_strata']=[]
        for cell,pixels in c['stratum_domains'].items():
            source['spatial_strata'].append(dict(stratum=cell,row=cell//3,column=cell%3,
                eligible_candidates=sum(s==cell for s in c['strata'].values()),selected_count=sum(c['strata'][i]==cell for i in selected),
                eligible_union_pixels=len(pixels),planned_union_pixels=int(np.count_nonzero(c['covered'][pixels])),
                planned_fraction=float(np.count_nonzero(c['covered'][pixels]))/len(pixels) if len(pixels) else 0.))
    plan['planned_connected_components']=[];plan['full_candidate_components']=[]
    for idx,c in caches.items():
        groups=_components(c['selected'],c['graph'],budget)
        plan['planned_connected_components'].append(dict(source_input_index=idx,source_decode_index=c['source']['source_decode_index'],
            components=[[_rid(idx,c['items'][i]) for i in part] for part in groups],component_count=len(groups),
            isolated_region_ids=[_rid(idx,c['items'][i]) for i in sorted(c['selected']) if not c['graph'][i]&c['selected']],
            meaning='source-space planned overlap graph only; not verified cross-target connectivity'))
        plan['full_candidate_components'].append(dict(source_input_index=idx,source_decode_index=c['source']['source_decode_index'],
            components=[[c['items'][i]['candidate_id'] for i in part] for part in c['full_components']],component_count=len(c['full_components'])))
    cap=plan['profile']['max_regions'];plan['selection_counts']=dict(sources=len(caches),candidates=sum(len(c['items']) for c in caches.values()),
        selected_regions=len(plan['regions']),edges=len(plan['edges']),remaining_region_slots=cap-len(plan['regions']),
        unselected_eligible_candidates=sum(i['eligible'] and not i['selected'] for c in caches.values() for i in c['items']))
    return plan


def prefix_plan(plan,max_regions=64,*,budget=lambda:None):
    """Crop the frozen sequence, never select replacements."""
    budget();_need(type(max_regions) is int and max_regions in (64,256),'CONNECTED_REGION_BOUND')
    _need(plan['profile'].get('selection_policy')==POLICY,'CONNECTED_PLAN_POLICY')
    _need(max_regions<=plan['profile']['max_regions'],'PREFIX_CANNOT_EXPAND_PLAN')
    result=deepcopy(plan);result['regions']=result['regions'][:max_regions];result['selection_steps']=result['selection_steps'][:max_regions]
    result['profile']['max_regions']=max_regions;result['profile']['frozen_sequence_capacity']=256
    result['profile']['prefix_only']=max_regions==64
    return _finalize(result,budget)


def build_connected_plan(source_plan,*,max_regions=256,budget=lambda:None):
    """Choose only from original-mask candidates, without observing outcomes."""
    budget();_need(type(max_regions) is int and max_regions in (64,256),'CONNECTED_REGION_BOUND')
    plan=build_overlap_plan(source_plan,budget=budget);plan['regions']=[];plan['edges']=[]
    plan['profile'].update(selection_policy=POLICY,max_regions=256,frozen_sequence_capacity=256,prefix_only=False,
        selection='3x3 spatial seeds; shortest graph bridges; low-stratum-coverage frontier; source round robin; no result refill',
        stratum_coverage='union of eligible candidate source foreground pixels assigned by candidate center; strata can overlap',
        seed_order='center nearest, then farthest minimum squared cell distance; row-major ties',
        bridge_order='fewest graph edges, then row-major candidate path; existing selected nodes reusable')
    caches=_cache(plan,budget);quota=256//max(1,len(caches))
    plan['profile'].update(per_source_region_limit=quota,
        source_quota_rule='equal frozen source quotas; unused source quota is not reassigned',
        stage_round_robin='single region per source per round within each stage; exhausted stage queues skipped')
    for c in caches.values():_sequence(c,quota,budget)
    plan['selection_steps']=[]
    # Finish every seed/bridge before filling any frontier, alternating sources.
    for stage in ('SEED','BRIDGE','FRONTIER'):
        queues={idx:deque(s for s in c['steps'] if s['stage']==stage) for idx,c in caches.items()}
        while any(queues.values()):
            for idx,queue in queues.items():
                budget()
                if not queue:continue
                step=queue.popleft();c=caches[idx];item=next(i for i in c['items'] if i['candidate_id']==step['candidate_id'])
                roi=item['roi'];rid=_rid(idx,item);step.update(ordinal=len(plan['selection_steps']),region_id=rid)
                plan['selection_steps'].append(step)
                plan['regions'].append(dict(region_id=rid,depth=0,tracklet_id=None,source_input_index=idx,
                    source_decode_index=c['source']['source_decode_index'],decode_path=list(c['source']['decode_path']),
                    center_raw_xy=[roi['left']+16.5,roi['top']+16.5]))
    _finalize(plan,budget)
    return prefix_plan(plan,64,budget=budget) if max_regions==64 else plan
