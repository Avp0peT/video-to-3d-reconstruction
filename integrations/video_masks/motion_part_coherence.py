"""Conservative connected temporal domains within existing motion candidates."""
from copy import deepcopy
import math
import cv2
import numpy as np

_DEFAULT = dict(min_shared_tracks=4, min_frames=3, min_warp_iou=0.25,
                min_inlier_ratio=0.75, max_residual=1.5, max_condition=100.0, max_parts=12, max_components=128)


def _policy(policy):
    p=dict(_DEFAULT)
    if policy is not None:
        if not isinstance(policy,dict) or set(policy)-set(p):
            raise ValueError('unrecognized coherence policy')
        p.update(policy)
    for k,v in p.items():
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:
            raise ValueError('invalid coherence policy: '+k)
    for k,minimum,maximum in [('min_shared_tracks',4,1200),('min_frames',3,6),('max_parts',1,32),('max_components',1,256)]:
        if int(p[k])!=p[k] or not minimum<=p[k]<=maximum:
            raise ValueError('invalid bounded integer: '+k)
        p[k]=int(p[k])
    if not 0<p['min_inlier_ratio']<=1: raise ValueError('invalid inlier ratio')
    if not 0<p['min_warp_iou']<=1 or p['max_condition']<1:
        raise ValueError('invalid geometric threshold')
    return p


def _iou(a,b):
    union=np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else 0.0


def _fit(a,b,p):
    centered=a-a.mean(axis=0)
    scale=np.linalg.norm(centered,axis=0)
    if np.any(scale<1e-8):
        return None,'DEGENERATE_TRACK_GEOMETRY',None
    singular=np.linalg.svd(centered/scale,compute_uv=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>1e-10 else float('inf')
    if not math.isfinite(condition) or condition>p['max_condition']:
        return None,'DEGENERATE_TRACK_GEOMETRY',None
    design=np.column_stack([a,np.ones(len(a))])
    model=np.linalg.lstsq(design,b,rcond=None)[0].T
    residual=np.linalg.norm(design@model.T-b,axis=1)
    evidence=dict(condition=condition,residual_max=float(residual.max()),residual_median=float(np.median(residual)),model=model.tolist())
    if residual.max()>p['max_residual']:
        return None,'AFFINE_RESIDUAL',evidence
    if abs(np.linalg.det(model[:,:2]))<1e-8:
        return None,'SINGULAR_AFFINE',evidence
    return model,None,evidence


def _edge(a,b,positions,p,track_ids=None):
    shared=sorted(a['ids']&b['ids'] if track_ids is None else set(track_ids)&a['ids']&b['ids'])
    record=dict(source_frame=a['frame'],source_component=a['component'],destination_frame=b['frame'],destination_component=b['component'],source_area=a['area'],destination_area=b['area'],shared_track_ids=shared,accepted_track_ids=[],inlier_track_ids=[],outlier_track_ids=list(shared),inlier_ratio=0.0)
    if len(shared)<p['min_shared_tracks']:
        return False,dict(record,reason='INSUFFICIENT_SHARED_TRACKS')
    aa=np.asarray([positions[a['frame']][tid] for tid in shared],float)
    bb=np.asarray([positions[b['frame']][tid] for tid in shared],float)
    # Rank checks are independent of robust consensus; three points cannot
    # supply an accepted identity core even if RANSAC internally samples three.
    _,initial_reason,_=_fit(aa,bb,p)
    if initial_reason=='DEGENERATE_TRACK_GEOMETRY':
        return False,dict(record,reason=initial_reason)
    consensus=[]
    for direction,source,destination in [('forward',aa,bb),('backward',bb,aa)]:
        cv2.setRNGSeed(0)
        rm,ri=cv2.estimateAffine2D(source,destination,method=cv2.RANSAC,
            ransacReprojThreshold=p['max_residual'],maxIters=1000,confidence=0.99,refineIters=10)
        keep=np.zeros(len(shared),bool) if rm is None or ri is None else ri.ravel().astype(bool)
        consensus.append(keep)
        record[direction+'_ransac']=dict(model=None if rm is None else rm.tolist(),inlier_track_ids=[tid for tid,k in zip(shared,keep) if k],outlier_track_ids=[tid for tid,k in zip(shared,keep) if not k],inlier_ratio=float(keep.mean()))
    core=consensus[0]&consensus[1]
    coreids=[tid for tid,k in zip(shared,core) if k]
    record.update(inlier_track_ids=coreids,outlier_track_ids=[tid for tid,k in zip(shared,core) if not k],inlier_ratio=float(core.mean()))
    if core.mean()<p['min_inlier_ratio']:
        return False,dict(record,reason='INSUFFICIENT_INLIER_RATIO')
    if core.sum()<p['min_shared_tracks']:
        return False,dict(record,reason='INSUFFICIENT_COMMON_INLIERS')
    aa,bb=aa[core],bb[core]
    model,reason,stats=_fit(aa,bb,p)
    record['forward_fit']=stats
    if reason:
        return False,dict(record,reason=reason)
    inverse,reason,stats=_fit(bb,aa,p)
    record['backward_fit']=stats
    if reason:
        return False,dict(record,reason=reason)
    am=a['labels']==a['component']; bm=b['labels']==b['component']
    h,w=am.shape
    forward=cv2.warpAffine(am.astype(np.uint8),model,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
    backward=cv2.warpAffine(bm.astype(np.uint8),inverse,(w,h),flags=cv2.INTER_NEAREST).astype(bool)
    record.update(forward_iou=_iou(forward,bm),backward_iou=_iou(backward,am))
    if min(record['forward_iou'],record['backward_iou'])<p['min_warp_iou']:
        return False,dict(record,reason='WARP_OVERLAP')
    return True,dict(record,reason='PASS',accepted_track_ids=coreids)


def refine_motion_parts(images,masks,raw_result,*,policy=None):
    """Only subtract raw motion pixels; temporal overlap is not an RGB contour."""
    p=_policy(policy)
    if not isinstance(images,list) or not isinstance(masks,list) or not 1<=len(images)<=6 or len(images)!=len(masks):
        raise ValueError('expected one to six matching image/mask frames')
    shape=masks[0].shape if isinstance(masks[0],np.ndarray) else None
    if shape is None or len(shape)!=2: raise ValueError('invalid mask dimensions')
    for image,mask in zip(images,masks):
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.shape!=(*shape,3) or not isinstance(mask,np.ndarray) or mask.dtype!=bool or mask.shape!=shape:
            raise ValueError('invalid uint8 RGB images or boolean masks')
    n=len(images)
    if not isinstance(raw_result,dict) or 'ledger' not in raw_result or 'global_masks' not in raw_result or 'part_masks' not in raw_result:
        raise ValueError('missing raw motion result')
    def arrays(key):
        value=raw_result.get(key)
        if not isinstance(value,list) or len(value)!=n or any(not isinstance(a,np.ndarray) or a.dtype!=bool or a.shape!=shape for a in value):
            raise ValueError('invalid raw arrays: '+key)
        return value
    whole=arrays('global_masks'); oldunknown=arrays('unknown_masks'); oldevidence=arrays('evidence_domain_masks')
    if any((e&~w).any() or (u&e).any() or not np.array_equal(u|e,w) for w,u,e in zip(whole,oldunknown,oldevidence)): raise ValueError('raw evidence and unknown must partition whole masks')
    if any((a&~b).any() for a,b in zip(oldunknown,whole)): raise ValueError('unknown outside whole mask')
    if any(not np.array_equal(a,b) for a,b in zip(whole,masks)):
        raise ValueError('raw whole masks must equal supplied masks')
    trusted={}
    for entry in raw_result['ledger'].get('parts',[]):
        name=entry.get('id'); tids=entry.get('track_ids')
        if not isinstance(name,str) or name in trusted or not isinstance(tids,list) or len(set(tids))!=len(tids) or any(isinstance(t,bool) or not isinstance(t,int) or t<0 for t in tids): raise ValueError('invalid raw part track identity')
        trusted[name]=set(tids)
    if set(trusted)!=set(raw_result['part_masks']): raise ValueError('raw part ledger identity mismatch')
    positions=[{} for _ in images]
    seen_pairs=set()
    for pair in raw_result['ledger'].get('pairs',[]):
        frames=pair.get('pair',[]); tids=pair.get('track_ids',[])
        if len(frames)!=2 or any(isinstance(f,bool) or not isinstance(f,int) for f in frames) or tuple(frames) in seen_pairs or frames[1]!=frames[0]+1 or not 0<=frames[0]<frames[1]<n or len(set(tids))!=len(tids):
            raise ValueError('invalid track pair ledger')
        seen_pairs.add(tuple(frames))
        for f,key in zip(frames,['source_points','destination_points']):
            coords=pair.get(key,[])
            if len(coords)!=len(tids): raise ValueError('track coordinate count mismatch')
            for tid,xy in zip(tids,coords):
                if isinstance(tid,bool) or not isinstance(tid,int) or tid<0 or len(xy)!=2 or not np.isfinite(xy).all(): raise ValueError('invalid track coordinate')
                x,y=np.rint(xy).astype(int)
                if not(0<=x<shape[1] and 0<=y<shape[0] and whole[f][y,x]): raise ValueError('track outside whole mask')
                if tid in positions[f] and not np.allclose(positions[f][tid],xy,atol=1e-4,rtol=0): raise ValueError('inconsistent shared track coordinate')
                positions[f][tid]=xy
    if not isinstance(raw_result['part_masks'],dict) or len(raw_result['part_masks'])>4: raise ValueError('at most four raw candidates supported')
    edges=[]; rejections=[]; accepted=[]; raw_union=[np.zeros(shape,bool) for _ in images]
    for rawid,values in sorted(raw_result['part_masks'].items()):
        if not isinstance(rawid,str) or not isinstance(values,list) or len(values)!=n: raise ValueError('invalid raw candidate')
        frames=[]
        for f,mask in enumerate(values):
            if not isinstance(mask,np.ndarray) or mask.dtype!=bool or mask.shape!=shape or (mask&~whole[f]).any(): raise ValueError('invalid raw candidate mask')
            if (raw_union[f]&mask).any(): raise ValueError('overlapping raw candidate masks')
            raw_union[f]|=mask
            count,labels=cv2.connectedComponents(mask.astype(np.uint8),connectivity=4)
            components=[]
            areas=np.bincount(labels.ravel(),minlength=count)
            for c in range(1,count):
                ids=set()
                for tid,xy in positions[f].items():
                    x,y=np.rint(xy).astype(int)
                    if tid in trusted[rawid] and labels[y,x]==c: ids.add(tid)
                components.append(dict(frame=f,component=c,labels=labels,area=int(areas[c]),ids=ids))
            components.sort(key=lambda c:(-len(c['ids']),-c['area'],c['component']))
            for dropped in components[p['max_components']:]:
                rejections.append(dict(source_raw_id=rawid,frames=[f],component=dropped['component'],reason='MAX_COMPONENTS_LIMIT',area=dropped['area']))
            frames.append(components[:p['max_components']])
        successors={}; predecessors={}
        for f in range(n-1):
            eligible=[]
            for a in frames[f]:
                for b in frames[f+1]:
                    ok,record=_edge(a,b,positions,p)
                    record['source_raw_id']=rawid
                    edges.append(record)
                    if ok: eligible.append((a,b,record))
            for a,b,record in eligible:
                outgoing=sum(x['component']==a['component'] for x,y,r in eligible)
                incoming=sum(y['component']==b['component'] for x,y,r in eligible)
                if outgoing!=1 or incoming!=1:
                    record['reason']='AMBIGUOUS_ASSOCIATION'
                    continue
                ak=(f,a['component']); bk=(f+1,b['component'])
                successors[ak]=(b,set(record['accepted_track_ids'])); predecessors[bk]=a
        visited=set()
        for comps in frames:
            for start in comps:
                sk=(start['frame'],start['component'])
                if sk in visited: continue
                chain=[start]; initial_ids=set(start['ids']); common=set(initial_ids); visited.add(sk)
                while True:
                    last=chain[-1]; lk=(last['frame'],last['component'])
                    link=successors.get(lk)
                    if link is None: break
                    target,pair_core=link
                    shared=common&target['ids']&pair_core
                    ok,anchor=_edge(start,target,positions,p,shared)
                    anchor.update(source_raw_id=rawid,check='FIXED_ANCHOR_COMMON_IDS',initial_support_count=len(initial_ids),accepted_core_count=len(anchor['accepted_track_ids']),start_support_retention=len(anchor['accepted_track_ids'])/len(initial_ids) if initial_ids else 0.0)
                    edges.append(anchor)
                    if not ok:
                        rejections.append(dict(source_raw_id=rawid,frames=[c['frame'] for c in chain]+[target['frame']],reason='ANCHOR_'+anchor['reason'],area=sum(c['area'] for c in chain)+target['area']))
                        break
                    chain.append(target); common=set(anchor['accepted_track_ids']); visited.add((target['frame'],target['component']))
                item=dict(source_raw_id=rawid,chain=chain,ids=common,initial_ids=initial_ids)
                if len(chain)>=p['min_frames']:
                    accepted.append(item)
                else:
                    rejections.append(dict(source_raw_id=rawid,frames=[c['frame'] for c in chain],reason='INSUFFICIENT_CONTINUOUS_FRAMES',area=sum(c['area'] for c in chain)))
    accepted.sort(key=lambda a:(a['source_raw_id'],a['chain'][0]['frame'],a['chain'][0]['component']))
    for item in accepted[p['max_parts']:]:
        rejections.append(dict(source_raw_id=item['source_raw_id'],frames=[c['frame'] for c in item['chain']],reason='MAX_PARTS_LIMIT',area=sum(c['area'] for c in item['chain'])))
    parts={}; partledger=[]; union=[np.zeros(shape,bool) for _ in images]
    for i,item in enumerate(accepted[:p['max_parts']]):
        name='coherent_%04d'%i
        values=[np.zeros(shape,bool) for _ in images]
        for c in item['chain']:
            f=c['frame']; values[f]=c['labels']==c['component']; union[f]|=values[f]
        parts[name]=values
        partledger.append(dict(id=name,source_raw_id=item['source_raw_id'],track_ids=sorted(item['ids']),initial_track_ids=sorted(item['initial_ids']),initial_support_count=len(item['initial_ids']),accepted_core_count=len(item['ids']),start_support_retention=len(item['ids'])/len(item['initial_ids']) if item['initial_ids'] else 0.0,active_frames=[c['frame'] for c in item['chain']]))
    output=dict(raw_result)
    output.update(global_masks=[m.copy() for m in whole],part_masks=parts,fallback_masks=[m&~u for m,u in zip(whole,union)],unknown_masks=[old|(raw&~kept) for old,raw,kept in zip(oldunknown,raw_union,union)],geometry_accepted=False,status='COHERENT_MOTION_CANDIDATES_NOT_PHYSICAL_PARTS' if parts else 'NO_TEMPORALLY_COHERENT_DOMAINS')
    output['raw_evidence_domain_masks']=[e.copy() for e in oldevidence]
    output['evidence_domain_masks']=[w&~u for w,u in zip(whole,output['unknown_masks'])]
    output['ledger']=deepcopy(raw_result['ledger']); output['ledger'].update(parts=partledger,geometry_accepted=False)
    metrics=[]
    for f in range(n):
        before=int(raw_union[f].sum()); after=int(union[f].sum())
        base=deepcopy(raw_result.get('metrics',[{} for _ in images])[f]) if isinstance(raw_result.get('metrics'),list) and len(raw_result['metrics'])==n else {}
        base.update(frame=f,global_pixels=int(whole[f].sum()),part_pixels=after,tracked_seed_count=base.get('tracked_seed_count',len(positions[f])))
        base.update(raw_connected_components=sum(cv2.connectedComponents(v[f].astype(np.uint8),connectivity=4)[0]-1 for v in raw_result['part_masks'].values()),refined_connected_components=sum(bool(v[f].any()) for v in parts.values()),raw_candidate_pixels=before,refined_candidate_pixels=after,removed_candidate_pixels=before-after,retained_fraction=after/before if before else None,fallback_pixels=int(output['fallback_masks'][f].sum()),unknown_pixels=int(output['unknown_masks'][f].sum()))
        metrics.append(base)
    output['metrics']=metrics
    output['refinement_ledger']=dict(policy=p,policy_version='motion-coherence-robust-v2',ransac_policy=dict(seed=0,max_iters=1000,confidence=0.99,threshold_pixels=p['max_residual'],ratio_scope='per_edge_shared_tracks',whole_window_ratio_gate=False,direction_core='forward_backward_intersection',identity_core='segment_intersection_plus_fixed_anchor_inliers'),geometry_accepted=False,raw_evidence_semantics='Original image motion observation domain before coherence rejection',evidence_semantics='Whole mask excluding refined unknown; may include observed fallback',boundary_semantics='TEMPORAL_MASK_OVERLAP_NOT_RGB_CONTOUR',interpretation='Connected temporal evidence domains only; affine overlap thresholds are heuristics, not physical accuracy.',threshold_units='native image pixels; resolution dependent',edges=edges,rejections=rejections,rejection_count=len(rejections)+sum(e['reason']!='PASS' for e in edges),coverage=metrics,raw_part_count=len(raw_result['part_masks']),output_part_count=len(parts))
    return output
