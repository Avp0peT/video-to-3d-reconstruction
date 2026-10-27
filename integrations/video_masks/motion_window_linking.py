"""Pure-data links between motion windows; never physical identity acceptance."""
import math
import re
import numpy as np

POLICY_VERSION='motion-window-links-v1'
_DEFAULT=dict(min_shared_frames=2,min_track_pairs=4,min_mask_iou=0.25,point_radius=2.0,max_displacement_difference=1.5,max_condition=100.0)


def _policy(policy):
    p=dict(_DEFAULT)
    if policy is not None:
        if not isinstance(policy,dict) or set(policy)-set(p): raise ValueError('unrecognized linking policy')
        p.update(policy)
    for k,v in p.items():
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float)) or not math.isfinite(v): raise ValueError('invalid policy value')
    for k,low,high in [('min_shared_frames',2,6),('min_track_pairs',4,1200)]:
        if int(p[k])!=p[k] or not low<=p[k]<=high: raise ValueError('invalid policy bound')
        p[k]=int(p[k])
    if not .25<=p['min_mask_iou']<=1 or not 0<p['point_radius']<=2 or not 0<p['max_displacement_difference']<=1.5 or p['max_condition']<1: raise ValueError('invalid linking threshold')
    return p


def _id(value):
    return type(value) in (str,int) and (type(value) is int and value>=0 or type(value) is str and bool(value))


def _sort(values):
    return sorted(values,key=lambda x:(type(x).__name__,str(x)))


def _validate(window):
    if not isinstance(window,dict) or type(window.get('window_id')) is not int or window['window_id']<0: raise ValueError('invalid window ID')
    frames=window.get('frames')
    if not isinstance(frames,list) or not 1<=len(frames)<=6: raise ValueError('expected one to six explicit frames')
    result={}; last=-math.inf; shape=None; allparts=set()
    for frame in frames:
        if not isinstance(frame,dict): raise ValueError('invalid frame')
        fid=frame.get('frame_id'); time=frame.get('time_seconds')
        if not isinstance(fid,str) or not fid or fid in result or isinstance(time,(bool,np.bool_)) or not isinstance(time,(int,float)) or not math.isfinite(time) or time<=last: raise ValueError('invalid frame identity/time order')
        for key in ['image_sha256','whole_sha256']:
            if not isinstance(frame.get(key),str) or re.fullmatch('[0-9a-fA-F]{64}',frame[key]) is None: raise ValueError('invalid frame hash')
        parts=frame.get('parts'); tracks=frame.get('tracks')
        if not isinstance(parts,dict) or not isinstance(tracks,dict) or set(parts)!=set(tracks) or any(not _id(k) for k in parts): raise ValueError('part/track keys mismatch')
        allparts.update(parts)
        if len(allparts)>12: raise ValueError('too many window parts')
        occupied=None; count=0
        for part,mask in parts.items():
            if not isinstance(mask,np.ndarray) or mask.dtype!=bool or mask.ndim!=2 or min(mask.shape)<1: raise ValueError('invalid part mask')
            if shape is None: shape=mask.shape
            if mask.shape!=shape: raise ValueError('mask grid mismatch')
            if occupied is None: occupied=np.zeros(shape,bool)
            if (occupied&mask).any(): raise ValueError('overlapping part masks')
            occupied|=mask
            points=tracks[part]
            if not isinstance(points,dict) or any(not _id(t) for t in points): raise ValueError('invalid track IDs')
            count+=len(points)
            if count>1200: raise ValueError('too many tracks')
            for xy in points.values():
                if not isinstance(xy,(list,tuple)) or len(xy)!=2 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in xy): raise ValueError('invalid track coordinates')
                x,y=map(float,xy)
                if not (0<=x<shape[1] and 0<=y<shape[0]): raise ValueError('track outside grid')
                ix,iy=int(round(x)),int(round(y))
                if not (0<=ix<shape[1] and 0<=iy<shape[0] and mask[iy,ix]): raise ValueError('track outside part')
        result[fid]=frame; last=time
    return result,shape,allparts


def _matching(left,right,radius):
    lids=_sort(left); rids=_sort(right)
    if not lids or not rids: return [],[]
    a=np.asarray([left[t] for t in lids],float); b=np.asarray([right[t] for t in rids],float)
    distance=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    lmin=distance.min(axis=1); rmin=distance.min(axis=0)
    lties=np.isclose(distance,lmin[:,None],rtol=0,atol=1e-12).sum(axis=1)>1
    rties=np.isclose(distance,rmin[None,:],rtol=0,atol=1e-12).sum(axis=0)>1
    ties=[dict(side='left',track_id=lids[i]) for i in range(len(lids)) if lties[i] and lmin[i]<=radius]
    ties += [dict(side='right',track_id=rids[j]) for j in range(len(rids)) if rties[j] and rmin[j]<=radius]
    pairs=[]
    nearest=distance.argmin(axis=1); reverse=distance.argmin(axis=0)
    for i,j in enumerate(nearest):
        if distance[i,j]<=radius and not lties[i] and not rties[j] and reverse[j]==i:
            pairs.append((lids[i],rids[j]))
    return pairs,ties


def _geometry(points,max_condition):
    a=np.asarray(points,float); a-=a.mean(axis=0)
    scale=np.linalg.norm(a,axis=0)
    if np.any(scale<=1e-10): return False,None
    s=np.linalg.svd(a/scale,compute_uv=False)
    condition=float(s[0]/s[-1]) if s[-1]>1e-10 else None
    return condition is not None and condition<=max_condition,condition


def associate_pair(left,right,policy=None):
    """Link only reciprocal unique candidates supported by repeated point pairs."""
    p=_policy(policy)
    lf,ls,lp=_validate(left); rf,rs,rp=_validate(right)
    if left['window_id']==right['window_id']: raise ValueError('distinct windows required')
    if ls is not None and rs is not None and ls!=rs: raise ValueError('cross-window grid mismatch')
    shared=sorted(set(lf)&set(rf),key=lambda fid:(lf[fid]['time_seconds'],fid))
    for fid in shared:
        if lf[fid]['time_seconds']!=rf[fid]['time_seconds'] or any(lf[fid][k].lower()!=rf[fid][k].lower() for k in ['image_sha256','whole_sha256']): raise ValueError('shared frame provenance mismatch')
    records=[]; eligible=[]
    for lid in _sort(lp):
        for rid in _sort(rp):
            active=[fid for fid in shared if lid in lf[fid]['parts'] and rid in rf[fid]['parts'] and lf[fid]['parts'][lid].any() and rf[fid]['parts'][rid].any()]
            record=dict(left_part=lid,right_part=rid,shared_active_frame_ids=active,frames=[],track_pairs=[],reason='PASS')
            common=None
            for fid in active:
                a=lf[fid]['parts'][lid]; b=rf[fid]['parts'][rid]
                iou=float(np.count_nonzero(a&b)/np.count_nonzero(a|b))
                matches,ties=_matching(lf[fid]['tracks'][lid],rf[fid]['tracks'][rid],p['point_radius'])
                common=set(matches) if common is None else common&set(matches)
                record['frames'].append(dict(frame_id=fid,time_seconds=float(lf[fid]['time_seconds']),image_sha256=lf[fid]['image_sha256'].lower(),whole_sha256=lf[fid]['whole_sha256'].lower(),mask_iou=iou,track_pairs=[list(v) for v in matches],point_correspondences=[dict(left_track=v[0],right_track=v[1],left_xy=list(lf[fid]['tracks'][lid][v[0]]),right_xy=list(rf[fid]['tracks'][rid][v[1]]),distance=float(np.linalg.norm(np.asarray(lf[fid]['tracks'][lid][v[0]],float)-rf[fid]['tracks'][rid][v[1]]))) for v in matches],ambiguous_nearest_points=ties))
            if len(active)<p['min_shared_frames']: record['reason']='INSUFFICIENT_SHARED_ACTIVE_FRAMES'
            elif any(f['mask_iou']<p['min_mask_iou'] for f in record['frames']): record['reason']='MASK_OVERLAP'
            elif any(f['ambiguous_nearest_points'] for f in record['frames']): record['reason']='AMBIGUOUS_NEAREST_POINTS'
            elif len(common or [])<p['min_track_pairs']: record['reason']='INSUFFICIENT_STABLE_TRACK_PAIRS'
            else:
                ordered=sorted(common,key=lambda v:(type(v[0]).__name__,str(v[0]),type(v[1]).__name__,str(v[1])))
                record['track_pairs']=[list(v) for v in ordered]
                displacement=[]
                for lt,rt in ordered:
                    offsets=np.asarray([np.asarray(lf[fid]['tracks'][lid][lt],float)-rf[fid]['tracks'][rid][rt] for fid in active])
                    maximum=float(np.linalg.norm(offsets[:,None,:]-offsets[None,:,:],axis=2).max())
                    displacement.append(dict(left_track=lt,right_track=rt,max_displacement_difference=maximum))
                    if maximum>p['max_displacement_difference']: record['reason']='DISPLACEMENT_INCONSISTENT'
                record['displacement_checks']=displacement
                for item in record['frames']:
                    fid=item['frame_id']; geometry={}
                    for side,frames,part,index in [('left',lf,lid,0),('right',rf,rid,1)]:
                        ok,condition=_geometry([frames[fid]['tracks'][part][v[index]] for v in ordered],p['max_condition'])
                        geometry[side]=dict(noncollinear=ok,condition=condition)
                        if not ok: record['reason']='DEGENERATE_TRACK_GEOMETRY'
                    item['geometry']=geometry
                if record['reason']=='PASS': eligible.append(record)
            # Preserve the cross-frame intersection even on a rejection.
            record['stable_track_pairs']=[list(v) for v in sorted(common or [],key=lambda v:(str(v[0]),str(v[1])))]
            records.append(record)
    accepted=[]
    for record in eligible:
        if sum(r['left_part']==record['left_part'] for r in eligible)!=1 or sum(r['right_part']==record['right_part'] for r in eligible)!=1:
            record['reason']='AMBIGUOUS_PART_ASSOCIATION'
        else:
            accepted.append(dict(left_part=record['left_part'],right_part=record['right_part'],track_pairs=record['track_pairs'],shared_frame_ids=record['shared_active_frame_ids']))
    return dict(policy=p,policy_version=POLICY_VERSION,left_window_id=left['window_id'],right_window_id=right['window_id'],resource_bounds=dict(max_frames_per_window=6,max_parts_per_window=12,max_tracks_per_frame=1200),coordinate_semantics='Native pixel coordinates; nearest rounded pixel must lie inside part; ties within 1e-12 rejected',shared_frame_ids=shared,pairs=records,accepted_links=accepted,geometry_accepted=False,interpretation='Engineering overlap and correspondence rules; not proof of physical identity.')
