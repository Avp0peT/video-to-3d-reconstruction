"""Frozen small foreground neighborhoods, measured independently with unchanged LK gates."""
from collections import Counter, defaultdict, deque
from copy import deepcopy
import math
import numpy as np
RIGHTS=dict(geometry_accepted=False,identity_verified=False,new_cameras=0,new_model_points=0,segmentation_calls=0)
def need(ok,reason):
    if not ok:raise ValueError(reason)
def xy(value):
    need(isinstance(value,(list,tuple)) and len(value)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in value),'INVALID_RAW_XY')
    return np.asarray(value,float)
def safe(value):
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,list):return [safe(v) for v in value]
    return value

def window(raw,domain,half=10):
    p=xy(raw)-.5;x0=int(math.floor(p[0]))-half;x1=int(math.ceil(p[0]))+half;y0=int(math.floor(p[1]))-half;y1=int(math.ceil(p[1]))+half
    if x0<0 or y0<0 or x1>=domain['width'] or y1>=domain['height']:return 'WINDOW_OUTSIDE_IMAGE'
    if domain['original_mask_status']=='OK':
        for y in range(y0,y1+1):
            if not any(a<=x0 and b>x1 for a,b in domain['mask_rle'][y]):return 'WINDOW_OUTSIDE_ORIGINAL_FOREGROUND'
    return None

def make_region_points(region_id,center,source_domain,*,budget=lambda:None):
    """JSON-safe independent81point grid, shared with bounded region growth."""
    center=xy(center);points=[]
    for ordinal,(dy,dx) in enumerate((dy,dx) for dy in range(-16,17,4) for dx in range(-16,17,4)):
        budget();raw=(center+[dx,dy]).tolist();reason=window(raw,source_domain)
        points.append(dict(point_id=f'{region_id}:grid:{ordinal}',grid_ordinal=ordinal,offset_xy=[dx,dy],source_raw_xy=raw,is_center=ordinal==40,status='EXCLUDED_SOURCE_'+reason if reason else 'READY',reason=reason))
    return points

def build_region_plan(ledger,frames,masks,decoded,*,budget=lambda:None):
    budget();need(len(frames)==len(masks),'FRAME_MASK_COUNT');original={};domains={};bbox={}
    if not ledger['records']:
        return dict(status='REGION_PLAN_FROZEN',regions=[],omitted_centers=[],frame_domains={},profile=dict(max_regions=8,min_center_distance_px=32,grid_width=9,grid_step_px=4,patch_width=13,lk_width=21,max_legs=4),**RIGHTS)
    for f,m in zip(frames,masks):
        budget();i=f.get('input_index');d=f.get('decode_index')
        need(type(i) is int and type(d) is int and i>=0 and d>=0 and i not in original,'INVALID_FRAME_IDENTITY')
        need(d not in {v[0]['decode_index'] for v in original.values()},'DUPLICATE_DECODE_IDENTITY');original[i]=(f,m)
    def domain(index):
        if str(index) in domains:return domains[str(index)]
        f=decoded[index];w=f['width'];h=f['height']
        need(type(w) is int and type(h) is int and w>0 and h>0,'INVALID_DIMENSIONS')
        need(isinstance(f['gray'],np.ndarray) and f['gray'].dtype==np.uint8 and f['gray'].shape==(h,w),'INVALID_GRAY')
        value=dict(width=w,height=h,path=f['path'],original_mask_status='UNKNOWN_NOT_SEGMENTED',mask_rle=None)
        known=[(o,m) for o,m in original.values() if o['decode_index']==index]
        if known:
            o,m=known[0];need((o['width'],o['height'])==(w,h),'FRAME_DIMENSION_MISMATCH')
            if o['original_mask_status']=='OK':
                need(isinstance(m,np.ndarray) and m.dtype==np.uint8 and m.shape==(h,w) and np.isin(m,(0,255)).all(),'INVALID_ORIGINAL_MASK')
                value['original_mask_status']='OK';runs=[]
                for line in m:
                    z=np.r_[False,line==255,False].astype(np.int8);edges=np.flatnonzero(np.diff(z));runs.append([[int(a),int(b)] for a,b in edges.reshape(-1,2)])
                value['mask_rle']=runs
        domains[str(index)]=value;return value
    buckets=defaultdict(deque);seen=set();omitted=[]
    for r in ledger['records']:
        budget()
        if r['status']!='CANDIDATE_2D_TRACKLET':continue
        tid=r['tracklet_id'];need(type(r.get('edge_ordinal')) is int and r['edge_ordinal']>=0 and tid==f"{r['point_id']}:edge:{r['edge_ordinal']}",'TRACKLET_IDENTITY_MISMATCH');need(tid not in seen,'DUPLICATE_TRACKLET');seen.add(tid)
        need(all(type(r.get(k)) is int and r[k]>=0 for k in ('source_input_index','track_id','identity_segment')),'INVALID_TRACK_IDENTITY')
        source=r['source_input_index'];need(source in original,'UNKNOWN_SOURCE_INPUT');f,m=original[source];obs=r['observations']
        need(r['point_id']==f"track:{r['track_id']}:segment:{r['identity_segment']}:source:{source}",'POINT_IDENTITY_MISMATCH')
        need(len(obs)>=3 and r['verified_legs']==len(obs)-1,'INVALID_PREFIX_LENGTH')
        path=[o['decode_index'] for o in obs[:5]];direction=1 if path[-1]>path[0] else -1
        need(all(type(i) is int and i>=0 for i in path) and all(b-a==direction for a,b in zip(path,path[1:])),'NONCONSECUTIVE_DECODE_PATH')
        center=xy(obs[0]['raw_xy']);p=r['source_point']
        need(path[0]==f['decode_index']==p['source_decode_index'] and np.array_equal(center,xy(p['source_raw_xy'])) and p['source_input_index']==source and p['point_id']==r['point_id'],'SOURCE_BINDING_MISMATCH')
        need(f['original_mask_status']=='OK' and m is not None,'SOURCE_ORIGINAL_MASK_REQUIRED')
        for d in path:need(domain(d)['width']==f['width'] and domain(d)['height']==f['height'],'ROUTE_DIMENSION_MISMATCH')
        reason=window(center.tolist(),domain(path[0]))
        if reason:omitted.append(dict(tracklet_id=tid,reason='CENTER_'+reason));continue
        if source not in bbox:
            ys,xs=np.nonzero(m==255);need(len(xs)>0,'EMPTY_ORIGINAL_MASK');bbox[source]=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        x0,y0,x1,y1=bbox[source];cell=min(3,max(0,int((center[1]-y0)*4/(y1-y0))))*4+min(3,max(0,int((center[0]-x0)*4/(x1-x0))))
        buckets[(source,cell)].append((r,path,center))
    for key,q in buckets.items():buckets[key]=deque(sorted(q,key=lambda v:(v[2][1],v[2][0],v[0]['track_id'],v[0]['identity_segment'])))
    source_queues={}
    for source in sorted({key[0] for key in buckets}):
        ordered=[];keys=[key for key in sorted(buckets) if key[0]==source]
        while any(buckets[key] for key in keys):
            for key in keys:
                if buckets[key]:ordered.append((key,buckets[key].popleft()))
        source_queues[source]=deque(ordered)
    regions=[];centers=defaultdict(list)
    while any(source_queues.values()):
        for source in sorted(source_queues):
            budget()
            if not source_queues[source]:continue
            key,(r,path,center)=source_queues[source].popleft()
            if len(regions)>=8:omitted.append(dict(tracklet_id=r['tracklet_id'],reason='REGION_BUDGET'));continue
            if any(np.linalg.norm(center-c)<32 for c in centers[source]):omitted.append(dict(tracklet_id=r['tracklet_id'],reason='CENTER_SPACING_32PX'));continue
            centers[source].append(center);rid=f"region:{len(regions)}:{r['tracklet_id']}"
            points=make_region_points(rid,center.tolist(),domain(path[0]),budget=budget)
            regions.append(dict(region_id=rid,tracklet_id=r['tracklet_id'],source_input_index=source,source_decode_index=path[0],decode_path=path,center_raw_xy=center.tolist(),source_bbox_raw_xy=[float(center[0]-16),float(center[1]-16),float(center[0]+16),float(center[1]+16)],points=points,selection_cell=key[1],depth=0,parent_region_id=None))
    return dict(status='REGION_PLAN_FROZEN',regions=regions,omitted_centers=omitted,frame_domains=domains,profile=dict(max_regions=8,min_center_distance_px=32,grid_width=9,grid_step_px=4,patch_width=13,lk_width=21,max_legs=4),**RIGHTS)

def measure_regions(plan,decoded,cv,*,budget=lambda:None):
    budget();result=dict(status='RUNNING',regions=deepcopy(plan['regions']),counts={},flow_calls=0,coordinate_convention='raw pixel centers; native input float32(raw-.5); FB compared to original unquantized raw source',**RIGHTS)
    need(len(result['regions'])<=24,'REGION_MEASUREMENT_BOUND_24')
    need(len({r['region_id'] for r in result['regions']})==len(result['regions']),'DUPLICATE_REGION_ID')
    for region in result['regions']:
        budget();route=region['decode_path']
        need(isinstance(route,list) and 3<=len(route)<=5 and all(type(i) is int and i>=0 for i in route),'INVALID_MEASUREMENT_ROUTE')
        direction=1 if route[-1]>route[0] else -1
        need(all(b-a==direction for a,b in zip(route,route[1:])),'NONCONSECUTIVE_MEASUREMENT_ROUTE')
        need(type(region.get('source_input_index')) is int and region['source_input_index']>=0 and type(region.get('source_decode_index')) is int and region['source_decode_index']==route[0],'INVALID_REGION_SOURCE_IDENTITY')
        center=xy(region['center_raw_xy'])
        for index in route:
            need(str(index) in plan['frame_domains'],'MISSING_FRAME_DOMAIN')
            d=plan['frame_domains'][str(index)]
            need(type(d.get('width')) is int and type(d.get('height')) is int and d['width']>0 and d['height']>0,'INVALID_DOMAIN_DIMENSIONS')
            need(d.get('original_mask_status') in ('OK','UNKNOWN_NOT_SEGMENTED'),'INVALID_DOMAIN_MASK_STATUS')
            if d['original_mask_status']=='OK':
                rle=d.get('mask_rle');need(isinstance(rle,list) and len(rle)==d['height'],'INVALID_MASK_RLE')
                for row in rle:
                    need(isinstance(row,list),'INVALID_MASK_RLE_ROW');previous=-1
                    for interval in row:
                        need(isinstance(interval,list) and len(interval)==2 and all(type(v) is int for v in interval),'INVALID_MASK_RLE_INTERVAL')
                        a,b=interval;need(0<=a<b<=d['width'] and a>=previous,'INVALID_MASK_RLE_INTERVAL');previous=b
            else:need(d.get('mask_rle') is None,'UNKNOWN_MASK_MUST_NOT_HAVE_RLE')
        d=plan['frame_domains'][str(route[0])];need(d['original_mask_status']=='OK' and window(center.tolist(),d) is None,'INVALID_SOURCE_CENTER_WINDOW')
        points=region['points'];need(isinstance(points,list) and len(points)==81 and len({p['point_id'] for p in points})==81,'REGION_POINT_BOUND_OR_DUPLICATE')
        need([p['grid_ordinal'] for p in points]==list(range(81)) and all(type(p['grid_ordinal']) is int for p in points),'INVALID_GRID_ORDINAL')
        for p in points:
            ordinal=p['grid_ordinal'];offset=[(ordinal%9-4)*4,(ordinal//9-4)*4];expected=(center+offset).tolist()
            need(p['point_id']==f"{region['region_id']}:grid:{ordinal}" and p['offset_xy']==offset and np.array_equal(xy(p['source_raw_xy']),expected) and type(p.get('is_center')) is bool and p['is_center']==(ordinal==40),'INVALID_GRID_SOURCE_GEOMETRY')
            reason=window(expected,d);need(p['status']==('EXCLUDED_SOURCE_'+reason if reason else 'READY') and p.get('reason')==reason,'INVALID_SOURCE_EXCLUSION_LEDGER')
    if not result['regions']:
        result.update(status='REGION_MEASUREMENTS_COMPLETE',counts={});return result
    lk=dict(winSize=(21,21),maxLevel=3,criteria=(cv.TERM_CRITERIA_COUNT|cv.TERM_CRITERIA_EPS,30,.01))
    def native(*args):
        budget();result['flow_calls']+=1
        out=cv.calcOpticalFlowPyrLK(*args,None,**lk)
        result['last_native_output']=dict(input_cv_xy=safe(args[2]),xy=safe(out[0]),status=safe(out[1]),error=safe(out[2]))
        budget();return out
    def domain(i):
        d=plan['frame_domains'][str(i)];f=decoded[i]
        need(f['width']==d['width'] and f['height']==d['height'] and f['path']==d['path'] and isinstance(f['gray'],np.ndarray) and f['gray'].dtype==np.uint8 and f['gray'].shape==(d['height'],d['width']),'DECODED_DOMAIN_CHANGED')
        return d
    def patch(gray,raw):
        # Bilinear sampling explicitly in raw-.5 index coordinates; no cv API dependency.
        x,y=xy(raw)-.5;xs=x+np.arange(-6,7);ys=y+np.arange(-6,7);x0=np.floor(xs).astype(int);y0=np.floor(ys).astype(int)
        if x0.min()<0 or y0.min()<0 or x0.max()+1>=gray.shape[1] or y0.max()+1>=gray.shape[0]:return None
        wx=xs-x0;wy=ys-y0
        v00=gray[y0[:,None],x0].astype(float);v01=gray[y0[:,None],x0+1].astype(float)
        v10=gray[y0[:,None]+1,x0].astype(float);v11=gray[y0[:,None]+1,x0+1].astype(float)
        return ((1-wy[:,None])*((1-wx)*v00+wx*v01)+wy[:,None]*((1-wx)*v10+wx*v11)).ravel()
    def ncc(a,b,source,target):
        p=patch(decoded[a]['gray'],source);q=patch(decoded[b]['gray'],target)
        if p is None or q is None:return dict(status='UNKNOWN_PATCH_BOUNDS',value=None)
        p-=p.mean();q-=q.mean();den=float(np.linalg.norm(p)*np.linalg.norm(q))
        if np.std(p)<1 or np.std(q)<1 or den<=0:return dict(status='UNKNOWN_LOW_VARIANCE',value=None)
        return dict(status='EVALUATED',value=float(np.clip(np.dot(p,q)/den,-1,1)))
    def arrays(out,n):
        pos,st,err=out
        need(isinstance(pos,np.ndarray) and pos.shape==(n,1,2) and pos.dtype.kind=='f' and isinstance(st,np.ndarray) and st.shape==(n,1) and st.dtype.kind in 'iu' and isinstance(err,np.ndarray) and err.shape==(n,1) and err.dtype.kind=='f','INVALID_BATCH_OUTPUT_SHAPE_OR_TYPE')
        return pos,st,err
    def record(pos,st,err,j,input_cv,input_raw):return dict(input_cv_xy=safe(input_cv),input_raw_xy=safe(input_raw),quantized_input_raw_xy=safe(np.asarray(input_cv,dtype=float)+.5),xy=safe(pos[j,0]),status=int(st[j,0]),error=safe(float(err[j,0])),observed_raw_xy=safe((pos[j,0].astype(float)+.5)))
    def reason(rec):
        if rec['status']!=1:return 'LK_STATUS_FAILED'
        if rec['observed_raw_xy'] is None or any(v is None for v in rec['observed_raw_xy']):return 'LK_NONFINITE_XY'
        if rec['error'] is None or not 0<=rec['error']<=25:return 'LK_ERROR_THRESHOLD'
        return None
    try:
        for region in result['regions']:
            budget();region['affine_diagnostics']=[];region['batch_outputs']=[];path=region['decode_path']
            for i in path:domain(i)
            for point in region['points']:
                point.update(verified_legs=0,steps=[],observations=[])
                if point['status']=='READY':
                    need(window(point['source_raw_xy'],domain(path[0])) is None,'SOURCE_PLAN_WINDOW_INVALID')
                    point['observations']=[dict(decode_index=path[0],raw_xy=list(point['source_raw_xy']))];point['status']='ACTIVE'
            for a,b in zip(path,path[1:]):
                budget();active=[p for p in region['points'] if p['status']=='ACTIVE']
                if not active:break
                starts=np.array([p['observations'][-1]['raw_xy'] for p in active],float);seeds=(starts-.5).astype(np.float32).reshape(-1,1,2)
                outputs=native(decoded[a]['gray'].copy(),decoded[b]['gray'].copy(),seeds.copy());region['batch_outputs'].append(dict(from_decode_index=a,to_decode_index=b,direction='forward',input_cv_xy=safe(seeds),input_raw_xy=safe(starts),output_shapes=[list(v.shape) if isinstance(v,np.ndarray) else None for v in outputs],point_ids=[p['point_id'] for p in active],xy=safe(outputs[0]),status=safe(outputs[1]),error=safe(outputs[2])))
                fw,fs,fe=arrays(outputs,len(active));reverse=[]
                for j,p in enumerate(active):
                    f=record(fw,fs,fe,j,seeds[j,0],starts[j]);step=dict(from_decode_index=a,to_decode_index=b,forward=f,backward=None,fb_error_px=None,ncc=dict(status='NOT_EVALUATED',value=None),status='REJECTED',reason=None,destination_mask_status=domain(b)['original_mask_status']);p['steps'].append(step)
                    why=reason(f)
                    if why is None:why=window(f['observed_raw_xy'],domain(b))
                    if why:p['status']='REJECTED';p['reason']=why;step['reason']=why
                    else:reverse.append((j,p,step))
                if reverse:
                    backseeds=np.array([fw[j,0] for j,p,s in reverse],np.float32).reshape(-1,1,2);outputs=native(decoded[b]['gray'].copy(),decoded[a]['gray'].copy(),backseeds.copy());region['batch_outputs'].append(dict(from_decode_index=b,to_decode_index=a,direction='backward',input_cv_xy=safe(backseeds),input_raw_xy=[s['forward']['observed_raw_xy'] for j,p,s in reverse],output_shapes=[list(v.shape) if isinstance(v,np.ndarray) else None for v in outputs],point_ids=[p['point_id'] for j,p,step in reverse],xy=safe(outputs[0]),status=safe(outputs[1]),error=safe(outputs[2])))
                    bw,bs,be=arrays(outputs,len(reverse))
                    for k,(j,p,step) in enumerate(reverse):
                        backward=record(bw,bs,be,k,backseeds[k,0],step['forward']['observed_raw_xy']);step['backward']=backward;why=reason(backward)
                        if why is None:why=window(backward['observed_raw_xy'],domain(a))
                        if why is None:
                            fb=float(np.linalg.norm(xy(backward['observed_raw_xy'])-starts[j]));step['fb_error_px']=fb
                            if fb>1:why='PER_LEG_FB_THRESHOLD'
                        step['ncc']=ncc(a,b,starts[j].tolist(),step['forward']['observed_raw_xy'])
                        if why:p['status']='REJECTED';p['reason']=why;step['reason']=why
                        else:
                            step['status']='PASSED';p['verified_legs']+=1;p['observations'].append(dict(decode_index=b,raw_xy=step['forward']['observed_raw_xy']))
                passed=[p for p in active if p['steps'][-1]['status']=='PASSED'];train=[p for p in passed if p['grid_ordinal']%4!=3];held=[p for p in passed if p['grid_ordinal']%4==3]
                affine=dict(from_decode_index=a,to_decode_index=b,status='BLOCKED',reason='INSUFFICIENT_FIXED_SPLIT_SUPPORT',train_ids=[p['point_id'] for p in train],holdout_ids=[p['point_id'] for p in held],matrix=None,train=None,holdout=None)
                if len(train)>=6 and len(held)>=3:
                    x=np.array([p['observations'][-2]['raw_xy']+[1.] for p in train]);y=np.array([p['observations'][-1]['raw_xy'] for p in train]);rank=int(np.linalg.matrix_rank(x));affine['rank']=rank
                    if rank==3:
                        matrix=np.linalg.lstsq(x,y,rcond=None)[0];affine.update(status='EVALUATED',reason=None,matrix=matrix.T.tolist())
                        for name,points in [('train',train),('holdout',held)]:
                            values=np.array([np.linalg.norm(np.r_[p['observations'][-2]['raw_xy'],1.]@matrix-p['observations'][-1]['raw_xy']) for p in points]);affine[name]=dict(max_px=float(values.max()),median_px=float(np.median(values)),residual_pixels=values.tolist())
                    else:affine['reason']='COLLINEAR_AFFINE_TRAINING_SUPPORT'
                region['affine_diagnostics'].append(affine)
            for p in region['points']:
                if p['status'].startswith('EXCLUDED_SOURCE_'):continue
                if p['verified_legs']>=2:p['status']='TRACKLET_2D'
                elif p['status']=='ACTIVE':p['status']='REJECTED';p['reason']='INSUFFICIENT_CONTINUOUS_LEGS'
            def summary(points):
                excluded=sum(p['status'].startswith('EXCLUDED_SOURCE_') for p in points);evaluated=len(points)-excluded
                return dict(total=len(points),excluded=excluded,evaluated=evaluated,tracklets_2d=sum(p['status']=='TRACKLET_2D' for p in points),tracking_rejected=sum(p['status']=='REJECTED' for p in points))
            region['counts']=dict(Counter(p['status'] for p in region['points']));region['center_baseline']=summary([p for p in region['points'] if p['is_center']]);region['expanded_grid']=summary([p for p in region['points'] if not p['is_center']])
        result['status']='REGION_MEASUREMENTS_COMPLETE';result['counts']=dict(Counter(p['status'] for r in result['regions'] for p in r['points']));return result
    except Exception as exc:
        result['status']='INTERRUPTED'
        try:exc.partial_result=deepcopy(result)
        except (TypeError,AttributeError):pass
        raise
