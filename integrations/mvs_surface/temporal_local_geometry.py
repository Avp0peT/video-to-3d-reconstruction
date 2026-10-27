"""CPU temporal image tracks and fixed-camera triangulation, experimental geometry."""
import cv2
import numpy as np
from .alignment_controls import _checked
from .local_depth_search import _rays

TRACK_REASONS={0:'NOT_ATTEMPTED_DIRECTION_STOPPED',1:'REFERENCE',2:'OBSERVED',3:'LK_FAILED',4:'FORWARD_BACKWARD',5:'UNKNOWN_FOREGROUND_PATCH',6:'LOW_TEXTURE_OR_NCC',7:'REFERENCE_NCC',8:'CHAIN_CLOSURE'}
GEOMETRY_REASONS={0:'INSUFFICIENT_OBSERVATIONS',1:'LOW_PARALLAX',2:'INVALID_TRIANGULATION',3:'HOLDOUT_REPROJECTION',4:'FULL_REPROJECTION',5:'PROJECTED_MASK_UNKNOWN',6:'ACCEPTED_EXPERIMENTAL_POINT',7:'TRAIN_LOW_PARALLAX'}
PARAMETERS=dict(max_controls=256,grid=8,max_per_cell=4,gftt_quality=.01,min_distance=5,block_size=5,patch_size=7,min_std=3.,min_ncc=.75,lk_window=31,lk_levels=4,lk_iterations=30,lk_epsilon=.01,max_fb=1.5,max_chain_closure=2.,min_observations=3,min_angle_degrees=1.,min_training_angle_degrees=1.,max_reprojection=2.,repetition_offsets=[-10,-5,5,10],max_repetition_ncc=.95,old_depth_used_for_selection=False,geometry_initialized=False,repetition_semantics='Fixed-offset patch self-similarity heuristic, not an identity guarantee',holdout_semantics='One temporal observation excluded from DLT fit only; historical calibration may share evidence')


def _patch(view,xy):
    h,w=view['gray'].shape
    x,y=map(float,xy)
    if not np.isfinite([x,y]).all() or x<3 or y<3 or x>w-4 or y>h-4: return None
    known=cv2.getRectSubPix(view['known'],(7,7),(x,y))
    if (known<1-1e-6).any(): return None
    return cv2.getRectSubPix(view['gray'],(7,7),(x,y)).astype(float)


def _ncc(a,b):
    if a is None or b is None or a.std()<3 or b.std()<3: return -1.
    a=a-a.mean();b=b-b.mean()
    return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))


def _step(a,b,xy,check):
    check()
    xy=np.asarray(xy,np.float32).reshape(-1,2)
    if not len(xy): return xy,np.empty(0,np.uint8)
    kw=dict(winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,.01),flags=0)
    out,status,_=cv2.calcOpticalFlowPyrLK(a['gray'],b['gray'],xy[:,None],None,**kw)
    reasons=np.full(len(xy),3,np.uint8)
    if out is None: return np.full_like(xy,np.nan),reasons
    out=out[:,0];finite=np.isfinite(out).all(axis=1)
    back,bs,_=cv2.calcOpticalFlowPyrLK(b['gray'],a['gray'],np.where(finite[:,None],out,xy)[:,None],None,**kw)
    if back is None: return out,reasons
    for i in range(len(xy)):
        if not finite[i] or not np.isfinite(back[i,0]).all() or not status[i,0] or not bs[i,0]: continue
        if np.linalg.norm(back[i,0]-xy[i])>1.5: reasons[i]=4;continue
        pa,pb=_patch(a,xy[i]),_patch(b,out[i])
        if pa is None or pb is None: reasons[i]=5;continue
        if _ncc(pa,pb)<.75: reasons[i]=6;continue
        reasons[i]=2
    return out,reasons


def _triangulate(views,xy):
    rows=[]
    for v,p in zip(views,xy):
        camera=np.column_stack((v['R'],-v['R']@v['C']))
        projection=v['K']@camera
        rows.extend((p[0]*projection[2]-projection[0],p[1]*projection[2]-projection[1]))
    _,_,vh=np.linalg.svd(np.asarray(rows))
    if abs(vh[-1,3])<1e-12: return None
    point=vh[-1,:3]/vh[-1,3]
    return point if np.isfinite(point).all() else None


def _reproject(point,views):
    uv=[];positive=[]
    for v in views:
        camera=(point-v['C'])@v['R'].T
        pix=camera@v['K'].T
        positive.append(camera[2]>1e-9)
        uv.append(pix[:2]/camera[2] if abs(camera[2])>1e-12 else [np.inf,np.inf])
    return np.asarray(uv),np.asarray(positive)


def recover_local_geometry(views,reference_id,roi,check=lambda:None):
    if not isinstance(views,list) or not 3<=len(views)<=32: raise ValueError('three to thirty-two calibrated temporal views required')
    checked=[]
    for view in views:
        v=_checked(view)
        if type(v.get('decode_index')) is not int: raise ValueError('integer decode_index required')
        v['gray']=cv2.cvtColor(v['image'],cv2.COLOR_RGB2GRAY)
        v['known']=(v['mask']&v['validity']).astype(np.float32)
        checked.append(v)
    checked.sort(key=lambda v:v['decode_index'])
    ids=[v['view_id'] for v in checked]
    if len(set(ids))!=len(ids) or len(set(v['decode_index'] for v in checked))!=len(ids) or reference_id not in ids: raise ValueError('distinct view/decode identities required')
    ri=ids.index(reference_id);ref=checked[ri]
    if any(v['gray'].shape!=ref['gray'].shape for v in checked): raise ValueError('native temporal image grids must match')
    if not isinstance(roi,np.ndarray) or roi.dtype!=bool or roi.shape!=ref['mask'].shape or (roi&~ref['mask']).any(): raise ValueError('invalid reference ROI')
    y,x=np.nonzero(roi);controls=[];ambiguous=0
    if len(x):
        xe=np.linspace(x.min(),x.max()+1,9,dtype=int);ye=np.linspace(y.min(),y.max()+1,9,dtype=int)
        interior=cv2.erode(ref['known'].astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        for row in range(8):
            for col in range(8):
                check();mask=np.zeros(roi.shape,np.uint8)
                mask[ye[row]:ye[row+1],xe[col]:xe[col+1]]=1
                mask&=interior&roi.astype(np.uint8)
                corners=cv2.goodFeaturesToTrack(ref['gray'],maxCorners=4,qualityLevel=.01,minDistance=5,mask=mask,blockSize=5)
                if corners is None: continue
                for xy in corners[:,0]:
                    patch=_patch(ref,xy)
                    repeat=False
                    for delta in (-10,-5,5,10):
                        for offset in ((delta,0),(0,delta)):
                            if _ncc(patch,_patch(ref,xy+offset))>=.95: repeat=True
                    if repeat: ambiguous+=1;continue
                    controls.append(xy)
    xy=np.asarray(controls,np.float32).reshape(-1,2);n=len(xy);m=len(checked)
    observations=np.full((n,m,2),np.nan,np.float32);observations[:,ri]=xy
    observed=np.zeros((n,m),bool);observed[:,ri]=True
    attempted=np.zeros((n,m),bool);attempted[:,ri]=True
    track_reason=np.zeros((n,m),np.uint8);track_reason[:,ri]=1
    closure_error=np.full((n,m),np.nan)
    for direction in (-1,1):
        active=np.arange(n);chain=[ri];prev=ri
        for index in range(ri+direction,m if direction>0 else -1,direction):
            check()
            if not len(active): break
            attempted[active,index]=True
            locations,reasons=_step(checked[prev],checked[index],observations[active,prev],check)
            survivors=[]
            for j,i in enumerate(active):
                code=int(reasons[j])
                if code==2 and _ncc(_patch(ref,xy[i]),_patch(checked[index],locations[j]))<.75: code=7
                if code==2:
                    back=locations[j:j+1]
                    for a,b in zip([index]+list(reversed(chain)),list(reversed(chain))):
                        back,back_reason=_step(checked[a],checked[b],back,check)
                        if back_reason[0]!=2: code=8;break
                    if code==2:
                        closure_error[i,index]=float(np.linalg.norm(back[0]-xy[i]))
                        if closure_error[i,index]>2: code=8
                track_reason[i,index]=code
                if code==2:
                    observations[i,index]=locations[j];observed[i,index]=True;survivors.append(i)
            active=np.asarray(survivors,int);chain.append(index);prev=index
    points=np.full((n,3),np.nan);accepted=np.zeros(n,bool);reason=np.zeros(n,np.uint8)
    angle=np.full(n,np.nan);train_angle=np.full(n,np.nan);holdout_error=np.full(n,np.nan);max_error=np.full(n,np.nan)
    holdout_id=np.full(n,-1,np.int32)
    for i in range(n):
        check();use=np.flatnonzero(observed[i]);vs=[checked[j] for j in use];obs=observations[i,use]
        if len(use)<3: continue
        rays=np.asarray([(_rays(v,p[0],p[1])@v['R'])[0] for v,p in zip(vs,obs)])
        rays/=np.linalg.norm(rays,axis=1)[:,None]
        angle[i]=float(np.degrees(np.arccos(np.clip(rays@rays.T,-1,1))).max())
        if angle[i]<1: reason[i]=1;continue
        # Farthest temporal observation is a fit-excluded diagnostic, not independent truth.
        hi=int(np.argmax(np.abs(np.asarray([v['decode_index'] for v in vs])-ref['decode_index'])))
        train=np.arange(len(vs))!=hi;holdout_id[i]=vs[hi]['view_id']
        train_rays=rays[train]
        train_angle[i]=float(np.degrees(np.arccos(np.clip(train_rays@train_rays.T,-1,1))).max())
        if train_angle[i]<1: reason[i]=7;continue
        point=_triangulate([v for j,v in enumerate(vs) if train[j]],obs[train])
        if point is None: reason[i]=2;continue
        uv,pos=_reproject(point,vs)
        holdout_error[i]=float(np.linalg.norm(uv[hi]-obs[hi]))
        if not pos.all() or holdout_error[i]>2: reason[i]=3;continue
        point=_triangulate(vs,obs)
        if point is None: reason[i]=2;continue
        uv,pos=_reproject(point,vs);max_error[i]=float(np.linalg.norm(uv-obs,axis=1).max())
        if not pos.all() or max_error[i]>2: reason[i]=4;continue
        if any(_patch(v,p) is None for v,p in zip(vs,uv)): reason[i]=5;continue
        points[i]=point;accepted[i]=True;reason[i]=6
    rgb=np.zeros((n,3),np.uint8)
    if n:
        ix=np.rint(xy[:,0]).astype(int);iy=np.rint(xy[:,1]).astype(int);rgb=ref['image'][iy,ix].copy()
    stats={str(v['view_id']):dict(attempted=int(attempted[:,j].sum()),observed=int(observed[:,j].sum()),external_observed=int(observed[:,j].sum()) if j!=ri else 0,reasons={name:int((track_reason[:,j]==code).sum()) for code,name in TRACK_REASONS.items()}) for j,v in enumerate(checked)}
    return dict(points=points,rgb=rgb,reference_xy=xy,accepted=accepted,reasons=reason,observations=observations,observation_mask=observed,attempted=attempted,track_reason=track_reason,chain_closure_error=closure_error,triangulation_angle_degrees=angle,train_angle_degrees=train_angle,holdout_error=holdout_error,max_reprojection_error=max_error,holdout_view_id=holdout_id,view_ids=ids,reference_id=reference_id,parameters=dict(PARAMETERS),track_reason_legend=TRACK_REASONS.copy(),reason_legend=GEOMETRY_REASONS.copy(),per_source=stats,observation_count_semantics='observation_mask includes reference; external_observed excludes reference',counters=dict(selected=n,external_observed=int(observed.sum()-n),rejected_repetitive_reference=ambiguous,accepted=int(accepted.sum()),reasons={name:int((reason==code).sum()) for code,name in GEOMETRY_REASONS.items()}),geometry_accepted=False,physical_scale='unknown',assumptions=['rigid scene','fixed calibrated K R C','correct foreground masks'],depth_used=False,status='EXPERIMENTAL_TRIANGULATED_POINTS_NOT_ACCEPTED_GEOMETRY')
