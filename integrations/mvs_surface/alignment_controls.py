"""Image-only controls versus frozen old-depth projection; diagnostic, never adopted."""
import cv2
import numpy as np
from .local_depth_search import _view, _rays, _project


def _checked(view):
    v = _view(view)
    valid = v.get('validity',np.ones(v['mask'].shape,bool))
    if not isinstance(valid,np.ndarray) or valid.dtype != bool or valid.shape != v['mask'].shape:
        raise ValueError('invalid validity mask')
    v['validity'] = valid
    return v


def select_reference_controls(reference,roi,check=lambda:None):
    ref = _checked(reference)
    if not isinstance(roi,np.ndarray) or roi.dtype != bool or roi.shape != ref['mask'].shape or (roi & ~ref['mask']).any():
        raise ValueError('invalid ROI')
    gray = cv2.cvtColor(ref['image'],cv2.COLOR_RGB2GRAY)
    known = ref['mask'] & ref['validity']
    interior = cv2.erode(known.astype(np.uint8),np.ones((17,17),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool) & (ref['depth']>0)
    h,w = roi.shape
    expanded_roi = cv2.dilate(roi.astype(np.uint8),np.ones((13,13),np.uint8)).astype(bool)
    xy,group,cells,split = [],[],[],[]
    y,x = np.nonzero(roi)
    bounds = [(0,0,w,h,8,6,0)]
    if len(x): bounds.append((int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1,4,4,1))
    for x0,y0,x1,y1,tiles,limit,g in bounds:
        xe = np.linspace(x0,x1,tiles+1,dtype=int)
        ye = np.linspace(y0,y1,tiles+1,dtype=int)
        for row in range(tiles):
            for col in range(tiles):
                check()
                domain = np.zeros((h,w),np.uint8)
                # Four-pixel tile margins keep 7x7 NCC footprints disjoint across tiles.
                if ye[row+1]-ye[row] > 8 and xe[col+1]-xe[col] > 8:
                    domain[ye[row]+4:ye[row+1]-4,xe[col]+4:xe[col+1]-4] = 1
                domain &= (interior & (roi if g else ~expanded_roi)).astype(np.uint8)
                points = cv2.goodFeaturesToTrack(gray,maxCorners=limit,qualityLevel=.01,minDistance=5,mask=domain,blockSize=5,useHarrisDetector=False)
                if points is None: continue
                # Native depth pixel and image tracker share the exact rounded pixel.
                for point in points[:,0]:
                    px,py = np.rint(point).astype(int)
                    xy.append((float(px),float(py)));group.append(g)
                    cells.append(g*64+row*tiles+col);split.append((row+col)%2 == 0)
    coords = np.asarray(xy,np.float32).reshape(-1,2)
    z = ref['depth'][coords[:,1].astype(int),coords[:,0].astype(int)].astype(np.float64)
    return dict(control_xy=coords,group=np.asarray(group,np.uint8),cell_id=np.asarray(cells,np.int32),train_split=np.asarray(split,bool),depth=z,
        parameters=dict(body_tiles=8,body_max_per_tile=6,roi_tiles=4,roi_max_per_tile=4,erosion_pixels=8,quality=.01,min_distance=5,block_size=5,tile_margin_pixels=4,body_roi_exclusion_radius=6,split_semantics='Disjoint level0 NCC footprints; coarse LK pyramid support can overlap and is not independent truth'),geometry_adopted=False)


def _median(values):
    return float(np.median(values)) if len(values) else None


def audit_control_alignment(reference,source,controls,check=lambda:None):
    ref,src = _checked(reference),_checked(source)
    xy = np.asarray(controls['control_xy'],np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or not np.array_equal(xy,np.rint(xy)):
        raise ValueError('controls must be finite rounded native pixels')
    h,w = ref['mask'].shape
    if ((xy<0).any() or (xy[:,0]>=w).any() or (xy[:,1]>=h).any()): raise ValueError('control outside reference')
    for key in ('group','cell_id','train_split'):
        if np.asarray(controls[key]).shape != (len(xy),): raise ValueError('invalid control metadata')
    check()
    a = cv2.cvtColor(ref['image'],cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(src['image'],cv2.COLOR_RGB2GRAY)
    z = ref['depth'][xy[:,1].astype(int),xy[:,0].astype(int)]
    predicted,_,positive = _project(ref,src,_rays(ref,xy[:,0],xy[:,1]),z)
    positive &= z>0
    n = len(xy)
    tracked = np.full((n,2),np.nan,np.float32)
    fb = np.full(n,np.nan,np.float64);ncc=np.full(n,np.nan,np.float64)
    accepted=np.zeros(n,bool)
    gates = {key:np.zeros(n,bool) for key in ('finite_tracking','forward_backward_status','reference_in_frame','source_in_frame','fb_error','known_reference_depth','known_foreground_patch','texture','ncc')}
    gates['known_reference_depth'] = positive.copy()
    if n:
        params = dict(winSize=(31,31),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),flags=0)
        tracked0,status,_ = cv2.calcOpticalFlowPyrLK(a,b,xy.reshape(-1,1,2),None,**params)
        check()
        if tracked0 is not None:
            tracked = tracked0[:,0]
            finite = np.isfinite(tracked).all(axis=1)
            safe = np.where(finite[:,None],tracked,xy).astype(np.float32)
            reverse,backstatus,_ = cv2.calcOpticalFlowPyrLK(b,a,safe.reshape(-1,1,2),None,**params)
            if reverse is not None:
                fb = np.linalg.norm(reverse[:,0]-xy,axis=1)
                sh,sw=b.shape
                inside = finite & (tracked[:,0]>=3)&(tracked[:,0]<=sw-4)&(tracked[:,1]>=3)&(tracked[:,1]<=sh-4)
                reference_inside = (xy[:,0]>=3)&(xy[:,0]<=w-4)&(xy[:,1]>=3)&(xy[:,1]<=h-4)
                gates['finite_tracking'] = finite
                gates['forward_backward_status'] = status.ravel().astype(bool)&backstatus.ravel().astype(bool)
                gates['reference_in_frame'] = reference_inside
                gates['source_in_frame'] = inside
                gates['fb_error'] = np.isfinite(fb)&(fb<=1)
                candidate = reference_inside & inside & status.ravel().astype(bool) & backstatus.ravel().astype(bool) & (fb<=1) & positive
                for i in np.flatnonzero(candidate):
                    check()
                    rx,ry=xy[i].astype(int);sx,sy=np.rint(tracked[i]).astype(int)
                    if not (ref['mask'][ry-3:ry+4,rx-3:rx+4] & ref['validity'][ry-3:ry+4,rx-3:rx+4]).all(): continue
                    known_patch = cv2.getRectSubPix((src['mask'] & src['validity']).astype(np.float32),(7,7),tuple(map(float,tracked[i])))
                    if not (known_patch >= 1-1e-6).all(): continue
                    gates['known_foreground_patch'][i] = True
                    pa=cv2.getRectSubPix(a,(7,7),tuple(map(float,xy[i]))).astype(float)
                    pb=cv2.getRectSubPix(b,(7,7),tuple(map(float,tracked[i]))).astype(float)
                    if pa.std()<3 or pb.std()<3: continue
                    gates['texture'][i] = True
                    pa-=pa.mean();pb-=pb.mean()
                    ncc[i]=float(np.sum(pa*pb)/np.sqrt(np.sum(pa*pa)*np.sum(pb*pb)))
                    gates['ncc'][i] = ncc[i]>=.8
                    accepted[i]=gates['ncc'][i]
    residual=tracked-predicted
    group=np.asarray(controls['group']);train=np.asarray(controls['train_split'],bool);cell=np.asarray(controls['cell_id'])
    bodytrain=accepted&(group==0)&train
    bodytest=accepted&(group==0)&~train
    proposal=None;reason='INSUFFICIENT_SPATIAL_BODY_CONTROLS'
    if bodytrain.sum()>=12 and bodytest.sum()>=8 and len(np.unique(cell[bodytrain]))>=3 and len(np.unique(cell[bodytest]))>=3:
        fit=np.median(residual[bodytrain],axis=0)
        if np.linalg.norm(fit)<=20:
            proposal=fit.tolist();reason='DIAGNOSTIC_TRANSLATION_NOT_ADOPTED'
        else: reason='TRANSLATION_EXCEEDS_20_PIXELS_NOT_EVALUABLE'
    norm=np.linalg.norm(residual,axis=1)
    summaries={}
    for name,mask in [('body_train',bodytrain),('body_test',bodytest),('roi',accepted&(group==1))]:
        attempt_mask = ((group==0)&train) if name=='body_train' else (((group==0)&~train) if name=='body_test' else (group==1))
        summaries[name]=dict(attempted=int(attempt_mask.sum()),rejected=int((attempt_mask&~accepted).sum()),gate_not_passed={key:int((attempt_mask&~value).sum()) for key,value in gates.items()},accepted=int(mask.sum()),distinct_cells=len(np.unique(cell[mask])),median_before=_median(norm[mask]),median_after=_median(np.linalg.norm(residual[mask]-np.asarray(proposal),axis=1)) if proposal is not None else None)
    return dict(ref_xy=xy.copy(),tracked_xy=tracked,predicted_xy=predicted,accepted=accepted,fb_error=fb,ncc=ncc,group=group.copy(),cell_id=cell.copy(),train_split=train.copy(),prediction_positive=positive,residual_xy=residual,
        proposal_translation_xy=proposal,reason=reason,status='EVALUATED_NOT_ADOPTED' if proposal is not None else 'NOT_EVALUABLE',per_group=summaries,selected_controls=n,accepted_controls=int(accepted.sum()),geometry_adopted=False,
        gate_not_passed={key:int((~value).sum()) for key,value in gates.items()},gate_count_semantics='Overlapping counts over all attempted controls; not passed includes gates not evaluated after an earlier failure',
        split_semantics='NCC level0 footprints disjoint across tiles; coarse pyramid support can overlap, not independent truth',
        parameters=dict(lk_window=31,lk_max_level=4,lk_iterations=30,lk_epsilon=.01,lk_flags=0,geometry_initialized=False,tracking_initialization='Image-only; accepted eligibility additionally requires positive known-depth geometric projection',max_fb_error=1.,patch_size=7,min_patch_std=3.,min_ncc=.8,max_translation=20.,min_train=12,min_test=8,min_tiles_each=3))
