"""Bounded CPU object feature births and row-preserving adjacent-frame tracking.

Coordinates use OpenCV pixel-center zero convention. No IDs are reassigned here:
callers retain their ID vector and filter it with the returned accepted vector.
Only actual supplied masks authorize births or sampled endpoint acceptance.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Profile:
    """Strict bounded profile for the frozen selection experiment."""
    max_live: int = 768
    grid: int = 8
    per_cell: int = 12
    min_distance: float = 5.0
    quality_level: float = 0.001
    erosion: int = 7
    lk_window: int = 21
    pyramid: int = 3
    fb_max: float = 1.0
    min_eigen: float = 1e-4
    zncc_min: float = 0.75
    patch: int = 15

    def __post_init__(self) -> None:
        bounds = {'max_live': (1,768), 'grid': (1,8), 'per_cell': (1,12),
                  'erosion': (3,16), 'lk_window': (3,31), 'pyramid': (0,3), 'patch': (3,31)}
        for name,(lo,hi) in bounds.items():
            value = getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or not lo <= value <= hi:
                raise ValueError(name)
        if self.lk_window % 2 != 1 or self.patch % 2 != 1:
            raise ValueError('windows must be odd')
        if self.erosion < self.patch // 2:
            raise ValueError('erosion must cover the ZNCC patch radius')
        for name,lo,hi in [('min_distance',5,100),('quality_level',0.001,1),
                           ('fb_max',0.001,1),('min_eigen',1e-4,1),('zncc_min',0.75,1)]:
            v=getattr(self,name)
            if not np.isfinite(v) or not lo <= v <= hi:
                raise ValueError(name)


def _gray(gray: np.ndarray) -> np.ndarray:
    a=np.asarray(gray)
    if a.ndim != 2 or a.dtype != np.uint8 or min(a.shape) < 3:
        raise ValueError('gray must be a nonempty uint8 image')
    return np.ascontiguousarray(a)


def _xy(xy: Any) -> np.ndarray:
    a=np.asarray(xy,dtype=np.float32)
    if a.size == 0:
        return np.empty((0,2),np.float32)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError('xy must have shape (N,2)')
    return a.copy()


def eroded_mask(mask: np.ndarray, cv: Any, profile: Profile = Profile()) -> np.ndarray:
    a=np.asarray(mask)
    if a.ndim != 2 or not np.isfinite(a).all() or not np.isin(a, [0,255]).all():
        raise ValueError('mask must be 2D with only binary 0/255 values')
    binary=(a>0).astype(np.uint8)*255
    k=2*profile.erosion+1
    return cv.erode(binary,np.ones((k,k),np.uint8),borderType=cv.BORDER_CONSTANT,borderValue=0)


def points_in_mask(mask: np.ndarray, xy: Any) -> np.ndarray:
    p=_xy(xy); h,w=mask.shape
    good=np.isfinite(p).all(axis=1)
    idx=np.flatnonzero(good)
    rounded=np.floor(p[idx]+0.5).astype(np.int64)
    inside=(rounded[:,0]>=0)&(rounded[:,0]<w)&(rounded[:,1]>=0)&(rounded[:,1]<h)
    good[idx]=False
    ii=idx[inside]; rr=rounded[inside]
    good[ii]=mask[rr[:,1],rr[:,0]]>0
    return good


def _cells(mask: np.ndarray, grid: int) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    yy,xx=np.nonzero(mask)
    labels=np.full(mask.shape,-1,np.int32)
    if not len(xx): return labels,(0,0,0,0)
    x0,y0,x1,y1=int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    labels[yy,xx]=np.minimum((yy-y0)*grid//(y1-y0),grid-1)*grid+np.minimum((xx-x0)*grid//(x1-x0),grid-1)
    return labels,(x0,y0,x1,y1)


def coverage(mask: np.ndarray, xy: Any, grid: int = 8) -> dict[str, Any]:
    if not isinstance(grid,int) or not 1<=grid<=8: raise ValueError('grid')
    labels,bbox=_cells(np.asarray(mask)>0,grid)
    supported=np.unique(labels[labels>=0]); p=_xy(xy)
    valid=points_in_mask(np.asarray(mask),p)
    rr=np.floor(p[valid]+0.5).astype(int)
    occupied=np.unique(labels[rr[:,1],rr[:,0]]) if len(rr) else np.empty(0,int)
    return {'supported_cells': int(len(supported)), 'occupied_cells': int(len(occupied)),
            'fraction': float(len(occupied)/len(supported)) if len(supported) else 0.0,
            'bbox': list(bbox)}


def rebalance(mask: np.ndarray, xy: Any, ids: Any,
              profile: Profile = Profile()) -> np.ndarray:
    """Retain oldest IDs under current bbox-cell quotas, without moving points.

    The returned selection applies only to future tracking. Callers retain all
    previous observations of retired IDs. Unsupported/nonfinite points retire.
    Mask lookup uses the same nearest pixel-center convention as points_in_mask.
    """
    a=np.asarray(mask)
    if a.ndim!=2 or not np.isfinite(a).all() or not np.isin(a,[0,255]).all():
        raise ValueError('mask must be 2D with only binary 0/255 values')
    p=_xy(xy); identity=np.asarray(ids)
    if identity.shape!=(len(p),) or identity.dtype.kind not in 'iu':
        raise ValueError('ids must be a row-aligned integer vector')
    if len(np.unique(identity))!=len(identity):
        raise ValueError('ids must be unique')
    labels,_=_cells(a>0,profile.grid)
    valid=points_in_mask(a,p)
    counts=np.zeros(profile.grid**2,int); keep=np.zeros(len(p),bool)
    live=0
    for index in np.argsort(identity,kind='stable'):
        if not valid[index]: continue
        x,y=np.floor(p[index]+0.5).astype(int)
        cell=labels[y,x]
        if counts[cell]>=profile.per_cell or live>=profile.max_live: continue
        keep[index]=True; counts[cell]+=1; live+=1
    return keep


def seed_balanced(gray: np.ndarray, mask: np.ndarray, existing_xy: Any, cv: Any,
                  profile: Profile = Profile()) -> tuple[np.ndarray,np.ndarray]:
    gray=_gray(gray)
    if np.asarray(mask).shape != gray.shape: raise ValueError('mask shape')
    safe=eroded_mask(mask,cv,profile); labels,_=_cells(np.asarray(mask)>0,profile.grid)
    old=_xy(existing_xy)
    if len(old)>profile.max_live or not np.isfinite(old).all(): raise ValueError('existing coordinates')
    if not ((old[:,0]>=0)&(old[:,0]<gray.shape[1])&(old[:,1]>=0)&(old[:,1]<gray.shape[0])).all():
        raise ValueError('existing coordinates out of image')
    counts=np.zeros(profile.grid**2,int)
    for p in old:
        x,y=np.floor(p+0.5).astype(int)
        if 0<=x<gray.shape[1] and 0<=y<gray.shape[0] and labels[y,x]>=0: counts[labels[y,x]]+=1
    response=cv.cornerMinEigenVal(gray,blockSize=3,ksize=3)
    if not np.isfinite(response).all() or not np.any(response[safe>0]>0):
        return np.empty((0,2),np.float32),np.empty(0,np.int32)
    pools=[]
    h,w=gray.shape
    for cell in range(profile.grid**2):
        yy,xx=np.nonzero((labels==cell)&(safe>0))
        if not len(xx) or counts[cell]>=profile.per_cell:
            pools.append([])
            continue
        # blockSize=3 and Sobel ksize=3 need two source pixels; add one
        # further pixel for the detector's local maximum neighborhood.
        halo=3
        x0=max(0,int(xx.min())-halo); x1=min(w,int(xx.max())+1+halo)
        y0=max(0,int(yy.min())-halo); y1=min(h,int(yy.max())+1+halo)
        cm=((labels[y0:y1,x0:x1]==cell)&(safe[y0:y1,x0:x1]>0)).astype(np.uint8)*255
        corners=cv.goodFeaturesToTrack(np.ascontiguousarray(gray[y0:y1,x0:x1]),
                    maxCorners=profile.per_cell*4,qualityLevel=profile.quality_level,
                    minDistance=profile.min_distance,mask=cm,blockSize=3,useHarrisDetector=False)
        pools.append([] if corners is None else list(corners.reshape(-1,2)+[x0,y0]))
    # Fixed bounded buffer avoids rebuilding the full population per candidate.
    population=np.empty((profile.max_live,2),np.float32)
    population[:len(old)]=old
    born=[]; sources=[]
    # Round-robin least occupied cells prevents one strong texture region winning.
    while len(old)+len(born)<profile.max_live:
        progressed=False
        for cell in sorted(range(len(pools)),key=lambda c:(counts[c],c)):
            if counts[cell]>=profile.per_cell: continue
            while pools[cell]:
                p=pools[cell].pop(0)
                if not np.isfinite(p).all() or not points_in_mask(safe,p[None])[0]: continue
                x,y=np.floor(p+0.5).astype(int)
                if response[y,x]<=0: continue
                allp=population[:len(old)+len(born)]
                if len(allp) and np.any(np.sum((allp-p)**2,axis=1)<profile.min_distance**2): continue
                population[len(old)+len(born)]=p
                born.append(p.copy()); sources.append(cell); counts[cell]+=1; progressed=True
                break
            if len(old)+len(born)>=profile.max_live: break
        if not progressed: break
    return np.asarray(born,np.float32).reshape(-1,2),np.asarray(sources,np.int32)


def step_tracks(prev_gray: np.ndarray, gray: np.ndarray, xy: Any, cv: Any,
                profile: Profile = Profile()) -> dict[str,np.ndarray]:
    prev_gray=_gray(prev_gray); gray=_gray(gray)
    if prev_gray.shape!=gray.shape: raise ValueError('image shape changed')
    p=_xy(xy); n=len(p)
    if n>profile.max_live: raise ValueError('live cap')
    out=p.copy(); accepted=np.zeros(n,bool); reasons=np.full(n,'invalid_coordinates',dtype='<U24')
    fb=np.full(n,np.inf,np.float32); scores=np.full(n,np.nan,np.float32)
    radius=profile.patch//2
    def interior(q: np.ndarray) -> np.ndarray:
        return np.isfinite(q).all(axis=1)&(q[:,0]>=radius)&(q[:,0]<=gray.shape[1]-1-radius)&(q[:,1]>=radius)&(q[:,1]<=gray.shape[0]-1-radius)
    ids=np.flatnonzero(interior(p))
    if len(ids):
        opts=dict(winSize=(profile.lk_window,profile.lk_window),maxLevel=profile.pyramid,
                  criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,30,0.01),minEigThreshold=profile.min_eigen)
        q,status,_=cv.calcOpticalFlowPyrLK(prev_gray,gray,p[ids,None,:],None,**opts)
        reasons[ids]='lk_forward'
        if q is not None and status is not None:
            q=q.reshape(-1,2); out[ids]=q
            good=(status.reshape(-1)>0)&interior(q)
            fids=ids[good]; fq=q[good]
            reasons[fids]='lk_backward'
            if len(fids):
                back,bs,_=cv.calcOpticalFlowPyrLK(gray,prev_gray,fq[:,None,:],None,**opts)
                if back is not None and bs is not None:
                    back=back.reshape(-1,2); errors=np.linalg.norm(back-p[fids],axis=1); fb[fids]=errors
                    keep=(bs.reshape(-1)>0)&np.isfinite(back).all(axis=1)&np.isfinite(errors)&(errors<=profile.fb_max)
                    reasons[fids[~keep]]='forward_backward'
                    for i,qpoint in zip(fids[keep],fq[keep]):
                        a=cv.getRectSubPix(prev_gray,(profile.patch,profile.patch),tuple(map(float,p[i]))).astype(np.float64)
                        b=cv.getRectSubPix(gray,(profile.patch,profile.patch),tuple(map(float,qpoint))).astype(np.float64)
                        a-=a.mean(); b-=b.mean(); denom=np.linalg.norm(a)*np.linalg.norm(b)
                        reasons[i]='zero_texture' if denom<=1e-12 else 'zncc'
                        if denom>1e-12:
                            score=float(np.sum(a*b)/denom); scores[i]=score
                            if np.isfinite(score) and score>=profile.zncc_min: accepted[i]=True; reasons[i]='accepted'
    return {'xy':out,'accepted':accepted,'reasons':reasons,'fb_error':fb,'zncc':scores}
