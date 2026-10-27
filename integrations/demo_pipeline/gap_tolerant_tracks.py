"""Measured-only CPU recovery across bounded decode-frame gaps.

Predictions are search proposals, never observations. No mask propagation, ID
merging, or observations interpolated into missing frames occur in this module.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np
from .balanced_object_tracks import Profile, step_tracks, _gray, _xy, eroded_mask, points_in_mask, rebalance


@dataclass(frozen=True)
class RecoveryProfile:
    max_gap: int = 3
    min_neighbors: int = 6
    neighbor_radius: float = 96.0
    min_neighbor_span: float = 12.0
    search_radius: int = 8
    zncc_min: float = .85
    peak_margin: float = .08
    neighbor_error_max: float = 2.0
    collision_distance: float = 3.0
    template: int = 15

    def __post_init__(self) -> None:
        for key,lo,hi in [('max_gap',1,3),('min_neighbors',6,32),('search_radius',2,8),('template',7,21)]:
            v=getattr(self,key)
            if isinstance(v,bool) or not isinstance(v,int) or not lo<=v<=hi: raise ValueError(key)
        if self.template%2!=1: raise ValueError('template must be odd')
        for key,lo,hi in [('neighbor_radius',24,192),('min_neighbor_span',6,48),('zncc_min',.8,1),
                          ('peak_margin',.03,.5),('neighbor_error_max',.5,3),('collision_distance',2,8)]:
            v=getattr(self,key)
            if not np.isfinite(v) or not lo<=v<=hi: raise ValueError(key)


def _similarity(a: np.ndarray,b: np.ndarray) -> np.ndarray | None:
    """Small bounded robust least squares similarity; no global object model."""
    keep=np.ones(len(a),bool)
    for _ in range(3):
        aa=a[keep]; bb=b[keep]
        if len(aa)<6: return None
        rows=np.zeros((2*len(aa),4)); target=bb.reshape(-1)
        rows[0::2]=np.column_stack((aa[:,0],-aa[:,1],np.ones(len(aa)),np.zeros(len(aa))))
        rows[1::2]=np.column_stack((aa[:,1],aa[:,0],np.zeros(len(aa)),np.ones(len(aa))))
        fit,_,rank,_=np.linalg.lstsq(rows,target,rcond=None)
        if rank<4 or not np.isfinite(fit).all(): return None
        u,v,tx,ty=fit; matrix=np.array([[u,-v,tx],[v,u,ty]])
        err=np.linalg.norm(a@matrix[:,:2].T+matrix[:,2]-b,axis=1)
        keep=err<=max(.5,min(2.,float(np.median(err)*2.5)))
    scale=np.hypot(u,v)
    return matrix if keep.sum()>=6 and .8<=scale<=1.25 else None


class GapTracker:
    def __init__(self,cv: Any,profile: Profile=Profile(),recovery: RecoveryProfile=RecoveryProfile()):
        self.cv=cv; self.profile=profile; self.recovery=recovery
        self.xy=np.empty((0,2),np.float32); self.ids=np.empty(0,np.int64)
        self.pending: dict[int,dict[str,Any]]={}; self.images: dict[int,np.ndarray]={}
        self.decode_index: int | None=None; self.next_id=0
        self.events: list[dict[str,Any]]=[]

    def seed(self,xy: Any,decode_index: int) -> np.ndarray:
        if decode_index!=self.decode_index: raise ValueError('seed only after same-frame update')
        p=_xy(xy)
        if not np.isfinite(p).all() or len(p)+len(self.xy)>self.profile.max_live: raise ValueError('seed bounds')
        h,w=self.images[decode_index].shape
        if not ((p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)).all(): raise ValueError('seed coordinates')
        ids=np.arange(self.next_id,self.next_id+len(p),dtype=np.int64); self.next_id+=len(p)
        self.xy=np.concatenate((self.xy,p)); self.ids=np.concatenate((self.ids,ids))
        return ids

    def clear(self,reason: str) -> dict[str,Any]:
        event={'kind':'clear','reason':str(reason),'ids':self.ids.tolist()+list(self.pending)}
        self.xy=np.empty((0,2),np.float32); self.ids=np.empty(0,np.int64); self.pending.clear()
        self.events.append(event)
        return event

    def filter_active(self,mask: np.ndarray,quotas: bool=True) -> dict[str,Any]:
        keep=points_in_mask(eroded_mask(mask,self.cv,self.profile),self.xy)
        mask_retired=self.ids[~keep].tolist()
        self.xy=self.xy[keep]; self.ids=self.ids[keep]
        quota_retired=[]
        if quotas:
            keep=rebalance(mask,self.xy,self.ids,self.profile)
            quota_retired=self.ids[~keep].tolist(); self.xy=self.xy[keep]; self.ids=self.ids[keep]
        # A pending ID supplies no current observation. A committed mask cannot
        # validate it, but does not terminate a bounded image-supported search.
        # Explicit NO_MATCH/unknown-mask policy is represented by clear().
        pending_retired=[]
        event={'kind':'mask_filter','mask_retired_ids':mask_retired,'quota_retired_ids':quota_retired,
               'pending_retired_ids':pending_retired,'mask_rejected':len(mask_retired),
               'quota_retired':len(quota_retired)}
        self.events.append(event)
        return event

    def _recover(self,item: dict[str,Any],gray: np.ndarray,current: dict[int,np.ndarray]) -> tuple[np.ndarray | None,str]:
        cfg=self.recovery; p=item['xy']; snapshot=item['neighbors']
        common=[i for i in current if i in snapshot and np.linalg.norm(snapshot[i]-p)<=cfg.neighbor_radius]
        if len(common)<cfg.min_neighbors: return None,'insufficient_neighbors'
        # Nearest 32 avoids an unbounded local fit.
        common=sorted(common,key=lambda i:float(np.linalg.norm(snapshot[i]-p)))[:32]
        a=np.array([snapshot[i] for i in common]); b=np.array([current[i] for i in common])
        singular=np.linalg.svd(a-a.mean(axis=0),compute_uv=False)
        if singular[-1]/np.sqrt(len(a))<cfg.min_neighbor_span: return None,'neighbor_geometry'
        matrix=_similarity(a,b)
        if matrix is None: return None,'neighbor_motion'
        residual=np.linalg.norm(a@matrix[:,:2].T+matrix[:,2]-b,axis=1)
        inliers=a[residual<=cfg.neighbor_error_max]
        if len(inliers)<cfg.min_neighbors: return None,'insufficient_motion_inliers'
        spread=np.linalg.svd(inliers-inliers.mean(axis=0),compute_uv=False)
        if spread[-1]/np.sqrt(len(inliers))<cfg.min_neighbor_span: return None,'inlier_geometry'
        # Require interpolation inside measured local support, not extrapolation
        # from a distant cluster whose wide initial set was due to outliers.
        hull=self.cv.convexHull(inliers.astype(np.float32))
        if self.cv.pointPolygonTest(hull,tuple(map(float,p)),True)<-cfg.neighbor_error_max:
            return None,'outside_neighbor_support'
        radius=cfg.search_radius; half=cfg.template//2; size=cfg.template+2*radius
        origin=p-[half+radius,half+radius]
        corners=np.array([[0,0],[size-1,0],[0,size-1],[size-1,size-1]])+origin
        mapped=corners@matrix[:,:2].T+matrix[:,2]
        h,w=gray.shape
        if not ((mapped[:,0]>=0)&(mapped[:,0]<w-1)&(mapped[:,1]>=0)&(mapped[:,1]<h-1)).all(): return None,'search_bounds'
        source=self.images.get(item['frame'])
        if source is None: return None,'anchor_expired'
        if not (half+radius<=p[0]<w-half-radius-1 and half+radius<=p[1]<h-half-radius-1): return None,'anchor_bounds'
        transform=matrix.copy(); transform[:,2]+=matrix[:,:2]@origin
        warped=self.cv.warpAffine(gray,transform,(size,size),flags=self.cv.INTER_LINEAR|self.cv.WARP_INVERSE_MAP)
        template=self.cv.getRectSubPix(source,(cfg.template,cfg.template),tuple(map(float,p)))
        if template.std()<1: return None,'zero_texture'
        scores=self.cv.matchTemplate(warped,template,self.cv.TM_CCOEFF_NORMED)
        if not np.isfinite(scores).all(): return None,'nonfinite_score'
        y,x=np.unravel_index(np.argmax(scores),scores.shape); score=float(scores[y,x])
        alternative=scores.copy(); alternative[max(0,y-2):y+3,max(0,x-2):x+3]=-1
        if score<cfg.zncc_min: return None,'recovery_zncc'
        if score-float(alternative.max())<cfg.peak_margin: return None,'ambiguous_peak'
        delta=np.array([x-radius,y-radius],float)
        if np.linalg.norm(matrix[:,:2]@delta)>cfg.neighbor_error_max: return None,'neighbor_inconsistent'
        # Reverse template localization in the real anchor image.
        patch=warped[y:y+cfg.template,x:x+cfg.template]
        anchor=self.cv.getRectSubPix(source,(size,size),tuple(map(float,p)))
        reverse=self.cv.matchTemplate(anchor,patch,self.cv.TM_CCOEFF_NORMED)
        if not np.isfinite(reverse).all(): return None,'reverse_nonfinite'
        ry,rx=np.unravel_index(np.argmax(reverse),reverse.shape)
        reverse_alternative=reverse.copy()
        reverse_alternative[max(0,ry-2):ry+3,max(0,rx-2):rx+3]=-1
        if float(reverse[ry,rx])-float(reverse_alternative.max())<cfg.peak_margin:
            return None,'reverse_ambiguous_peak'
        if np.hypot(rx-radius,ry-radius)>1.5 or reverse[ry,rx]<cfg.zncc_min: return None,'reverse_inconsistent'
        q=matrix[:,:2]@(p+delta)+matrix[:,2]
        return q.astype(np.float32),'recovered'

    def update(self,gray: np.ndarray,decode_index: int) -> dict[str,Any]:
        gray=_gray(gray)
        if not isinstance(decode_index,int) or decode_index<0: raise ValueError('decode index')
        previous=self.decode_index
        if previous is not None and decode_index!=previous+1: raise ValueError('every decoded frame is required')
        self.events=[]; rejections={}
        if previous is not None:
            oldxy=self.xy.copy(); oldids=self.ids.copy(); neighbors=dict(zip(map(int,oldids),oldxy))
            move=step_tracks(self.images[previous],gray,oldxy,self.cv,self.profile)
            good=move['accepted']; self.xy=move['xy'][good]; self.ids=oldids[good]
            for ident,p,reason in zip(oldids[~good],oldxy[~good],move['reasons'][~good]):
                self.pending[int(ident)]={'frame':previous,'xy':p.copy(),'neighbors':neighbors}
                key=str(reason); rejections[key]=rejections.get(key,0)+1
                self.events.append({'kind':'lost','id':int(ident),'decode_index':decode_index,'reason':key})
        self.images[decode_index]=gray.copy(); self.decode_index=decode_index
        # Bound recovery work before any local fits/template searches.
        if len(self.pending)>self.profile.max_live:
            for ident in sorted(self.pending,reverse=True)[:len(self.pending)-self.profile.max_live]:
                del self.pending[ident]; self.events.append({'kind':'pending_capacity_retired','id':ident})
        current=dict(zip(map(int,self.ids),self.xy)); proposals=[]
        for ident,item in list(self.pending.items()):
            gap=decode_index-item['frame']
            if gap>self.recovery.max_gap:
                del self.pending[ident]; self.events.append({'kind':'expired','id':ident,'decode_index':decode_index}); continue
            q,reason=self._recover(item,gray,current)
            if q is not None and len(self.xy) and np.any(np.linalg.norm(self.xy-q,axis=1)<self.recovery.collision_distance):
                q=None; reason='active_collision'
            if q is None:
                rejections[reason]=rejections.get(reason,0)+1
            else: proposals.append((ident,q,item['frame']))
        recovered=[]
        for ident,q,anchor in proposals:
            if any(other!=ident and np.linalg.norm(q-oq)<self.recovery.collision_distance for other,oq,_ in proposals):
                rejections['recovery_collision']=rejections.get('recovery_collision',0)+1; continue
            if len(self.ids)>=self.profile.max_live: continue
            self.ids=np.append(self.ids,ident); self.xy=np.vstack((self.xy,q)); del self.pending[ident]; recovered.append(ident)
            self.events.append({'kind':'recovered','id':ident,'anchor_decode_index':anchor,'decode_index':decode_index,
                                'missing_decode_indices':list(range(anchor+1,decode_index)),'xy':q.tolist()})
        for frame in list(self.images):
            if decode_index-frame>self.recovery.max_gap: del self.images[frame]
        return {'active_xy':self.xy.copy(),'active_ids':self.ids.copy(),'recovered_ids':np.array(recovered,np.int64),
                'strict_ids':np.array(list(current),np.int64),
                'pending_count':len(self.pending),'events':list(self.events),'rejections':rejections}
