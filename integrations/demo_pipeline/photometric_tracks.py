"""Local contrast pre-normalization before LK, with raw-image verification.

This is not joint photometric/geometric optimization. Original images remain the
source for detection, recovery and final photometric admission.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np
from .balanced_object_tracks import Profile, _gray, _xy, step_tracks
from .gap_tolerant_tracks import GapTracker, RecoveryProfile


@dataclass(frozen=True)
class PhotometricProfile:
    local_window: int = 31
    contrast_floor: float = 8.0
    normalized_scale: float = 32.0
    gain_min: float = .25
    gain_max: float = 4.0
    bias_abs_max: float = 128.0
    texture_std_min: float = 2.0
    saturated_fraction_max: float = .2
    fit_residual_max: float = 20.0

    def __post_init__(self) -> None:
        if isinstance(self.local_window,bool) or not isinstance(self.local_window,int) or not 15<=self.local_window<=63 or self.local_window%2!=1:
            raise ValueError('local_window')
        bounds={'contrast_floor':(1,32),'normalized_scale':(8,48),'gain_min':(.1,1),
                'gain_max':(1,8),'bias_abs_max':(0,192),'texture_std_min':(1,16),
                'saturated_fraction_max':(0,.5),'fit_residual_max':(1,32)}
        for name,(lo,hi) in bounds.items():
            value=getattr(self,name)
            if not np.isfinite(value) or not lo<=value<=hi: raise ValueError(name)
        if self.gain_min>=self.gain_max: raise ValueError('gain range')


def _normalize(gray: np.ndarray,cv: Any,cfg: PhotometricProfile) -> np.ndarray:
    raw=gray.astype(np.float32)
    size=(cfg.local_window,cfg.local_window)
    mean=cv.boxFilter(raw,-1,size,normalize=True,borderType=cv.BORDER_REFLECT101)
    second=cv.boxFilter(raw*raw,-1,size,normalize=True,borderType=cv.BORDER_REFLECT101)
    std=np.sqrt(np.maximum(second-mean*mean,0))
    normalized=128+cfg.normalized_scale*(raw-mean)/np.maximum(std,cfg.contrast_floor)
    if not np.isfinite(normalized).all(): raise ValueError('nonfinite normalized image')
    return np.clip(np.rint(normalized),0,255).astype(np.uint8)


def photometric_step(prev: np.ndarray,gray: np.ndarray,xy: Any,cv: Any,
                     profile: Profile=Profile(),photometric: PhotometricProfile=PhotometricProfile()) -> dict[str,np.ndarray]:
    prev=_gray(prev); gray=_gray(gray); p=_xy(xy)
    n=len(p)
    if prev.shape!=gray.shape: raise ValueError('image shape changed')
    if n>profile.max_live: raise ValueError('live cap')
    normalized_prev=_normalize(prev,cv,photometric); normalized_gray=_normalize(gray,cv,photometric)
    result={'xy':p.copy(),'accepted':np.zeros(n,bool),'reasons':np.full(n,'invalid_coordinates',dtype='<U24'),
            'fb_error':np.full(n,np.inf,np.float32),'zncc':np.full(n,np.nan,np.float32)}
    radius=profile.patch//2
    def interior(q: np.ndarray) -> np.ndarray:
        return np.isfinite(q).all(axis=1)&(q[:,0]>=radius)&(q[:,0]<=gray.shape[1]-1-radius)&(q[:,1]>=radius)&(q[:,1]<=gray.shape[0]-1-radius)
    indices=np.flatnonzero(interior(p))
    if len(indices):
        opts=dict(winSize=(profile.lk_window,profile.lk_window),maxLevel=profile.pyramid,
                  criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,30,.01),minEigThreshold=profile.min_eigen)
        q,status,_=cv.calcOpticalFlowPyrLK(normalized_prev,normalized_gray,p[indices,None,:],None,**opts)
        result['reasons'][indices]='lk_forward'
        if q is not None and status is not None:
            q=q.reshape(-1,2); result['xy'][indices]=q
            good=(status.reshape(-1)>0)&interior(q); forward_ids=indices[good]; forward_xy=q[good]
            result['reasons'][forward_ids]='lk_backward'
            if len(forward_ids):
                back,bs,_=cv.calcOpticalFlowPyrLK(normalized_gray,normalized_prev,forward_xy[:,None,:],None,**opts)
                if back is not None and bs is not None:
                    back=back.reshape(-1,2); error=np.linalg.norm(back-p[forward_ids],axis=1)
                    result['fb_error'][forward_ids]=error
                    keep=(bs.reshape(-1)>0)&np.isfinite(back).all(axis=1)&np.isfinite(error)&(error<=profile.fb_max)
                    result['reasons'][forward_ids[~keep]]='forward_backward'
                    result['accepted'][forward_ids[keep]]=True
    gains=np.full(n,np.nan,np.float32); biases=gains.copy(); residuals=gains.copy()
    # Only the original image supplies ZNCC/photometric admission. Normalized
    # images affect LK search and FB, without a second correlation gate.
    cfg=photometric
    for index in np.flatnonzero(result['accepted']):
        a=cv.getRectSubPix(prev,(profile.patch,profile.patch),tuple(map(float,p[index]))).astype(np.float64)
        b=cv.getRectSubPix(gray,(profile.patch,profile.patch),tuple(map(float,result['xy'][index]))).astype(np.float64)
        reason='accepted'
        if max(np.mean((a<=1)|(a>=254)),np.mean((b<=1)|(b>=254)))>cfg.saturated_fraction_max:
            reason='photometric_saturation'
        ac=a-a.mean(); bc=b-b.mean(); std_a=float(a.std()); std_b=float(b.std())
        if min(std_a,std_b)<cfg.texture_std_min: reason='photometric_texture'
        denom=float(np.sum(ac*ac)); cross=float(np.sum(ac*bc))
        if denom>1e-12 and std_b>0:
            gain=cross/denom; bias=float(b.mean()-gain*a.mean())
            residual=float(np.sqrt(np.mean((b-(gain*a+bias))**2)))
            score=float(cross/(np.linalg.norm(ac)*np.linalg.norm(bc)))
            gains[index]=gain; biases[index]=bias; residuals[index]=residual; result['zncc'][index]=score
            if reason=='accepted':
                if not np.isfinite([gain,bias,residual,score]).all(): reason='photometric_nonfinite'
                elif not cfg.gain_min<=gain<=cfg.gain_max: reason='photometric_gain'
                elif abs(bias)>cfg.bias_abs_max: reason='photometric_bias'
                elif residual>cfg.fit_residual_max: reason='photometric_residual'
                elif score<profile.zncc_min: reason='photometric_zncc'
        else: reason='photometric_texture'
        result['accepted'][index]=reason=='accepted'; result['reasons'][index]=reason
    result.update(gain=gains,bias=biases,photometric_residual=residuals)
    return result


class PhotometricGapTracker(GapTracker):
    """Base gap policy with only the strict measured-step implementation replaced."""
    def __init__(self,cv: Any,profile: Profile=Profile(),recovery: RecoveryProfile=RecoveryProfile(),
                 photometric: PhotometricProfile=PhotometricProfile()):
        super().__init__(cv,profile,recovery)
        self.photometric=photometric

    def update(self,gray: np.ndarray,decode_index: int) -> dict[str,Any]:
        gray=_gray(gray)
        if not isinstance(decode_index,int) or decode_index<0: raise ValueError('decode index')
        previous=self.decode_index
        if previous is not None and decode_index!=previous+1: raise ValueError('every decoded frame is required')
        self.events=[]; rejections={}
        if previous is not None:
            oldxy=self.xy.copy(); oldids=self.ids.copy(); neighbors=dict(zip(map(int,oldids),oldxy))
            move=photometric_step(self.images[previous],gray,oldxy,self.cv,self.profile,self.photometric)
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
