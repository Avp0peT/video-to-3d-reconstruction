"""Progressive sampled silhouette intersections; no measured surface claim."""
import math
import cv2
import numpy as np
from .local_depth_search import _inputs, _view, _rays
from .silhouette_intervals import _classify, REASONS

_BRACKETS = ('front_transition_low', 'front_transition_high', 'back_transition_low', 'back_transition_high')


def _intervals(admitted, background, inverse):
    n = len(admitted)
    reason = np.ones(n, np.uint8)
    front = np.zeros(n, np.float64); back = front.copy()
    brackets = {k: np.zeros(n, np.float64) for k in _BRACKETS}
    depths = 1/inverse
    for j, a in enumerate(admitted):
        ids = np.flatnonzero(a)
        if not len(ids): continue
        first, last = ids[0], ids[-1]
        if len(ids) != last-first+1: reason[j] = 2
        elif first == 0 or last == 127: reason[j] = 3
        elif not (background[j,first-1] and background[j,last+1]): reason[j] = 4
        else:
            reason[j] = 5
            front[j] = 1/((inverse[first-1]+inverse[first])/2)
            back[j] = 1/((inverse[last]+inverse[last+1])/2)
            for key,index in zip(_BRACKETS,(first-1,first,last,last+1)):
                brackets[key][j] = depths[index]
    return reason, front, back, brackets


def _summary():
    return dict(surviving_samples=0, any_surviving_rays=0, bounded=0, disjoint=0,
                unbounded=0, no_survivor=0, unknown_bracket=0)


def tighten_silhouette_intervals(reference, sources, roi, depth_bounds, check=lambda:None):
    """First three sources form baseline; remaining admission views intersect in order.

    The final source is diagnostic only. Source depth values are never admission
    evidence; zero depth denotes unknown native geometry, not a measured surface.
    """
    if not isinstance(sources,list) or not 5 <= len(sources) <= 15:
        raise ValueError('five to fifteen sources required, including final diagnostic source')
    ref, _, _ = _inputs(reference, sources[:2], roi, check)
    src = []
    for source in sources:
        check(); src.append(_view(source))
    ids = [s['view_id'] for s in src]
    if len(set(ids)) != len(ids) or ref['view_id'] in ids:
        raise ValueError('source IDs must be distinct and non-reference')
    for v in [ref]+src:
        validity = v.get('validity',np.ones(v['mask'].shape,bool))
        if not isinstance(validity,np.ndarray) or validity.dtype != bool or validity.shape != v['mask'].shape:
            raise ValueError('invalid validity mask')
        v['validity'] = validity
    if (not isinstance(depth_bounds,(list,tuple)) or len(depth_bounds)!=2 or
        any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in depth_bounds) or depth_bounds[0]>=depth_bounds[1]):
        raise ValueError('finite positive increasing axial depth bounds required')
    low,high=map(float,depth_bounds)
    if low < np.finfo(np.float32).tiny or high > np.finfo(np.float32).max:
        raise ValueError('bounds exceed supported float32 depth range')
    inverse=np.linspace(1/low,1/high,128); depths=1/inverse
    shape=roi.shape
    reason=np.zeros(shape,np.uint8); front=np.zeros(shape,np.float64); back=front.copy()
    baseline_reason=reason.copy(); baseline_front=front.copy(); baseline_back=front.copy()
    brackets={k:front.copy() for k in _BRACKETS}
    sampled_counts=np.zeros(shape,np.uint16)
    baseline_sample_counts=sampled_counts.copy()
    eroded=cv2.erode((ref['mask'] & ref['validity']).astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    yy,xx=np.indices(shape); sampled=roi & eroded & (xx%3==0) & (yy%3==0)
    y,x=np.nonzero(sampled)
    baseline_surviving_samples=np.zeros((len(x),128),bool)
    final_surviving_samples=np.zeros((len(x),128),bool)
    baseline_surviving_span=np.zeros(shape,np.float64)
    final_surviving_span=baseline_surviving_span.copy()
    sample_gains=0
    stages=[dict(admission_source_ids=ids[:n],**_summary()) for n in range(3,len(src))]
    loo={str(s['view_id']):dict(background_samples=0,background_rays=0,unknown_samples=0,unknown_rays=0) for s in src[:-1]}
    train_counts=dict(support=0,known_background=0,unknown=0,evaluated_samples=0)
    held=dict(train_counts)
    endpoints={name:dict(foreground=0,background=0,unknown=0,evaluated=0) for name in ('front','back')}
    for start in range(0,len(x),256):
        check(); bx,by=x[start:start+256],y[start:start+256]
        rays=np.repeat(_rays(ref,bx,by),128,axis=0); z=np.tile(depths,len(bx))
        fgs=[]; bgs=[]; unknowns=[]
        for source in src[:-1]:
            check(); fg,bg,unknown=_classify(ref,source,rays,z)
            fgs.append(fg.reshape(-1,128)); bgs.append(bg.reshape(-1,128)); unknowns.append(unknown.reshape(-1,128))
            for key,value in [('support',fg),('known_background',bg),('unknown',unknown)]: train_counts[key]+=int(value.sum())
            train_counts['evaluated_samples']+=len(z)
        admitted=np.ones((len(bx),128),bool); background=np.zeros_like(admitted)
        for index,(fg,bg) in enumerate(zip(fgs,bgs)):
            previous=admitted.copy()
            admitted &= fg; background |= bg
            sample_gains+=int((admitted & ~previous).sum())
            if index<2: continue
            r,f,b,br=_intervals(admitted,background,inverse)
            stage=stages[index-2]
            stage['surviving_samples']+=int(admitted.sum())
            stage['any_surviving_rays']+=int(admitted.any(axis=1).sum())
            for key,code in [('bounded',5),('disjoint',2),('unbounded',3),('no_survivor',1),('unknown_bracket',4)]: stage[key]+=int((r==code).sum())
            if index==2:
                baseline_reason[by,bx]=r; baseline_front[by,bx]=f; baseline_back[by,bx]=b
                baseline_sample_counts[by,bx]=admitted.sum(axis=1)
                baseline_surviving_samples[start:start+len(bx)]=admitted
                for j,a in enumerate(admitted):
                    surviving_ids=np.flatnonzero(a)
                    if len(surviving_ids)>1:
                        baseline_surviving_span[by[j],bx[j]]=depths[surviving_ids[-1]]-depths[surviving_ids[0]]
        reason[by,bx]=r; front[by,bx]=f; back[by,bx]=b
        for key in brackets: brackets[key][by,bx]=br[key]
        sampled_counts[by,bx]=admitted.sum(axis=1)
        final_surviving_samples[start:start+len(bx)]=admitted
        for j,a in enumerate(admitted):
            surviving_ids=np.flatnonzero(a)
            if len(surviving_ids)>1:
                final_surviving_span[by[j],bx[j]]=depths[surviving_ids[-1]]-depths[surviving_ids[0]]
        fg_total=np.sum(fgs,axis=0)
        for index,source in enumerate(src[:-1]):
            other_fg=(fg_total-fgs[index])==len(fgs)-1
            for label,values in [('background',bgs[index]),('unknown',unknowns[index])]:
                found=other_fg & values
                loo[str(source['view_id'])][label+'_samples']+=int(found.sum())
                loo[str(source['view_id'])][label+'_rays']+=int(found.any(axis=1).sum())
        # Final diagnostic source cannot influence any admission calculation.
        check(); fg,bg,unknown=_classify(ref,src[-1],rays,z)
        selected=np.repeat(r==5,128) & admitted.ravel()
        for key,value in [('support',fg),('known_background',bg),('unknown',unknown)]: held[key]+=int((selected & value).sum())
        held['evaluated_samples']+=int(selected.sum())
        select=r==5; endpoint_rays=_rays(ref,bx[select],by[select])
        for name,values in [('front',f),('back',b)]:
            check(); fg,bg,unknown=_classify(ref,src[-1],endpoint_rays,values[select])
            for key,value in [('foreground',fg),('background',bg),('unknown',unknown)]: endpoints[name][key]+=int(value.sum())
            endpoints[name]['evaluated']+=len(fg)
    bounded=reason==5
    final_outside_baseline=int((final_surviving_samples & ~baseline_surviving_samples).sum())
    if final_outside_baseline or sample_gains:
        raise AssertionError('progressive silhouette intersection gained samples')
    if not (np.isfinite(front[bounded]).all() and np.isfinite(back[bounded]).all() and (front[bounded]>0).all() and (back[bounded]>=front[bounded]).all()): raise ValueError('invalid computed silhouette bounds')
    known=bounded & (ref['depth']>0)
    counts=dict(roi_pixels=int(roi.sum()),sampled_pixels=int(sampled.sum()),bounded_pixels=int(bounded.sum()),
                whole_depth_evaluated=int(known.sum()),whole_depth_outside_midpoint_estimate=int((known & ((ref['depth']<front)|(ref['depth']>back))).sum()),
                whole_depth_outside_outer_sample_bracket=int((known & ((ref['depth']<brackets['front_transition_low'])|(ref['depth']>brackets['back_transition_high']))).sum()),
                reasons={name:int((reason==code).sum()) for code,name in REASONS.items()})
    whole_projection={}
    wy,wx=np.nonzero(roi & (ref['depth']>0))
    for source in src:
        diagnostic=dict(foreground=0,background=0,unknown=0,evaluated=0)
        for start in range(0,len(wx),4096):
            check(); dx,dy=wx[start:start+4096],wy[start:start+4096]
            fg,bg,unknown=_classify(ref,source,_rays(ref,dx,dy),ref['depth'][dy,dx])
            for key,value in [('foreground',fg),('background',bg),('unknown',unknown)]: diagnostic[key]+=int(value.sum())
            diagnostic['evaluated']+=len(fg)
        whole_projection[str(source['view_id'])]=diagnostic
    return dict(**brackets,front_depth=front,back_depth=back,interval_width=back-front,
                sampled_mask=sampled,bounded_mask=bounded,reason=reason,reason_legend=dict(REASONS),sample_counts=sampled_counts,
                counts=counts,counters=counts,train_sample_counts=train_counts,withheld_sample_counts=held,withheld_endpoint_counts=endpoints,
                baseline_reason=baseline_reason,baseline_front_depth=baseline_front,baseline_back_depth=baseline_back,
                baseline_sample_counts=baseline_sample_counts,baseline_summary=dict(stages[0]),stage_counts=stages,
                ray_x=x.copy(),ray_y=y.copy(),baseline_surviving_samples=baseline_surviving_samples,
                final_surviving_samples=final_surviving_samples,
                baseline_surviving_span=baseline_surviving_span,final_surviving_span=final_surviving_span,
                monotonicity=dict(no_sample_gains=sample_gains==0,sample_gains_across_stages=sample_gains,
                                  final_samples_outside_baseline=final_outside_baseline),
                leave_one_out=loo,whole_depth_source_classifications=whole_projection,
                admission_source_ids=ids[:-1],withheld_source_id=ids[-1],depth_bounds=[low,high],depth_samples=depths,grid_step=3,erosion_pixels=1,
                geometry_accepted=False,surface_accepted=False,status='SILHOUETTE_BOUND_NOT_MEASURED_SURFACE',
                assumptions=['rigid scene','calibrated cameras','correct silhouettes'],
                depth_semantics='Zero native depth means unknown geometry; source depth is never used for silhouette admission. Historical depth projections are diagnostic, not ground truth.',
                discretization='128 inverse-depth samples can miss thinner intervals or holes; midpoint bounds and transition sample brackets are estimates, not a certified continuous visual hull. Historical depth outside a midpoint estimate is not a silhouette conflict.',
                semantics='Outer-shape bound, not hidden-cavity recovery; withheld view is not independent world truth. Historical depth and frozen bounds may share reconstruction evidence.')
