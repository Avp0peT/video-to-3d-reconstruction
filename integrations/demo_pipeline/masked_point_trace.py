"""Adjacent single-point checks using supplied current masks, never warped masks.

Mask membership and image consistency do not establish physical identity.
"""
from typing import Any
import numpy as np
from .balanced_object_tracks import Profile, _gray, points_in_mask
from .photometric_tracks import PhotometricProfile, photometric_step
from .single_point_trace import _FORWARD_PROPOSAL_REASONS, _finite_scalar


class _InitialFlow:
    """Local CV proxy; only the two LK calls receive initial flow."""
    def __init__(self, cv, previous, predicted):
        self.cv, self.previous, self.predicted = cv, previous, predicted
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.cv, name)

    def calcOpticalFlowPyrLK(self, a, b, p, unused, **opts):
        initial = self.predicted if self.calls == 0 else self.previous
        self.calls += 1
        opts['flags'] = opts.get('flags', 0) | self.cv.OPTFLOW_USE_INITIAL_FLOW
        return self.cv.calcOpticalFlowPyrLK(a, b, p, initial.reshape(1, 1, 2).copy(), **opts)


def _mask_metrics(mask, point):
    if mask is None:
        return dict(status='UNKNOWN_NO_MATCH', center_inside=None, patch15_foreground_fraction=None)
    if point is None or not np.isfinite(point).all():
        return dict(status='SUPPLIED_CURRENT_MASK', center_inside=False, patch15_foreground_fraction=None)
    inside = bool(points_in_mask(mask, np.asarray(point)[None])[0])
    x, y = np.floor(np.asarray(point) + .5).astype(int)
    # Fixed 15x15 nearest-center footprint; outside image contributes background.
    h, w = mask.shape
    fraction = float(np.count_nonzero(mask[max(0,y-7):min(h,y+8), max(0,x-7):min(w,x+8)]) / 225) if -7<=x<w+7 and -7<=y<h+7 else 0.
    return dict(status='SUPPLIED_CURRENT_MASK', center_inside=inside, patch15_foreground_fraction=fraction)


def trace(images: list[np.ndarray], masks: list[np.ndarray | None],
          decode_indices: list[int], seed_xy: Any, cv: Any, *, predict_motion: bool,
          times: list[float] | None = None) -> dict[str, Any]:
    """B is unchanged photometric_step; C adds at most one velocity initial guess.

    Optional times are strictly increasing real timestamps. Without them velocity
    uses decode indices. Certification never resumes after its first failure.
    """
    if not isinstance(images,list) or not images or not isinstance(masks,list) or not isinstance(decode_indices,list) or not len(images)==len(masks)==len(decode_indices):
        raise ValueError('aligned nonempty image/mask/index lists required')
    if any(type(i) is not int or i<0 for i in decode_indices) or any(b!=a+1 for a,b in zip(decode_indices,decode_indices[1:])):
        raise ValueError('every consecutive decoded image required')
    if type(predict_motion) is not bool: raise ValueError('predict_motion must be bool')
    grays=[_gray(im) for im in images]; h,w=grays[0].shape
    if any(g.shape!=(h,w) for g in grays): raise ValueError('image shape changed')
    for mask in masks:
        if mask is not None and (not isinstance(mask,np.ndarray) or mask.dtype!=np.uint8 or mask.shape!=(h,w)):
            raise ValueError('mask must be aligned uint8 HxW or None')
        if mask is not None and not np.isin(mask,[0,255]).all():
            raise ValueError('mask labels must be 0/255')
    ts=np.asarray(decode_indices if times is None else times,dtype=float)
    if ts.shape!=(len(images),) or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0): raise ValueError('strictly increasing aligned finite times required')
    seed=np.asarray(seed_xy,dtype=np.float32)
    if seed.shape!=(2,) or not np.isfinite(seed).all(): raise ValueError('finite single seed xy required')
    profile=Profile(); photo=PhotometricProfile()
    def row(i):
        return dict(decode_index=decode_indices[i],time=float(ts[i]),candidate_xy=None,raw_proposal_xy=None,accepted_xy=None,
                    continuous_accepted=False,step_accepted=False,raw_step_accepted=None,reason='NOT_TRACKED',candidate_status='NOT_TRACKED',
                    mask=_mask_metrics(masks[i],None),previous_mask=None,motion=dict(attempted=False,reason='disabled' if not predict_motion else 'no_continuous_velocity'),
                    candidates=[],selected_candidate_source=None,prediction_used=False,fb_error=None,zncc=None,gain=None,bias=None,photometric_residual=None)
    frames=[]; first=row(0); first['mask']=_mask_metrics(masks[0],seed)
    interior=7<=seed[0]<=w-8 and 7<=seed[1]<=h-8
    continuous=bool(interior and first['mask']['center_inside'])
    first.update(reason='selected_saved_observation' if continuous else 'seed_mask_unknown_or_outside',step_accepted=None,
                 continuous_accepted=continuous,candidate_xy=seed.tolist() if continuous else None,
                 accepted_xy=seed.tolist() if continuous else None,candidate_status='SELECTED_SAVED_OBSERVATION' if continuous else 'REJECTED')
    frames.append(first); position=seed.copy() if continuous else None
    first_failure=None if continuous else dict(decode_index=decode_indices[0],reason=first['reason'],shadow_available=False,candidate_xy=None)
    for i in range(1,len(grays)):
        r=row(i)
        if position is None: r['reason']='NOT_TRACKED_AFTER_FORWARD_FAILURE'
        elif masks[i] is None: r['reason']='current_mask_unknown'
        elif masks[i-1] is None or not _mask_metrics(masks[i-1],position)['center_inside']: r['reason']='previous_mask_unknown_or_outside'
        else:
            r['previous_mask']=_mask_metrics(masks[i-1],position)
            guesses=[('baseline',None)]
            if predict_motion and continuous and i>=2 and frames[i-2]['accepted_xy'] is not None:
                delta=(position-np.asarray(frames[i-2]['accepted_xy']))*((ts[i]-ts[i-1])/(ts[i-1]-ts[i-2]))
                length=float(np.linalg.norm(delta))
                r['motion'].update(displacement_px=length,reason='over_96px' if length>96 else 'constant_velocity')
                if length<=96:
                    predicted=(position+delta).astype(np.float32); guesses.append(('motion',predicted))
                    r['motion'].update(attempted=True,predicted_xy=predicted.tolist(),backward_initial_xy=position.tolist())
            elif predict_motion and not continuous: r['motion']['reason']='certification_failed_no_shadow_velocity'
            for name,guess in guesses:
                proxy=cv if guess is None else _InitialFlow(cv,position,guess)
                result=photometric_step(grays[i-1],grays[i],position[None],proxy,profile,photo)
                q=np.asarray(result['xy'],dtype=np.float32)
                if q.shape!=(1,2): raise ValueError('single-row native proposal required')
                q=q[0]; reason=str(result['reasons'][0]); raw=bool(result['accepted'][0])
                valid=bool(reason in _FORWARD_PROPOSAL_REASONS and np.isfinite(q).all() and 0<=q[0]<w and 0<=q[1]<h)
                metrics=_mask_metrics(masks[i],q if valid else None)
                reported_reason=('invalid_native_proposal' if not valid and reason in _FORWARD_PROPOSAL_REASONS
                                 else reason if not valid or metrics['center_inside'] else 'current_mask_outside')
                c=dict(source=name,raw_proposal_xy=q.tolist() if np.isfinite(q).all() else None,
                       candidate_xy=q.tolist() if valid and metrics['center_inside'] else None,
                       reason=reported_reason,raw_step_accepted=raw,
                       qualified=bool(valid and metrics['center_inside'] and raw and reason=='accepted'),mask=metrics,
                       forward_initial_xy=None if guess is None else guess.tolist(),backward_initial_xy=None if guess is None else position.tolist())
                for key in ('fb_error','zncc','gain','bias','photometric_residual'): c[key]=_finite_scalar(result[key][0])
                r['candidates'].append(c)
            good=[c for c in r['candidates'] if c['qualified']]
            ambiguous=len(good)==2 and np.linalg.norm(np.asarray(good[0]['candidate_xy'])-good[1]['candidate_xy'])>2
            chosen=good[0] if good else r['candidates'][0]
            for key in ('raw_proposal_xy','raw_step_accepted','mask','fb_error','zncc','gain','bias','photometric_residual'): r[key]=chosen[key]
            r.update(reason='AMBIGUOUS' if ambiguous else chosen['reason'],step_accepted=bool(good and not ambiguous),
                     candidate_xy=None if ambiguous else chosen['candidate_xy'],
                     selected_candidate_source=None if ambiguous else chosen['source'],
                     prediction_used=bool(not ambiguous and chosen['source']=='motion'))
        if continuous and not r['step_accepted']:
            continuous=False
            first_failure=dict(decode_index=decode_indices[i],reason=r['reason'],shadow_available=r['candidate_xy'] is not None,candidate_xy=r['candidate_xy'])
        r['continuous_accepted']=continuous
        r['accepted_xy']=r['candidate_xy'] if continuous else None
        r['candidate_status']='CONTINUOUS_CHECKS_PASSED' if continuous else ('UNVERIFIED_SHADOW' if r['candidate_xy'] is not None else 'REJECTED')
        position=None if r['candidate_xy'] is None else np.asarray(r['candidate_xy'],dtype=np.float32)
        frames.append(r)
    return dict(schema='masked-single-point-trace-v1',coordinate_convention='opencv_pixel_center_0',mask_rounding='floor(x+0.5); rounded indices must be within image',
                seed_origin='selected_saved_observation',seed_xy=seed.tolist(),frames=frames,first_failure=first_failure,
                predict_motion=predict_motion,time_basis='decode_indices' if times is None else 'supplied_times',
                physical_identity_verified=False,geometry_accepted=False,reseed_calls=0,recovery_calls=0,interpolation_calls=0,sfm_calls=0,
                profile=profile.__dict__,photometric_profile=photo.__dict__,patch_fraction_is_diagnostic_only=True)
