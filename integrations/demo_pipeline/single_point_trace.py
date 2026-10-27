"""Single saved observation followed through adjacent decoded raw images.

Passing image-consistency checks does not establish physical point identity.
After the first failed step, any surviving LK proposal is diagnostic shadow
only. No re-seeding, gap recovery, external matches, or missing observations
are introduced here.
"""
from typing import Any
import numpy as np
from .balanced_object_tracks import Profile, _gray
from .photometric_tracks import PhotometricProfile, photometric_step


# These stages establish that a valid forward LK proposal existed. In contrast,
# lk_forward/invalid_coordinates can contain photometric_step's unchanged input
# fallback, which must never be emitted as a new observation or shadow point.
_FORWARD_PROPOSAL_REASONS=frozenset(('accepted','lk_backward','forward_backward',
    'photometric_saturation','photometric_texture','photometric_gain',
    'photometric_bias','photometric_residual','photometric_zncc','photometric_nonfinite'))


def _finite_scalar(value: Any) -> float | None:
    number=float(value)
    return number if np.isfinite(number) else None


def _anchor_score(anchor: np.ndarray,gray: np.ndarray,point: np.ndarray,cv: Any,patch: int) -> float | None:
    radius=patch//2; h,w=gray.shape
    if not (radius<=point[0]<=w-1-radius and radius<=point[1]<=h-1-radius): return None
    candidate=cv.getRectSubPix(gray,(patch,patch),tuple(map(float,point))).astype(np.float64)
    centered=candidate-candidate.mean(); denominator=float(np.linalg.norm(anchor)*np.linalg.norm(centered))
    if not np.isfinite(centered).all() or not np.isfinite(denominator) or denominator<=1e-12: return None
    return _finite_scalar(np.sum(anchor*centered)/denominator)


def trace(images: list[np.ndarray],decode_indices: list[int],seed_xy: Any,cv: Any) -> dict[str,Any]:
    """Trace one selected saved observation, without revival of certification.

    ``candidate_xy`` after a failure is UNVERIFIED_SHADOW, even if later adjacent
    image checks pass. ``accepted_xy`` remains null permanently after that first
    failure. An unusable forward proposal terminates even the shadow chain.
    Diagnostic metrics are JSON-safe (nonfinite native values become null).
    """
    if not isinstance(images,list) or not images or not isinstance(decode_indices,list) or len(images)!=len(decode_indices):
        raise ValueError('aligned nonempty image/index lists required')
    if any(type(index) is not int or index<0 for index in decode_indices): raise ValueError('integer nonnegative decode indices required')
    if any(b!=a+1 for a,b in zip(decode_indices,decode_indices[1:])): raise ValueError('every consecutive decoded image required')
    grays=[_gray(image) for image in images]
    if any(gray.shape!=grays[0].shape for gray in grays): raise ValueError('image shape changed')
    seed=np.asarray(seed_xy,dtype=np.float32)
    if seed.shape!=(2,) or not np.isfinite(seed).all(): raise ValueError('finite single seed xy required')
    profile=Profile(); photometric=PhotometricProfile(); radius=profile.patch//2; h,w=grays[0].shape
    if not (radius<=seed[0]<=w-1-radius and radius<=seed[1]<=h-1-radius): raise ValueError('seed patch outside first real image')
    anchor=cv.getRectSubPix(grays[0],(profile.patch,profile.patch),tuple(map(float,seed))).astype(np.float64)
    anchor-=anchor.mean()
    first=dict(decode_index=decode_indices[0],candidate_xy=seed.tolist(),accepted_xy=seed.tolist(),
        continuous_accepted=True,step_accepted=None,raw_step_accepted=None,reason='selected_saved_observation',
        candidate_status='SELECTED_SAVED_OBSERVATION',fb_error=None,zncc=None,gain=None,bias=None,
        photometric_residual=None,anchor_zncc=_anchor_score(anchor,grays[0],seed,cv,profile.patch),
        displacement_px=0.,step_displacement_px=None)
    frames=[first]; first_failure=None; continuous=True; position=seed.copy()
    for index in range(1,len(grays)):
        row=dict(decode_index=decode_indices[index],candidate_xy=None,accepted_xy=None,
            continuous_accepted=False,step_accepted=False,raw_step_accepted=None,reason='NOT_TRACKED_AFTER_FORWARD_FAILURE',
            candidate_status='NOT_TRACKED',fb_error=None,zncc=None,gain=None,bias=None,
            photometric_residual=None,anchor_zncc=None,displacement_px=None,step_displacement_px=None)
        if position is None:
            frames.append(row); continue
        result=photometric_step(grays[index-1],grays[index],position[None,:],cv,profile,photometric)
        proposal=np.asarray(result['xy'],dtype=np.float32)
        if proposal.shape!=(1,2): raise ValueError('single-row native proposal required')
        proposal=proposal[0]; reason=str(result['reasons'][0]); raw_accepted=bool(result['accepted'][0])
        valid=(reason in _FORWARD_PROPOSAL_REASONS and np.isfinite(proposal).all()
               and 0<=proposal[0]<w and 0<=proposal[1]<h)
        accepted=bool(raw_accepted and valid and reason=='accepted')
        for key in ('fb_error','zncc','gain','bias','photometric_residual'):
            row[key]=_finite_scalar(result[key][0])
        row.update(reason=reason,raw_step_accepted=raw_accepted,step_accepted=accepted)
        if not valid and reason in _FORWARD_PROPOSAL_REASONS:
            row['reason']='invalid_native_proposal'
        if continuous and not accepted:
            continuous=False
            first_failure=dict(decode_index=decode_indices[index],reason=row['reason'],
                               shadow_available=bool(valid),candidate_xy=proposal.tolist() if valid else None)
        row['continuous_accepted']=continuous
        if valid:
            row.update(candidate_xy=proposal.tolist(),accepted_xy=proposal.tolist() if continuous else None,
                candidate_status='CONTINUOUS_CHECKS_PASSED' if continuous else 'UNVERIFIED_SHADOW',
                anchor_zncc=_anchor_score(anchor,grays[index],proposal,cv,profile.patch),
                displacement_px=float(np.linalg.norm(proposal.astype(float)-seed)),
                step_displacement_px=float(np.linalg.norm(proposal.astype(float)-position)))
            position=proposal.copy()
        else:
            position=None
        frames.append(row)
    return dict(schema='single-point-trace-v1',coordinate_convention='opencv_pixel_center_0',
        seed_origin='selected_saved_observation',seed_xy=seed.tolist(),frames=frames,first_failure=first_failure,
        physical_identity_verified=False,geometry_accepted=False,reseed_calls=0,recovery_calls=0,
        interpolation_calls=0,sfm_calls=0,profile=profile.__dict__,photometric_profile=photometric.__dict__,
        meaning='Only uninterrupted adjacent image checks authorize accepted_xy. Shadow proposals are unverified; no physical identity truth is implied.')
