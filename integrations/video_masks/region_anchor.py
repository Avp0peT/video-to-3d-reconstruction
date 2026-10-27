"""Opt-in direct endpoint patch audit; no mutation or correspondence truth claim.

Local LK may be unobservable over large endpoint motion and repeated texture can
remain ambiguous even after admission. UNKNOWN never establishes a wrong match.
"""
from copy import deepcopy
import numpy as np
from .region_tracks import step, PROFILE as TRACK_PROFILE

PROFILE = dict(id='glove-region-anchor-v1', algorithm='bidirectional-direct-endpoint-patch',
               endpoint_error_max_pixels=2.,
               base_patch_profile=deepcopy(TRACK_PROFILE))


def _validate(gray, mask):
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError('uint8 grayscale required')
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != gray.shape or not np.isin(mask, [0, 255]).all():
        raise ValueError('same-grid uint8 mask labels 0/255 required')


def _observations(rows):
    result = {}
    for row in rows:
        ident = row['track_id']
        xy = np.asarray(row['xy'], dtype=np.float32)
        if type(ident) is not int or ident < 0 or ident in result or xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('unique integer IDs and finite xy required')
        result[ident] = xy.copy()
    return result


def audit_pair(source_gray, target_gray, source_mask, target_mask,
               source_observations, target_observations, cv):
    """Measure only shared IDs in the supplied endpoint observations.

    Caller explicitly opts in by invoking this function. No files, recovery,
    interpolation, ID reassignment, or geometry execution occur here. Forward
    and backward audits each use raw full-patch ZNCC plus internal LK FB checks.
    Endpoint error compares that direct measured position to the serial track's
    supplied endpoint. Failure remains UNKNOWN, never a proved mismatch.
    """
    _validate(source_gray, source_mask); _validate(target_gray, target_mask)
    if source_gray.shape != target_gray.shape:
        raise ValueError('same endpoint grids required')
    a = _observations(source_observations); b = _observations(target_observations)
    ids = sorted(a.keys() & b.keys())
    result = dict(schema='glove-region-anchor-v1', profile=deepcopy(PROFILE),
                  identity_verified=False, geometry_accepted=False,
                  coordinate_convention='opencv_integer_centers', candidates=[],
                  source_only_ids=sorted(a.keys()-b.keys()), target_only_ids=sorted(b.keys()-a.keys()),
                  limitations='Local motion may be unobservable; repeated texture and gradual drift remain possible. UNKNOWN does not prove a mismatch.')
    if not ids:
        result['metrics'] = dict(shared=0, admitted=0, unknown=0)
        return result
    cv.ocl.setUseOpenCL(False)
    cv.setNumThreads(TRACK_PROFILE['cpu_threads'])
    forward, freasons, fscores = step(source_gray, target_gray, source_mask, target_mask, [a[i] for i in ids], cv)
    backward, breasons, bscores = step(target_gray, source_gray, target_mask, source_mask, [b[i] for i in ids], cv)
    def measured(q, reason, score, supplied):
        finite = bool(np.isfinite(q).all())
        return dict(status='MEASURED_APPROVED' if reason == 'accepted' else 'UNKNOWN',
                    reason=reason, measured_xy=q.astype(float).tolist() if reason == 'accepted' and finite else None,
                    patch_zncc=float(score) if score is not None and np.isfinite(score) else None,
                    endpoint_error_pixels=float(np.linalg.norm(q-supplied)) if reason == 'accepted' and finite else None,
                    internal_forward_backward_gate_pixels=TRACK_PROFILE['fb_max'])
    for n, ident in enumerate(ids):
        f = measured(forward[n], freasons[n], fscores[n], b[ident])
        back = measured(backward[n], breasons[n], bscores[n], a[ident])
        reasons = []
        for direction, measurement in (('forward', f), ('backward', back)):
            if measurement['status'] != 'MEASURED_APPROVED':
                reasons.append(direction+':'+measurement['reason'])
            elif measurement['endpoint_error_pixels'] > PROFILE['endpoint_error_max_pixels']:
                reasons.append(direction+':serial_endpoint_disagreement')
        result['candidates'].append(dict(track_id=ident, source_xy=a[ident].astype(float).tolist(),
                                         target_xy=b[ident].astype(float).tolist(),
                                         forward=f, backward=back,
                                         status='unknown' if reasons else 'admitted', reasons=reasons))
    admitted = sum(r['status'] == 'admitted' for r in result['candidates'])
    result['metrics'] = dict(shared=len(ids), admitted=admitted, unknown=len(ids)-admitted)
    return result
