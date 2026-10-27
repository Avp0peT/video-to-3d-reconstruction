"""Array-only local search proposals; never promote a rejected full mask."""
from __future__ import annotations

import math
import numpy as np


def _binary(value, shape=None):
    if (not isinstance(value, np.ndarray) or value.ndim != 2 or not value.size
            or (shape is not None and value.shape != shape)):
        raise ValueError('matching nonempty two-dimensional arrays required')
    if value.dtype == bool:
        return value.copy()
    if value.dtype != np.uint8 or not np.isin(value, [0,255]).all():
        raise ValueError('binary bool or uint8 0/255 arrays required')
    return value > 0


def derive(forward, backward, fvalid, bvalid, *, leaveout_passed, metrics=None):
    """Derive a search-only intersection without changing full-mask acceptance.

    Optional metrics may contain the three recomputed metric fields and an
    opaque full_shape_status copied to the result. Any supplied metric must
    agree with these arrays within absolute 1e-9. A leaveout result must be a
    literal boolean; binding its evidence is the caller's responsibility.
    """
    f = _binary(forward)
    b, fv, bv = [_binary(a, f.shape) for a in (backward,fvalid,bvalid)]
    if type(leaveout_passed) is not bool:
        raise ValueError('leaveout_passed must be a boolean')
    if metrics is None:
        metrics = {}
    if type(metrics) is not dict:
        raise ValueError('metrics must be a dict or None')
    domain = fv & bv
    union = int(np.count_nonzero((f | b) & domain))
    overlap = float(np.count_nonzero(f & b & domain)/union) if union else 0.
    nf, nb = int(f.sum()), int(b.sum())
    measured = dict(bidirectional_iou=overlap,
        area_ratio_forward_backward=float(nf/nb) if nb else None,
        area_ratio_backward_forward=float(nb/nf) if nf else None)
    for key, actual in measured.items():
        if key not in metrics:
            continue
        supplied = metrics[key]
        if actual is None:
            if supplied is not None:
                raise ValueError('metric differs from arrays: '+key)
        elif (isinstance(supplied, (bool,np.bool_)) or not isinstance(supplied,(int,float,np.integer,np.floating))
                or not math.isfinite(float(supplied)) or abs(float(supplied)-actual)>1e-9):
            raise ValueError('metric differs from arrays: '+key)
    reasons = []
    if not leaveout_passed:
        reasons.append('LEAVEOUT_CONTROL_NOT_PASSED')
    if overlap < .65:
        reasons.append('BIDIRECTIONAL_IOU_LOW')
    if any(measured[k] is None or not .7 <= measured[k] <= 1.4
           for k in ('area_ratio_forward_backward','area_ratio_backward_forward')):
        reasons.append('DIRECTION_AREA_RATIO_OUT_OF_RANGE')
    proposal = f & b & domain
    if not proposal.any():
        reasons.append('EMPTY_VALID_INTERSECTION')
    search = proposal if not reasons else np.zeros_like(f)
    return dict(search=search, uncertain=(f | b) & ~search,
        allowed=not reasons, reasons=reasons, full_mask_accepted=False,
        usage='SEARCH_PROPOSAL_ONLY', search_only=True, sam_mask=False,
        identity_verified=False, full_shape_status=metrics.get('full_shape_status'),
        metrics=measured)
