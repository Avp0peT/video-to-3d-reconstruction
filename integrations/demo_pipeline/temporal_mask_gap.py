"""Bounded CPU temporal support; never SAM output or point identity evidence."""
from __future__ import annotations

import numpy as np


SCHEMA = 'temporal-mask-gap-support-v1'
PROFILE = dict(max_long_side=512, max_gap=6, cycle_max_work_px=1.5,
               iou_min=.65, valid_foreground_min=.80,
               area_ratio_min=.7, area_ratio_max=1.4)


def _flow(source, target, cv):
    """Injection seam for tests; exactly one Farneback call per ordered edge."""
    return cv.calcOpticalFlowFarneback(source, target, None, .5, 5, 25, 5, 7, 1.5, 0)


def _iou(a, b, domain):
    union = np.count_nonzero((a | b) & domain)
    return float(np.count_nonzero(a & b & domain) / union) if union else 0.


def _retained(prediction, valid):
    count = np.count_nonzero(prediction)
    return float(np.count_nonzero(prediction & valid) / count) if count else 0.


def _warp(mask, valid, forward, backward, cv):
    """Inverse sample: backward is target->source, forward checks its cycle."""
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    mx, my = xx + backward[..., 0], yy + backward[..., 1]
    finite = np.isfinite(mx) & np.isfinite(my)
    bounds = finite & (mx >= 0) & (mx <= w-1) & (my >= 0) & (my <= h-1)
    # Nonfinite coordinates cannot be passed to native remap as evidence.
    sx, sy = np.where(finite, mx, -1).astype(np.float32), np.where(finite, my, -1).astype(np.float32)
    kwargs = dict(borderMode=cv.BORDER_CONSTANT, borderValue=0)
    prediction = cv.remap(mask.astype(np.uint8), sx, sy, cv.INTER_NEAREST, **kwargs) > 0
    inherited = cv.remap(valid.astype(np.uint8), sx, sy, cv.INTER_NEAREST, **kwargs) > 0
    sampled = cv.remap(forward, sx, sy, cv.INTER_LINEAR, **kwargs)
    cycle = np.linalg.norm(backward + sampled, axis=2)
    rx, ry = mx + sampled[..., 0], my + sampled[..., 1]
    good = (bounds & inherited & np.isfinite(sampled).all(axis=2) & np.isfinite(cycle)
            & (cycle <= PROFILE['cycle_max_work_px'])
            & (rx >= 0) & (rx <= w-1) & (ry >= 0) & (ry <= h-1))
    return prediction, good, dict(valid_fraction=float(good.mean()),
        inverse_map_inbounds_fraction=float(bounds.mean()),
        cycle_consistent_fraction=float((bounds & np.isfinite(cycle) & (cycle <= 1.5)).mean()),
        valid_foreground_fraction=_retained(prediction, good))


def _validate(grays, first_mask, last_mask):
    if type(grays) is not list or not 3 <= len(grays) <= 8:
        raise ValueError('grays must include two anchors and one to six gap frames')
    shape = None
    for gray in grays:
        if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2 or not gray.size:
            raise ValueError('nonempty uint8 grayscale arrays required')
        if shape is not None and gray.shape != shape:
            raise ValueError('same-shape grays required')
        shape = gray.shape
    for mask in (first_mask, last_mask):
        if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != shape
                or not np.isin(mask, [0, 255]).all() or not mask.any()):
            raise ValueError('two nonempty actual binary uint8 0/255 anchor masks required')
    return shape


def propagate_gap(grays, first_mask, last_mask, cv):
    """Return original-size boolean evidence arrays and JSON-safe diagnostics.

    Caller must bind first_mask/last_mask to actual original SAM anchors. This
    function validates their format, not provenance. Thresholds are engineering
    presets and have not been calibrated. Inputs are never modified.
    """
    h, w = _validate(grays, first_mask, last_mask)
    scale = min(1., PROFILE['max_long_side'] / max(h, w))
    size = (max(1, int(round(w*scale))), max(1, int(round(h*scale))))
    images = [cv.resize(a, size, interpolation=cv.INTER_AREA) for a in grays]
    anchors = [cv.resize(m, size, interpolation=cv.INTER_NEAREST) > 0 for m in (first_mask, last_mask)]
    if not all(m.any() for m in anchors):
        raise ValueError('anchor foreground lost at bounded working resolution')
    n = len(images)
    cache = []
    for i in range(n-1):
        f = np.asarray(_flow(images[i], images[i+1], cv))
        b = np.asarray(_flow(images[i+1], images[i], cv))
        if f.shape != (*images[0].shape, 2) or b.shape != f.shape or f.dtype != np.float32 or b.dtype != np.float32:
            raise ValueError('flow helper must return working-size float32 HxWx2')
        cache.append((f, b))
    predictions, valids, direction_rows = {}, {}, {}
    endpoint_pass = True
    for name, order, start, endpoint in (
            ('forward', list(range(n)), anchors[0], anchors[1]),
            ('backward', list(range(n-1, -1, -1)), anchors[1], anchors[0])):
        pred = [None]*n
        valid = [None]*n
        pred[order[0]], valid[order[0]] = start.copy(), np.ones(start.shape, bool)
        steps = []
        for source, target in zip(order, order[1:]):
            f, b = cache[min(source, target)]
            if source > target:
                f, b = b, f
            pred[target], valid[target], row = _warp(pred[source], valid[source], f, b, cv)
            row.update(source_index=source, target_index=target, flow_calls=0,
                       cached_ordered_flows_used=2)
            steps.append(row)
        end = order[-1]
        overlap_valid = _iou(pred[end], endpoint, valid[end])
        # Full-domain overlap conservatively penalizes missing support; it does
        # not certify invalid pixels as background or segmentation truth.
        overlap = _iou(pred[end] & valid[end], endpoint, np.ones(endpoint.shape, bool))
        retained = _retained(pred[end], valid[end])
        actual_retained = _retained(endpoint, valid[end])
        reasons = []
        if overlap < PROFILE['iou_min']:
            reasons.append('ENDPOINT_IOU_LOW')
        if retained < PROFILE['valid_foreground_min']:
            reasons.append('ENDPOINT_VALID_FOREGROUND_LOW')
        if actual_retained < PROFILE['valid_foreground_min']:
            reasons.append('ACTUAL_ENDPOINT_FOREGROUND_VALID_LOW')
        endpoint_pass &= not reasons
        predictions[name], valids[name] = pred, valid
        direction_rows[name] = dict(steps=steps, endpoint_iou=overlap,
            endpoint_iou_valid=overlap_valid, endpoint_iou_full=overlap,
            endpoint_valid_foreground_fraction=retained, endpoint_reasons=reasons,
            actual_endpoint_foreground_valid_fraction=actual_retained,
            endpoint_pass=not reasons)
    supports, uncertainties, gap_rows = [], [], []
    for i in range(1, n-1):
        a, b = predictions['forward'][i], predictions['backward'][i]
        va, vb = valids['forward'][i], valids['backward'][i]
        domain = va & vb
        overlap = _iou(a, b, domain)
        counts = [int(np.count_nonzero(p)) for p in (a, b)]
        ratio = float(counts[0]/counts[1]) if counts[1] else None
        reverse_ratio = float(counts[1]/counts[0]) if counts[0] else None
        fractions = [_retained(a, va), _retained(b, vb)]
        reasons = []
        if not endpoint_pass:
            reasons.append('ENDPOINT_CHECK_FAILED')
        if overlap < PROFILE['iou_min']:
            reasons.append('BIDIRECTIONAL_IOU_LOW')
        if ratio is None or reverse_ratio is None or any(not .7 <= r <= 1.4 for r in (ratio, reverse_ratio)):
            reasons.append('DIRECTION_AREA_RATIO_OUT_OF_RANGE')
        if any(f < PROFILE['valid_foreground_min'] for f in fractions):
            reasons.append('VALID_FOREGROUND_LOW')
        support = (a & b & domain) if not reasons else np.zeros(a.shape, bool)
        supports.append(support)
        uncertainties.append((a | b) & ~support)
        gap_rows.append(dict(frame_index=i, status='ACCEPTED_TEMPORAL_SUPPORT' if not reasons else 'REJECTED',
            reasons=reasons, bidirectional_iou=overlap, area_ratio_forward_backward=ratio,
            area_ratio_backward_forward=reverse_ratio, valid_foreground_fractions=fractions,
            support_pixels_work=int(support.sum()), uncertainty_pixels_work=int(uncertainties[-1].sum())))
    def restore(array):
        return cv.resize(array.astype(np.uint8), (w, h), interpolation=cv.INTER_NEAREST).astype(bool)
    return dict(schema=SCHEMA, semantics='TEMPORAL_SUPPORT_ONLY', actual_SAM=False,
        tracking_identity=False,
        predictions={k: [restore(a) for a in rows] for k, rows in predictions.items()},
        valid={k: [restore(a) for a in rows] for k, rows in valids.items()},
        support=[restore(a) for a in supports], uncertainty=[restore(a) for a in uncertainties],
        diagnostics=dict(schema=SCHEMA, semantics='TEMPORAL_SUPPORT_ONLY', actual_SAM=False,
            tracking_identity=False, thresholds_calibrated=False, profile=dict(PROFILE),
            original_shape=[h, w], work_shape=list(images[0].shape), gap_frame_indices=list(range(1,n-1)),
            flow_calls=2*(n-1), flow_edges=[dict(source_index=i,target_index=i+1,
                forward_calls=1,backward_calls=1) for i in range(n-1)],
            directions=direction_rows, gaps=gap_rows))
