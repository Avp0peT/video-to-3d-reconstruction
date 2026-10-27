"""Measured reference-exterior image tracks; never background truth or 3D."""
from __future__ import annotations

import numpy as np

PROFILE = dict(max_corners=160, seed_distance_min_px=8, seed_distance_max_px=48,
               seed_patch_margin_px=5, track_patch_margin_px=4, quality_level=.01,
               min_distance=8, block_size=7, window_size=21, max_level=3,
               iterations=30, epsilon=.01, fb_max_px=1., lk_error_max=25.,
               patch_size=9, patch_ncc_min=.75, patch_std_min=5.,
               endpoint_distance_min_px=8, max_frames=7, max_long_edge=1024)


def track_reference_exterior(images_gray, before_mask, after_mask, frame_indices, cv):
    """Track every seed until first failed transition, without reseeding.

    Coordinates are OpenCV pixel centres. Endpoint selection is retrospective
    and survivor-conditioned; physical identity remains UNVERIFIED.
    """
    if not isinstance(images_gray, (list, tuple)) or not 3 <= len(images_gray) <= 7:
        raise ValueError('require 3..7 grayscale frames')
    ims = [np.asarray(x) for x in images_gray]
    shape = ims[0].shape
    if len(shape) != 2 or min(shape) < 11 or max(shape) > 1024 or any(x.dtype != np.uint8 or x.shape != shape for x in ims):
        raise ValueError('images must be matching uint8 HxW, long edge <=1024')
    if len(frame_indices) != len(ims) or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 0 for x in frame_indices):
        raise ValueError('frame indices must be integers')
    ids = [int(x) for x in frame_indices]
    if any(b != a + 1 for a, b in zip(ids, ids[1:])):
        raise ValueError('frame indices must be consecutive and unique')
    masks = [np.asarray(before_mask), np.asarray(after_mask)]
    if any(m.dtype != np.uint8 or m.shape != shape or not np.isin(m, [0, 255]).all() or not m.any() for m in masks):
        raise ValueError('original masks must be nonempty matching binary uint8')
    distances = [cv.distanceTransform((m == 0).astype(np.uint8), cv.DIST_L2, cv.DIST_MASK_PRECISE) for m in masks]
    h, w = shape
    band = ((distances[0] >= 8) & (distances[0] <= 48)).astype(np.uint8)
    band[:5] = 0; band[-5:] = 0; band[:, :5] = 0; band[:, -5:] = 0
    corners = cv.goodFeaturesToTrack(ims[0], maxCorners=160, qualityLevel=.01,
                                   minDistance=8, mask=band, blockSize=7, useHarrisDetector=False)
    seeds = [] if corners is None else np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    if len(seeds) > 160:
        raise ValueError('corner provider exceeded bound')
    def safe(p, margin=4):
        return p is not None and np.isfinite(p).all() and margin <= p[0] <= w-1-margin and margin <= p[1] <= h-1-margin
    def xy(p):
        return [float(v) for v in p] if p is not None and np.isfinite(p).all() else None
    def number(v):
        return float(v) if v is not None and np.isfinite(v) else None
    def distance_at(d, p):
        return float(cv.getRectSubPix(d, (1, 1), tuple(map(float, p)))[0, 0])
    ledger, chains, records = [], [], []
    for i, p in enumerate(seeds):
        if not safe(p, 5) or not 8 <= distance_at(distances[0], p) <= 48:
            raise ValueError('corner provider returned point outside original seed band')
        ledger.append(dict(seed_id=i, frame_index=ids[0], point=xy(p), before_exterior_distance_px=distance_at(distances[0], p)))
        chains.append(dict(seed_id=i, points=[dict(frame_index=ids[0], point=xy(p))], steps=[], continuous=True, accepted=False, endpoint_distance_px=None, endpoint_outside=None, reasons=[]))
    live = list(range(len(seeds)))
    positions = {i: np.asarray(p, dtype=np.float32) for i, p in enumerate(seeds)}
    kwargs = dict(winSize=(21, 21), maxLevel=3, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, .01))
    for t in range(1, len(ims)):
        if not live:
            break
        p = np.array([positions[i] for i in live], dtype=np.float32).reshape(-1, 1, 2)
        q, fs, fe = cv.calcOpticalFlowPyrLK(ims[t-1], ims[t], p, None, **kwargs)
        def unpack(a, n, coords=False):
            if a is None:
                return np.full((n, 2) if coords else (n,), np.nan)
            return np.asarray(a).reshape(n, 2) if coords else np.asarray(a).reshape(n)
        q, fs, fe = unpack(q, len(live), True), unpack(fs, len(live)), unpack(fe, len(live))
        back = np.full_like(q, np.nan); bs = np.full(len(live), np.nan); be = bs.copy()
        eligible = [j for j in range(len(live)) if fs[j] == 1 and safe(q[j]) and np.isfinite(fe[j]) and 0 <= fe[j] <= 25]
        if eligible:
            r, s, e = cv.calcOpticalFlowPyrLK(ims[t], ims[t-1], q[eligible].astype(np.float32).reshape(-1, 1, 2), None, **kwargs)
            back[eligible], bs[eligible], be[eligible] = unpack(r, len(eligible), True), unpack(s, len(eligible)), unpack(e, len(eligible))
        next_live = []
        for j, seed in enumerate(live):
            reasons = []
            row = dict(seed_id=seed, from_frame_index=ids[t-1], to_frame_index=ids[t], from_point=xy(p[j, 0]), forward_point=xy(q[j]), backward_point=xy(back[j]), forward_status=number(fs[j]), backward_status=number(bs[j]), forward_lk_error=number(fe[j]), backward_lk_error=number(be[j]), fb_error_px=None, patch_ncc=None, patch_std=None, reasons=reasons, passed=False)
            if fs[j] != 1: reasons.append('forward_status')
            if not safe(q[j]): reasons.append('forward_nonfinite_or_patch_bounds')
            if not np.isfinite(fe[j]) or not 0 <= fe[j] <= 25: reasons.append('forward_lk_error')
            if bs[j] != 1: reasons.append('backward_status')
            if not safe(back[j]): reasons.append('backward_nonfinite_or_patch_bounds')
            if not np.isfinite(be[j]) or not 0 <= be[j] <= 25: reasons.append('backward_lk_error')
            if np.isfinite(back[j]).all():
                row['fb_error_px'] = float(np.linalg.norm(back[j] - p[j, 0]))
                if row['fb_error_px'] > 1: reasons.append('fb_error')
            if safe(q[j]):
                patches = [cv.getRectSubPix(im, (9, 9), tuple(map(float, point))).astype(float) for im, point in [(ims[t-1], p[j, 0]), (ims[t], q[j])]]
                row['patch_std'] = min(float(z.std()) for z in patches)
                a, b = [z-z.mean() for z in patches]
                den = np.linalg.norm(a)*np.linalg.norm(b)
                row['patch_ncc'] = float(np.clip((a*b).sum()/den, -1, 1)) if den > 0 else None
                if row['patch_std'] < 5: reasons.append('patch_low_texture')
                if row['patch_ncc'] is None or row['patch_ncc'] < .75: reasons.append('patch_appearance')
            row['passed'] = not reasons
            records.append(row); chains[seed]['steps'].append(row)
            if row['forward_point'] is not None:
                chains[seed]['points'].append(dict(frame_index=ids[t], point=row['forward_point'], transition_passed=row['passed']))
            if reasons:
                chains[seed]['continuous'] = False; chains[seed]['reasons'] = list(reasons)
            else:
                positions[seed] = q[j].astype(np.float32); next_live.append(seed)
        live = next_live
    accepted = []
    for seed in live:
        chain = chains[seed]; point = positions[seed]
        d = distance_at(distances[1], point)
        chain['endpoint_distance_px'] = d; chain['endpoint_outside'] = d > 0
        if d < 8:
            chain['reasons'] = ['endpoint_inside_original_after_mask' if d == 0 else 'endpoint_distance_below_8']
        else:
            chain['accepted'] = True
            accepted.append(dict(chain, witness_id=seed, observations=[dict(frame_index=v['frame_index'], xy=v['point']) for v in chain['points']], reference_exterior_witness=True, verified_background=False, tag='REF_EXTERIOR_CONDITIONAL', physical_identity='UNVERIFIED', background_truth=False, three_dimensional=False))
    return dict(schema_version=1, profile=dict(PROFILE), frame_indices=ids,
                coordinate_convention='OpenCV pixel centres; no half-pixel shift',
                endpoint_selection='retrospective survivor-conditioned original AFTER mask exterior >=8px; no upper endpoint distance bound',
                physical_identity='UNVERIFIED', background_truth=False, three_dimensional=False,
                seed_ledger=ledger, step_records=records, all_chains=chains, accepted_witnesses=accepted,
                counts=dict(seeds=len(seeds), continuous_survivors=len(live), rejected_continuous=len(seeds)-len(live), rejected_endpoint=len(live)-len(accepted), accepted=len(accepted)))
