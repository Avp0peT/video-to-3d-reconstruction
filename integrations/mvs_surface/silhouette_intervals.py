"""Sampled silhouette transition estimates, never measured hidden surfaces."""
import math
import cv2
import numpy as np
from .local_depth_search import _inputs, _rays, _project

REASONS = {0: 'UNSAMPLED', 1: 'NO_SURVIVING_SAMPLE', 2: 'DISJOINT_INTERVALS',
           3: 'UNBOUNDED', 4: 'UNKNOWN_BRACKET', 5: 'BOUNDED'}


def _classify(ref, source, rays, z):
    uv, _, positive = _project(ref, source, rays, z)
    h, w = source['mask'].shape
    inside = positive & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
    ix = np.rint(np.clip(uv[:, 0], 0, w-1)).astype(int)
    iy = np.rint(np.clip(uv[:, 1], 0, h-1)).astype(int)
    known = inside & source['validity'][iy, ix]
    fg = known & source['mask'][iy, ix]
    return fg, known & ~source['mask'][iy, ix], ~known


def estimate_silhouette_intervals(reference, sources, roi, depth_bounds, check=lambda: None):
    """Use all but the final source for admission; final source is diagnostic only.

    Front/back are axial z at inverse-depth midpoints of adjacent samples.
    No values are filled between sampled reference pixels.
    """
    ref, src, _ = _inputs(reference, sources, roi, check)
    if len(src) not in (4, 5):
        raise ValueError('four to five sources required, including one withheld source')
    for v in [ref] + src:
        valid = v.get('validity', np.ones(v['mask'].shape, bool))
        if not isinstance(valid, np.ndarray) or valid.dtype != bool or valid.shape != v['mask'].shape:
            raise ValueError('invalid validity mask')
        v['validity'] = valid
    if (not isinstance(depth_bounds, (list, tuple)) or len(depth_bounds) != 2 or
        any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in depth_bounds) or
        depth_bounds[0] >= depth_bounds[1]):
        raise ValueError('finite positive increasing axial depth bounds required')
    low, high = map(float, depth_bounds)
    if low < np.finfo(np.float32).tiny or high > np.finfo(np.float32).max:
        raise ValueError('bounds exceed supported float32 depth range')
    inverse = np.linspace(1/low, 1/high, 128)
    depths = 1/inverse
    shape = roi.shape
    front = np.zeros(shape, np.float64); back = front.copy()
    transitions = {key: np.zeros(shape, np.float64) for key in
                   ('front_transition_low', 'front_transition_high',
                    'back_transition_low', 'back_transition_high')}
    reason = np.zeros(shape, np.uint8)
    eroded = cv2.erode((ref['mask'] & ref['validity']).astype(np.uint8), np.ones((3,3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    yy, xx = np.indices(shape)
    sampled = roi & eroded & (xx % 3 == 0) & (yy % 3 == 0)
    y, x = np.nonzero(sampled)
    surviving = np.zeros(shape, np.uint16)
    endpoint_counts = {name: dict(foreground=0, background=0, unknown=0, evaluated=0)
                       for name in ('front', 'back')}
    final_counts = dict(support=0, known_background=0, unknown=0, evaluated_samples=0)
    train_counts = dict(support=0, known_background=0, unknown=0, evaluated_samples=0)
    for start in range(0, len(x), 256):
        check()
        bx, by = x[start:start+256], y[start:start+256]
        rays = np.repeat(_rays(ref, bx, by), 128, axis=0)
        z = np.tile(depths, len(bx))
        admitted = np.ones(len(z), bool); background = np.zeros(len(z), bool)
        for source in src[:-1]:
            check()
            fg, bg, unknown = _classify(ref, source, rays, z)
            admitted &= fg; background |= bg
            for key, value in [('support',fg),('known_background',bg),('unknown',unknown)]:
                train_counts[key] += int(value.sum())
            train_counts['evaluated_samples'] += len(z)
        admitted = admitted.reshape(-1,128); background = background.reshape(-1,128)
        for j, (px, py) in enumerate(zip(bx, by)):
            a = admitted[j]; ids = np.flatnonzero(a)
            surviving[py,px] = len(ids)
            code = 1
            if len(ids):
                first, last = ids[0], ids[-1]
                if len(ids) != last-first+1: code = 2
                elif first == 0 or last == 127: code = 3
                elif not (background[j,first-1] and background[j,last+1]): code = 4
                else:
                    code = 5
                    front[py,px] = 1/((inverse[first-1]+inverse[first])/2)
                    back[py,px] = 1/((inverse[last]+inverse[last+1])/2)
                    for key, index in [('front_transition_low',first-1),
                                       ('front_transition_high',first),
                                       ('back_transition_low',last),
                                       ('back_transition_high',last+1)]:
                        transitions[key][py,px] = depths[index]
            reason[py,px] = code
        # Withheld evidence is collected only after admission/bounds are fixed.
        fg, bg, unknown = _classify(ref, src[-1], rays, z)
        selected = np.repeat(reason[by,bx] == 5, 128) & admitted.ravel()
        for key, value in [('support',fg),('known_background',bg),('unknown',unknown)]:
            final_counts[key] += int((selected & value).sum())
        final_counts['evaluated_samples'] += int(selected.sum())
        selected_rays = reason[by,bx] == 5
        endpoint_rays = _rays(ref, bx[selected_rays], by[selected_rays])
        for name, field in [('front', front), ('back', back)]:
            check()
            fg, bg, unknown = _classify(ref, src[-1], endpoint_rays,
                                       field[by[selected_rays],bx[selected_rays]])
            for key, value in [('foreground',fg), ('background',bg), ('unknown',unknown)]:
                endpoint_counts[name][key] += int(value.sum())
            endpoint_counts[name]['evaluated'] += len(fg)
    bounded = reason == 5
    if not (np.isfinite(front[bounded]).all() and np.isfinite(back[bounded]).all() and (front[bounded] > 0).all() and (back[bounded] >= front[bounded]).all()):
        raise ValueError('invalid computed silhouette bounds')
    old_known = bounded & (ref['depth'] > 0)
    outside = old_known & ((ref['depth'] < front) | (ref['depth'] > back))
    counts = dict(roi_pixels=int(roi.sum()), sampled_pixels=int(sampled.sum()), bounded_pixels=int(bounded.sum()),
                  whole_depth_evaluated=int(old_known.sum()), whole_depth_outside_midpoint_estimate=int(outside.sum()),
                  whole_depth_outside_outer_sample_bracket=int((old_known & ((ref['depth'] < transitions['front_transition_low']) | (ref['depth'] > transitions['back_transition_high']))).sum()),
                  reasons={name:int((reason == code).sum()) for code,name in REASONS.items()})
    return dict(**transitions, front_depth=front, back_depth=back, interval_width=back-front,
                sampled_mask=sampled, bounded_mask=bounded, reason=reason, reason_legend=dict(REASONS),
                sample_counts=surviving, counts=counts, counters=counts,
                train_sample_counts=train_counts, withheld_sample_counts=final_counts,
                withheld_endpoint_counts=endpoint_counts,
                admission_source_ids=[s['view_id'] for s in src[:-1]], withheld_source_id=src[-1]['view_id'],
                depth_bounds=[low,high], depth_samples=depths, grid_step=3, erosion_pixels=1,
                geometry_accepted=False, status='SILHOUETTE_BOUND_NOT_MEASURED_SURFACE',
                assumptions=['rigid scene', 'calibrated cameras', 'correct silhouettes'],
                discretization='128 inverse-depth samples can miss thinner intervals or holes; midpoint bounds and transition sample brackets are estimates, not a certified continuous visual hull. Historical depth outside a midpoint estimate is not a silhouette conflict.',
                semantics='Outer-shape bound, not hidden-cavity recovery; withheld view is not independent world truth. Historical depth and frozen bounds may share reconstruction evidence.')
