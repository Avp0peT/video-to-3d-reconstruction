"""Windowed diagnostics of frozen global-MVS depth estimates, never reconstruction."""
from fractions import Fraction
import numpy as np
from .region_depth_support import project_pixels, separated_support


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def analyze_window(points, states, origins, cameras, target_index, times, region,
                   raw_coverage, global_selected, *, window=Fraction(2),
                   view_ids=None, check=lambda: None):
    """Return (row, masks) on the unchanged candidate universe.

    Cameras and times are in exactly the ledger-column order, supplied by the
    caller's bound identity join. Other-view support excludes generator/target.
    The generator must belong to the closed temporal window. The same angular
    support cohort underlies local and all-view veto comparisons.
    """
    points, states, origins, global_selected = map(np.asarray, (points, states, origins, global_selected))
    _need(points.ndim == 2 and points.shape[1] == 3 and points.dtype.kind in 'fiu' and np.isfinite(points).all(), 'finite Nx3 points required')
    n, m = len(points), len(cameras)
    _need(n <= 50000 and 0 < m <= 256, 'bounded candidate/view arrays required')
    _need(states.shape == (n, m) and states.dtype.kind in 'iu' and np.isin(states, [0, 1, 2, 3, 4]).all(), 'integer states 0..4 required')
    _need(origins.shape == (n,) and origins.dtype.kind in 'iu' and (not n or (origins.min() >= 0 and origins.max() < m)), 'origin indices required')
    _need(global_selected.shape == (n,) and global_selected.dtype == np.bool_, 'bool global_selected required')
    _need(type(target_index) is int and 0 <= target_index < m, 'target index required')
    _need(not np.any(origins == target_index), 'target view cannot generate candidates')
    _need(len(times) == m and all(type(t) is Fraction for t in times) and len(set(times)) == m and all(times[j] < times[j+1] for j in range(m-1)), 'unique increasing Fraction times in ledger order required')
    _need(type(window) is Fraction and window > 0, 'positive Fraction window required')
    if view_ids is None:
        view_ids = list(range(m))
    _need(len(view_ids) == m and all(type(v) in (str, int) for v in view_ids) and len(set(view_ids)) == m, 'unique ordered view IDs required')
    for camera in cameras:
        _need(all(np.asarray(camera[k]).shape == shape and np.isfinite(camera[k]).all() for k, shape in [('K', (3, 3)), ('R', (3, 3)), ('C', (3,))]), 'finite camera K/R/C required')
        _need(all(type(camera[k]) is int and 1 <= camera[k] <= 2048 for k in ('width', 'height')), 'bounded camera dimensions required')
        _need(np.linalg.det(camera['K']) > 0 and np.allclose(np.asarray(camera['R']) @ np.asarray(camera['R']).T, np.eye(3), atol=1e-7, rtol=0) and np.isclose(np.linalg.det(camera['R']), 1, atol=1e-7, rtol=0), 'valid calibrated camera required')
    region, raw_coverage = np.asarray(region), np.asarray(raw_coverage)
    shape = (cameras[target_index]['height'], cameras[target_index]['width'])
    _need(region.dtype == np.bool_ and raw_coverage.dtype == np.bool_ and region.shape == raw_coverage.shape == shape, 'same-domain bool target rasters required')
    near = np.array([abs(t-times[target_index]) <= window for t in times], bool)
    eligible = near[origins] & np.isin(states[:, target_index], [1, 2])
    uv, _, inside = project_pixels(points, cameras[target_index])
    in_region = np.zeros(n, bool); ids = np.flatnonzero(inside)
    in_region[ids] = region[uv[ids, 1], uv[ids, 0]]
    eligible &= in_region
    other = np.ones((n, m), bool)
    other[:, target_index] = False
    other[np.arange(n), origins] = False
    agreeing = (states == 2) & other & near
    candidate_ids = np.flatnonzero(eligible & (agreeing.sum(axis=1) >= 3))
    support = np.zeros(n, bool)
    check()
    support[candidate_ids] = separated_support(points[candidate_ids], cameras, agreeing[candidate_ids], check=check)
    local_front = ((states == 3) & other & near).any(axis=1)
    far_front = ((states == 3) & other & ~near).any(axis=1)
    all_front = local_front | far_front
    masks = dict(SUPPORT_ONLY=support, NO_LOCAL_FRONT_CONFLICT=support & ~local_front,
                 NO_ALL_VIEW_FRONT_CONFLICT=support & ~all_front,
                 REMOTE_ONLY_CONFLICT=support & ~local_front & far_front)
    missing = region & ~raw_coverage
    layers = {}
    for label, take in masks.items():
        hits = np.zeros(shape, bool); ii = np.flatnonzero(take)
        hits[uv[ii, 1], uv[ii, 0]] = True
        layers[label] = dict(samples=int(take.sum()), fraction_of_frozen_candidates=float(take.sum()/n) if n else None,
                             target_unique_pixels=int(hits.sum()), raw_missing_unique_pixels=int((hits & missing).sum()),
                             global_selected_overlap=int((take & global_selected).sum()))
    exposures = {}
    for label, band in [('near', near), ('far', ~near)]:
        active = other & band
        per_point = {name: ((states == code) & active).sum(axis=1, dtype=np.int16) for code, name in enumerate(('outside', 'unknown', 'agree', 'front', 'behind'))}
        # Known includes occluded/behind estimates; observable excludes behind.
        per_point['known'] = per_point['agree'] + per_point['front'] + per_point['behind']
        per_point['observable'] = per_point['agree'] + per_point['front']
        def aggregate(take):
            totals = {name: int(value[take].sum()) for name, value in per_point.items()}
            totals['front_fraction_of_observable'] = totals['front']/totals['observable'] if totals['observable'] else None
            return totals
        exposures[label] = dict(frozen_universe=aggregate(np.ones(n, bool)), same_support_cohort=aggregate(support))
    row = dict(window_seconds=dict(numerator=window.numerator, denominator=window.denominator),
               frozen_candidate_denominator=n, origin_and_target_eligible=int(eligible.sum()),
               three_other_agree_before_angle=len(candidate_ids), support_cohort=int(support.sum()),
               near_view_ids=[view_ids[j] for j in range(m) if near[j] and j != target_index],
               target_view_id=view_ids[target_index], layers=layers, exposures=exposures,
               local_front_veto_count=int((support & local_front).sum()),
               all_front_veto_count=int((support & all_front).sum()),
               remote_only_conflict_count=int(masks['REMOTE_ONLY_CONFLICT'].sum()),
               status='FROZEN_GLOBAL_DEPTH_WINDOW_DIAGNOSTIC_NOT_ADOPTED',
               depth_source='FROZEN_GLOBAL_MVS',
               temporal_filter_scope='ORIGIN_AND_VALIDATION_VIEWS_ONLY',
               depth_estimation_neighbors_window_restricted=False,
               limitations=['Near-view agreement selects this cohort; near/far rates are conditioned and not causal.',
                            'Any-view veto depends on observation opportunities, which differ across time bands.',
                            'Direction support uses conservative greedy triples; valid triples may be missed.'])
    return row, masks


def analyze_windows(*args, **kwargs):
    """All predeclared windows, no best-window selection; returns rows and masks."""
    _need('window' not in kwargs, 'windows are fixed at 1/2/4 seconds')
    rows, masks = [], {}
    for seconds in (1, 2, 4):
        row, layers = analyze_window(*args, window=Fraction(seconds), **kwargs)
        rows.append(row); masks[str(seconds)] = layers
    return rows, masks
