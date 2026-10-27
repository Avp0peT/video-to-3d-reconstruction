"""Frozen local measured-point diagnostics; never select or warp a mask.

Similarity and affine are separate hypotheses, not holdout-selected alternatives.
Passing evidence does not establish physical identity or geometry acceptance.
"""
from numbers import Integral

import numpy as np

VERSION = "local-motion-v1"
CONSTANTS = dict(max_points=768, grid_size=3, holdout_stride=5, min_fit=12,
    min_holdout=4, min_occupied_quadrants=3, min_eigenvalue_px2=4.0,
    ransac_threshold_px=2.5, max_iters=1000, confidence=0.99, refine_iters=10,
    seed=0, min_inlier_fraction=0.65, min_singular_value=0.7,
    max_singular_value=1.4, max_condition=2.0, holdout_p90_px=4.0,
    holdout_max_px=10.0, max_extrapolation_px=8.0, point_residual_px=4.0)


def _mask(mask, shape=None):
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or
        mask.ndim != 2 or not mask.size or (shape is not None and mask.shape != shape)
        or not np.isin(mask, [0, 255]).all() or not mask.any()):
        raise ValueError("masks must be same-shape nonempty uint8 binary 0/255 arrays")
    return mask


def _points(value, shape):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.dtype.kind not in "fiu":
        raise ValueError("points must be real numeric Nx2")
    p = raw.astype(np.float64)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > [shape[1]-1, shape[0]-1]):
        raise ValueError("points must be finite and image-bounded")
    return p


def _inside(p, bounds):
    lo, hi = np.asarray(bounds[:2]), np.asarray(bounds[2:])
    return np.all((p >= lo) & (p < hi), axis=1)


def _spread(p):
    if len(p) < 3:
        return 0.0
    d = p-p.mean(axis=0)
    return float(np.linalg.eigvalsh(d.T @ d / len(p))[0])


def _group(p, q, bounds):
    lo, hi = np.asarray(bounds[:2]), np.asarray(bounds[2:])
    quadrants = (p >= (lo+hi)/2).astype(int)
    occupied = sorted(set(map(tuple, quadrants.tolist())))
    return dict(occupied_quadrants=[list(v) for v in occupied],
        occupied_quadrant_count=len(occupied), source_min_eigenvalue_px2=_spread(p),
        query_min_eigenvalue_px2=_spread(q))


def _check_group(name, group, reasons):
    if group['occupied_quadrant_count'] < 3:
        reasons.append(name+'_insufficient_quadrants')
    if min(group['source_min_eigenvalue_px2'], group['query_min_eigenvalue_px2']) < 4:
        reasons.append(name+'_insufficient_2d_spread')


def _query_coverage(p, bounds, matrix):
    x0, y0, x1, y1 = bounds
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    predicted = corners @ matrix[:, :2].T + matrix[:, 2]
    lo, hi = predicted.min(0), predicted.max(0)
    quadrant = (p >= (lo+hi)/2).astype(int)
    inside = np.all((p >= lo) & (p <= hi), axis=1)
    occupied = sorted(set(map(tuple, quadrant[inside].tolist())))
    return dict(predicted_bounds_xyxy=[*lo.tolist(), *hi.tolist()],
        occupied_quadrants=[list(v) for v in occupied], occupied_quadrant_count=len(occupied))


def _cell(p, q, ids, fit, held, core, window, cell_id, estimator, cv):
    reasons = []
    record = dict(cell_id=cell_id, core_bounds_xyxy=core, fit_window_bounds_xyxy=window,
        fit_track_ids=[ids[i] for i in fit], holdout_track_ids=[ids[i] for i in held],
        fit_inlier_track_ids=[], matrix=None, fit_inlier_fraction=None,
        singular_values=None, determinant=None, condition_number=None,
        fit_residuals_px=[], holdout_residuals_px=[], holdout_p90_px=None,
        holdout_max_px=None, holdout_extrapolation_distances_px=[],
        holdout_query_extrapolation_distances_px=[],
        groups=dict(fit=_group(p[fit], q[fit], window), holdout=_group(p[held], q[held], core)),
        passed=False, reasons=reasons)
    if len(fit) < 12: reasons.append('insufficient_fit_points')
    if len(held) < 4: reasons.append('insufficient_holdout_points')
    for name, group in record['groups'].items():
        _check_group(name, group, reasons)
    if reasons:
        return record
    cv.setRNGSeed(0)
    matrix, inliers = estimator(p[fit].copy(), q[fit].copy(), method=cv.RANSAC,
        ransacReprojThreshold=2.5, maxIters=1000, confidence=0.99, refineIters=10)
    if matrix is None or inliers is None:
        reasons.append('estimation_failed')
        return record
    matrix = np.asarray(matrix, dtype=float)
    inliers = np.asarray(inliers).reshape(-1)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or len(inliers) != len(fit):
        reasons.append('invalid_estimation_output')
        return record
    selected = fit[inliers != 0]
    residuals = np.linalg.norm(p @ matrix[:, :2].T + matrix[:, 2]-q, axis=1)
    singular = np.linalg.svd(matrix[:, :2], compute_uv=False)
    determinant = float(np.linalg.det(matrix[:, :2]))
    condition = float(singular[0]/singular[-1]) if singular[-1] else None
    group = _group(p[selected], q[selected], window)
    record['groups']['fit_inliers'] = group
    _check_group('fit_inliers', group, reasons)
    for name, indices, bounds in [('fit', fit, window), ('holdout', held, core), ('fit_inliers', selected, window)]:
        coverage = _query_coverage(q[indices], bounds, matrix)
        record['groups'][name]['query_coverage'] = coverage
        if coverage['occupied_quadrant_count'] < 3:
            reasons.append(name+'_query_insufficient_quadrants')
    hull = cv.convexHull(p[selected].astype(np.float32)) if len(selected) >= 3 else None
    distances = ([max(0.0, -float(cv.pointPolygonTest(hull, tuple(map(float, v)), True)))
        for v in p[held]] if hull is not None else [None]*len(held))
    query_hull = cv.convexHull(q[selected].astype(np.float32)) if len(selected) >= 3 else None
    query_distances = ([max(0.0, -float(cv.pointPolygonTest(query_hull, tuple(map(float, v)), True)))
        for v in q[held]] if query_hull is not None else [None]*len(held))
    record.update(matrix=matrix.tolist(), fit_inlier_track_ids=[ids[i] for i in selected],
        fit_inlier_fraction=float(len(selected)/len(fit)), singular_values=singular.tolist(),
        determinant=determinant, condition_number=condition,
        fit_residuals_px=residuals[fit].tolist(), holdout_residuals_px=residuals[held].tolist(),
        holdout_p90_px=float(np.percentile(residuals[held], 90)),
        holdout_max_px=float(residuals[held].max()), holdout_extrapolation_distances_px=distances,
        holdout_query_extrapolation_distances_px=query_distances)
    if record['fit_inlier_fraction'] < .65: reasons.append('fit_inlier_fraction_low')
    if determinant <= 0: reasons.append('nonpositive_determinant')
    if singular[-1] < .7 or singular[0] > 1.4: reasons.append('singular_values_out_of_bounds')
    if condition is None or condition > 2: reasons.append('condition_number_high')
    if record['holdout_p90_px'] > 4 or record['holdout_max_px'] > 10:
        reasons.append('holdout_residual_high')
    if any(d is None or d > 8 for d in distances): reasons.append('holdout_extrapolation_unsupported')
    if any(d is None or d > 8 for d in query_distances): reasons.append('holdout_query_extrapolation_unsupported')
    record['passed'] = not reasons
    return record


def evaluate_local_motion(before_mask, after_mask, before_points, query_points,
                          after_points, track_ids, cv):
    """Return JSON-safe diagnostics using supplied real measured correspondences.

    Sorted ID positions 0,5,... stay held out from every overlapping fit window.
    Source-mask bounding boxes alone determine regions. No query mask is used.
    Empty Nx2 arrays are valid insufficient evidence; invalid schemas raise.
    """
    bm = _mask(before_mask)
    am = _mask(after_mask, bm.shape)
    bp, qp, ap = [_points(v, bm.shape) for v in (before_points, query_points, after_points)]
    try:
        ids = list(track_ids)
    except TypeError as exc:
        raise ValueError('track_ids must be unique integers excluding bool') from exc
    if (len(bp) != len(qp) or len(bp) != len(ap) or len(ids) != len(bp) or len(ids) > 768
        or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in ids)):
        raise ValueError('equal lengths, maximum 768, unique integer IDs excluding bool required')
    ids = [int(v) for v in ids]
    if len(set(ids)) != len(ids): raise ValueError('duplicate track IDs')
    order = np.asarray(sorted(range(len(ids)), key=ids.__getitem__), dtype=int)
    ids = [ids[i] for i in order]
    bp, qp, ap = bp[order], qp[order], ap[order]
    held = np.arange(len(ids)) % 5 == 0
    result = dict(version=VERSION, status='DIAGNOSTIC_ONLY', physical_identity='UNVERIFIED',
        publishable_for_geometry=False, constants=dict(CONSTANTS),
        fit_track_ids=[v for i, v in enumerate(ids) if not held[i]],
        holdout_track_ids=[v for i, v in enumerate(ids) if held[i]], profiles={})
    for profile, estimator in [('local_similarity', cv.estimateAffinePartial2D), ('local_affine', cv.estimateAffine2D)]:
        references = {}
        for name, mask, p in [('before', bm, bp), ('after', am, ap)]:
            ys, xs = np.where(mask > 0)
            lo = np.array([xs.min(), ys.min()], dtype=float)
            hi = np.array([xs.max()+1, ys.max()+1], dtype=float)
            step = (hi-lo)/3
            xedges, yedges = np.linspace(lo[0], hi[0], 4), np.linspace(lo[1], hi[1], 4)
            pixel = np.floor(p+.5).astype(int)
            member = mask[pixel[:, 1], pixel[:, 0]] > 0
            cells = []
            for row in range(3):
                for col in range(3):
                    # Shared edge values make half-open cores disjoint even
                    # when the mask width is not divisible by three.
                    start = np.array([xedges[col], yedges[row]])
                    end = np.array([xedges[col+1], yedges[row+1]])
                    core = [*start.tolist(), *end.tolist()]
                    center = (start+end)/2
                    window = [*(center-step).tolist(), *(center+step).tolist()]
                    fit = np.flatnonzero(member & ~held & _inside(p, window))
                    test = np.flatnonzero(member & held & _inside(p, core))
                    cells.append(_cell(p, qp, ids, fit, test, core, window,
                        f'r{row}c{col}', estimator, cv))
            references[name] = dict(source_mask_bbox_xyxy=[*lo.tolist(), *hi.tolist()],
                source_mask_member_track_ids=[ids[i] for i in np.flatnonzero(member)], cells=cells)
        supports = {}
        for name, reference in references.items():
            supports[name] = {tid: (cell['cell_id'], residual) for cell in reference['cells']
                if cell['passed'] for tid, residual in zip(cell['holdout_track_ids'], cell['holdout_residuals_px']) if residual <= 4}
        common = sorted(set(supports['before']) & set(supports['after']))
        lookup = {v: i for i, v in enumerate(ids)}
        evidence = [dict(track_id=tid, query_point_xy=qp[lookup[tid]].tolist(),
            before_cell_id=supports['before'][tid][0], after_cell_id=supports['after'][tid][0],
            before_residual_px=supports['before'][tid][1], after_residual_px=supports['after'][tid][1]) for tid in common]
        result['profiles'][profile] = dict(status='LOCAL_SUPPORT' if common else 'UNRESOLVED',
            references=references, common_supported_track_ids=common, measured_query_evidence=evidence)
    return result
