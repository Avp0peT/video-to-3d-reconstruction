"""Deterministic local motion diagnostic using actual forward registration centers.

Regions may share pixels: support counts are not independent physical observations.
Candidate motion is consulted only after neighbor-only model selection. This module
does no I/O and never grants geometry acceptance or verifies identity.
"""
from itertools import combinations
import math
import numpy as np

THRESHOLDS = dict(min_distance_px=16., max_distance_px=96., max_neighbors=8,
                  min_support=4, support_fraction=.6, inlier_error_px=2.,
                  max_design_condition=100., min_singular_value=.65,
                  max_singular_value=1.5, max_affine_condition=1.8,
                  max_loo_error_px=4., max_control_error_px=4.)


def _array(value, shape, name):
    # Inspect object scalars before NumPy can silently coerce mixed bool/float.
    try:
        objects = np.asarray(value, dtype=object)
        if any(isinstance(item, (bool, np.bool_)) for item in objects.flat):
            raise ValueError('INVALID_' + name)
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('INVALID_' + name) from exc
    if raw.shape != shape or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError('INVALID_' + name)
    return raw.astype(float)


def _hull_contains(points, point):
    """Monotone-chain convex hull with boundary inclusion."""
    pts = sorted(set(map(tuple, points)))
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    halves = []
    for seq in (pts, pts[::-1]):
        half = []
        for p in seq:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.append(half[:-1])
    hull = halves[0] + halves[1]
    return len(hull) >= 3 and all(cross(hull[i], hull[(i+1) % len(hull)], point) >= -1e-8
                                     for i in range(len(hull)))


def assess_motion(candidate_id, source_center, source_controls, candidate_matrix,
                  neighbors, *, budget=lambda: None):
    """Return finite JSON diagnostic; malformed inputs raise ValueError.

    ``neighbors`` contains unique region_id, source_raw_xy, target_raw_xy and
    strict_accepted (an actual bool). All records are validated, including self
    and records outside the selected radius. Budget exceptions propagate.
    ``matrix_global_raw3x3`` maps source raw pixels to predicted target raw
    pixels; ``neighbor_matrix`` is its compatibility alias. The legacy
    ``max_neighbor_error_px`` and explicit ``max_selected_neighbor_error_px``
    include outliers, whereas ``max_inlier_error_px`` covers only model support.
    """
    budget()
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError('INVALID_CANDIDATE_ID')
    center = _array(source_center, (2,), 'SOURCE_CENTER')
    controls = _array(source_controls, (5, 2), 'SOURCE_CONTROLS')
    candidate = _array(candidate_matrix, (3, 3), 'CANDIDATE_MATRIX')
    if not isinstance(neighbors, list):
        raise ValueError('INVALID_NEIGHBORS')
    rows, seen = [], set()
    for row in neighbors:
        budget()
        if not isinstance(row, dict):
            raise ValueError('INVALID_NEIGHBOR')
        ident = row.get('region_id')
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError('INVALID_OR_DUPLICATE_REGION_ID')
        seen.add(ident)
        src = _array(row.get('source_raw_xy'), (2,), 'NEIGHBOR_SOURCE')
        dst = _array(row.get('target_raw_xy'), (2,), 'NEIGHBOR_TARGET')
        if type(row.get('strict_accepted')) is not bool:
            raise ValueError('INVALID_STRICT_ACCEPTED')
        distance = float(np.linalg.norm(src-center))
        if not math.isfinite(distance):
            raise ValueError('NONFINITE_DISTANCE')
        if ident != candidate_id and 16 <= distance <= 96:
            rows.append((distance, ident, src, dst, row['strict_accepted']))
    rows = sorted(rows, key=lambda r: (r[0], r[1]))[:8]
    ids = [r[1] for r in rows]
    source = np.array([r[2] for r in rows]).reshape(-1, 2)
    target = np.array([r[3] for r in rows]).reshape(-1, 2)
    result = dict(status='NO_CONSENSUS', reasons=[], selected_neighbor_ids=ids,
                  inlier_ids=[], outlier_ids=ids.copy(), source_raw_xy=source.tolist(),
                  actual_target_raw_xy=target.tolist(), neighbor_matrix=None,
                  matrix_global_raw3x3=None,
                  loo_errors_px=[], design_condition=None, affine_condition=None,
                  neighbor_residuals_px=[], max_neighbor_error_px=None,
                  max_selected_neighbor_error_px=None, max_inlier_error_px=None,
                  max_loo_error_px=None, control_residuals_px=[], max_control_error_px=None,
                  thresholds=THRESHOLDS.copy(), geometry_accepted=False, identity_verified=False,
                  support_is_independent_physical_points=False)
    if len(rows) < 4:
        result.update(status='INSUFFICIENT_SUPPORT', reasons=['FEWER_THAN_FOUR_NEIGHBORS'])
        return result
    design = np.column_stack(((source-center)/96., np.ones(len(rows))))
    control_design = np.column_stack(((controls-center)/96., np.ones(5)))
    rejected = set()

    def fit(indices):
        budget()
        x = design[list(indices)]
        condition = float(np.linalg.cond(x))
        if np.linalg.matrix_rank(x) != 3 or not math.isfinite(condition) or condition > 100:
            rejected.add('DEGENERATE_DESIGN')
            return None
        coeff = np.linalg.lstsq(x, target[list(indices)], rcond=None)[0]
        if not np.isfinite(coeff).all():
            raise ValueError('NONFINITE_FITTED_COEFFICIENTS')
        affine = coeff[:2].T / 96.
        singular = np.linalg.svd(affine, compute_uv=False)
        if (np.linalg.det(affine) <= 0 or singular[-1] < .65 or singular[0] > 1.5
                or singular[0]/singular[-1] > 1.8):
            rejected.add('INVALID_AFFINE_GEOMETRY')
            return None
        return coeff, condition, float(singular[0]/singular[-1])

    models, seen_sets = [], set()
    for triple in combinations(range(len(rows)), 3):
        fitted = fit(triple)
        if fitted is None:
            continue
        residual = np.linalg.norm(design @ fitted[0]-target, axis=1)
        if not np.isfinite(residual).all():
            raise ValueError('NONFINITE_NEIGHBOR_RESIDUAL')
        inliers = tuple(np.flatnonzero(residual <= 2.).tolist())
        if len(inliers) < max(4, math.ceil(.6*len(rows))) or inliers in seen_sets:
            continue
        seen_sets.add(inliers)
        fitted = fit(inliers)
        if fitted is None:
            continue
        coeff, condition, affine_condition = fitted
        residual = np.linalg.norm(design @ coeff-target, axis=1)
        if not np.isfinite(residual).all():
            raise ValueError('NONFINITE_NEIGHBOR_RESIDUAL')
        if any(residual[i] > 2. for i in inliers):
            rejected.add('REFIT_INLIER_ERROR')
            continue
        loo = []
        for i in inliers:
            held_fit = fit(tuple(j for j in inliers if j != i))
            if held_fit is None:
                break
            loo.append(float(np.linalg.norm(design[i] @ held_fit[0]-target[i])))
            if not math.isfinite(loo[-1]):
                raise ValueError('NONFINITE_LOO_RESIDUAL')
        if len(loo) != len(inliers) or max(loo) > 4.:
            rejected.add('LEAVE_ONE_NEIGHBOR_OUT_FAILED')
            continue
        if not _hull_contains(source[list(inliers)], center):
            rejected.add('SOURCE_CENTER_OUTSIDE_INLIER_HULL')
            continue
        rmse = float(np.sqrt(np.mean(residual[list(inliers)]**2)))
        models.append((inliers, coeff, condition, affine_condition, residual, loo, rmse))
    result['reasons'] = sorted(rejected)
    if not models:
        return result
    models.sort(key=lambda m: (-len(m[0]), m[6], tuple(ids[i] for i in m[0])))
    best = models[0]
    inliers, coeff, condition, affine_condition, residual, loo, _ = best
    matrix = np.eye(3)
    matrix[:2, :2] = coeff[:2].T/96.
    matrix[:2, 2] = coeff[2]-matrix[:2, :2] @ center
    predicted = control_design @ coeff
    if not np.isfinite(matrix).all() or not np.isfinite(predicted).all():
        raise ValueError('NONFINITE_NEIGHBOR_PREDICTION')
    result.update(inlier_ids=[ids[i] for i in inliers],
                  outlier_ids=[ident for i, ident in enumerate(ids) if i not in inliers],
                  neighbor_matrix=matrix.tolist(), loo_errors_px=loo,
                  matrix_global_raw3x3=matrix.tolist(),
                  max_loo_error_px=float(max(loo)),
                  design_condition=condition, affine_condition=affine_condition,
                  max_neighbor_error_px=float(max(residual)),
                  max_selected_neighbor_error_px=float(max(residual)),
                  max_inlier_error_px=float(max(residual[list(inliers)])),
                  neighbor_residuals_px=[dict(region_id=ident, error_px=float(residual[i]))
                                         for i, ident in enumerate(ids)],
                  neighbor_control_predictions=predicted.tolist())
    if any(len(m[0]) >= len(inliers)-1 and
           np.max(np.linalg.norm(control_design @ m[1]-predicted, axis=1)) > 4.
           for m in models[1:]):
        result.update(status='AMBIGUOUS', reasons=result['reasons']+['COMPETING_NEIGHBOR_MODELS'])
        return result
    mapped = np.column_stack((controls, np.ones(5))) @ candidate.T
    if not np.isfinite(mapped).all() or np.any(mapped[:, 2] == 0):
        raise ValueError('INVALID_CANDIDATE_FORWARD_MAPPING')
    actual = mapped[:, :2]/mapped[:, 2, None]
    errors = np.linalg.norm(actual-predicted, axis=1)
    if not np.isfinite(actual).all() or not np.isfinite(errors).all():
        raise ValueError('NONFINITE_CANDIDATE_FORWARD_MAPPING')
    result.update(actual_candidate_control_targets=actual.tolist(), control_residuals_px=errors.tolist(),
                  max_control_error_px=float(max(errors)))
    if max(errors) > 4.:
        status, reason = 'NEIGHBOR_CONFLICT', 'CANDIDATE_CONTROL_ERROR_EXCEEDS_THRESHOLD'
    elif not any(rows[i][4] for i in inliers):
        status, reason = 'UNANCHORED_CONSENSUS', 'NO_STRICT_ACCEPTED_INLIER'
    else:
        status, reason = 'NEIGHBOR_CONSISTENT_CANDIDATE', 'CONTROL_ERROR_WITHIN_THRESHOLD'
    result.update(status=status, reasons=result['reasons']+[reason])
    return result
