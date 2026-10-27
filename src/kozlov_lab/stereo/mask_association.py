"""Frozen CPU v1 evidence for choosing among existing masks, candidate-only.

This function cannot distinguish a hand or another object sharing the same
motion, and cannot verify physical identity, segmentation truth or 3D geometry.
No synthetic mask is returned or persisted. Input/schema failures raise
ValueError; valid inputs with insufficient evidence return UNRESOLVED.
"""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np


VERSION = "bidirectional-mask-association-v1"
MIN_POINTS = 25
MIN_HOLDOUT = 5
RANSAC_THRESHOLD_PX = 2.5
RANSAC_MAX_ITERS = 1000
RANSAC_CONFIDENCE = 0.99
RANSAC_REFINE_ITERS = 10
FIT_INLIER_MIN = 0.65
HOLDOUT_P90_MAX_PX = 4.0
HOLDOUT_MAX_PX = 10.0
SCALE_MIN, SCALE_MAX = 0.7, 1.4
MIN_OCCUPIED_GRID_CELLS = 6
MIN_HOLDOUT_GRID_CELLS = 4
MIN_COVARIANCE_EIGENVALUE = 4.0
QUERY_VALID_MIN = 0.95
SOURCE_FOREGROUND_RETAINED_MIN = 0.95
PREDICTION_COMMON_RETAINED_MIN = 0.95
EXTRAPOLATION_DISTANCE_PX = 64.0
EXTRAPOLATION_FRACTION_MAX = 0.20
COMMON_VALID_MIN = 0.90
IOU_MIN = 0.65
QUERY_COVERAGE_MIN = 0.85
OUTSIDE_DILATION_MAX = 0.20
CANDIDATE_VALID_MIN = 0.95
WINNER_MARGIN_MIN = 0.10
DILATION_RADIUS_PX = 6.0

LIMITATIONS = [
    "Candidate association only; no physical identity or segmentation truth verified.",
    "A hand or another object with shared motion cannot be distinguished by this evidence.",
    "No 3D geometry, physical scale or reconstruction quality verified.",
    "Affine partial motion and two reference masks are engineering hypotheses.",
    "Warp padding is unknown and excluded from mask overlap measurements.",
]


def _mask(value, shape=None, name="mask"):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError(f"{name} must be a uint8 ndarray")
    if value.ndim != 2 or not value.size or (shape is not None and value.shape != shape):
        raise ValueError(f"{name} has invalid shape")
    if not np.isin(value, [0, 255]).all() or not np.any(value):
        raise ValueError(f"{name} must contain nonempty binary 0/255 foreground")
    return value


def _points(value, shape, name):
    if value is None:
        raise ValueError(f"{name} is required")
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != 2 or raw.shape[1:] != (2,) or not len(raw):
        raise ValueError(f"{name} must be a nonempty numeric Nx2 array")
    points = raw.astype(np.float64)
    h, w = shape
    if not np.isfinite(points).all() or np.any(points < 0) or np.any(points[:, 0] > w-1) or np.any(points[:, 1] > h-1):
        raise ValueError(f"{name} must be finite and in bounds")
    return points


def _sample(mask, points):
    pixel = np.floor(points + 0.5).astype(np.int64)
    return mask[pixel[:, 1], pixel[:, 0]] > 0


def _iou(a, b, domain):
    union = np.count_nonzero((a | b) & domain)
    return float(np.count_nonzero(a & b & domain) / union) if union else 0.0


def _spread(points):
    if len(points) < 3:
        return 0.0
    centered = points - points.mean(axis=0)
    return float(np.linalg.eigvalsh(centered.T @ centered / len(points))[0])


def _grid_cells(points, bounds):
    x, y, width, height = bounds
    cells = np.floor((points - [x, y]) / [width, height] * 4).astype(int)
    return cells


def _group(points, query, bounds):
    cells = _grid_cells(points, bounds)
    occupied = sorted({(int(y), int(x)) for x, y in cells if 0 <= x < 4 and 0 <= y < 4})
    return dict(occupied_grid_cells=[list(v) for v in occupied], occupied_grid_cell_count=len(occupied),
                source_min_covariance_eigenvalue_px2=_spread(points),
                query_min_covariance_eigenvalue_px2=_spread(query))


def _outside_hull_fraction(foreground, anchors, cv):
    """Exact Euclidean distance to the fitted source-anchor convex hull."""
    if len(anchors) < 3:
        return 1.0
    hull = cv.convexHull(anchors.astype(np.float32)).reshape(-1, 2).astype(float)
    if len(hull) < 3:
        return 1.0
    minimum = np.full(len(foreground), np.inf)
    all_positive, all_negative = np.ones(len(foreground), bool), np.ones(len(foreground), bool)
    for start, end in zip(hull, np.roll(hull, -1, axis=0)):
        edge = end-start
        offset = foreground-start
        cross = edge[0]*offset[:, 1]-edge[1]*offset[:, 0]
        all_positive &= cross >= -1e-7
        all_negative &= cross <= 1e-7
        length = float(edge @ edge)
        t = np.clip(offset @ edge / length, 0, 1) if length else np.zeros(len(offset))
        minimum = np.minimum(minimum, np.linalg.norm(offset-t[:, None]*edge, axis=1))
    minimum[all_positive | all_negative] = 0
    return float(np.mean(minimum > EXTRAPOLATION_DISTANCE_PX))


def _reference(mask, points, query, fit, held, ids, cv):
    reasons = []
    membership = _sample(mask, points)
    ys, xs = np.where(mask > 0)
    width, height = int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)
    bounds = (int(xs.min()), int(ys.min()), width, height)
    cells = _grid_cells(points, bounds)
    occupied = sorted({(int(y), int(x)) for x, y in cells[membership] if 0 <= x < 4 and 0 <= y < 4})
    record = dict(matrix=None, reasons=reasons, source_mask_membership_count=int(membership.sum()),
                  occupied_grid_cells=[list(v) for v in occupied], occupied_grid_cell_count=len(occupied),
                  fit_track_ids=[ids[i] for i in fit], holdout_track_ids=[ids[i] for i in held],
                  fit_inlier_fraction=None, fit_inlier_track_ids=[], fit_residuals_px=[],
                  holdout_residuals_px=[], holdout_p90_px=None, holdout_max_px=None, scale=None)
    record['groups'] = dict(fit=_group(points[fit], query[fit], bounds),
                            holdout=_group(points[held], query[held], bounds))
    for name, minimum in [('fit', MIN_OCCUPIED_GRID_CELLS), ('holdout', MIN_HOLDOUT_GRID_CELLS)]:
        group = record['groups'][name]
        if group['occupied_grid_cell_count'] < minimum:
            reasons.append(name+'_insufficient_grid_coverage')
        if group['source_min_covariance_eigenvalue_px2'] < MIN_COVARIANCE_EIGENVALUE or group['query_min_covariance_eigenvalue_px2'] < MIN_COVARIANCE_EIGENVALUE:
            reasons.append(name+'_insufficient_2d_spread')
    if not membership.all():
        reasons.append("reference_points_outside_original_mask")
    if len(occupied) < MIN_OCCUPIED_GRID_CELLS:
        reasons.append("insufficient_reference_grid_coverage")
    # Always use fit points only; holdout points never influence estimation.
    cv.setRNGSeed(0)
    matrix, inliers = cv.estimateAffinePartial2D(points[fit], query[fit], method=cv.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD_PX, maxIters=RANSAC_MAX_ITERS,
        confidence=RANSAC_CONFIDENCE, refineIters=RANSAC_REFINE_ITERS)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        reasons.append("affine_estimation_failed")
        return record, None, None
    matrix = np.asarray(matrix, dtype=float)
    residuals = np.linalg.norm(points @ matrix[:, :2].T + matrix[:, 2] - query, axis=1)
    fraction = float(np.mean(np.asarray(inliers).reshape(-1) != 0))
    inlier_indices = fit[np.asarray(inliers).reshape(-1) != 0]
    group = _group(points[inlier_indices], query[inlier_indices], bounds)
    record['groups']['fit_inliers'] = group
    if group['occupied_grid_cell_count'] < MIN_OCCUPIED_GRID_CELLS:
        reasons.append('fit_inliers_insufficient_grid_coverage')
    if group['source_min_covariance_eigenvalue_px2'] < MIN_COVARIANCE_EIGENVALUE or group['query_min_covariance_eigenvalue_px2'] < MIN_COVARIANCE_EIGENVALUE:
        reasons.append('fit_inliers_insufficient_2d_spread')
    scale = float(math.hypot(matrix[0, 0], matrix[1, 0]))
    record.update(matrix=matrix.tolist(), fit_inlier_fraction=fraction,
        fit_inlier_track_ids=[ids[i] for i, yes in zip(fit, np.asarray(inliers).reshape(-1)) if yes],
        fit_residuals_px=residuals[fit].tolist(), holdout_residuals_px=residuals[held].tolist(),
        holdout_p90_px=float(np.percentile(residuals[held], 90)),
        holdout_max_px=float(np.max(residuals[held])), scale=scale)
    if fraction < FIT_INLIER_MIN:
        reasons.append("fit_inlier_fraction_low")
    if record["holdout_p90_px"] > HOLDOUT_P90_MAX_PX or record["holdout_max_px"] > HOLDOUT_MAX_PX:
        reasons.append("holdout_residual_high")
    if not SCALE_MIN <= scale <= SCALE_MAX:
        reasons.append("scale_out_of_bounds")
    h, w = mask.shape
    foreground = np.column_stack((xs, ys)).astype(float)
    transformed = foreground @ matrix[:, :2].T + matrix[:, 2]
    retained = float(np.mean((transformed[:, 0] >= 0) & (transformed[:, 0] <= w-1) &
                            (transformed[:, 1] >= 0) & (transformed[:, 1] <= h-1)))
    record['source_foreground_retained_fraction'] = retained
    if retained < SOURCE_FOREGROUND_RETAINED_MIN:
        reasons.append('source_foreground_clipped_by_warp')
    extrapolated = _outside_hull_fraction(foreground, points[inlier_indices], cv)
    record['source_foreground_beyond_inlier_hull_64px_fraction'] = extrapolated
    if extrapolated > EXTRAPOLATION_FRACTION_MAX:
        reasons.append('source_foreground_extrapolation_unsupported')
    kwargs = dict(flags=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)
    prediction = cv.warpAffine(mask, matrix, (w, h), **kwargs) > 0
    valid = cv.warpAffine(np.full(mask.shape, 255, np.uint8), matrix, (w, h), **kwargs) > 0
    record["valid_image_fraction"] = float(valid.mean())
    return record, prediction, valid


def associate_candidates(before_mask, after_mask, before_points, query_points,
                         after_points, track_ids, candidate_masks, cv):
    """Associate existing masks using separately fitted references sharing tracks.

    Point coordinates use OpenCV pixel centers. IDs are unique integers (bool
    prohibited). Correspondences must be supplied as independent evidence by
    the caller. Sorted IDs at positions 0,5,10,... form the frozen holdout.
    Empty candidate lists yield NO_CANDIDATE; empty or malformed masks are schema
    errors. The caller retains original NO_MATCH and identity provenance.
    """
    before_mask = _mask(before_mask, name="before_mask")
    after_mask = _mask(after_mask, before_mask.shape, "after_mask")
    bp = _points(before_points, before_mask.shape, "before_points")
    qp = _points(query_points, before_mask.shape, "query_points")
    ap = _points(after_points, before_mask.shape, "after_points")
    if len(bp) != len(qp) or len(ap) != len(bp):
        raise ValueError("point arrays must have equal lengths")
    if track_ids is None:
        raise ValueError("track_ids is required")
    try:
        ids = list(track_ids)
    except TypeError as exc:
        raise ValueError("track_ids must be a sequence") from exc
    if len(ids) != len(bp) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in ids):
        raise ValueError("track_ids must be same-length unique integers, excluding bool")
    ids = [int(v) for v in ids]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate track_ids")
    if not isinstance(candidate_masks, list):
        raise ValueError("candidate_masks must be a list")
    seen = set()
    for c in candidate_masks:
        if not isinstance(c, dict) or set(c) != {"candidate_id", "mask"}:
            raise ValueError("candidate entry must have exactly candidate_id and mask")
        cid = c["candidate_id"]
        if isinstance(cid, (bool, np.bool_)) or not isinstance(cid, (str, Integral)) or (isinstance(cid, str) and not cid):
            raise ValueError("candidate_id must be a nonempty string or integer")
        if cid in seen:
            raise ValueError("duplicate candidate_id")
        seen.add(cid)
        _mask(c["mask"], before_mask.shape, "candidate mask")
    order = np.argsort(ids)
    ids = [ids[i] for i in order]
    bp, qp, ap = bp[order], qp[order], ap[order]
    held = np.arange(0, len(ids), 5)
    fit = np.flatnonzero(np.arange(len(ids)) % 5 != 0)
    result = dict(version=VERSION, status="UNRESOLVED", selected_candidate_id=None,
        reasons=[], fit_track_ids=[ids[i] for i in fit], holdout_track_ids=[ids[i] for i in held],
        references={}, common_valid_image_fraction=None, prediction_agreement_iou=None,
        candidates=[], winner_margin=None, limitations=list(LIMITATIONS))
    if not candidate_masks:
        result.update(status="NO_CANDIDATE", reasons=["empty_candidate_list"])
        return result
    if len(ids) < MIN_POINTS or len(held) < MIN_HOLDOUT:
        result["reasons"].append("insufficient_correspondences")
        return result
    br, prediction_b, valid_b = _reference(before_mask, bp, qp, fit, held, ids, cv)
    ar, prediction_a, valid_a = _reference(after_mask, ap, qp, fit, held, ids, cv)
    result["references"] = dict(before=br, after=ar)
    for name, record in result["references"].items():
        result["reasons"].extend(name + ":" + reason for reason in record["reasons"])
    if valid_b is None or valid_a is None:
        return result
    domain = valid_b & valid_a
    result["common_valid_image_fraction"] = float(domain.mean())
    if domain.mean() < COMMON_VALID_MIN:
        result["reasons"].append("insufficient_common_warp_valid_domain")
    agreement = _iou(prediction_b, prediction_a, domain)
    result["prediction_agreement_iou"] = agreement
    if agreement < IOU_MIN:
        result["reasons"].append("reference_prediction_disagreement")
    valid_query = _sample(domain, qp)
    result['query_common_valid_fraction'] = float(valid_query.mean())
    if valid_query.mean() < QUERY_VALID_MIN:
        result['reasons'].append('query_points_outside_common_valid_domain')
    for name, prediction in [('before', prediction_b), ('after', prediction_a)]:
        fraction = float(np.count_nonzero(prediction & domain) / np.count_nonzero(prediction)) if prediction.any() else 0.0
        result['references'][name]['prediction_common_valid_fraction'] = fraction
        if fraction < PREDICTION_COMMON_RETAINED_MIN:
            result['reasons'].append(name+':prediction_foreground_outside_common_valid_domain')
    if not valid_query.any():
        result["reasons"].append("no_query_points_in_common_valid_domain")
    dilations = [cv.distanceTransform((~p).astype(np.uint8), cv.DIST_L2,
                 cv.DIST_MASK_PRECISE) <= DILATION_RADIUS_PX for p in (prediction_b, prediction_a)]
    for c in candidate_masks:
        mask = c["mask"] > 0
        inside = int(np.count_nonzero(mask & domain))
        valid_fraction = float(inside / np.count_nonzero(mask))
        ious = [_iou(mask, pred, domain) for pred in (prediction_b, prediction_a)]
        outside = [float(np.count_nonzero(mask & domain & ~dilated) / inside) if inside else 1.0 for dilated in dilations]
        covered = _sample(mask, qp) & valid_query
        coverage = float(covered.mean())
        fit_coverage, holdout_coverage = float(covered[fit].mean()), float(covered[held].mean())
        lower = qp.min(axis=0)
        span = qp.max(axis=0)-lower+1
        query_cells = _grid_cells(qp, (*lower, *span))
        cell_metrics = []
        for cell in sorted(set(map(tuple, query_cells))):
            chosen = np.all(query_cells == cell, axis=1)
            cell_metrics.append(dict(cell=list(map(int, cell)), count=int(chosen.sum()), coverage=float(covered[chosen].mean())))
        cell_mean = float(np.mean([v['coverage'] for v in cell_metrics]))
        reasons = []
        if min(ious) < IOU_MIN: reasons.append("candidate_reference_iou_low")
        if coverage < QUERY_COVERAGE_MIN: reasons.append("query_point_coverage_low")
        if fit_coverage < QUERY_COVERAGE_MIN: reasons.append('fit_query_point_coverage_low')
        if holdout_coverage < QUERY_COVERAGE_MIN: reasons.append('holdout_query_point_coverage_low')
        if cell_mean < QUERY_COVERAGE_MIN or any(v['coverage'] < QUERY_COVERAGE_MIN for v in cell_metrics): reasons.append('spatial_cell_query_coverage_low')
        if max(outside) > OUTSIDE_DILATION_MAX: reasons.append("candidate_outside_dilation_high")
        if valid_fraction < CANDIDATE_VALID_MIN: reasons.append("candidate_valid_domain_fraction_low")
        cid = c["candidate_id"]
        result["candidates"].append(dict(candidate_id=int(cid) if isinstance(cid, Integral) else cid,
            iou_before=ious[0], iou_after=ious[1], score=min(ious), query_point_coverage=coverage,
            fit_query_point_coverage=fit_coverage, holdout_query_point_coverage=holdout_coverage,
            spatial_cell_query_coverage=cell_metrics, spatial_cell_mean_query_coverage=cell_mean,
            outside_dilation_before=outside[0], outside_dilation_after=outside[1],
            valid_domain_fraction=valid_fraction, valid_domain_pixels=inside, passed=not reasons, reasons=reasons))
    if result["reasons"]:
        return result
    passing = [c for c in result["candidates"] if c["passed"]]
    if not passing:
        result["reasons"].append("no_candidate_passed")
        return result
    winner = max(passing, key=lambda c: c["score"])
    # Every schema-valid competitor matters, even if it failed another gate.
    others = [c["score"] for c in result["candidates"] if c is not winner]
    margin = winner["score"] - max(others) if others else 1.0
    result["winner_margin"] = float(margin)
    if margin + 1e-12 < WINNER_MARGIN_MIN:
        result["reasons"].append("ambiguous_candidate_scores")
        return result
    result.update(status="TEMPORAL_CANDIDATE_SELECTED", selected_candidate_id=winner["candidate_id"])
    return result
