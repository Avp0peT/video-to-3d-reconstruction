"""Bounded conditional point checks; fitted cameras are not independent truth.

Only SIMPLE_RADIAL with exactly zero k is admitted. This is a restricted
camera model assumption, not a claim of physical camera calibration.
"""
from __future__ import annotations

import numpy as np


def assess_point(xyz, observations, *, max_error_px=2.0, min_angle_deg=2.0,
                 max_loo_error_px=3.0):
    """Return finite JSON data; LOO refits xyz from pixels with cameras fixed.

    LOO means leave-one-observation-out conditional internal consistency.
    Cameras fitted from the same data make this unsuitable as independent
    geometry/scale/quality ground truth.
    """
    result = {
        "status": "REJECTED", "reasons": [], "reprojection": [], "loo": [],
        "max_angle_deg": None, "measurement_ray_angle_deg": None,
        "candidate_ray_angle_deg": None, "jacobian_rank": None,
        "jacobian_condition": None, "max_condition": 1e4, "source_identities": [],
        "camera_model": "SIMPLE_RADIAL", "camera_condition": "k_exactly_zero",
        "evidence_scope": "conditional_internal_consistency",
        "independent_quality_truth": False,
        "loo_method": "leave_one_observation_out_fixed_camera_DLT_from_pixels",
    }
    reasons = result["reasons"]

    def reject(reason):
        if reason not in reasons:
            reasons.append(reason)

    def array(value, shape):
        a = np.asarray(value, dtype=float)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError("invalid numeric array")
        return a

    try:
        thresholds = array([max_error_px, min_angle_deg, max_loo_error_px], (3,))
        if (thresholds < 0).any() or thresholds[1] > 180:
            raise ValueError("invalid thresholds")
        point = array(xyz, (3,))
    except (ValueError, TypeError, OverflowError):
        reject("INVALID_POINT_OR_THRESHOLDS")
        return result
    if not isinstance(observations, (list, tuple)) or not 3 <= len(observations) <= 32:
        reject("OBSERVATION_COUNT_OUT_OF_BOUNDS")
        return result
    cameras = []
    ids = set()
    try:
        for observation in observations:
            if not isinstance(observation, dict):
                raise ValueError("observation is not a mapping")
            image_id, row = observation["image_id"], observation["feature_row"]
            if type(image_id) is not int or type(row) is not int or image_id < 0 or row < 0:
                raise ValueError("invalid identity")
            if image_id in ids:
                reject("DUPLICATE_IMAGE_ID")
            ids.add(image_id)
            result["source_identities"].append({"image_id": image_id, "feature_row": row})
            pixels = array(observation["raw_pixels"], (2,))
            params = array(observation["params"], (4,))
            rotation = array(observation["R"], (3, 3))
            translation = array(observation["t"], (3,))
            if params[0] <= 0 or params[3] != 0:
                reject("UNSUPPORTED_INTRINSICS")
            if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-7) or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-7):
                reject("INVALID_ROTATION")
            if observation["foreground"] is not True:
                reject("NON_FOREGROUND_OBSERVATION")
            cameras.append((rotation, translation, params, pixels))
    except (KeyError, ValueError, TypeError, OverflowError):
        reject("INVALID_OBSERVATION")
    if reasons:
        return result

    def project(p, camera):
        r, t, intrinsics, pixels = camera
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            local = r @ p + t
            projected = intrinsics[0] * local[:2] / local[2] + intrinsics[1:3]
            error = np.linalg.norm(projected - pixels)
        if not np.isfinite(local).all() or not np.isfinite(error):
            return None, None
        return float(local[2]), float(error)

    measured_rays = []
    jacobians = []
    rays = []
    for identity, camera in zip(result["source_identities"], cameras):
        depth, error = project(point, camera)
        result["reprojection"].append({**identity, "depth": depth, "error_px": error})
        if depth is None or depth <= 0:
            reject("NON_POSITIVE_DEPTH")
        if error is None or error > max_error_px:
            reject("REPROJECTION_ERROR_EXCEEDED")
        r, t, intrinsics, pixels = camera
        measured = r.T @ np.array([*( (pixels - intrinsics[1:3]) / intrinsics[0]), 1.0])
        measured_norm = np.linalg.norm(measured)
        if not np.isfinite(measured_norm) or measured_norm <= 0:
            reject("INVALID_MEASUREMENT_RAY")
        else:
            measured_rays.append(measured / measured_norm)
        if depth is not None and depth > 0:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                local = r @ point + t
                jacobians.extend(intrinsics[0] / depth *
                    (r[:2] - local[:2, None] / depth * r[2]))
        with np.errstate(over="ignore", invalid="ignore"):
            ray = point + r.T @ t
            norm = np.linalg.norm(ray)
        if not np.isfinite(norm) or norm <= 0:
            reject("INVALID_VIEW_RAY")
        else:
            rays.append(ray / norm)
    if len(rays) == len(cameras):
        angle = max(float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
                    for i, a in enumerate(rays) for b in rays[i + 1:])
        result["candidate_ray_angle_deg"] = angle
    if len(measured_rays) == len(cameras):
        angle = max(float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
                    for i, a in enumerate(measured_rays) for b in measured_rays[i + 1:])
        result["max_angle_deg"] = angle
        result["measurement_ray_angle_deg"] = angle
        if angle < min_angle_deg:
            reject("INSUFFICIENT_TRIANGULATION_ANGLE")
    if len(jacobians) == 2 * len(cameras) and np.isfinite(jacobians).all():
        try:
            singular = np.linalg.svd(jacobians, compute_uv=False)
            rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
            result["jacobian_rank"] = rank
            condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
            result["jacobian_condition"] = condition if condition is not None and np.isfinite(condition) else None
            if rank < 3 or condition is None or not np.isfinite(condition) or condition > 1e4:
                reject("ILL_CONDITIONED_POINT")
        except np.linalg.LinAlgError:
            reject("INVALID_JACOBIAN")
    else:
        reject("INVALID_JACOBIAN")

    for held_index, identity in enumerate(result["source_identities"]):
        detail = {**identity, "error_px": None, "depth": None,
                  "all_depths_positive": False, "status": "REJECTED"}
        result["loo"].append(detail)
        rows = []
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            for i, (r, t, params, pixels) in enumerate(cameras):
                if i == held_index:
                    continue
                u, v = (pixels - params[1:3]) / params[0]
                p = np.column_stack((r, t))
                rows.extend([u * p[2] - p[0], v * p[2] - p[1]])
        try:
            a = np.asarray(rows)
            if not np.isfinite(a).all():
                raise ValueError("nonfinite DLT")
            _, singular, vh = np.linalg.svd(a, full_matrices=False)
            if singular[-2] <= 1e-12 * singular[0] or abs(vh[-1, 3]) <= 1e-12:
                raise ValueError("degenerate DLT")
            fitted = vh[-1, :3] / vh[-1, 3]
            if not np.isfinite(fitted).all():
                raise ValueError("nonfinite triangulation")
            projections = [project(fitted, camera) for camera in cameras]
            depth, error = projections[held_index]
            positive = all(d is not None and d > 0 for d, _ in projections)
            detail.update(error_px=error, depth=depth, all_depths_positive=positive)
            if not positive:
                reject("LOO_NON_POSITIVE_DEPTH")
            if error is None or error > max_loo_error_px:
                reject("LOO_ERROR_EXCEEDED")
            if positive and error is not None and error <= max_loo_error_px:
                detail["status"] = "CONDITIONAL_POINT_SUPPORTED"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            reject("LOO_DEGENERATE_TRIANGULATION")
    if not reasons:
        result["status"] = "CONDITIONAL_POINT_SUPPORTED"
    return result
