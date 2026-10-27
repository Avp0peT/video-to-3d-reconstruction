"""Bounded adjacent-frame image evidence, not physical identity certification.

Local median flow is an engineering consistency check, not the book's exact
affine theorem. Callers must supply successful forward AND backward statuses
through ``flow_valid`` (or prefilter candidates before calling).
"""
from __future__ import annotations

import numpy as np


DEFAULT_PROFILE = {
    "fb_max_px": 1.0, "lk_error_max": 25.0, "patch_size": 9,
    "patch_ncc_min": 0.75, "patch_std_min": 5.0,
    "neighbor_radius_px": 64.0, "min_neighbors": 4,
    "neighbor_spread_min": 4.0, "motion_residual_max_px": 2.5,
}


def boundary_distance(mask, cv):
    """Distance to foreground, zero inside; empty masks have infinite distance."""
    foreground = np.asarray(mask) > 0
    if foreground.ndim != 2:
        raise ValueError("mask must be two dimensional")
    if not foreground.any():
        return np.full(foreground.shape, np.inf, dtype=np.float32)
    return cv.distanceTransform((~foreground).astype(np.uint8), cv.DIST_L2,
                                cv.DIST_MASK_PRECISE).astype(np.float32)


def assess_temporal_support(previous_gray, current_gray, previous_points,
                            current_points, fb_errors, lk_errors,
                            eligible_anchor, cv, profile=None, *, flow_valid=None):
    """Return measured support per candidate without modifying any input.

    ``eligible_anchor`` must come from an original foreground source fixed by
    the caller and an independently checked patch/flow chain. A prior supported
    result must never recursively upgrade a speculative point into an anchor.
    This remains candidate-only evidence, not physical identity verification.
    A point never counts itself. Profile overrides may tighten defaults only.
    """
    cfg = dict(DEFAULT_PROFILE)
    if profile is not None:
        if set(profile) - set(cfg):
            raise ValueError("unknown temporal support profile field")
        for key, value in profile.items():
            if not np.isfinite(value):
                raise ValueError("profile values must be finite")
            default = cfg[key]
            minimum = key in {"patch_ncc_min", "patch_std_min", "min_neighbors", "neighbor_spread_min"}
            if key == "patch_size":
                if value != 9:
                    raise ValueError("patch_size is fixed at 9")
            elif (minimum and value < default) or (not minimum and value > default) or value <= 0:
                raise ValueError("profile cannot relax bounded defaults")
            cfg[key] = value
    a, b = np.asarray(previous_gray), np.asarray(current_gray)
    p, q = np.asarray(previous_points, dtype=float), np.asarray(current_points, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("grays must be matching two dimensional images")
    if p.ndim != 2 or p.shape[1:] != (2,) or q.shape != p.shape:
        raise ValueError("points must have matching Nx2 shapes")
    n = len(p)
    fb, err = np.asarray(fb_errors).reshape(-1), np.asarray(lk_errors).reshape(-1)
    anchors = np.asarray(eligible_anchor, dtype=bool).reshape(-1)
    valid = np.ones(n, dtype=bool) if flow_valid is None else np.asarray(flow_valid, dtype=bool).reshape(-1)
    if any(len(x) != n for x in (fb, err, anchors, valid)):
        raise ValueError("candidate arrays must have equal lengths")
    rows, image_ok = [], np.zeros(n, dtype=bool)
    half = cfg["patch_size"] // 2
    height, width = a.shape
    for i in range(n):
        row = dict(supported=False, patch_ncc=None, patch_std=None,
                   neighbor_count=0, neighbor_spread=None, neighbor_indices=[],
                   agreeing_neighbor_indices=[],
                   local_motion_residual_px=None, reasons=[])
        rows.append(row)
        reasons = row["reasons"]
        if not valid[i]:
            reasons.append("flow_failed")
        if not (np.isfinite(p[i]).all() and np.isfinite(q[i]).all() and np.isfinite(fb[i]) and np.isfinite(err[i])):
            reasons.append("nonfinite_flow")
            continue
        if fb[i] < 0 or fb[i] > cfg["fb_max_px"]:
            reasons.append("fb_error")
        if err[i] < 0 or err[i] > cfg["lk_error_max"]:
            reasons.append("lk_error")
        if any(x < half or y < half or x > width - 1 - half or y > height - 1 - half for x, y in (p[i], q[i])):
            reasons.append("outside_patch")
            continue
        patches = [np.asarray(cv.getRectSubPix(im, (9, 9), tuple(map(float, pt))), dtype=float)
                   for im, pt in ((a, p[i]), (b, q[i]))]
        centered = [patch - patch.mean() for patch in patches]
        stds = [float(patch.std()) for patch in patches]
        row["patch_std"] = min(stds)
        denom = np.linalg.norm(centered[0]) * np.linalg.norm(centered[1])
        if denom > 0:
            row["patch_ncc"] = float(np.clip(np.sum(centered[0] * centered[1]) / denom, -1, 1))
        if min(stds) < cfg["patch_std_min"]:
            reasons.append("patch_low_texture")
        if row["patch_ncc"] is None or row["patch_ncc"] < cfg["patch_ncc_min"]:
            reasons.append("patch_appearance")
        image_ok[i] = not reasons
    anchor_ok = anchors & image_ok
    for i, row in enumerate(rows):
        if not image_ok[i]:
            continue
        indices = np.flatnonzero(anchor_ok & (np.linalg.norm(p - p[i], axis=1) <= cfg["neighbor_radius_px"]))
        indices = indices[indices != i]
        row["neighbor_indices"] = indices.tolist()
        row["neighbor_count"] = int(len(indices))
        if len(indices) < cfg["min_neighbors"]:
            row["reasons"].append("insufficient_neighbors")
            continue
        coordinates = p[indices] - p[indices].mean(axis=0)
        spread = float(np.linalg.eigvalsh(coordinates.T @ coordinates / len(indices))[0])
        row["neighbor_spread"] = spread
        if spread < cfg["neighbor_spread_min"]:
            row["reasons"].append("collinear_neighbors")
        flows = q[indices] - p[indices]
        median = np.median(flows, axis=0)
        residual = float(np.linalg.norm((q[i] - p[i]) - median))
        row["local_motion_residual_px"] = residual
        # Require four agreeing anchors, not merely a median of disagreeing flows.
        agreeing = np.linalg.norm(flows - median, axis=1) <= cfg["motion_residual_max_px"]
        row["agreeing_neighbor_indices"] = indices[agreeing].tolist()
        if residual > cfg["motion_residual_max_px"] or agreeing.sum() < cfg["min_neighbors"]:
            row["reasons"].append("local_motion_disagreement")
        elif np.linalg.eigvalsh(np.cov(p[indices[agreeing]].T, bias=True))[0] < cfg["neighbor_spread_min"]:
            row["reasons"].append("collinear_agreeing_neighbors")
        row["supported"] = not row["reasons"]
    return rows
