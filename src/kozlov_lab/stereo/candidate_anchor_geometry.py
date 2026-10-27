"""Saved-camera diagnostics from original foreground anchors only.

The caller binds rows to immutable source observations and saved cameras. This
kernel checks their data contract; it cannot authenticate those bindings.
"""
from copy import deepcopy
import math


def _integer(value):
    return type(value) is int and value >= 0


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _vector(value, size):
    return isinstance(value, list) and len(value) == size and all(_number(v) for v in value)


def evaluate_anchor_group(observations: list[dict]) -> dict:
    """Use frozen strong-flow bounds (FB <=1px, LK <=25) and strict geometry.

    Two-view output is provisional. Fixed-camera LOO is conditional internal
    consistency, including when three or more views satisfy the strict gate.
    """
    out = dict(status="INVALID_ORIGINAL_ANCHORS", anchor_count=len(observations) if isinstance(observations, list) else 0,
               xyz=None, raw_helper_result=None, reasons=[], publishable_for_geometry=False,
               geometry_accepted=False, independent_quality_truth=False, scale_status="unknown",
               feature_identity_domain="candidate_anchor_adapter_row_not_native_feature_index",
               evidence_scope="original_foreground_fixed_saved_camera_conditional_internal_consistency",
               loo_scope="fixed_camera_internal_consistency_not_independent_validation",
               strong_flow_bounds=dict(fb_error_px_max=1., lk_error_max=25.))
    def reject(reason, status="INVALID_ORIGINAL_ANCHORS"):
        out["status"] = status
        out["reasons"].append(reason)
        return out
    if not isinstance(observations, list):
        return reject("OBSERVATIONS_MUST_BE_LIST")
    identities = set()
    frames, images, indexes = set(), set(), set()
    adapted = []
    for i, row in enumerate(observations):
        if not isinstance(row, dict):
            return reject("INVALID_OBSERVATION")
        if any(not _integer(row.get(k)) for k in ("input_index", "image_id", "track_id", "identity_segment")):
            return reject("INVALID_IDENTITY")
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("frame_id", "model_key")):
            return reject("INVALID_IDENTITY")
        identity = row["track_id"], row["identity_segment"], row["model_key"]
        identities.add(identity)
        if len(identities) != 1:
            return reject("MIXED_TRACK_SEGMENT_OR_MODEL")
        if row["frame_id"] in frames or row["image_id"] in images or row["input_index"] in indexes:
            return reject("DUPLICATE_FRAME_OR_IMAGE")
        frames.add(row["frame_id"]); images.add(row["image_id"]); indexes.add(row["input_index"])
        if (row.get("source_label") != "foreground" or row.get("source_target_verified") is not True
                or row.get("original_mask_inside") is not True):
            return reject("CONDITIONAL_SCOPE_NOT_ORIGINAL_FOREGROUND", "CONDITIONAL_SCOPE_NOT_ORIGINAL_FOREGROUND")
        if row.get("strong") is not True:
            return reject("NOT_STRONG_ORIGINAL_ANCHOR")
        for field, bound in (("fb_error_px", 1.), ("lk_error", 25.)):
            value = row.get(field)
            if not _number(value) or not 0 <= value <= bound:
                return reject("INVALID_OR_EXCEEDED_FLOW_ERROR")
        if row.get("camera_model") != "SIMPLE_RADIAL":
            return reject("UNSUPPORTED_OR_MISSING_CAMERA_MODEL")
        if not _vector(row.get("params"), 4) or row["params"][0] <= 0 or row["params"][3] != 0:
            return reject("UNSUPPORTED_OR_INVALID_INTRINSICS")
        if (not _vector(row.get("raw_pixels"), 2) or not _vector(row.get("t"), 3)
                or not isinstance(row.get("R"), list) or len(row["R"]) != 3
                or not all(_vector(r, 3) for r in row["R"])):
            return reject("INVALID_CAMERA_OR_PIXELS")
        adapted.append(dict(image_id=row["image_id"], feature_row=i, raw_pixels=deepcopy(row["raw_pixels"]),
                            camera_model=row["camera_model"], params=deepcopy(row["params"]),
                            R=deepcopy(row["R"]), t=deepcopy(row["t"]), foreground=True))
    if len(observations) < 2:
        return reject("FEWER_THAN_TWO_ORIGINAL_ANCHORS", "INSUFFICIENT_ORIGINAL_ANCHORS")
    if len(observations) > 32:
        return reject("OBSERVATION_COUNT_OUT_OF_BOUNDS")
    from integrations.demo_pipeline.coverage_triangulation import triangulate_candidate
    raw = triangulate_candidate(adapted)
    out["raw_helper_result"] = deepcopy(raw)
    out["adapter_rows"] = [dict(adapter_row=i, input_index=r["input_index"], image_id=r["image_id"],
                                frame_id=r["frame_id"], track_id=r["track_id"],
                                identity_segment=r["identity_segment"], model_key=r["model_key"])
                           for i, r in enumerate(observations)]
    if len(observations) == 2 and raw["tier"] == "TWO_VIEW_PROVISIONAL":
        out.update(status="TWO_VIEW_PROVISIONAL", xyz=deepcopy(raw["xyz"]))
    elif len(observations) >= 3 and raw["tier"] == "STRICT_SUPPORTED":
        out.update(status="CONDITIONAL_ANCHOR_SUPPORTED", xyz=deepcopy(raw["xyz"]))
    else:
        out.update(status="STRICT_GATE_REJECTED", reasons=list(raw["reasons"]))
        if raw.get("strict_decision"):
            out["reasons"] = list(dict.fromkeys(out["reasons"] + raw["strict_decision"]["reasons"]))
    return out
