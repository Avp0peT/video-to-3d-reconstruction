"""object-mask-v1 contract constants (authority: docs/contracts_object_masks_v1.md).

This module intentionally contains no third-party imports so that the whole
adapter can be imported and unit-tested without torch / sam3 present.
"""

from __future__ import annotations

CONTRACT_VERSION = "object-mask-v1"
PHOTO_IO_CONTRACT_VERSION = "photo-io-v1"

#: image domain of the inputs consumed here. Distortion-corrected consumers must
#: register their own transform; this mask must never be labelled undistorted.
IMAGE_DOMAIN = "photo_io_canonical"

SELECTION_POLICY = "unique_candidate_or_explicit"
MASK_SOURCE = "automatic_sam3"

RUN_STATUSES = (
    "SUCCESS",
    "PARTIAL",
    "INPUT_ERROR",
    "BACKEND_UNAVAILABLE",
    "BACKEND_ERROR",
    "AUTH_REQUIRED",
    "RESOURCE_LIMIT",
)

FRAME_STATUSES = (
    "OK",
    "NO_MATCH",
    "NEEDS_SELECTION",
    "INPUT_ERROR",
    "BACKEND_ERROR",
    "RESOURCE_LIMIT",
    "NOT_RUN",
)

#: frames whose status counts as `error` in the counts block
ERROR_FRAME_STATUSES = ("INPUT_ERROR", "BACKEND_ERROR", "RESOURCE_LIMIT")

EXIT_CODES = {
    "SUCCESS": 0,
    "PARTIAL": 1,
    "INPUT_ERROR": 2,
    "BACKEND_UNAVAILABLE": 3,
    "AUTH_REQUIRED": 3,
    "BACKEND_ERROR": 4,
    "RESOURCE_LIMIT": 5,
}

#: per-candidate score semantics. SAM3 scores are sigmoid(pred_logits) multiplied
#: by sigmoid(presence_logit); this is a ranking score, NOT a calibrated
#: probability and NOT a mask IoU. See sam3/model/sam3_image_processor.py.
SAM3_SCORE_SEMANTICS = "presence_calibrated_sigmoid_product_not_probability"

#: SAM3 binarises its own resampled mask logits at this value
#: (sam3/model/sam3_image_processor.py: `state["masks"] = out_masks > 0.5`).
SAM3_MASK_BINARIZATION = "model_side_sigmoid_logits_gt_0.5_after_bilinear_resample"

#: internal forward grid of Sam3Processor (square, aspect NOT preserved)
SAM3_INTERNAL_RESOLUTION = 1008

#: quality / cross-view verdicts must stay NOT_EVALUATED for this task
NOT_EVALUATED = "NOT_EVALUATED"

#: evidence tiers. Only the last one may be reported as "SAM3 ran".
EVIDENCE_LEVEL_REAL = "sam3_real_inference"
EVIDENCE_LEVEL_FAKE = "fake_backend_interface_test"

FAKE_BACKEND_NAME = "fake"
SAM3_BACKEND_NAME = "sam3"

#: default wall-clock budgets (seconds). The load budget and the batch budget
#: share ONE absolute origin: a later phase never resets the total.
LOAD_BUDGET_SECONDS = 300.0
FRAME_BUDGET_SECONDS = 300.0
BATCH_BUDGET_SECONDS = 900.0
#: minimum sampling period of the soft monitor (seconds)
MONITOR_MIN_INTERVAL_SECONDS = 0.05
#: best-effort default sampling period of the background soft monitor (seconds)
MONITOR_INTERVAL_DEFAULT_SECONDS = 0.25


def counts_from_frames(frames: list[dict]) -> dict:
    """Recompute the `counts` block from the frames list (must stay consistent)."""
    statuses = [f["status"] for f in frames]
    return {
        "requested": len(frames),
        "success": statuses.count("OK"),
        "no_match": statuses.count("NO_MATCH"),
        "needs_selection": statuses.count("NEEDS_SELECTION"),
        "error": sum(statuses.count(s) for s in ERROR_FRAME_STATUSES),
        "not_run": statuses.count("NOT_RUN"),
    }


def counts_consistent(counts: dict, frames: list[dict]) -> list[str]:
    """Return a list of human-readable inconsistencies (empty == consistent)."""
    problems = []
    recomputed = counts_from_frames(frames)
    for key, value in recomputed.items():
        if counts.get(key) != value:
            problems.append(f"counts[{key}]={counts.get(key)!r} != recomputed {value!r}")
    total = sum(counts.get(k, 0) for k in ("success", "no_match", "needs_selection", "error", "not_run"))
    if counts.get("requested") != total:
        problems.append(f"requested={counts.get('requested')!r} != sum of parts {total}")
    if counts.get("requested") != len(frames):
        problems.append(f"requested={counts.get('requested')!r} != len(frames) {len(frames)}")
    for key in ("requested", "success", "no_match", "needs_selection", "error", "not_run"):
        value = counts.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            problems.append(f"counts[{key}] must be a non-negative int, got {value!r}")
    return problems
