"""Pure, object-independent proposals for intermediate video frames.

This module does not decode images, execute geometry, or assess reconstruction
quality. The default caps are resource limits, not acceptance thresholds.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
import math
from typing import Any


POLICY = {
    "schema": "object-region-bridge-policy-v1",
    "native_max_frames": 96,
    "max_added": 24,
}
_VALID_DOMAIN = "CAMERA_DOMAIN_VALID"


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; bool is not allowed")
    return value


def _sequence(value: Any, name: str) -> Sequence:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    return value


def _configuration(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    config = dict(POLICY)
    if policy is not None:
        if not isinstance(policy, Mapping):
            raise ValueError("policy must be a mapping")
        if set(policy) - set(POLICY):
            raise ValueError("unknown policy keys")
        config.update(policy)
    if config["schema"] != POLICY["schema"]:
        raise ValueError("unsupported policy schema")
    for key, minimum in (("native_max_frames", 1), ("max_added", 0)):
        value = _integer(config[key], key, minimum=minimum)
        if value > POLICY[key]:
            raise ValueError(f"{key} may only tighten the default resource limit")
    return config


def _timeline(timeline: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict], str]:
    rows = []
    endpoints = []
    previous_decode = previous_input = -1
    for position, row in enumerate(_sequence(timeline, "timeline")):
        if not isinstance(row, Mapping):
            raise ValueError("each timeline row must be a mapping")
        decode = _integer(row.get("decode_index"), f"timeline[{position}].decode_index")
        if decode <= previous_decode:
            raise ValueError("decode_index must be unique and strictly increasing")
        previous_decode = decode
        item = {"decode_index": decode, "input_index": row.get("input_index")}
        if item["input_index"] is not None:
            index = _integer(item["input_index"], f"timeline[{position}].input_index")
            if index <= previous_input:
                raise ValueError("endpoint input_index must be unique and strictly increasing")
            previous_input = index
            endpoints.append(item)
        if "timestamp_seconds" in row:
            time = row["timestamp_seconds"]
            if (type(time) not in (int, float) or time < 0
                    or (type(time) is float and not math.isfinite(time))):
                raise ValueError("timestamp_seconds must be a finite nonnegative number, not bool")
            item["timestamp_seconds"] = time
        rows.append(item)
    timed = ["timestamp_seconds" in row for row in rows]
    if any(timed) and not all(timed):
        raise ValueError("timestamp_seconds must be present on every row or none")
    coordinate = "timestamp_seconds" if any(timed) else "decode_index"
    previous = None
    for row in rows:
        row["coordinate"] = Fraction(str(row[coordinate]))
        if previous is not None and row["coordinate"] <= previous:
            raise ValueError("timestamp_seconds must be strictly increasing")
        previous = row["coordinate"]
    return rows, endpoints, coordinate


def _models(models: Sequence[Mapping[str, Any]], endpoint_domain: set[int]) -> tuple[list[dict], list[dict], str]:
    normalized = []
    seen_ids = set()
    labels = set()
    for position, model in enumerate(_sequence(models, "models")):
        if not isinstance(model, Mapping):
            raise ValueError("each model must be a mapping")
        model_id = _integer(model.get("model_id"), f"models[{position}].model_id")
        if model_id in seen_ids:
            raise ValueError("model_id must be unique within the supplied model set")
        seen_ids.add(model_id)
        label = model.get("label")
        if label is not None and (type(label) is not str or label not in ("new", "bridge")):
            raise ValueError("model label must be new or bridge; supply an explicitly filtered model set")
        labels.add(label)
        status = model.get("camera_domain_status")
        if type(status) is not str or not status.strip():
            raise ValueError("camera_domain_status must be a nonempty string")
        indices = []
        previous = -1
        for view in _sequence(model.get("registered_indices"), "registered_indices"):
            index = _integer(view, "registered_indices entry")
            if index <= previous:
                raise ValueError("registered_indices must be unique and strictly increasing")
            if index not in endpoint_domain:
                raise ValueError("model view reference is outside the endpoint input_index domain")
            previous = index
            indices.append(index)
        normalized.append({"model_id": model_id, "registered_indices": indices,
                           "camera_domain_status": status})
    if len(labels) > 1:
        raise ValueError("model labels must identify one model set; mixed labels are not allowed")
    valid = sorted((m for m in normalized if m["camera_domain_status"] == _VALID_DOMAIN),
                   key=lambda m: (-len(m["registered_indices"]), m["model_id"]))
    rejected = [dict(m, reason="CAMERA_DOMAIN_NOT_VALID") for m in normalized
                if m["camera_domain_status"] != _VALID_DOMAIN]
    return valid, sorted(rejected, key=lambda m: m["model_id"]), next(iter(labels), None) or "CALLER_FILTERED"


def select_intermediates(
    timeline: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Return sorted existing decode IDs proposed for addition, plus a ledger.

    ``timeline`` contains unique, increasing nonnegative integer ``decode_index``
    values. An optional non-null ``input_index`` identifies an existing endpoint;
    endpoint IDs must also increase but need not be consecutive. A complete,
    increasing ``timestamp_seconds`` field selects time-based thirds; otherwise
    decode IDs define thirds. Only actual intermediate rows can be selected.

    Models contain nonnegative integer ``model_id``, sorted unique endpoint
    ``registered_indices``, and a string ``camera_domain_status``. Only exact
    CAMERA_DOMAIN_VALID records establish registration. Labelled input must be
    one ``new`` or ``bridge`` set. Omit labels only after the caller filters the
    set; mixed sets are rejected. Overlap belongs to the largest valid model,
    with smaller model ID breaking ties, independent of model list order.

    Adjacent registered endpoints from different components define bridge gaps.
    Same-component endpoints define a recovery gap only when unregistered
    endpoints lie between them. Shorter endpoint spans are visited first. Each
    constituent original interval proposes its nearest available actual frame
    to each third, with smaller decode ID breaking ties. Already selected frames
    are excluded before the next choice. The caps can only be tightened.
    """
    config = _configuration(policy)
    rows, endpoints, coordinate = _timeline(timeline)
    valid, rejected, model_set = _models(models, {r["input_index"] for r in endpoints})
    groups = {position: next((m["model_id"] for m in valid
                              if row["input_index"] in m["registered_indices"]), None)
              for position, row in enumerate(endpoints)}
    registered = [p for p in groups if groups[p] is not None]
    gaps = []
    for left, right in zip(registered, registered[1:]):
        if groups[left] != groups[right]:
            reason = "DIFFERENT_VALID_COMPONENTS"
        elif right - left > 1:
            reason = "SAME_COMPONENT_ENCLOSES_UNREGISTERED_ENDPOINTS"
        else:
            continue
        gaps.append({"left_original_position": left, "right_original_position": right,
                     "span": right - left, "reason": reason,
                     "left_model_id": groups[left], "right_model_id": groups[right],
                     "unregistered_original_positions": list(range(left + 1, right))})
    gaps.sort(key=lambda gap: (gap["span"], gap["left_original_position"]))
    remaining = max(0, config["native_max_frames"] - len(endpoints))
    budget = min(config["max_added"], remaining)
    ledger = {"schema": "object-region-bridge-selection-v1", "policy": config,
              "resource_limits_are_quality_criteria": False,
              "geometry_executed": False, "quality_status": "NOT_EVALUATED",
              "model_set": model_set, "coordinate": coordinate,
              "endpoint_count": len(endpoints), "available_addition_budget": budget,
              "valid_model_order": [m["model_id"] for m in valid], "rejected_models": rejected,
              "group_by_original_position": groups, "gaps": gaps, "decisions": [],
              "unregistered_original_positions": [p for p in groups if groups[p] is None]}
    reasons = []
    if len(endpoints) < 2:
        reasons.append("INSUFFICIENT_ENDPOINT_DOMAIN")
    if not valid:
        reasons.append("NO_VALID_CAMERA_DOMAIN_MODEL")
    elif len(registered) < 2:
        reasons.append("INSUFFICIENT_REGISTERED_ENDPOINTS")
    if not gaps:
        reasons.append("NO_ELIGIBLE_GAPS")
    if len(endpoints) > config["native_max_frames"]:
        reasons.append("EXISTING_ENDPOINTS_EXCEED_NATIVE_RESOURCE_LIMIT")
    elif not budget:
        reasons.append("RESOURCE_BUDGET_EXHAUSTED")
    selected = set()
    intermediates = [r for r in rows if r["input_index"] is None]
    for gap in gaps:
        for position in range(gap["left_original_position"], gap["right_original_position"]):
            left, right = endpoints[position:position + 2]
            available = [r for r in intermediates
                         if left["decode_index"] < r["decode_index"] < right["decode_index"]]
            for numerator in (1, 2):
                target = left["coordinate"] + (right["coordinate"] - left["coordinate"]) * Fraction(numerator, 3)
                candidates = [r for r in available if r["decode_index"] not in selected]
                record = {"gap_reason": gap["reason"], "interval_positions": [position, position + 1],
                          "interval_decode_indices": [left["decode_index"], right["decode_index"]],
                          "fraction": [numerator, 3], "target_coordinate": str(target),
                          "decode_index": None}
                if not candidates:
                    record["decision"] = "NO_UNUSED_INTERMEDIATE_FRAME"
                else:
                    candidate = min(candidates, key=lambda r: (abs(r["coordinate"] - target), r["decode_index"]))
                    record["decode_index"] = candidate["decode_index"]
                    if len(selected) >= budget:
                        record["decision"] = "RESOURCE_BUDGET_EXHAUSTED"
                    else:
                        record["decision"] = "SELECTED"
                        selected.add(candidate["decode_index"])
                ledger["decisions"].append(record)
    if gaps and not selected and budget and not any(d["decode_index"] is not None for d in ledger["decisions"]):
        reasons.append("NO_AVAILABLE_INTERMEDIATE_FRAMES")
    ledger.update(selected_decode_indices=sorted(selected), selected_count=len(selected),
                  total_frame_count=len(endpoints) + len(selected),
                  resource_limit_satisfied=len(endpoints) + len(selected) <= config["native_max_frames"],
                  status="PROPOSED_ONLY" if selected else "NO_CANDIDATES", reasons=reasons)
    return sorted(selected), ledger
