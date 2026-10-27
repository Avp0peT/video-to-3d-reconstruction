"""One identity-bound AI-assisted positive pixel box per requested frame.

This module is standard-library only. A box is a prompt, never a truth mask.
"""
from __future__ import annotations

import hashlib
import json
import re


class VisualPromptError(ValueError):
    """Unsupported, malformed or unbound visual input."""


def _need(condition, message):
    if not condition:
        raise VisualPromptError(message)


def validate_visual_prompt(value: dict | None, indices: list[int], text_prompt: str | None) -> dict | None:
    if value is None:
        _need(isinstance(text_prompt, str), "text-only requests require text_prompt")
        return None
    _need(text_prompt is None, "positive_boxes requires text_prompt=None; mixed prompts are unsupported")
    _need(type(value) is dict and set(value) == {"kind", "image_domain", "annotation_source", "frames"},
          "visual_prompt must carry exactly kind, image_domain, annotation_source and frames")
    _need(value["kind"] == "positive_boxes", "only positive_boxes is supported")
    _need(value["image_domain"] == "photo_io_canonical", "box domain must be photo_io_canonical")
    _need(value["annotation_source"] == "ai_assisted", "box annotation_source must be ai_assisted")
    _need(type(indices) is list and bool(indices) and all(type(i) is int and i >= 0 for i in indices)
          and len(set(indices)) == len(indices), "requested indices must be unique nonnegative plain integers")
    frames = value["frames"]
    _need(type(frames) is list and len(frames) == len(indices), "exactly one positive box per requested frame is required")
    rows = []
    for row in frames:
        _need(type(row) is dict and set(row) == {"input_index", "frame_id", "image_sha256", "width", "height", "box_xyxy"},
              "invalid positive box frame fields")
        _need(type(row["input_index"]) is int and row["input_index"] >= 0, "input_index must be a nonnegative plain integer")
        _need(type(row["frame_id"]) is str and bool(row["frame_id"].strip()), "frame_id must be a nonempty string")
        _need(type(row["image_sha256"]) is str and re.fullmatch("[0-9a-f]{64}", row["image_sha256"]) is not None,
              "image_sha256 must be lowercase SHA256")
        width, height = row["width"], row["height"]
        _need(type(width) is int and width > 0 and type(height) is int and height > 0,
              "width and height must be positive plain integers")
        box = row["box_xyxy"]
        _need(type(box) is list and len(box) == 4 and all(type(x) is int for x in box),
              "box_xyxy must contain four integer pixels, excluding bool")
        x0, y0, x1, y1 = box
        _need(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, "box_xyxy is reversed, empty or outside canonical image")
        rows.append({**row, "box_xyxy": list(box)})
    ids = [r["input_index"] for r in rows]
    _need(len(set(ids)) == len(ids) and set(ids) == set(indices), "box frame membership must equal requested indices")
    # Preserve submitted frame order/values in the manifest; never mutate caller data.
    return {"kind": "positive_boxes", "image_domain": "photo_io_canonical",
            "annotation_source": "ai_assisted", "frames": rows}


def visual_prompt_sha256(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def frame_box(value: dict, input_index: int) -> dict:
    _need(type(input_index) is int, "input_index must be a plain integer")
    found = [r for r in value["frames"] if r["input_index"] == input_index]
    _need(len(found) == 1, "no unique positive box for this input_index")
    return found[0]


def validate_bound_frame(value: dict, frame) -> None:
    row = frame_box(value, frame.input_index)
    _need(frame.error is None and frame.source_image is not None, "box frame has no valid canonical image")
    _need(type(frame.width) is int and type(frame.height) is int, "canonical dimensions must be plain integers")
    for key, actual in (("frame_id", frame.frame_id), ("image_sha256", frame.source_image_sha256),
                        ("width", frame.width), ("height", frame.height)):
        _need(row[key] == actual, "positive box canonical identity drift: " + key)


def normalized_box(row: dict) -> list[float]:
    x0, y0, x1, y1 = row["box_xyxy"]
    return [(x0 + x1) / (2 * row["width"]), (y0 + y1) / (2 * row["height"]),
            (x1 - x0) / row["width"], (y1 - y0) / row["height"]]
