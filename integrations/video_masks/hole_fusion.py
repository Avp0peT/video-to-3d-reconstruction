"""Conservative, CPU-only repair of enclosed mask holes using independent roots."""
import math
import numbers

import numpy as np

POLICY_VERSION = "independent-root-enclosed-holes-v1"


def _mask(value, shape=None):
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError("mask must be a uint8 ndarray")
    if value.ndim != 2 or 0 in value.shape or (shape is not None and value.shape != shape):
        raise ValueError("mask must be nonempty HxW with matching shape")
    if not np.all((value == 0) | (value == 255)):
        raise ValueError("mask must contain only 0 and 255")
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return int(value)


def union_masks(base, supports):
    """Return the literal pixel OR of base and every supplied binary mask.

    Roots describe provenance only; they are neither votes nor eligibility
    filters. Union can admit false positives and is not accuracy validation.
    """
    _mask(base)
    if not isinstance(supports, (list, tuple)):
        raise ValueError("supports must be a list or tuple")
    output = base.copy()
    roots = set()
    for support in supports:
        if not isinstance(support, dict) or "mask" not in support or "root_seed_decode_index" not in support:
            raise ValueError("support requires mask and root_seed_decode_index")
        mask = _mask(support["mask"], base.shape)
        roots.add(_integer(support["root_seed_decode_index"], "root_seed_decode_index"))
        np.bitwise_or(output, mask, out=output)
    before, after = hole_components(base), hole_components(output)
    base_count = int(np.count_nonzero(base))
    output_count = int(np.count_nonzero(output))
    return output, {"policy_version": "literal-mask-union-v1", "method": "pixel_or",
                    "base_foreground_pixels": base_count,
                    "output_foreground_pixels": output_count,
                    "added_pixels": output_count - base_count, "removed_pixels": 0,
                    "holes_before": len(before), "holes_after": len(after),
                    "hole_pixels_before": sum(c["pixels"] for c in before),
                    "hole_pixels_after": sum(c["pixels"] for c in after),
                    "support_roots": sorted(roots), "support_count": len(supports),
                    "roots_are_votes": False, "accuracy_verified": False,
                    "can_merge_false_positives": True}


def hole_components(base):
    """Return enclosed 8-connected background components as row runs [y,x0,x1).

    Run union-find avoids a Python visit for every background pixel. Border
    components are excluded, including those connected through diagonal channels.
    """
    _mask(base)
    height, width = base.shape
    parents, runs, exterior = [], [], []

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    previous = []
    for y in range(height):
        padded = np.empty(width + 2, dtype=np.int8)
        padded[0] = padded[-1] = 0
        padded[1:-1] = base[y] == 0
        transitions = np.diff(padded)
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        current = []
        left = 0
        for x0, x1 in zip(starts.tolist(), ends.tolist()):
            i = len(parents)
            parents.append(i)
            runs.append((y, x0, x1))
            exterior.append(y == 0 or y == height - 1 or x0 == 0 or x1 == width)
            current.append(i)
            while left < len(previous) and runs[previous[left]][2] < x0:
                left += 1
            k = left
            while k < len(previous) and runs[previous[k]][1] <= x1:
                a, b = root(i), root(previous[k])
                if a != b:
                    parents[b] = a
                k += 1
        previous = current
    grouped = {}
    for i, run in enumerate(runs):
        r = root(i)
        entry = grouped.setdefault(r, {"runs": [], "exterior": False})
        entry["runs"].append(run)
        entry["exterior"] |= exterior[i]
    result = []
    for entry in grouped.values():
        if entry["exterior"]:
            continue
        component_runs = entry["runs"]
        result.append({"runs": component_runs,
                       "pixels": sum(b - a for _, a, b in component_runs),
                       "bbox": [min(a for _, a, _ in component_runs), component_runs[0][0],
                                max(b for _, _, b in component_runs), component_runs[-1][0] + 1]})
    return sorted(result, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def repair_holes(base, supports, base_root, *, max_hole_pixels=1024,
                 max_hole_foreground_ratio=0.005, min_support_fraction=0.9,
                 min_global_iou=0.8, max_total_added_fraction=0.05):
    """Return a fresh uint8 mask and JSONable audit; never mutate inputs.

    Each component uses one independent original root. Repeated masks from a
    root are alternatives, never additional votes. Selection maximizes supported
    pixels, then global IoU, then smallest root; equal ties have equal additions.
    """
    _mask(base)
    base_root = _integer(base_root, "base_root")
    max_hole_pixels = _integer(max_hole_pixels, "max_hole_pixels")
    fractions = {"max_hole_foreground_ratio": max_hole_foreground_ratio,
                 "min_support_fraction": min_support_fraction,
                 "min_global_iou": min_global_iou,
                 "max_total_added_fraction": max_total_added_fraction}
    for name, value in fractions.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(name + " must be finite and between 0 and 1")
    foreground = base == 255
    fg_count = int(np.count_nonzero(foreground))
    candidates = []
    if not isinstance(supports, (list, tuple)):
        raise ValueError("supports must be a list or tuple")
    for support in supports:
        if not isinstance(support, dict) or "mask" not in support or "root_seed_decode_index" not in support:
            raise ValueError("support requires mask and root_seed_decode_index")
        mask = _mask(support["mask"], base.shape)
        seed = _integer(support["root_seed_decode_index"], "root_seed_decode_index")
        union = int(np.count_nonzero(foreground | (mask == 255)))
        iou = float(np.count_nonzero(foreground & (mask == 255)) / union) if union else 1.0
        if seed != base_root and iou >= min_global_iou:
            candidates.append((mask, seed, iou))
    holes = hole_components(base)
    limit = min(max_hole_pixels, math.floor(fg_count * max_hole_foreground_ratio))
    output = base.copy()
    details = []
    total_added = 0
    eligible = 0
    for index, component in enumerate(holes):
        item = {"component": index, "bbox": component["bbox"], "pixels": component["pixels"],
                "support_root": None, "global_iou": None, "support_fraction": 0.0, "added": 0}
        details.append(item)
        if component["pixels"] > limit:
            item["reason"] = "hole_too_large"
            continue
        eligible += 1
        choices = []
        for mask, seed, iou in candidates:
            count = sum(int(np.count_nonzero(mask[y, a:b])) for y, a, b in component["runs"])
            # Byte tie-break ensures order independence even for repeated roots.
            evidence = b"".join(mask[y, a:b].tobytes() for y, a, b in component["runs"])
            choices.append((count, iou, -seed, evidence, mask))
        if not choices:
            item["reason"] = "no_independent_eligible_support"
            continue
        count, iou, neg_seed, _, mask = max(choices, key=lambda choice: choice[:4])
        fraction = count / component["pixels"]
        item.update(support_root=-neg_seed, global_iou=iou, support_fraction=fraction)
        if fraction < min_support_fraction or count == 0:
            item["reason"] = "insufficient_hole_support"
            continue
        for y, a, b in component["runs"]:
            output[y, a:b] = mask[y, a:b]
        item.update(reason="repaired", added=count)
        total_added += count
    cap_rejected = total_added > fg_count * max_total_added_fraction
    if cap_rejected:
        output = base.copy()
        for item in details:
            if item["reason"] == "repaired":
                item.update(reason="total_added_cap_exceeded", added=0)
        total_added = 0
    after = hole_components(output)
    filled = sum(item["reason"] == "repaired" for item in details)
    report = {"policy_version": POLICY_VERSION, "base_root": base_root,
              "base_foreground_pixels": fg_count, "hole_pixel_limit": limit,
              "policy": {"max_hole_pixels": max_hole_pixels, **{k: float(v) for k, v in fractions.items()}},
              "holes_before": len(holes), "holes_after": len(after),
              "hole_pixels_before": sum(c["pixels"] for c in holes),
              "hole_pixels_after": sum(c["pixels"] for c in after),
              "eligible_components": eligible, "filled_components": filled,
              "rejected_components": len(holes) - filled, "added_pixels": total_added,
              "total_cap_rejected": cap_rejected, "components": details}
    return output, report
