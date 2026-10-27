"""Small, dependency-free mesh helpers; no welding, repair, or native execution.

Faces must be three distinct, nonnegative integer indices (booleans rejected).
Only helpers receiving points can check their upper bounds. Metric checks do not
replace manifold, orientation, edge-connectivity, or self-intersection checks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path


def _sequence(value, name):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, not bool")
    return int(value)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def _points(points):
    result = []
    for index, point in enumerate(_sequence(points, "points")):
        point = _sequence(point, f"point[{index}]")
        if len(point) != 3:
            raise ValueError(f"point[{index}] must have exactly three coordinates")
        for axis, value in enumerate(point):
            _finite(value, f"point[{index}][{axis}]")
        # Preserve the supplied scalar values; no resampling or coordinate cast.
        result.append(tuple(point))
    return result


def _faces(faces, point_count=None):
    result = []
    for index, face in enumerate(_sequence(faces, "faces")):
        face = _sequence(face, f"face[{index}]")
        if len(face) != 3:
            raise ValueError(f"face[{index}] must contain exactly three indices")
        values = tuple(_integer(value, f"face[{index}] index") for value in face)
        if len(set(values)) != 3:
            raise ValueError(f"face[{index}] must contain three distinct indices")
        if point_count is not None and any(value >= point_count for value in values):
            raise ValueError(f"face[{index}] index is outside points")
        result.append(values)
    return result


def edge_components(faces):
    """Return sorted face-index components connected by shared undirected edges.

    Sharing a single vertex never connects components. Nonmanifold edges still
    connect all incident faces; this function does not certify manifoldness.
    Empty input returns an empty list.
    """
    triangles = _faces(faces)
    parents = list(range(len(triangles)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    edges = {}
    for face_id, (a, b, c) in enumerate(triangles):
        for left, right in ((a, b), (b, c), (c, a)):
            edge = (min(left, right), max(left, right))
            previous = edges.setdefault(edge, face_id)
            first, second = root(face_id), root(previous)
            if first != second:
                parents[max(first, second)] = min(first, second)
    groups = {}
    for face_id in range(len(triangles)):
        groups.setdefault(root(face_id), []).append(face_id)
    return sorted(groups.values(), key=lambda group: group[0])


def compact_component(points, faces, face_ids):
    """Select faces in requested order, remapping vertices in original-ID order.

    Returns (list of point tuples, list of face tuples, original vertex IDs).
    Coincident vertices remain distinct. Empty selection returns three lists.
    """
    vertices = _points(points)
    triangles = _faces(faces, len(vertices))
    selected = [_integer(index, "face_ids index")
                for index in _sequence(face_ids, "face_ids")]
    if len(set(selected)) != len(selected):
        raise ValueError("face_ids must not contain duplicates")
    if any(index >= len(triangles) for index in selected):
        raise ValueError("face_ids index is outside faces")
    source_vertex_ids = sorted({vertex for index in selected for vertex in triangles[index]})
    remap = {source: target for target, source in enumerate(source_vertex_ids)}
    new_points = [vertices[index] for index in source_vertex_ids]
    new_faces = [tuple(remap[vertex] for vertex in triangles[index]) for index in selected]
    return new_points, new_faces, source_vertex_ids


def write_double_ply(path, points, faces):
    """Exclusively create ASCII PLY with double XYZ and 17 significant digits.

    All geometry is validated before opening the file. Existing output raises
    FileExistsError. Parent directories must already exist. No native writer is
    used; face ordering and winding are preserved.
    """
    vertices = _points(points)
    triangles = _faces(faces, len(vertices))
    try:
        target = Path(path)
    except (TypeError, ValueError) as exc:
        raise ValueError("path must be a valid filesystem path") from exc
    with target.open("x", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(vertices)}\n")
        stream.write("property double x\nproperty double y\nproperty double z\n")
        stream.write(f"element face {len(triangles)}\n")
        stream.write("property list uchar int vertex_indices\nend_header\n")
        for point in vertices:
            stream.write(" ".join(format(value, ".17g") for value in point) + "\n")
        for a, b, c in triangles:
            stream.write(f"3 {a} {b} {c}\n")


def _stats(stats, name):
    if not isinstance(stats, Mapping):
        raise ValueError(f"{name} must be a mapping")
    required = ("components", "vertices", "edges", "faces", "signed_volume")
    absent = [key for key in required if key not in stats]
    if absent:
        raise ValueError(f"{name} missing fields: {', '.join(absent)}")
    result = {key: _integer(stats[key], f"{name}.{key}") for key in required[:-1]}
    result["signed_volume"] = _finite(stats["signed_volume"], f"{name}.signed_volume")
    result["euler"] = result["vertices"] - result["edges"] + result["faces"]
    return result


def validate_metrics(before_stats, after_stats, max_relative_volume_change):
    """Check single components, preserved Euler characteristic and oriented volume.

    Stats require components/vertices/edges/faces (nonnegative integers) and
    signed_volume (finite real). Invalid types, missing fields, or nonfinite
    values raise ValueError. Well-formed measurements failing the gates return
    passed=False. Relative volume change uses the positive *before* volume.
    """
    limit = _finite(max_relative_volume_change, "max_relative_volume_change")
    if limit < 0:
        raise ValueError("max_relative_volume_change must be nonnegative")
    before, after = _stats(before_stats, "before_stats"), _stats(after_stats, "after_stats")
    problems = []
    for name, stats in (("before", before), ("after", after)):
        if stats["components"] != 1:
            problems.append(f"{name} components must equal 1")
        if stats["signed_volume"] <= 0:
            problems.append(f"{name} signed_volume must be positive")
    if before["euler"] != after["euler"]:
        problems.append("Euler characteristic changed")
    relative = None
    if before["signed_volume"] > 0 and after["signed_volume"] > 0:
        relative = abs(after["signed_volume"] - before["signed_volume"]) / before["signed_volume"]
        if not math.isfinite(relative):
            relative = None
            problems.append("relative volume change is not finite")
        elif relative > limit:
            problems.append("relative volume change exceeds limit")
    return {
        "passed": not problems,
        "problems": problems,
        "before_euler": before["euler"],
        "after_euler": after["euler"],
        "before_signed_volume": before["signed_volume"],
        "after_signed_volume": after["signed_volume"],
        "relative_volume_change": relative,
        "max_relative_volume_change": limit,
        "scope": "metric checks only; not topology, orientation, or self-intersection certification",
    }
