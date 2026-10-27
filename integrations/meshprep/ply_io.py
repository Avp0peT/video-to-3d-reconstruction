"""Read a bounded triangle-only PLY without reducing double coordinates.

This deliberately restricted reader never repairs, triangulates, merges points,
or imports a native geometry library. Additional scalar attributes are parsed
and range-checked, then discarded with their declarations recorded in metadata.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
import re
import struct


MAX_HEADER_BYTES = 64 * 1024
_INTEGER = re.compile(rb"[+-]?[0-9]+\Z")
_FLOAT = re.compile(
    rb"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf(?:inity)?|nan)\Z",
    re.IGNORECASE,
)
# PLY 1.0 scalar spellings and their commonly used fixed-width aliases.
_TYPES = {
    "char": ("b", -128, 127), "int8": ("b", -128, 127),
    "uchar": ("B", 0, 255), "uint8": ("B", 0, 255),
    "short": ("h", -32768, 32767), "int16": ("h", -32768, 32767),
    "ushort": ("H", 0, 65535), "uint16": ("H", 0, 65535),
    "int": ("i", -2147483648, 2147483647), "int32": ("i", -2147483648, 2147483647),
    "uint": ("I", 0, 4294967295), "uint32": ("I", 0, 4294967295),
    "float": ("f", None, None), "float32": ("f", None, None),
    "double": ("d", None, None), "float64": ("d", None, None),
}
_FORMATS = {"ascii", "binary_little_endian", "binary_big_endian"}


def _type(name: str):
    if name not in _TYPES:
        raise ValueError(f"unsupported PLY scalar type: {name}")
    return _TYPES[name]


def _header(data: bytes, max_faces: int, max_vertices: int):
    stream = io.BytesIO(data)

    def line():
        remaining = MAX_HEADER_BYTES - stream.tell()
        if remaining <= 0:
            raise ValueError("PLY header exceeds 64 KiB")
        raw = stream.readline(remaining + 1)
        if stream.tell() > MAX_HEADER_BYTES:
            raise ValueError("PLY header exceeds 64 KiB")
        if not raw or not raw.endswith(b"\n"):
            raise ValueError("truncated or unterminated PLY header")
        try:
            return raw.rstrip(b"\r\n").decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("PLY header must be ASCII") from exc

    if line() != "ply":
        raise ValueError("missing PLY magic")
    encoding = None
    elements = []
    current = None
    while True:
        parts = line().split()
        if not parts:
            raise ValueError("empty PLY header directive")
        directive = parts[0]
        if directive in ("comment", "obj_info"):
            continue
        if directive == "format":
            if encoding is not None or elements or len(parts) != 3 or parts[1] not in _FORMATS or parts[2] != "1.0":
                raise ValueError("invalid, duplicate, or misplaced PLY format")
            encoding = parts[1]
        elif directive == "element":
            if encoding is None or len(parts) != 3 or parts[1] not in ("vertex", "face"):
                raise ValueError("only vertex and face elements are supported")
            if any(item["name"] == parts[1] for item in elements):
                raise ValueError("duplicate PLY element")
            if not parts[2].isascii() or not parts[2].isdecimal():
                raise ValueError("element count must be a nonnegative integer")
            count = int(parts[2])
            limit = max_vertices if parts[1] == "vertex" else max_faces
            if count > limit:
                raise ValueError(f"{parts[1]} count exceeds budget")
            current = {"name": parts[1], "count": count, "properties": []}
            elements.append(current)
        elif directive == "property":
            if current is None:
                raise ValueError("property without element")
            if len(parts) == 3 and parts[1] != "list":
                _type(parts[1])
                prop = {"kind": "scalar", "type": parts[1], "name": parts[2]}
            elif len(parts) == 5 and parts[1] == "list":
                if current["name"] != "face" or parts[4] not in ("vertex_indices", "vertex_index"):
                    raise ValueError("only face vertex index lists are supported")
                if _type(parts[2])[1] is None or _type(parts[3])[1] is None:
                    raise ValueError("face list count and indices must use integer types")
                prop = {"kind": "list", "count_type": parts[2], "item_type": parts[3], "name": parts[4]}
            else:
                raise ValueError("invalid PLY property declaration")
            if any(p["name"] == prop["name"] for p in current["properties"]):
                raise ValueError("duplicate PLY property")
            current["properties"].append(prop)
        elif directive == "end_header":
            if len(parts) != 1:
                raise ValueError("invalid end_header directive")
            break
        else:
            raise ValueError(f"unsupported PLY header directive: {directive}")
    if encoding is None or {e["name"] for e in elements} != {"vertex", "face"}:
        raise ValueError("format and exactly one vertex and face element are required")
    vertex = next(e for e in elements if e["name"] == "vertex")
    face = next(e for e in elements if e["name"] == "face")
    coordinates = {p["name"]: p for p in vertex["properties"] if p["name"] in ("x", "y", "z")}
    if set(coordinates) != {"x", "y", "z"} or any(_type(p["type"])[1] is not None for p in coordinates.values()):
        raise ValueError("vertex x/y/z must be scalar float or double")
    lists = [p for p in face["properties"] if p["kind"] == "list"]
    if len(lists) != 1:
        raise ValueError("face requires exactly one integer vertex index list")
    return encoding, elements, stream.tell(), vertex["count"], coordinates


class _Values:
    def __init__(self, data: bytes, offset: int, encoding: str):
        self.data = data
        self.offset = offset
        self.tokens = iter(re.finditer(rb"\S+", data[offset:])) if encoding == "ascii" else None
        self.endian = "<" if encoding == "binary_little_endian" else ">"

    def scalar(self, name: str):
        code, minimum, maximum = _type(name)
        if self.tokens is None:
            size = struct.calcsize(code)
            if self.offset + size > len(self.data):
                raise ValueError("truncated binary PLY payload")
            value = struct.unpack_from(self.endian + code, self.data, self.offset)[0]
            self.offset += size
            return value
        match = next(self.tokens, None)
        if match is None:
            raise ValueError("truncated ASCII PLY payload")
        token = match.group()
        if minimum is not None:
            if not _INTEGER.fullmatch(token):
                raise ValueError("invalid integer PLY scalar")
            value = int(token)
            if not minimum <= value <= maximum:
                raise ValueError("integer PLY scalar outside declared range")
            return value
        if not _FLOAT.fullmatch(token):
            raise ValueError("invalid floating PLY scalar")
        value = float(token)
        # Check the declared storage range without quantizing ASCII coordinates.
        if math.isfinite(value) and code == "f" and abs(value) > 3.4028234663852886e38:
            raise ValueError("floating PLY scalar outside declared range")
        return value

    def finish(self):
        if self.tokens is not None:
            if next(self.tokens, None) is not None:
                raise ValueError("extra ASCII PLY payload")
        elif self.offset != len(self.data):
            raise ValueError("extra binary PLY payload")


def read_ply(path, max_bytes=64 * 1024 * 1024, max_faces=200000, max_vertices=400000):
    """Return ``(points, triangle_indices, metadata)``; malformed input raises ValueError.

    Both vertex and face elements are required, in either declared order. Empty
    elements are parsed faithfully; downstream topology gates decide usability.
    Additional scalar attributes are discarded, never mistaken for coordinates.
    """
    for name, value in (("max_bytes", max_bytes), ("max_faces", max_faces), ("max_vertices", max_vertices)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"cannot read PLY: {exc}") from exc
    if len(data) > max_bytes:
        raise ValueError("PLY file exceeds byte budget")
    encoding, elements, header_bytes, vertex_count, coordinates = _header(data, max_faces, max_vertices)
    values = _Values(data, header_bytes, encoding)
    points, faces, ignored = [], [], []
    for element in elements:
        for prop in element["properties"]:
            if prop["kind"] == "scalar" and (element["name"] == "face" or prop["name"] not in ("x", "y", "z")):
                ignored.append({"element": element["name"], "name": prop["name"], "type": prop["type"]})
        for _ in range(element["count"]):
            xyz = {}
            indices = None
            for prop in element["properties"]:
                if prop["kind"] == "scalar":
                    value = values.scalar(prop["type"])
                    if element["name"] == "vertex" and prop["name"] in coordinates:
                        if not math.isfinite(value):
                            raise ValueError("vertex coordinates must be finite")
                        xyz[prop["name"]] = float(value)
                else:
                    count = values.scalar(prop["count_type"])
                    if count != 3:
                        raise ValueError("every face must have exactly three indices")
                    indices = tuple(values.scalar(prop["item_type"]) for _ in range(3))
                    if any(type(i) is not int or not 0 <= i < vertex_count for i in indices):
                        raise ValueError("face index outside vertex range")
            if element["name"] == "vertex":
                points.append(tuple(xyz[axis] for axis in ("x", "y", "z")))
            else:
                faces.append(indices)
    values.finish()
    metadata = {"format": encoding, "version": "1.0", "bytes": len(data), "header_bytes": header_bytes,
                "vertex_count": len(points), "face_count": len(faces), "elements": elements,
                "coordinate_types": {axis: coordinates[axis]["type"] for axis in ("x", "y", "z")},
                "ignored_scalar_properties": ignored, "coordinate_storage": "python_float_binary64",
                "repairs_applied": False}
    return points, faces, metadata
