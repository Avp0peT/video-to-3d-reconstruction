"""Bounded, lossless-order PLY input for the offline viewer (standard library only).

The reader accepts point clouds and triangle meshes, never adds faces, reduces
vertices, sorts records, or repairs topology. Extra scalar fields are validated
and reported; unsupported elements and extra list fields are rejected.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import re
import struct
from typing import Any


_MAX_HEADER_BYTES = 64 * 1024
_INTEGER = re.compile(rb"[+-]?[0-9]+\Z")
_FLOAT = re.compile(
    rb"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf(?:inity)?|nan)\Z",
    re.IGNORECASE,
)
_TOKENS = re.compile(rb"\S+")
_TYPES: dict[str, tuple[str, int | None, int | None]] = {
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
_XYZ = ("x", "y", "z")
_RGB = ("red", "green", "blue")


def _scalar_type(name: str) -> tuple[str, int | None, int | None]:
    if name not in _TYPES:
        raise ValueError(f"unsupported PLY scalar type: {name}")
    return _TYPES[name]


def _header(data: bytes, max_vertices: int, max_faces: int) -> tuple[str, list[dict[str, Any]], int, int, bool]:
    stream = io.BytesIO(data)

    def line() -> str:
        remaining = _MAX_HEADER_BYTES - stream.tell()
        if remaining <= 0:
            raise ValueError("PLY header exceeds 64 KiB")
        raw = stream.readline(remaining + 1)
        if stream.tell() > _MAX_HEADER_BYTES:
            raise ValueError("PLY header exceeds 64 KiB")
        if not raw or not raw.endswith(b"\n"):
            raise ValueError("truncated or unterminated PLY header")
        try:
            return raw.rstrip(b"\r\n").decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("PLY header must be ASCII") from exc

    if line() != "ply":
        raise ValueError("missing PLY magic")
    encoding: str | None = None
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
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
            if any(element["name"] == parts[1] for element in elements):
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
                _scalar_type(parts[1])
                prop = {"kind": "scalar", "type": parts[1], "name": parts[2]}
            elif len(parts) == 5 and parts[1] == "list":
                if current["name"] != "face" or parts[4] not in ("vertex_indices", "vertex_index"):
                    raise ValueError("only face vertex index lists are supported")
                if _scalar_type(parts[2])[1] is None or _scalar_type(parts[3])[1] is None:
                    raise ValueError("face list count and indices must use integer types")
                prop = {"kind": "list", "count_type": parts[2], "item_type": parts[3], "name": parts[4]}
            else:
                raise ValueError("invalid PLY property declaration")
            if any(previous["name"] == prop["name"] for previous in current["properties"]):
                raise ValueError("duplicate PLY property")
            current["properties"].append(prop)
        elif directive == "end_header":
            if len(parts) != 1:
                raise ValueError("invalid end_header directive")
            break
        else:
            raise ValueError(f"unsupported PLY header directive: {directive}")

    vertex = next((element for element in elements if element["name"] == "vertex"), None)
    if encoding is None or vertex is None:
        raise ValueError("format and exactly one vertex element are required")
    properties = {prop["name"]: prop for prop in vertex["properties"]}
    if not all(axis in properties for axis in _XYZ):
        raise ValueError("vertex x/y/z properties are required")
    colors = [name in properties for name in _RGB]
    if any(colors) and not all(colors):
        raise ValueError("vertex RGB properties must be present together")
    if all(colors) and any(_scalar_type(properties[name]["type"])[0] != "B" for name in _RGB):
        raise ValueError("vertex RGB properties must use uchar or uint8")
    for element in elements:
        if element["name"] == "face" and sum(prop["kind"] == "list" for prop in element["properties"]) != 1:
            raise ValueError("face requires exactly one integer vertex index list")
    return encoding, elements, stream.tell(), vertex["count"], all(colors)


class _Values:
    def __init__(self, data: bytes, offset: int, encoding: str) -> None:
        self.data = data
        self.offset = offset
        self.tokens = _TOKENS.finditer(data, offset) if encoding == "ascii" else None
        self.endian = "<" if encoding == "binary_little_endian" else ">"

    def scalar(self, name: str) -> int | float:
        code, minimum, maximum = _scalar_type(name)
        if self.tokens is None:
            size = struct.calcsize(code)
            if self.offset + size > len(self.data):
                raise ValueError("truncated binary PLY payload")
            value = struct.unpack_from(self.endian + code, self.data, self.offset)[0]
            self.offset += size
        else:
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
            else:
                if not _FLOAT.fullmatch(token):
                    raise ValueError("invalid floating PLY scalar")
                value = float(token)
                # Check float32 range without quantizing ASCII double precision.
                if code == "f" and math.isfinite(value) and abs(value) > 3.4028234663852886e38:
                    raise ValueError("floating PLY scalar outside declared range")
        if not math.isfinite(value):
            raise ValueError("all PLY scalar values must be finite")
        return value

    def finish(self) -> None:
        if self.tokens is not None:
            if next(self.tokens, None) is not None:
                raise ValueError("trailing ASCII PLY data")
        elif self.offset != len(self.data):
            raise ValueError("trailing binary PLY data")


def read_ply(
    path: str | Path,
    max_bytes: int = 16 * 1024 * 1024,
    max_vertices: int = 100000,
    max_faces: int = 200000,
) -> dict[str, Any]:
    """Return vertices, optional RGB colors, faces, and source metadata.

    ``vertices`` and ``faces`` retain file order and use lists of three values.
    ``colors`` is an equally ordered list of RGB integer lists, or ``None`` if
    absent. ``ignored_properties`` records each discarded scalar declaration as
    ``{element, name, type}``. ``sha256`` hashes the entire original file.

    Only PLY 1.0 ASCII and little/big-endian binary are supported. A vertex
    element with x/y/z is required; a face element is optional. Declared faces
    must be triangles with three distinct, in-range vertex indices. Every
    scalar, including ignored attributes, must be finite and within its declared
    range. Empty elements are retained. Limits must be positive integers;
    malformed, unsupported, unreadable, or over-budget input raises ValueError.
    ASCII trailing whitespace is allowed; trailing tokens or binary bytes are
    rejected. The input file is never modified.
    """
    for name, value in (("max_bytes", max_bytes), ("max_vertices", max_vertices), ("max_faces", max_faces)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"cannot read PLY: {exc}") from exc
    if len(data) > max_bytes:
        raise ValueError("PLY file exceeds byte budget")
    encoding, elements, offset, vertex_count, has_colors = _header(data, max_vertices, max_faces)
    values = _Values(data, offset, encoding)
    vertices: list[list[float]] = []
    colors: list[list[int]] | None = [] if has_colors else None
    faces: list[list[int]] = []
    ignored: list[dict[str, str]] = []
    for element in elements:
        for prop in element["properties"]:
            if prop["kind"] == "scalar" and (element["name"] == "face" or prop["name"] not in _XYZ + _RGB):
                ignored.append({"element": element["name"], "name": prop["name"], "type": prop["type"]})
        for _ in range(element["count"]):
            vertex_values: dict[str, int | float] = {}
            indices: list[int] | None = None
            for prop in element["properties"]:
                if prop["kind"] == "scalar":
                    value = values.scalar(prop["type"])
                    if element["name"] == "vertex" and prop["name"] in _XYZ + _RGB:
                        vertex_values[prop["name"]] = value
                else:
                    if values.scalar(prop["count_type"]) != 3:
                        raise ValueError("every face must have exactly three indices")
                    indices = [int(values.scalar(prop["item_type"])) for _ in range(3)]
                    if any(index < 0 or index >= vertex_count for index in indices):
                        raise ValueError("face index outside vertex range")
                    if len(set(indices)) != 3:
                        raise ValueError("triangle contains repeated vertex indices")
            if element["name"] == "vertex":
                vertices.append([float(vertex_values[axis]) for axis in _XYZ])
                if colors is not None:
                    colors.append([int(vertex_values[channel]) for channel in _RGB])
            else:
                # The header has already required exactly one face index list.
                assert indices is not None
                faces.append(indices)
    values.finish()
    return {
        "vertices": vertices,
        "colors": colors,
        "faces": faces,
        "format": encoding,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "ignored_properties": ignored,
    }
