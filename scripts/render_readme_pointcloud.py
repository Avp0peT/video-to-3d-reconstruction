"""Render every vertex of a hash-bound PLY as an offline PNG (stdlib only).

The repository PLY reader validates the source, including any triangle records.
Faces are counted for provenance but never drawn. Projection, centering and
depth colors affect this preview only; source geometry is never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys
from typing import Any
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from integrations.viewer.ply_data import read_ply


def binding(path: Path) -> dict[str, Any]:
    """Use repository-relative paths when possible, plus exact byte bindings."""
    path = path.resolve(strict=True)
    name = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    data = path.read_bytes()
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def png_bytes(width: int, height: int, pixels: bytearray) -> bytes:
    """Encode deterministic RGB PNG bytes without image-library dependencies."""
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    stride = width * 3
    scanlines = b"".join(b"\0" + pixels[y * stride:(y + 1) * stride] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines, level=9)) + chunk(b"IEND", b""))


def build(source: Path, expected_sha256: str, output: Path, *, width: int = 960,
          height: int = 720, azimuth: float = -60, elevation: float = 25,
          point_radius: int = 1) -> dict[str, Any]:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists() or source.is_relative_to(output):
        raise ValueError("fresh output directory that cannot contain the source required")
    if not 64 <= width <= 4096 or not 64 <= height <= 4096 or not 0 <= point_radius <= 8:
        raise ValueError("width/height must be 64..4096; point radius must be 0..8")
    if not all(math.isfinite(angle) for angle in (azimuth, elevation)):
        raise ValueError("view angles must be finite")
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256.lower()):
        raise ValueError("expected SHA256 must contain 64 hexadecimal characters")
    before = binding(source)
    if before["sha256"] != expected_sha256.lower():
        raise ValueError("source SHA256 differs from the expected binding")
    scene = read_ply(source, max_bytes=64 * 1024 * 1024,
                     max_vertices=400000, max_faces=800000)
    if scene["sha256"] != before["sha256"]:
        raise ValueError("source changed while reading")
    points = scene["vertices"]
    if not points:
        raise ValueError("at least one vertex required")
    low = [min(p[i] for p in points) for i in range(3)]
    high = [max(p[i] for p in points) for i in range(3)]
    center = [(a + b) / 2 for a, b in zip(low, high)]
    a, e = math.radians(azimuth), math.radians(elevation)
    right = (-math.sin(a), math.cos(a), 0.0)
    up = (-math.sin(e) * math.cos(a), -math.sin(e) * math.sin(a), math.cos(e))
    toward_camera = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))

    def dot(point: list[float], axis: tuple[float, float, float]) -> float:
        return sum((point[i] - center[i]) * axis[i] for i in range(3))

    projected = [(dot(p, right), dot(p, up), dot(p, toward_camera)) for p in points]
    bounds = [[min(p[i] for p in projected), max(p[i] for p in projected)] for i in range(3)]
    spans = [b - a for a, b in bounds]
    scale = min(width * 0.88 / max(spans[0], 1e-12), height * 0.88 / max(spans[1], 1e-12))
    middle = [(a + b) / 2 for a, b in bounds]
    pixels = bytearray((12, 22, 36)) * (width * height)
    depths = [-math.inf] * (width * height)
    offsets = [(dx, dy) for dy in range(-point_radius, point_radius + 1)
               for dx in range(-point_radius, point_radius + 1)
               if dx * dx + dy * dy <= point_radius * point_radius]
    for x, y, z in projected:
        px = round((x - middle[0]) * scale + (width - 1) / 2)
        py = round(-(y - middle[1]) * scale + (height - 1) / 2)
        brightness = 0.45 + 0.55 * ((z - bounds[2][0]) / spans[2] if spans[2] else 1.0)
        color = bytes(round(channel * brightness) for channel in (94, 206, 255))
        for dx, dy in offsets:
            xx, yy = px + dx, py + dy
            if 0 <= xx < width and 0 <= yy < height:
                index = yy * width + xx
                if z > depths[index]:
                    depths[index] = z
                    pixels[index * 3:index * 3 + 3] = color
    preview = png_bytes(width, height, pixels)
    if binding(source) != before:
        raise ValueError("source changed while rendering")
    output.mkdir(parents=True, exist_ok=False)
    (output / "preview.png").write_bytes(preview)
    summary = {
        "status": "PLY_VERTEX_PREVIEW", "source": before,
        "source_kind": "prepared_surface_mesh_vertices" if scene["faces"] else "point_cloud_vertices",
        "counts": {"source_vertices": len(points), "source_faces": len(scene["faces"]),
                   "rendered_vertices": len(points), "faces_drawn": 0},
        "sampling": {"method": "all_vertices_in_file_order", "limit": None, "seed": None},
        "render": {"projection": "orthographic", "azimuth_degrees": azimuth,
                   "elevation_degrees": elevation, "width": width, "height": height,
                   "point_radius_pixels": point_radius, "fit_fraction": 0.88,
                   "depth_test": "nearest_vertex_wins", "color": "presentation_depth_blue",
                   "background_rgb": [12, 22, 36], "source_bounds": [low, high],
                   "display_translation": center, "pixels_per_source_unit": scale},
        "sources_unchanged": True, "geometry_modified": False, "geometry_accepted": False,
        "quality_status": "NOT_EVALUATED", "scale_status": "unknown",
        "independently_verified_physical_point_ids": 0,
        "limitations": ["Mesh vertex density is not independent observation coverage.",
                        "Occlusion and raster overlap hide vertices in this single projection.",
                        "No new reconstruction, metric scale, physical point identity or accuracy evaluation."],
        "preview": binding(output / "preview.png"),
        "implementation": binding(Path(__file__)),
        "ply_reader": binding(ROOT / "integrations/viewer/ply_data.py"),
        "environment": {"python": platform.python_version(),
                        "implementation": platform.python_implementation(), "zlib": zlib.ZLIB_VERSION,
                        "dependencies": "Python standard library only"},
    }
    (output / "RESULT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sha256", required=True, help="expected SHA256 of the input PLY")
    parser.add_argument("--output", required=True, type=Path, help="fresh output directory")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=-60)
    parser.add_argument("--elevation", type=float, default=25)
    parser.add_argument("--point-radius", type=int, default=1)
    args = parser.parse_args()
    result = build(args.input, args.sha256, args.output, width=args.width, height=args.height,
                   azimuth=args.azimuth, elevation=args.elevation, point_radius=args.point_radius)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
