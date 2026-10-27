"""Bounded CPU comparisons of unchanged meshes and shared-input projections.

These images are presentation diagnostics. Shared cameras, masks and depth do
not provide independent shape truth, and this module never edits geometry.
"""
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .global_local_pipeline import binding

PANEL_WIDTH, PANEL_HEIGHT = 360, 560
HEADER, FOOTER = 40, 36
RASTER_HEIGHT = PANEL_HEIGHT - HEADER - FOOTER
BACKGROUND = (17, 26, 40)
COLORS = ((70, 170, 245), (255, 150, 30), (70, 205, 130))
TITLES = ("Native depth", "Robust estimate", "Strict estimate")
MAX_VERTICES, MAX_FACES, MAX_PIXELS = 200_000, 400_000, 16_000_000


def _target(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        raise ValueError("fresh comparison path required")
    if not target.parent.is_dir():
        raise ValueError("existing output directory required")
    return target


def _array_binding(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {"shape": list(array.shape), "dtype": array.dtype.str,
            "sha256": sha256(memoryview(contiguous.reshape(-1)).cast("B")).hexdigest()}


def _vertices(value: Any, label: str) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.ndim != 2
            or value.shape[1] != 3 or value.dtype.kind != "f"
            or len(value) > MAX_VERTICES or not np.isfinite(value).all()):
        raise ValueError(f"{label}: finite floating Nx3 vertices, at most 200000, required")
    if value.size and np.max(np.abs(value)) > 1e12:
        raise ValueError(f"{label}: unsafe coordinate magnitude")
    return value


def _rotation(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != (3, 3) or raw.dtype.kind not in "fiu"
            or any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat)
            or not np.isfinite(raw).all()):
        raise ValueError("finite 3x3 rotation required")
    rotation = raw.astype(np.float64)
    if (np.max(np.abs(rotation)) > 1.000001
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
        raise ValueError("proper orthonormal rotation required")
    return rotation


def _mesh(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("mesh mapping required")
    title = value.get("title")
    if (not isinstance(title, str) or not 1 <= len(title) <= 40
            or any(ord(char) < 32 or ord(char) > 126 for char in title)):
        raise ValueError("printable ASCII mesh title of 1..40 characters required")
    vertices = _vertices(value.get("vertices"), title)
    faces = value.get("faces")
    if (not isinstance(faces, np.ndarray) or faces.ndim != 2
            or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or len(faces) > MAX_FACES
            or (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))):
        raise ValueError("valid triangle indices, at most 400000 faces, required")
    color = np.asarray(value.get("color"))
    if (color.shape != (3,) or color.dtype.kind not in "iu"
            or (color < 0).any() or (color > 255).any()):
        raise ValueError("three integer RGB color channels in 0..255 required")
    return dict(title=title, vertices=vertices, faces=faces, color=color)


def _rasterize(mesh: Mapping[str, Any], rotation: np.ndarray,
               center: np.ndarray, midpoint: np.ndarray, scale: float,
               check: Callable[[], None]) -> tuple[np.ndarray, dict[str, Any]]:
    points = (mesh["vertices"].astype(np.float64) - center) @ rotation.T
    xy = (points[:, :2] - midpoint) * scale + [PANEL_WIDTH / 2, RASTER_HEIGHT / 2]
    if not np.isfinite(xy).all() or (xy.size and np.max(np.abs(xy)) > 1e9):
        raise ValueError("unsafe projected coordinate magnitude")
    zbuffer = np.full((RASTER_HEIGHT, PANEL_WIDTH), np.inf)
    owners = np.full(zbuffer.shape, -1, np.int32)
    image = np.full((*zbuffer.shape, 3), BACKGROUND, np.uint8)
    light = np.array([-.4, -.6, -1.])
    light /= np.linalg.norm(light)
    degenerate = offscreen = rasterized = 0
    for index, face in enumerate(mesh["faces"]):
        if index % 2048 == 0:
            check()
        triangle = xy[face]
        lower = np.maximum(0, np.floor(triangle.min(axis=0)).astype(np.int64))
        upper = np.minimum([PANEL_WIDTH - 1, RASTER_HEIGHT - 1],
                           np.ceil(triangle.max(axis=0)).astype(np.int64))
        if (upper < lower).any():
            offscreen += 1
            continue
        a, b, c = triangle
        denominator = ((b[1] - c[1]) * (a[0] - c[0])
                       + (c[0] - b[0]) * (a[1] - c[1]))
        if abs(denominator) < 1e-8:
            degenerate += 1
            continue
        # At most one fixed-size panel is allocated per face, never a face batch.
        yy, xx = np.ogrid[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
        xx, yy = xx + .5, yy + .5
        w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        w2 = 1. - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        rasterized += 1
        depth = w0 * points[face[0], 2] + w1 * points[face[1], 2] + w2 * points[face[2], 2]
        region = np.s_[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1]
        nearer = inside & (depth < zbuffer[region])
        if not nearer.any():
            continue
        p = points[face]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        length = np.linalg.norm(normal)
        shade = .27 + .73 * abs(float((normal / max(length, 1e-30)) @ light))
        zbuffer[region][nearer] = depth[nearer]
        owners[region][nearer] = index
        image[region][nearer] = np.clip(mesh["color"] * shade, 0, 255).astype(np.uint8)
    occupied = owners >= 0
    visible_ids = np.unique(owners[occupied])
    counts = dict(vertices=len(mesh["vertices"]), faces=len(mesh["faces"]),
                  rasterized_faces=rasterized, visible_faces=len(visible_ids),
                  degenerate_projected_faces=degenerate, offscreen_faces=offscreen,
                  visible_pixels=int(occupied.sum()),
                  visible_depth_range=([float(zbuffer[occupied].min()), float(zbuffer[occupied].max())]
                                       if occupied.any() else None))
    return image, counts


def render_comparison(meshes: Sequence[Mapping[str, Any]], out_path: str | Path, *,
                      rotations: Sequence[Any], reference_vertices: np.ndarray,
                      check: Callable[[], None] = lambda: None) -> dict[str, Any]:
    """Render rows of views and columns of native/robust/strict geometry.

    One reference AABB center and one pixels-per-world-unit scale are shared by
    all rows and columns. Each row also shares its rotation and reference image
    midpoint. Smaller camera z wins; both windings render with identical lighting.
    Display transforms never move or modify any supplied world-space array.
    The output is published only after all panels and the final check complete.
    """
    check()
    target = _target(out_path)
    if not isinstance(meshes, (list, tuple)) or len(meshes) != 3:
        raise ValueError("exactly three meshes in native/robust/strict order required")
    if not isinstance(rotations, (list, tuple)) or not 1 <= len(rotations) <= 3:
        raise ValueError("one to three rotations required")
    checked_meshes = [_mesh(mesh) for mesh in meshes]
    checked_rotations = [_rotation(rotation) for rotation in rotations]
    reference = _vertices(reference_vertices, "reference")
    if not len(reference):
        raise ValueError("nonempty reference vertices required")
    reference64 = reference.astype(np.float64)
    center = reference64.min(axis=0) + (reference64.max(axis=0) - reference64.min(axis=0)) / 2.
    cameras = []
    scales = []
    for rotation in checked_rotations:
        projected = (reference64 - center) @ rotation.T
        lower, upper = projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)
        span = upper - lower
        if not (span > 0).any():
            raise ValueError("nonzero projected reference extent required")
        scales.extend(float(available / extent) for available, extent in
                      zip((PANEL_WIDTH - 44, RASTER_HEIGHT - 44), span) if extent > 0)
        cameras.append(dict(rotation=rotation, midpoint=(lower + upper) / 2,
                            reference_bounds=[lower.tolist(), upper.tolist()]))
    scale = min(scales)
    if not np.isfinite(scale) or not 1e-12 <= scale <= 1e12:
        raise ValueError("unsafe reference projection scale")
    canvas = Image.new("RGB", (PANEL_WIDTH * 3, PANEL_HEIGHT * len(cameras)), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    panels = []
    projections = []
    for row, camera in enumerate(cameras):
        check()
        projection = dict(kind="orthographic", row=row, world_reference_center=center.tolist(),
                          rotation=camera["rotation"].tolist(), camera_xy_midpoint=camera["midpoint"].tolist(),
                          reference_camera_xy_bounds=camera["reference_bounds"],
                          pixels_per_world_unit=scale, raster_size=[PANEL_WIDTH, RASTER_HEIGHT],
                          pixel_center_offset=[.5, .5], screen_center=[PANEL_WIDTH / 2, RASTER_HEIGHT / 2],
                          screen_y="positive camera y down", depth_order="smaller camera z wins",
                          formula="camera=(world-reference_center)@rotation.T; xy=(camera.xy-midpoint)*scale+screen_center")
        projections.append(projection)
        for column, mesh in enumerate(checked_meshes):
            image, counts = _rasterize(mesh, camera["rotation"], center, camera["midpoint"], scale, check)
            x, y = column * PANEL_WIDTH, row * PANEL_HEIGHT
            canvas.paste(Image.fromarray(image), (x, y + HEADER))
            draw.text((x + 10, y + 12), f"View {row + 1} | {mesh['title']}", fill="white", font=font)
            draw.text((x + 10, y + PANEL_HEIGHT - 27),
                      f"V={counts['vertices']} F={counts['faces']} pixels={counts['visible_pixels']}",
                      fill=(210, 220, 230), font=font)
            panels.append(dict(row=row, column=column, title=mesh["title"],
                               color=mesh["color"].tolist(), counts=counts, projection=projection,
                               canvas_raster_origin=[x, y + HEADER]))
    check()
    with target.open("xb") as stream:
        canvas.save(stream, format="PNG")
    return dict(**binding(target), status="COMPARISON_RENDERED", panels=panels,
                projections=projections, panel_size=[PANEL_WIDTH, PANEL_HEIGHT],
                reference_binding=_array_binding(reference),
                source_bindings=[dict(title=mesh["title"], vertices=_array_binding(mesh["vertices"]),
                                      faces=_array_binding(mesh["faces"])) for mesh in checked_meshes],
                geometry_modified=False, geometry_accepted=False, independent_shape_truth=False,
                semantics="CPU orthographic z-buffer display of unchanged world coordinates; no smoothing, geometry repair, completeness or correctness claim.")


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def write_projection_comparison(path: str | Path, view: Mapping[str, Any],
                                footprints: Sequence[np.ndarray]) -> dict[str, Any]:
    """Write original plus three overlays; all counts remain in native pixels.

    ``view`` supplies RGB uint8 image, boolean mask and explicit boolean validity.
    ``fg`` is projected valid foreground, ``bg`` projected valid background,
    ``missing`` unprojected valid foreground; IoU excludes invalid pixels. Empty
    union gives None, never a fabricated perfect score. Shared-input overlap is
    diagnostic only, not independent geometry truth or visibility evidence.
    """
    target = _target(path)
    if not isinstance(view, Mapping):
        raise ValueError("view mapping required")
    image, mask, valid = view.get("image"), view.get("mask"), view.get("validity")
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1
            or max(image.shape[:2]) > 8192 or np.prod(image.shape[:2]) > MAX_PIXELS):
        raise ValueError("bounded RGB uint8 source image required")
    shape = image.shape[:2]
    for array in (mask, valid):
        if not isinstance(array, np.ndarray) or array.dtype != bool or array.shape != shape:
            raise ValueError("explicit boolean source mask and validity required")
    if not isinstance(footprints, (list, tuple)) or len(footprints) != 3:
        raise ValueError("exactly three boolean footprints required")
    for footprint in footprints:
        if not isinstance(footprint, np.ndarray) or footprint.dtype != bool or footprint.shape != shape:
            raise ValueError("boolean footprint in native image domain required")
    foreground, background = mask & valid, ~mask & valid
    source_boundary = _boundary(foreground)
    h, w = shape
    display_scale = min(PANEL_WIDTH / w, RASTER_HEIGHT / h, 1.)
    display_size = [max(1, int(w * display_scale)), max(1, int(h * display_scale))]
    origin = [(PANEL_WIDTH - display_size[0]) // 2, HEADER + (RASTER_HEIGHT - display_size[1]) // 2]
    canvas = Image.new("RGB", (PANEL_WIDTH * 4, PANEL_HEIGHT), BACKGROUND)
    draw, font = ImageDraw.Draw(canvas), ImageFont.load_default()
    counts = []
    for column in range(4):
        overlay = image.copy()
        title = "Original source"
        if column:
            index = column - 1
            footprint, color = footprints[index], np.asarray(COLORS[index])
            overlay[footprint] = (.6 * overlay[footprint] + .4 * color).astype(np.uint8)
            overlay[_boundary(footprint)] = color
            overlay[source_boundary] = (245, 80, 220)
            fg, bg = int((footprint & foreground).sum()), int((footprint & background).sum())
            missing = int((foreground & ~footprint).sum())
            union = fg + bg + missing
            count = dict(model_index=index, title=TITLES[index], fg=fg, bg=bg, missing=missing,
                         iou=fg / union if union else None, union=union,
                         invalid=int((footprint & ~valid).sum()), total_footprint=int(footprint.sum()))
            counts.append(count)
            title = TITLES[index] + " projection"
            score = f"{count['iou']:.4f}" if count["iou"] is not None else "undefined"
            draw.text((column * PANEL_WIDTH + 8, PANEL_HEIGHT - 31),
                      f"FG={fg} BG={bg} missing={missing}", fill="white", font=font)
            draw.text((column * PANEL_WIDTH + 8, PANEL_HEIGHT - 17),
                      f"IoU={score}; invalid={count['invalid']}", fill="white", font=font)
        preview = Image.fromarray(overlay).resize(display_size, resample=Image.Resampling.NEAREST)
        canvas.paste(preview, (column * PANEL_WIDTH + origin[0], origin[1]))
        draw.text((column * PANEL_WIDTH + 8, 12), title, fill="white", font=font)
    draw.text((8, PANEL_HEIGHT - 31), "Magenta: valid source-mask boundary", fill="white", font=font)
    draw.text((8, PANEL_HEIGHT - 17), "Shared input, not independent truth", fill=(245, 220, 125), font=font)
    with target.open("xb") as stream:
        canvas.save(stream, format="PNG")
    return dict(**binding(target), status="PROJECTION_COMPARISON_RENDERED", view_id=view.get("view_id"),
                counts=counts, valid_foreground_pixels=int(foreground.sum()),
                native_size=[w, h], display_size=display_size, display_scale=display_scale,
                display_resampling="nearest", panel_size=[PANEL_WIDTH, PANEL_HEIGHT],
                source_bindings={key: _array_binding(view[key]) for key in ("image", "mask", "validity")},
                footprint_bindings=[_array_binding(footprint) for footprint in footprints],
                geometry_modified=False, geometry_accepted=False, independent_shape_truth=False,
                semantics="Shared-input native-pixel projection overlap only, not visibility, independent physical truth or point accuracy. Invalid pixels are excluded from FG/BG/missing/IoU; no geometry is inferred.")
