"""Conservative face/silhouette conflict candidates; no IO or mesh mutation.

Native OpenMVS integer pixel centers are required. Masks are visible labels,
not ground truth: external occlusion, inaccurate poses and nonrigid motion can
still create conflicts. Neither NO_CONFLICT nor kept means accepted geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from collections.abc import Mapping
import math

import numpy as np

UNKNOWN, NO_CONFLICT, CONFLICT = 0, 1, 2
STATUS_NAMES = {UNKNOWN: 'UNKNOWN', NO_CONFLICT: 'NO_CONFLICT', CONFLICT: 'CONFLICT'}
REASON_NAMES = dict(enumerate((
    'NOT_EVALUATED', 'MISSING_DEPTH', 'CAMERA_PLANE', 'OUT_OF_FRAME',
    'INVALID_VERTEX_DOMAIN', 'VERTEX_OUTSIDE_FOREGROUND_MARGIN',
    'OCCLUDED_VERTEX', 'INSUFFICIENT_DEPTH_SUPPORT', 'SUBPIXEL_OR_DEGENERATE',
    'RASTER_BUDGET', 'NO_INTERIOR_PIXEL_CENTERS', 'INVALID_SAMPLE_DOMAIN',
    'BACKGROUND_CONFLICT', 'NO_DEEP_BACKGROUND', 'BELOW_CONFLICT_THRESHOLD',
    'BOUNDING_BOX_NO_DEEP_BACKGROUND',
)))
_REASON = {name: key for key, name in REASON_NAMES.items()}


@dataclass(frozen=True)
class Policy:
    boundary_px: float = 3.0
    min_background_pixels: int = 4
    min_background_fraction: float = 0.01
    min_angle_degrees: float = 8.0
    min_depth_supported_vertices: int = 2
    depth_relative_tolerance: float = 0.02
    max_raster_pixels_per_face: int = 200000

    def __post_init__(self):
        for name in ('boundary_px', 'min_background_fraction', 'min_angle_degrees',
                     'depth_relative_tolerance'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('finite numeric policy required: ' + name)
        if not 0 <= self.boundary_px <= 100 or not 0 < self.min_background_fraction <= 1:
            raise ValueError('invalid boundary/fraction policy')
        if not 0 < self.min_angle_degrees <= 180 or not 0 <= self.depth_relative_tolerance < 1:
            raise ValueError('invalid angle/depth tolerance policy')
        for name in ('min_background_pixels', 'min_depth_supported_vertices', 'max_raster_pixels_per_face'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('positive integer policy required: ' + name)
        if not 2 <= self.min_depth_supported_vertices <= 3:
            raise ValueError('at least two, at most three depth-supported vertices required')


def _mesh(vertices, faces):
    v, f = np.asarray(vertices, dtype=float), np.asarray(faces)
    if (v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1:] != (3,) or not np.issubdtype(f.dtype, np.integer)
            or (f.size and (f.min() < 0 or f.max() >= len(v)))):
        raise ValueError('finite Nx3 vertices and valid integer Mx3 face indices required')
    return v, f


@lru_cache(maxsize=1)
def _cv():
    # Use the repository's existing, origin-checked CPU OpenCV lazily.
    from integrations.demo_pipeline.temporal_review import load_cv
    module = load_cv([])
    if module is None:
        raise RuntimeError('existing repository OpenCV runtime unavailable')
    return module


def _distances(mask, validity):
    cv = _cv()
    foreground = mask & validity
    to_foreground = cv.distanceTransform((~foreground).astype(np.uint8), cv.DIST_L2, cv.DIST_MASK_PRECISE)
    # Outside the image is unknown too; padding must not create deep background.
    background = np.pad((~mask & validity).astype(np.uint8), 1)
    deep_background = cv.distanceTransform(background, cv.DIST_L2, cv.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return to_foreground, deep_background


def _camera(K, R, C):
    k, r, c = (np.asarray(x, dtype=float) for x in (K, R, C))
    if (k.shape != (3, 3) or r.shape != (3, 3) or c.shape != (3,)
            or not all(np.isfinite(x).all() for x in (k, r, c))
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(r @ r.T, np.eye(3), atol=1e-6, rtol=0)
            or abs(np.linalg.det(r) - 1) > 1e-6):
        raise ValueError('finite pinhole K, proper world-to-camera R, and center C required')
    return k, r, c


def _view_id(value):
    if type(value) not in (int, str):
        raise ValueError('view_id must be a stable integer or string')


def _raster_pixels(triangle, width, height, budget):
    """Full center-in-triangle raster plus <=1 pixel projected edge sampling."""
    lo, hi = np.ceil(triangle.min(axis=0)).astype(int), np.floor(triangle.max(axis=0)).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [width - 1, height - 1])
    lengths = np.linalg.norm(np.roll(triangle, -1, axis=0) - triangle, axis=1)
    steps = np.maximum(1, np.ceil(lengths).astype(int))
    bbox_count = max(0, int(hi[0] - lo[0] + 1)) * max(0, int(hi[1] - lo[1] + 1))
    if bbox_count + int(np.sum(steps + 1)) > budget:
        return None, None, 'RASTER_BUDGET'
    if not bbox_count:
        return None, None, 'NO_INTERIOR_PIXEL_CENTERS'
    xx, yy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    edge_values = []
    for start, end in zip(triangle, np.roll(triangle, -1, axis=0)):
        edge_values.append((end[0] - start[0]) * (yy - start[1]) - (end[1] - start[1]) * (xx - start[0]))
    signed = np.stack(edge_values)
    inside = np.all(signed >= -1e-10, axis=0) | np.all(signed <= 1e-10, axis=0)
    interior = (yy[inside] * width + xx[inside]).ravel()
    if not len(interior):
        return None, None, 'NO_INTERIOR_PIXEL_CENTERS'
    edges = []
    for i, nsteps in enumerate(steps):
        u = np.linspace(0, 1, int(nsteps) + 1)[:, None]
        uv = triangle[i] * (1 - u) + triangle[(i + 1) % 3] * u
        pixels = np.floor(uv + 0.5).astype(int)
        # Complete in-frame vertices imply in-frame segments by convexity.
        if ((pixels < 0).any() or (pixels[:, 0] >= width).any() or (pixels[:, 1] >= height).any()):
            raise ValueError('edge raster escaped verified image domain')
        edges.append(pixels[:, 1] * width + pixels[:, 0])
    return np.unique(interior), np.unique(np.concatenate(edges)), None


def _integral_counts(binary, low, high):
    integral = np.pad(np.cumsum(np.cumsum(binary, axis=0, dtype=np.int64), axis=1, dtype=np.int64), ((1, 0), (1, 0)))
    x0, y0 = low.T
    x1, y1 = (high + 1).T
    return integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]


def evaluate_view(vertices, faces, *, K, R, C, mask, validity, depth, view_id,
                  policy=Policy(), check=lambda: None):
    """Return indexed numpy arrays; no input arrays are changed.

    A vertex behind a valid central depth is occluded even if an adjacent depth
    happens to match. Otherwise a 3x3 foreground-depth match can support it.
    Missing center depth plus only shallower neighbors also means occluded.
    A projected area below 0.5 px^2 or no raster pixel center is UNKNOWN.
    Raster budget includes bounding-box centers and edge sample work, not just
    retained pixels. Any unknown validity sample makes that face UNKNOWN.
    A valid full bounding box with no deep background may prove NO_CONFLICT
    without enumerating samples. Its raster_evaluated is False and sample
    counts stay zero (not measured); bounding_box_pixels records proof scope.
    """
    if not isinstance(policy, Policy):
        raise ValueError('Policy required')
    _view_id(view_id)
    v, f = _mesh(vertices, faces)
    k, r, c = _camera(K, R, C)
    mask, validity = np.asarray(mask), np.asarray(validity)
    if (mask.dtype != np.bool_ or validity.dtype != np.bool_ or mask.ndim != 2
            or mask.shape != validity.shape or not all(mask.shape)):
        raise ValueError('same-size nonempty bool mask and validity required')
    h, w = mask.shape
    if depth is not None:
        depth = np.asarray(depth, dtype=float)
        if depth.shape != mask.shape:
            raise ValueError('depth must use exactly the same pixel domain as masks and K')
    n = len(f)
    out = dict(view_id=view_id, status=np.zeros(n, np.uint8), reason=np.zeros(n, np.uint8),
               status_names=STATUS_NAMES.copy(), reason_names=REASON_NAMES.copy(),
               face_indices=np.arange(n, dtype=np.int64),
               background_pixels=np.zeros(n, np.int32), total_pixels=np.zeros(n, np.int32),
               interior_pixels=np.zeros(n, np.int32), edge_pixels=np.zeros(n, np.int32),
               edge_background_pixels=np.zeros(n, np.int32),
               raster_evaluated=np.zeros(n, bool), bounding_box_pixels=np.zeros(n, np.int32),
               depth_supported_vertices=np.zeros(n, np.uint8), occluded_vertices=np.zeros(n, np.uint8),
               projected_area_pixels=np.zeros(n, float), geometry_accepted=False)
    check()
    if depth is None:
        out['reason'][:] = _REASON['MISSING_DEPTH']
        return out
    if not n:
        return out
    cam = (v - c) @ r.T
    hom = cam @ k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = hom[:, :2] / hom[:, 2, None]
    front = (cam[:, 2] > 1e-12) & np.isfinite(uv).all(axis=1)
    in_frame = front & (uv[:, 0] >= -0.5) & (uv[:, 0] < w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < h - 0.5)
    pix = np.zeros((len(v), 2), dtype=np.int64)
    pix[in_frame] = np.floor(uv[in_frame] + 0.5).astype(np.int64)
    vertex_valid = in_frame & validity[pix[:, 1], pix[:, 0]]
    to_fg, bg_distance = _distances(mask, validity)
    vertex_foreground = vertex_valid & (to_fg[pix[:, 1], pix[:, 0]] <= policy.boundary_px)
    support = np.zeros(len(v), bool)
    center_has_depth = np.zeros(len(v), bool)
    center_occluded = np.zeros(len(v), bool)
    max_depth = np.zeros(len(v), float)
    ids = np.flatnonzero(vertex_foreground)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = pix[ids, 0] + dx, pix[ids, 1] + dy
            good = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            ii, x, y = ids[good], x[good], y[good]
            d = depth[y, x]
            known = validity[y, x] & mask[y, x] & np.isfinite(d) & (d > 0)
            ii, d = ii[known], d[known]
            max_depth[ii] = np.maximum(max_depth[ii], d)
            support[ii] |= np.abs(cam[ii, 2] - d) <= policy.depth_relative_tolerance * d
            if dx == 0 and dy == 0:
                center_has_depth[ii] = True
                center_occluded[ii] = cam[ii, 2] > d * (1 + policy.depth_relative_tolerance)
    occluded = center_occluded | (~center_has_depth & (max_depth > 0) & (cam[:, 2] > max_depth * (1 + policy.depth_relative_tolerance)))
    support &= ~occluded
    out['depth_supported_vertices'][:] = support[f].sum(axis=1)
    out['occluded_vertices'][:] = occluded[f].sum(axis=1)
    pending = np.ones(n, bool)

    def reject(condition, reason):
        selected = pending & condition
        out['reason'][selected] = _REASON[reason]
        pending[selected] = False

    reject(~front[f].all(axis=1), 'CAMERA_PLANE')
    reject(~in_frame[f].all(axis=1), 'OUT_OF_FRAME')
    reject(~vertex_valid[f].all(axis=1), 'INVALID_VERTEX_DOMAIN')
    reject(~vertex_foreground[f].all(axis=1), 'VERTEX_OUTSIDE_FOREGROUND_MARGIN')
    reject(out['occluded_vertices'] > 0, 'OCCLUDED_VERTEX')
    reject(out['depth_supported_vertices'] < policy.min_depth_supported_vertices, 'INSUFFICIENT_DEPTH_SUPPORT')
    triangles = uv[f]
    with np.errstate(invalid='ignore'):
        ab, ac = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        area = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) * 0.5
    out['projected_area_pixels'][:] = np.where(np.isfinite(area), area, 0)
    reject(~np.isfinite(area) | (area < 0.5), 'SUBPIXEL_OR_DEGENERATE')
    check()
    flat_valid, deep_bg = validity.ravel(), (bg_distance > policy.boundary_px).ravel()
    # A bounding-box proof is conservative and avoids Python work on the many
    # ordinary faces. Rounded bounds also enclose nearest-pixel edge samples.
    selected = np.flatnonzero(pending)
    if len(selected):
        tri = triangles[selected]
        lows, highs = tri.min(axis=1), tri.max(axis=1)
        lo, hi = np.floor(lows + 0.5).astype(int), np.floor(highs + 0.5).astype(int)
        out['bounding_box_pixels'][selected] = np.prod(hi - lo + 1, axis=1)
        raster_sizes = np.maximum(0, np.floor(highs) - np.ceil(lows) + 1)
        edge_work = np.maximum(1, np.ceil(np.linalg.norm(np.roll(tri, -1, axis=1) - tri, axis=2))).sum(axis=1) + 3
        over_budget = np.prod(raster_sizes, axis=1) + edge_work > policy.max_raster_pixels_per_face
        out['reason'][selected[over_budget]] = _REASON['RASTER_BUDGET']
        pending[selected[over_budget]] = False
        no_bg = _integral_counts(deep_bg.reshape(h, w), lo, hi) == 0
        all_valid = _integral_counts(~validity, lo, hi) == 0
        probes = np.floor(np.concatenate((tri.mean(axis=1, keepdims=True), tri,
                                         (tri + np.roll(tri, -1, axis=1)) / 2), axis=1) + 0.5)
        signed = []
        for side in range(3):
            start, end = tri[:, side], tri[:, (side + 1) % 3]
            signed.append((end[:, 0] - start[:, 0])[:, None] * (probes[:, :, 1] - start[:, 1, None])
                          - (end[:, 1] - start[:, 1])[:, None] * (probes[:, :, 0] - start[:, 0, None]))
        signed = np.stack(signed)
        interior_witness = ((signed >= -1e-10).all(axis=0) | (signed <= 1e-10).all(axis=0)).any(axis=1)
        fast = selected[~over_budget & no_bg & all_valid & interior_witness]
        out['status'][fast] = NO_CONFLICT
        out['reason'][fast] = _REASON['BOUNDING_BOX_NO_DEEP_BACKGROUND']
        pending[fast] = False
    for ordinal, idx in enumerate(np.flatnonzero(pending)):
        if ordinal % 128 == 0:
            check()
        interior, edges, unknown = _raster_pixels(triangles[idx], w, h, policy.max_raster_pixels_per_face)
        if unknown:
            out['reason'][idx] = _REASON[unknown]
            continue
        samples = np.union1d(interior, edges)
        out['raster_evaluated'][idx] = True
        out['interior_pixels'][idx], out['edge_pixels'][idx] = len(interior), len(edges)
        out['total_pixels'][idx] = len(samples)
        if not flat_valid[samples].all():
            out['reason'][idx] = _REASON['INVALID_SAMPLE_DOMAIN']
            continue
        background = int(np.count_nonzero(deep_bg[samples]))
        out['background_pixels'][idx] = background
        out['edge_background_pixels'][idx] = np.count_nonzero(deep_bg[edges])
        conflict = background >= policy.min_background_pixels and background >= len(samples) * policy.min_background_fraction
        out['status'][idx] = CONFLICT if conflict else NO_CONFLICT
        out['reason'][idx] = _REASON['BACKGROUND_CONFLICT' if conflict else ('NO_DEEP_BACKGROUND' if background == 0 else 'BELOW_CONFLICT_THRESHOLD')]
    check()
    return out


def aggregate_views(vertices, faces, view_results, camera_centers, policy=Policy(), *, check=lambda: None):
    """Quarantine only two angularly separated conflicts; keep original indices.

    camera_centers is {view_id: C}, or an array in view_results order.
    max_conflict_angle_degrees=0 with angle_pair_evaluated=False means unknown,
    not a measured zero angle. Witnesses are the pair with largest valid angle.
    """
    if not isinstance(policy, Policy):
        raise ValueError('Policy required')
    v, f = _mesh(vertices, faces)
    results = list(view_results)
    ids = [item['view_id'] for item in results]
    for key in ids:
        _view_id(key)
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate view identity cannot supply another vote')
    centers = np.asarray([camera_centers[key] for key in ids] if isinstance(camera_centers, Mapping) else camera_centers, dtype=float)
    if not len(results) and centers.size == 0:
        centers = centers.reshape(0, 3)
    if centers.shape != (len(results), 3) or not np.isfinite(centers).all():
        raise ValueError('one finite center per unique view required')
    statuses = []
    for row in results:
        status = np.asarray(row['status'])
        if status.shape != (len(f),) or not np.isin(status, [UNKNOWN, NO_CONFLICT, CONFLICT]).all():
            raise ValueError('one valid status per original face required')
        if 'face_indices' in row and not np.array_equal(row['face_indices'], np.arange(len(f))):
            raise ValueError('view result face order differs')
        statuses.append(status)
    matrix = np.stack(statuses) if statuses else np.empty((0, len(f)), np.uint8)
    out = dict(status='CANDIDATE_NOT_ACCEPTED', geometry_accepted=False, view_count=len(results),
               quarantine=np.zeros(len(f), bool),
               conflict_views=(matrix == CONFLICT).sum(axis=0),
               no_conflict_views=(matrix == NO_CONFLICT).sum(axis=0),
               unknown_views=(matrix == UNKNOWN).sum(axis=0),
               max_conflict_angle_degrees=np.zeros(len(f), float),
               angle_pair_evaluated=np.zeros(len(f), bool), witness_view_ids=[None] * len(f))
    face_centers = v[f].mean(axis=1)
    check()
    pair = 0
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            if pair % 128 == 0:
                check()
            pair += 1
            selected = np.flatnonzero((matrix[i] == CONFLICT) & (matrix[j] == CONFLICT))
            if not len(selected):
                continue
            a, b = centers[i] - face_centers[selected], centers[j] - face_centers[selected]
            na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            valid = (na > 1e-12) & (nb > 1e-12)
            selected, a, b, na, nb = selected[valid], a[valid], b[valid], na[valid], nb[valid]
            cosine = np.einsum('ij,ij->i', a, b) / (na * nb)
            angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            improved = (~out['angle_pair_evaluated'][selected]) | (angles > out['max_conflict_angle_degrees'][selected])
            out['angle_pair_evaluated'][selected] = True
            out['max_conflict_angle_degrees'][selected[improved]] = angles[improved]
            for idx in selected[improved]:
                out['witness_view_ids'][idx] = [ids[i], ids[j]]
    out['quarantine'][:] = out['angle_pair_evaluated'] & (out['max_conflict_angle_degrees'] >= policy.min_angle_degrees)
    out['quarantine_face_indices'] = np.flatnonzero(out['quarantine'])
    out['kept_face_indices'] = np.flatnonzero(~out['quarantine'])
    out['kept_status'] = 'KEPT_NOT_ACCEPTED'
    out['limitations'] = ['Visible masks, poses and rigid-scene correctness remain assumptions.',
                          'External occlusion is unresolved; conflicts are candidates, not established errors.',
                          'No new geometry, visibility proof, physical accuracy or quality acceptance.']
    check()
    return out
