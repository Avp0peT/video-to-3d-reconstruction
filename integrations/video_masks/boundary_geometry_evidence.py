"""RGB/depth disagreement diagnostics, without mesh edits or acceptance.

Inputs must already share the native OpenMVS pinhole domain: integer pixel
centres and camera-Z depth. RGB support is an appearance observation, not a new
segmentation label. A conflict covers every enumerated triangle pixel centre,
not its entire continuous footprint, and remains conditional on camera/depth
accuracy, rigidity and visibility. Missing evidence never becomes free space.
"""
from __future__ import annotations

import numpy as np

from .face_edge_gate import _project
from .face_mask_gate import _camera, _mesh, _view_id, UNKNOWN, CONFLICT


MAX_FACES = 400_000
MAX_IMAGE_PIXELS = 4_000_000
MAX_RASTER_PIXELS_PER_FACE = 200_000
MIN_PIXEL_CENTERS = 4
DEPTH_RELATIVE_TOLERANCE = .02
REASON_NAMES = dict(enumerate((
    'NOT_EVALUATED', 'CAMERA_PLANE', 'OUT_OF_FRAME',
    'INVALID_VERTEX_DOMAIN', 'DEGENERATE_PROJECTION',
    'NO_SUPPORTED_GAP_IN_BBOX', 'MISSING_DEPTH', 'RASTER_BUDGET',
    'INSUFFICIENT_PIXEL_CENTERS', 'INVALID_RASTER_DOMAIN',
    'PARTIAL_RGB_SUPPORT', 'UNKNOWN_DEPTH', 'INVALID_FACE_DEPTH',
    'OCCLUDED_SAMPLES', 'SURFACE_MATCH_OR_INSUFFICIENT_CLEARANCE',
    'CORE_PROTECTED', 'RGB_AND_DEPTH_CONFLICT',
)))
_REASON = {name: key for key, name in REASON_NAMES.items()}


def _positive_probes(vertices, faces, k, r, c, mask, valid, depth):
    """Original foreground is protected independently of the RGB evidence map.

    This intentionally protects boundary foreground too: any central-depth
    match at a vertex or the projected 3D centroid is a conservative veto. It
    is sampled protection, not a claim that all interior support was found.
    """
    def supported(points):
        z, _, pixels, inside = _project(points, k, r, c, mask.shape)
        x, y = pixels.T
        saved = depth[y, x]
        return (inside & valid[y, x] & mask[y, x]
                & np.isfinite(saved) & (saved > 0)
                & (np.abs(z - saved) <= DEPTH_RELATIVE_TOLERANCE * saved))

    vertex = supported(vertices)
    centroid = supported(vertices[faces].mean(axis=1))
    return vertex[faces].sum(axis=1, dtype=np.uint8) + centroid.astype(np.uint8)


def _centers(triangle, low, high):
    """Enumerate the closed triangle's integer centres and screen barycentrics.

    A tiny outward roundoff allowance adds checks; it never drops boundary
    samples from an otherwise complete raster. Budget is checked by the caller.
    """
    xx, yy = np.meshgrid(np.arange(low[0], high[0] + 1),
                         np.arange(low[1], high[1] + 1))
    a, b, c = triangle
    ab, ac = b - a, c - a
    determinant = ab[0] * ac[1] - ab[1] * ac[0]
    dx, dy = xx - a[0], yy - a[1]
    second = (dx * ac[1] - dy * ac[0]) / determinant
    third = (ab[0] * dy - ab[1] * dx) / determinant
    weights = np.stack((1 - second - third, second, third), axis=-1)
    inside = (weights >= -1e-12).all(axis=-1) & (weights <= 1 + 1e-12).all(axis=-1)
    return xx[inside], yy[inside], weights[inside]


def evaluate_rgb_view(vertices, faces, *, K, R, C, mask, validity, depth,
                      supported_gap, view_id, check=lambda: None):
    """Return face-indexed arrays; status 0=UNKNOWN, 2=CONFLICT (never PASS).

    ``supported_gap`` may overlap foreground in ``mask`` (e.g. union additions).
    It must not replace that mask or its validity. CONFLICT requires >=4 actual
    integer centres, all of them valid and RGB-supported, and finite positive
    saved depth strictly farther than the perspective-correct face depth by 2%.
    Any original foreground probe supported by saved depth vetoes the vote.

    Counts are exact for completed rasters; zero with raster_evaluated=False is
    unmeasured. Bbox work, rather than only accepted pixels, is budgeted. The
    caller must bind provenance, aggregate independent views and preserve the
    NOT_ACCEPTED status. No files, inputs or geometry are changed here.
    """
    check()
    _view_id(view_id)
    v, f = _mesh(vertices, faces)
    if len(f) > MAX_FACES:
        raise ValueError('at most 400000 faces required')
    k, r, c = _camera(K, R, C)
    mask, valid, gap = (np.asarray(a) for a in (mask, validity, supported_gap))
    if (mask.ndim != 2 or not all(mask.shape) or mask.size > MAX_IMAGE_PIXELS
            or any(a.dtype != np.bool_ or a.shape != mask.shape for a in (mask, valid, gap))
            or np.any(mask & ~valid) or np.any(gap & ~valid)):
        raise ValueError('same-domain bool masks within valid 4M-pixel domain required')
    if depth is not None:
        depth = np.asarray(depth)
        if (depth.shape != mask.shape or not np.issubdtype(depth.dtype, np.number)
                or np.issubdtype(depth.dtype, np.complexfloating)):
            raise ValueError('real depth in the exact mask domain required')

    n = len(f)
    out = dict(view_id=view_id, status=np.full(n, UNKNOWN, np.uint8),
               negative=np.zeros(n, bool), reason=np.zeros(n, np.uint8),
               reason_names=REASON_NAMES.copy(), core_protected=np.zeros(n, bool),
               protected_sample_count=np.zeros(n, np.uint8),
               bbox_candidate=np.zeros(n, bool), raster_evaluated=np.zeros(n, bool),
               actual_pixel_count=np.zeros(n, np.uint32),
               supported_pixel_count=np.zeros(n, np.uint32),
               valid_pixel_count=np.zeros(n, np.uint32),
               depth_checked_pixel_count=np.zeros(n, np.uint32),
               front_pixel_count=np.zeros(n, np.uint32),
               evidence_pixel_count=np.zeros(n, np.uint32),
               whole_raster_supported=np.zeros(n, bool),
               all_depth_known=np.zeros(n, bool), all_samples_in_front=np.zeros(n, bool),
               minimum_relative_clearance=np.full(n, np.nan, np.float32),
               projected_candidate_count=0, depth_checked_face_count=0,
               negative_face_count=0, geometry_changed=False, source_mask_changed=False,
               quality_status='NOT_ACCEPTED', business_status='DIAGNOSTIC_ONLY',
               coverage='all enumerated integer pixel centres; not continuous triangle coverage',
               protection='original foreground central-depth matches at vertices and 3D centroid only',
               visibility='conditional on bound cameras, camera-Z depth and rigid scene; external occlusion not independently verified',
               limits=dict(max_faces=MAX_FACES, max_image_pixels=MAX_IMAGE_PIXELS,
                           max_bbox_pixels_per_face=MAX_RASTER_PIXELS_PER_FACE,
                           minimum_actual_pixel_centers=MIN_PIXEL_CENTERS,
                           depth_relative_tolerance=DEPTH_RELATIVE_TOLERANCE))
    if not n:
        return out

    z, uv, pixels, inframe = _project(v, k, r, c, mask.shape)
    front = (z[f] > 1e-12).all(axis=1)
    on_image = inframe[f].all(axis=1)
    vertex_valid = valid[pixels[:, 1], pixels[:, 0]][f].all(axis=1)
    out['reason'][:] = _REASON['CAMERA_PLANE']
    out['reason'][front] = _REASON['OUT_OF_FRAME']
    out['reason'][front & on_image] = _REASON['INVALID_VERTEX_DOMAIN']
    usable = front & on_image & vertex_valid

    if depth is not None:
        out['protected_sample_count'] = _positive_probes(v, f, k, r, c, mask, valid, depth)
        out['core_protected'] = out['protected_sample_count'] > 0
    check()
    ids = np.flatnonzero(usable)
    if len(ids):
        triangles = uv[f[ids]]
        ab, ac = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        determinant = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        scale = np.maximum(1., np.max(np.abs(triangles), axis=(1, 2)))
        nondegenerate = np.abs(determinant) > 64 * np.finfo(float).eps * scale ** 2
        out['reason'][ids] = _REASON['DEGENERATE_PROJECTION']
        ids = ids[nondegenerate]

    h, w = mask.shape
    if len(ids):
        triangles = uv[f[ids]]
        low = np.maximum(np.ceil(triangles.min(axis=1)).astype(np.int64), [0, 0])
        high = np.minimum(np.floor(triangles.max(axis=1)).astype(np.int64), [w - 1, h - 1])
        size = np.maximum(0, high - low + 1)
        work = size[:, 0] * size[:, 1]
        out['reason'][ids] = _REASON['NO_SUPPORTED_GAP_IN_BBOX']
        integral = np.pad(gap.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64),
                          ((1, 0), (1, 0)))
        x0, y0 = low.T
        # Empty bboxes cannot contribute and must not use inverted indexing.
        x1, y1 = (low + size).T
        count = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        candidate = (work > 0) & (count > 0)
        out['bbox_candidate'][ids[candidate]] = True
        out['projected_candidate_count'] = int(candidate.sum())
        out['reason'][ids[work == 0]] = _REASON['INSUFFICIENT_PIXEL_CENTERS']
        for iteration, j in enumerate(np.flatnonzero(candidate)):
            if iteration % 128 == 0:
                check()
            face_id = ids[j]
            if out['core_protected'][face_id]:
                continue
            if depth is None:
                out['reason'][face_id] = _REASON['MISSING_DEPTH']
                continue
            if work[j] > MAX_RASTER_PIXELS_PER_FACE:
                out['reason'][face_id] = _REASON['RASTER_BUDGET']
                continue
            x, y, barycentric = _centers(triangles[j], low[j], high[j])
            out['raster_evaluated'][face_id] = True
            total = len(x)
            out['actual_pixel_count'][face_id] = total
            out['valid_pixel_count'][face_id] = int(valid[y, x].sum())
            out['supported_pixel_count'][face_id] = int(gap[y, x].sum())
            if total < MIN_PIXEL_CENTERS:
                out['reason'][face_id] = _REASON['INSUFFICIENT_PIXEL_CENTERS']
                continue
            if not valid[y, x].all():
                out['reason'][face_id] = _REASON['INVALID_RASTER_DOMAIN']
                continue
            if not gap[y, x].all():
                out['reason'][face_id] = _REASON['PARTIAL_RGB_SUPPORT']
                continue
            out['whole_raster_supported'][face_id] = True
            saved = depth[y, x]
            known = np.isfinite(saved) & (saved > 0)
            out['depth_checked_pixel_count'][face_id] = int(known.sum())
            if not known.all():
                out['reason'][face_id] = _REASON['UNKNOWN_DEPTH']
                continue
            out['all_depth_known'][face_id] = True
            # Screen barycentrics interpolate reciprocal camera depth, not Z.
            reciprocal = barycentric @ (1. / z[f[face_id]])
            if not (np.isfinite(reciprocal) & (reciprocal > 0)).all():
                out['reason'][face_id] = _REASON['INVALID_FACE_DEPTH']
                continue
            with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                face_depth = 1. / reciprocal
            if not (np.isfinite(face_depth) & (face_depth > 0)).all():
                out['reason'][face_id] = _REASON['INVALID_FACE_DEPTH']
                continue
            out['depth_checked_face_count'] += 1
            threshold = saved * (1 - DEPTH_RELATIVE_TOLERANCE)
            # Require numerical separation too; exact 2% contact must never gain
            # a vote from reciprocal-interpolation roundoff.
            roundoff = 64 * np.finfo(float).eps * np.maximum(np.abs(face_depth), np.abs(threshold))
            separated = (threshold - face_depth) > roundoff
            out['front_pixel_count'][face_id] = int(separated.sum())
            out['minimum_relative_clearance'][face_id] = float(np.min((saved - face_depth) / saved))
            if not separated.all():
                reason = ('OCCLUDED_SAMPLES' if np.any(face_depth > saved * (1 + DEPTH_RELATIVE_TOLERANCE))
                          else 'SURFACE_MATCH_OR_INSUFFICIENT_CLEARANCE')
                out['reason'][face_id] = _REASON[reason]
                continue
            out['all_samples_in_front'][face_id] = True
            out['evidence_pixel_count'][face_id] = total
            out['negative'][face_id] = True
            out['status'][face_id] = CONFLICT
            out['reason'][face_id] = _REASON['RGB_AND_DEPTH_CONFLICT']

    out['reason'][out['core_protected']] = _REASON['CORE_PROTECTED']
    out['negative_face_count'] = int(out['negative'].sum())
    check()
    return out
