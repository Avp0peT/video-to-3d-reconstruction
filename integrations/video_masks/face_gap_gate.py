"""Negative silhouette evidence in bounded foreground-background-foreground gaps.

Foreground banks belong to the mask, not to vertices of a possibly incorrect
mesh. In particular, a subdivided curtain whose vertices are all in a gap must
not escape examination merely because it lacks positive depth support.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .face_mask_gate import (
    UNKNOWN, NO_CONFLICT, CONFLICT, _mesh, _camera, _view_id, _cv,
    _distances, _integral_counts, _raster_pixels,
)

REASONS = dict(enumerate(('NOT_EVALUATED', 'NO_BRACKETED_GAP', 'CAMERA_PLANE',
    'OUT_OF_FRAME', 'INVALID_VERTEX_DOMAIN', 'SUBPIXEL_OR_DEGENERATE',
    'RASTER_BUDGET', 'NO_INTERIOR_PIXEL_CENTERS', 'INVALID_SAMPLE_DOMAIN',
    'GAP_CONFLICT', 'BELOW_GAP_THRESHOLD')))
_R = {value: key for key, value in REASONS.items()}


@dataclass(frozen=True)
class GapPolicy:
    boundary_px: float = 3.0
    min_gap_pixels: int = 4
    min_gap_fraction: float = 0.01
    max_gap_span_fraction: float = 0.20
    min_angle_degrees: float = 8.0
    min_views: int = 2
    max_raster_pixels_per_face: int = 200000

    def __post_init__(self):
        for key in ('boundary_px', 'min_gap_fraction', 'max_gap_span_fraction', 'min_angle_degrees'):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('finite gap policy required: '+key)
        if not 0 <= self.boundary_px <= 100 or not 0 < self.min_gap_fraction <= 1:
            raise ValueError('invalid boundary/fraction')
        if not 0 < self.max_gap_span_fraction <= .5 or not 0 < self.min_angle_degrees <= 180:
            raise ValueError('invalid gap span/angle')
        for key in ('min_gap_pixels', 'min_views', 'max_raster_pixels_per_face'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError('positive gap integer required: '+key)
        if self.min_views < 2:
            raise ValueError('at least two unique views required')


def _brackets(banks, validity, axis, maximum_steps):
    """Nearest opposed foreground banks with no unknown pixel between them."""
    b, valid = np.moveaxis(banks, axis, -1), np.moveaxis(validity, axis, -1)
    n = b.shape[-1]
    x = np.arange(n, dtype=np.int32)
    left = np.maximum.accumulate(np.where(b, x, -1), axis=-1)
    right = np.minimum.accumulate(np.where(b, x, n)[..., ::-1], axis=-1)[..., ::-1]
    bad_left = np.maximum.accumulate(np.where(~valid, x, -1), axis=-1)
    bad_right = np.minimum.accumulate(np.where(~valid, x, n)[..., ::-1], axis=-1)[..., ::-1]
    result = ((left >= 0) & (right < n) & (left < x) & (x < right)
              & ((right-left) <= maximum_steps) & (bad_left < left) & (bad_right > right))
    return np.moveaxis(result, -1, axis)


def bracketed_gap_mask(mask, validity, policy=GapPolicy(), check=lambda: None):
    mask, validity = np.asarray(mask), np.asarray(validity)
    if (mask.dtype != np.bool_ or validity.dtype != np.bool_ or mask.ndim != 2
            or mask.shape != validity.shape or not all(mask.shape) or mask.size > 4_194_304):
        raise ValueError('same-domain bounded boolean masks required')
    if np.any(mask & ~validity):
        raise ValueError('foreground cannot occupy unknown domain')
    check()
    foreground = mask & validity
    yy, xx = np.nonzero(foreground)
    if not len(xx):
        return np.zeros_like(mask), dict(gap_pixels=0, foreground_pixels=0, reason='NO_FOREGROUND')
    extent = int(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1))
    max_span = float(extent*policy.max_gap_span_fraction)
    cv = _cv()
    # One-pixel foreground erosion gives each bank a local 3x3 support, but does
    # not change the supplied object mask or add any foreground.
    banks = cv.erode(foreground.astype('uint8'), np.ones((3, 3), np.uint8),
                     borderType=cv.BORDER_CONSTANT, borderValue=0) != 0
    bracket = _brackets(banks, validity, 1, max_span) | _brackets(banks, validity, 0, max_span)
    h, w = mask.shape
    directions = ['horizontal', 'vertical']
    # Integer skewing preserves diagonal pixel identities (no image resampling).
    if h*(w+h-1) <= 8_388_608:
        y, x = np.indices(mask.shape, dtype=np.int32)
        for sign in (-1, 1):
            check()
            column = x+y if sign == 1 else x+(h-1-y)
            skew_banks = np.zeros((h, w+h-1), bool)
            skew_valid = np.zeros_like(skew_banks)
            skew_banks[y, column] = banks
            skew_valid[y, column] = validity
            allowed = _brackets(skew_banks, skew_valid, 0, max_span/math.sqrt(2))
            bracket |= allowed[y, column]
            directions.append('diagonal_'+str(sign))
    _, deep = _distances(mask, validity)
    gap = bracket & (deep > policy.boundary_px) & ~mask & validity
    check()
    return gap, dict(gap_pixels=int(gap.sum()), foreground_pixels=int(foreground.sum()),
                     bank_pixels=int(banks.sum()), object_extent_pixels=extent,
                     maximum_bank_distance_pixels=max_span, tested_directions=directions,
                     bank_support='3x3 foreground, opposed nearest banks, no unknown pixel in between',
                     source_mask_changed=False, physical_visibility='NOT_EVALUATED')


def evaluate_gap_view(vertices, faces, *, K, R, C, mask, validity, view_id,
                      policy=GapPolicy(), check=lambda: None):
    if not isinstance(policy, GapPolicy):
        raise ValueError('GapPolicy required')
    _view_id(view_id)
    v, f = _mesh(vertices, faces)
    k, r, c = _camera(K, R, C)
    gap, metadata = bracketed_gap_mask(mask, validity, policy, check)
    n = len(f); h, w = mask.shape
    out = dict(view_id=view_id, status=np.zeros(n, np.uint8), reason=np.zeros(n, np.uint8),
               reason_names=REASONS.copy(), face_indices=np.arange(n),
               background_pixels=np.zeros(n, np.int32), total_pixels=np.zeros(n, np.int32),
               projected_area_pixels=np.zeros(n, float), raster_evaluated=np.zeros(n, bool),
               gap_mask=gap, gap_metadata=metadata, geometry_accepted=False)
    if not gap.any():
        out['reason'][:] = _R['NO_BRACKETED_GAP']
        return out
    cam = (v-c)@r.T; homogeneous = cam@k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = homogeneous[:, :2]/homogeneous[:, 2:]
    front = (cam[:, 2] > 1e-12)&np.isfinite(uv).all(axis=1)
    inside = front&(uv[:, 0] >= -.5)&(uv[:, 0] < w-.5)&(uv[:, 1] >= -.5)&(uv[:, 1] < h-.5)
    pix = np.zeros((len(v), 2), dtype=int); pix[inside] = np.floor(uv[inside]+.5).astype(int)
    valid = inside & validity[pix[:, 1], pix[:, 0]]
    pending = np.ones(n, bool)
    def reject(condition, reason):
        ix = pending & condition
        out['reason'][ix] = _R[reason]; pending[ix] = False
    reject(~front[f].all(axis=1), 'CAMERA_PLANE')
    reject(~inside[f].all(axis=1), 'OUT_OF_FRAME')
    reject(~valid[f].all(axis=1), 'INVALID_VERTEX_DOMAIN')
    triangles = uv[f]
    with np.errstate(invalid='ignore'):
        ab, ac = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
        area = .5*np.abs(ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0])
    out['projected_area_pixels'][:] = np.where(np.isfinite(area), area, 0)
    reject(~np.isfinite(area)|(area < .5), 'SUBPIXEL_OR_DEGENERATE')
    selected = np.flatnonzero(pending)
    if len(selected):
        tri = triangles[selected]
        lo, hi = np.floor(tri.min(axis=1)+.5).astype(int), np.floor(tri.max(axis=1)+.5).astype(int)
        absent = selected[_integral_counts(gap, lo, hi) == 0]
        out['reason'][absent] = _R['NO_BRACKETED_GAP']; pending[absent] = False
    flat_gap, flat_valid = gap.ravel(), validity.ravel()
    for ordinal, fid in enumerate(np.flatnonzero(pending)):
        if ordinal % 128 == 0:
            check()
        interior, edges, error = _raster_pixels(triangles[fid], w, h, policy.max_raster_pixels_per_face)
        if error:
            out['reason'][fid] = _R[error]
            continue
        samples = np.union1d(interior, edges)
        out['raster_evaluated'][fid] = True; out['total_pixels'][fid] = len(samples)
        if not flat_valid[samples].all():
            out['reason'][fid] = _R['INVALID_SAMPLE_DOMAIN']
            continue
        count = int(np.count_nonzero(flat_gap[samples]))
        out['background_pixels'][fid] = count
        conflict = count >= policy.min_gap_pixels and count >= len(samples)*policy.min_gap_fraction
        out['status'][fid] = CONFLICT if conflict else NO_CONFLICT
        out['reason'][fid] = _R['GAP_CONFLICT' if conflict else 'BELOW_GAP_THRESHOLD']
    check()
    return out
