"""Binary mask remapping between cameras with the same pose and pixel axes.

The caller must bind image identities and verify the poses. No model, image,
camera, or source array is changed here. Invalid mappings have a separate
validity mask; zero output alone must not be interpreted as observed background.
"""
from dataclasses import dataclass
import numpy as np

ALGORITHM = "colmap-center-nearest-label-v1"


@dataclass(frozen=True)
class WarpedMask:
    mask: np.ndarray
    valid: np.ndarray
    counts: dict


def _size(camera):
    width, height = camera.width, camera.height
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("camera dimensions must be positive integers")
    params = np.asarray(camera.params, dtype=np.float64)
    if not camera.verify_params() or not np.isfinite(params).all():
        raise ValueError("invalid camera parameters")
    matrix = np.asarray(camera.calibration_matrix(), dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 0:
        raise ValueError("camera must have finite positive focal lengths")
    return width, height


def warp_binary_mask(mask, source_camera, target_camera, *, tile_rows=64, max_pixels=16_777_216):
    """Inverse-map uint8 {0,255} labels into an existing undistorted camera.

    Destination pixel centers are (column+.5,row+.5), as in COLMAP. Its inverse
    ray is projected with the complete source model. Nearest label sampling uses
    floor(projected COLMAP coordinates); exact half-way ties go to the larger
    array index. Raster support is [0,width) x [0,height), not clamped.
    This does not certify RGB bilinear support, segmentation quality or coverage.
    """
    if type(tile_rows) is not int or tile_rows <= 0 or type(max_pixels) is not int or max_pixels <= 0:
        raise ValueError("tile_rows and max_pixels must be positive integers")
    sw, sh = _size(source_camera)
    tw, th = _size(target_camera)
    if target_camera.model_name not in ("PINHOLE", "SIMPLE_PINHOLE"):
        raise ValueError("target camera must be undistorted pinhole")
    if sw * sh > max_pixels or tw * th > max_pixels:
        raise ValueError("image exceeds max_pixels")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != (sh, sw):
        raise ValueError("mask must be uint8 HxW matching source camera")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("mask labels must be 0/255; no implicit threshold")
    out = np.zeros((th, tw), dtype=np.uint8)
    validity = np.zeros((th, tw), dtype=np.uint8)
    x = np.arange(tw, dtype=np.float64) + 0.5
    invalid_projection = 0
    outside = 0
    for start in range(0, th, tile_rows):
        end = min(start + tile_rows, th)
        xx, yy = np.meshgrid(x, np.arange(start, end, dtype=np.float64) + 0.5)
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        rays = np.asarray(target_camera.cam_from_img(xy), dtype=np.float64)
        if rays.shape != xy.shape:
            raise ValueError("unprojection returned wrong shape")
        finite_ray = np.isfinite(rays).all(axis=1)
        mapped = np.full(xy.shape, np.nan, dtype=np.float64)
        if finite_ray.any():
            projected = np.asarray(source_camera.img_from_cam(
                np.column_stack((rays[finite_ray], np.ones(int(finite_ray.sum()))))), dtype=np.float64)
            if projected.shape != (int(finite_ray.sum()), 2):
                raise ValueError("projection returned wrong shape")
            mapped[finite_ray] = projected
        finite = np.isfinite(mapped).all(axis=1)
        inside = (finite & (mapped[:, 0] >= 0) & (mapped[:, 0] < sw)
                  & (mapped[:, 1] >= 0) & (mapped[:, 1] < sh))
        # Do not cast NaN/Inf or out-of-range values to integer indices.
        indices = np.floor(mapped[inside]).astype(np.int64)
        tile = out[start:end].reshape(-1)
        tile[inside] = mask[indices[:, 1], indices[:, 0]]
        validity[start:end].reshape(-1)[inside] = 255
        invalid_projection += int((~finite).sum())
        outside += int((finite & ~inside).sum())
    valid_pixels = int(np.count_nonzero(validity))
    counts = dict(total=tw * th, valid=valid_pixels, unknown=tw * th - valid_pixels,
                  invalid_projection=invalid_projection, outside_source=outside,
                  foreground=int(np.count_nonzero(out)))
    assert counts["unknown"] == invalid_projection + outside
    return WarpedMask(out, validity, counts)
