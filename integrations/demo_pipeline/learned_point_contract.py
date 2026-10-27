"""Pure coordinate and candidate gates; these do not establish a real measurement."""

from numbers import Integral, Real

import numpy as np


def _size(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _points(value, shape_tail=2, finite=True):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != shape_tail:
        raise ValueError("points must be a numeric Qx2 array")
    result = raw.astype(np.float64)
    if finite and not np.isfinite(result).all():
        raise ValueError("points must be finite")
    return result


def cv_to_model_queries(points_xy, width, height, model_size=256):
    """Convert OpenCV integer-centered xy into model boundary-origin [0,y,x]."""
    width, height, model_size = (_size(v, n) for v, n in
                                 ((width, "width"), (height, "height"), (model_size, "model_size")))
    points = _points(points_xy)
    scaled = (points + 0.5) * np.array([model_size / width, model_size / height])
    return np.column_stack((np.zeros(len(points)), scaled[:, 1], scaled[:, 0]))


def model_to_cv(points_yx, width, height, model_size=256):
    """Convert model boundary-origin yx to OpenCV integer-centered xy."""
    width, height, model_size = (_size(v, n) for v, n in
                                 ((width, "width"), (height, "height"), (model_size, "model_size")))
    points = _points(points_yx)
    return points[:, ::-1] * np.array([width / model_size, height / model_size]) - 0.5


def _probability(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability")
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0,1]")
    return float(value)


def _validate_mask(mask, width, height):
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (height, width) or mask.dtype.kind not in "iuf" or not np.isin(mask, [0, 255]).all():
            raise ValueError("mask must have image dimensions and binary values 0 or 255")
    return mask


def _probabilities(values, count, name):
    values = np.asarray(values)
    if values.shape != (count,) or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a numeric Q vector")
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError(f"{name} must contain finite probabilities in [0,1]")
    return values.astype(np.float64)


def classify_points(points_xy, probabilities, certainties, mask, width, height):
    """Gate a learned candidate; VISIBLE_CANDIDATE is not measurement acceptance.

    Coordinates use [0,width) x [0,height). Mask lookup rounds half up;
    rounding beyond the mask has no foreground support and is OUTSIDE_MASK.
    All supplied inputs are validated even if an earlier gate would reject.
    """
    width, height = _size(width, "width"), _size(height, "height")
    points = _points(points_xy, finite=False)
    probabilities = _probabilities(probabilities, len(points), "probabilities")
    certainties = _probabilities(certainties, len(points), "certainties")
    mask = _validate_mask(mask, width, height)
    statuses = []
    for point, pv, cert in zip(points, probabilities, certainties):
        if pv * cert <= 0.5:
            statuses.append("OCCLUDED_OR_UNCERTAIN")
            continue
        x, y = point
        if not np.isfinite(point).all() or not (0 <= x < width and 0 <= y < height):
            statuses.append("OUTSIDE_IMAGE")
            continue
        if mask is None:
            statuses.append("MASK_UNAVAILABLE")
            continue
        ix, iy = int(np.floor(x + 0.5)), int(np.floor(y + 0.5))
        statuses.append("OUTSIDE_MASK" if ix >= width or iy >= height or mask[iy, ix] != 255
                        else "VISIBLE_CANDIDATE")
    return statuses


def classify_point(xy, p_visible, certainty, mask, width, height):
    """Single-point adapter to the shared batch rules; not measurement acceptance."""
    raw = np.asarray(xy)
    if raw.shape != (2,):
        raise ValueError("xy must have shape (2,)")
    pv, cert = _probability(p_visible, "p_visible"), _probability(certainty, "certainty")
    return classify_points(raw[None], np.array([pv]), np.array([cert]), mask, width, height)[0]
