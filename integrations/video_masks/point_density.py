"""Image-domain support audit; thresholds are engineering starting values.

Coordinates use integer pixel centres (OpenCV), rounded by floor(x + .5).
Passing this audit says nothing about correspondence or geometry validity.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class DensityPolicy:
    grid_rows: int = 8
    grid_cols: int = 8
    min_points_per_cell: int = 4
    max_pixels: int = 16_777_216
    max_points: int = 1_000_000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(key + ' must be a positive integer, not bool')
        if self.grid_rows > 128 or self.grid_cols > 128:
            raise ValueError('grid dimensions exceed 128')
        if self.max_pixels > 16_777_216 or self.max_points > 1_000_000:
            raise ValueError('resource bounds may only be tightened')


def assess_density(mask, xy, policy=DensityPolicy()):
    """Audit every foreground-bearing bbox cell, without dropping boundaries.

    Cell target is ceil(min_points_per_cell * foreground / full cell area),
    at least one and capped by distinct foreground pixels. Coordinates outside the image or foreground are excluded;
    malformed/nonfinite coordinates are rejected. Duplicate rounded pixels
    count once. Largest deficient region uses 4-connected *image pixels*.
    """
    if not isinstance(policy, DensityPolicy):
        raise ValueError('DensityPolicy required')
    image = np.asarray(mask)
    if image.ndim != 2 or not image.size or image.size > policy.max_pixels:
        raise ValueError('nonempty bounded two-dimensional mask required')
    if image.dtype.kind not in 'biuf' or not np.isfinite(image).all():
        raise ValueError('finite numeric or boolean mask required')
    if not np.isin(image, (0, 1, 255)).all():
        raise ValueError('binary mask values required')
    # Lists preserve explicit bool intent; a pre-coerced numeric ndarray cannot.
    if isinstance(xy, (list, tuple)):
        for pair in xy:
            if isinstance(pair, (list, tuple, np.ndarray)) and any(
                    isinstance(value, (bool, np.bool_)) for value in pair):
                raise ValueError('boolean coordinates forbidden')
    values = np.asarray(xy)
    if values.size == 0 and values.shape == (0,):
        values = values.reshape(0, 2)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) > policy.max_points
            or values.dtype.kind not in 'iuf' or not np.isfinite(values).all()):
        raise ValueError('bounded finite numeric Nx2 coordinates required; bool forbidden')
    height, width = image.shape
    # Filter before conversion to avoid integer overflow from huge finite floats.
    inside = ((values[:, 0] >= -.5) & (values[:, 0] < width-.5)
              & (values[:, 1] >= -.5) & (values[:, 1] < height-.5))
    pixels = np.floor(values[inside].astype(float)+.5).astype(np.int64)
    if len(pixels):
        pixels = pixels[image[pixels[:, 1], pixels[:, 0]] != 0]
        pixels = np.unique(pixels, axis=0)
    fg = image != 0
    ys, xs = np.nonzero(fg)
    result = dict(schema='object-point-density-v1', policy=asdict(policy),
                  threshold_basis='ENGINEERING_STARTING_VALUES', geometry_accepted=False,
                  coordinate_domain='OPENCV_INTEGER_PIXEL_CENTRES', input_points=len(values),
                  unique_in_mask_points=len(pixels), foreground_pixels=len(xs), cells=[],
                  largest_deficient_region_fraction=0.0, met_area_fraction=0.0)
    if not len(xs):
        return dict(result, status='EMPTY_MASK', bbox_xyxy=None)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
    result['bbox_xyxy'] = [x0, y0, x1, y1]
    xb = np.linspace(x0, x1, min(policy.grid_cols, x1-x0)+1, dtype=int)
    yb = np.linspace(y0, y1, min(policy.grid_rows, y1-y0)+1, dtype=int)
    deficient = np.zeros(fg.shape, dtype=bool)
    met_area = 0
    for row, (top, bottom) in enumerate(zip(yb[:-1], yb[1:])):
        for col, (left, right) in enumerate(zip(xb[:-1], xb[1:])):
            patch = fg[top:bottom, left:right]
            area = int(patch.sum())
            if not area:
                continue
            target = min(area, max(1, math.ceil(policy.min_points_per_cell * area / patch.size)))
            count = int(((pixels[:, 0] >= left) & (pixels[:, 0] < right)
                         & (pixels[:, 1] >= top) & (pixels[:, 1] < bottom)).sum())
            deficit = max(0, target-count)
            result['cells'].append(dict(row=row, col=col, cell_id=row*(len(xb)-1)+col, bounds=list(map(int, (left, top, right, bottom))),
                                       foreground_pixels=area, point_count=count, target_points=target, deficit=deficit))
            if deficit:
                deficient[top:bottom, left:right] = patch
            else:
                met_area += area
    # Run-length components keep memory bounded by foreground runs, not Python pixels.
    parent, sizes = [], []
    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    previous = []
    largest = 0
    for line in deficient[y0:y1, x0:x1]:
        edges = np.diff(np.r_[False, line, False].astype(np.int8))
        current = []
        cursor = 0
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            index = len(parent)
            parent.append(index); sizes.append(int(right-left))
            while cursor < len(previous) and previous[cursor][1] <= left:
                cursor += 1
            other = cursor
            while other < len(previous) and previous[other][0] < right:
                a, b = root(index), root(previous[other][2])
                if a != b:
                    parent[b] = a; sizes[a] += sizes[b]
                other += 1
            largest = max(largest, sizes[root(index)])
            current.append((left, right, index))
        previous = current
    result.update(met_area_fraction=met_area/len(xs),
                  largest_deficient_region_fraction=largest/len(xs),
                  status='DENSITY_TARGET_MET' if met_area == len(xs) else 'BELOW_DENSITY_TARGET')
    return result
