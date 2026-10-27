"""Fixed image/descriptor appearance evidence, without search or correspondence truth."""
import numpy as np


def _image(gray, mask):
    if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError('2D uint8 grayscale array required')
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != gray.shape:
        raise ValueError('same-size uint8 mask required')
    if not np.isin(mask, [0, 255]).all():
        raise ValueError('mask labels must be 0/255')


def _center(xy):
    try:
        if len(xy) != 2 or any(isinstance(v, (bool, np.bool_)) or
                not isinstance(v, (int, float, np.integer, np.floating)) for v in xy):
            raise ValueError('finite real two-coordinate center required')
        center = np.array(xy, dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError('finite real two-coordinate center required') from exc
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('finite real two-coordinate center required')
    return center


def _patch(gray, mask, xy, radius):
    x, y = np.meshgrid(xy[0]+np.arange(-radius, radius+1),
                       xy[1]+np.arange(-radius, radius+1))
    h, w = gray.shape
    if np.any(x < 0) or np.any(x > w-1) or np.any(y < 0) or np.any(y > h-1):
        return None, dict(status='NOT_EVALUATED', reason='OUT_OF_IMAGE_BOUNDS',
                          foreground_fraction=None, center=xy.tolist())
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    dx = x-x0; dy = y-y0; values = np.zeros(x.shape); inside = np.ones(x.shape, bool)
    for ox, oy, weight in ((0, 0, (1-dx)*(1-dy)), (1, 0, dx*(1-dy)),
                           (0, 1, (1-dx)*dy), (1, 1, dx*dy)):
        used = weight > 0
        # Only positive-weight neighbors count; zero-weight edge neighbors can be out of bounds.
        xx = np.clip(x0+ox, 0, w-1); yy = np.clip(y0+oy, 0, h-1)
        values += weight*gray[yy, xx]
        inside &= ~used | (mask[yy, xx] == 255)
    fraction = float(np.mean(inside))
    stats = dict(status='EVALUATED', center=xy.tolist(), foreground_fraction=fraction,
                 sample_count=int(values.size), mean=float(values.mean()), std=float(values.std()),
                 min=float(values.min()), max=float(values.max()))
    if fraction < 1:
        stats.update(status='NOT_EVALUATED_MASK_BOUNDARY', reason='POSITIVE_WEIGHT_SOURCE_PIXEL_OUTSIDE_MASK')
    elif stats['std'] <= 1e-6:
        stats.update(status='NOT_EVALUATED', reason='NUMERICALLY_ZERO_TEXTURE')
    return values, stats


def _support(original, alternative, higher):
    difference = original-alternative
    if abs(difference) <= 1e-10:
        return 'tie'
    return 'original' if (difference > 0) == higher else 'alternative'


def compare_patches(reference_gray, reference_mask, reference_xy, target_gray,
                    target_mask, original_xy, alternative_xy, radius):
    """Sample three fixed bilinear patches; do not select a radius or change coordinates."""
    if type(radius) is not int or radius not in (8, 16):
        raise ValueError('radius must be frozen at 8 or 16')
    _image(reference_gray, reference_mask); _image(target_gray, target_mask)
    centers = [_center(xy) for xy in (reference_xy, original_xy, alternative_xy)]
    sampled = [_patch(g, m, c, radius) for g, m, c in
               zip((reference_gray, target_gray, target_gray),
                   (reference_mask, target_mask, target_mask), centers)]
    stats = dict(zip(('reference', 'original', 'alternative'), (s for _, s in sampled)))
    result = dict(status='EVALUATED', radius=radius, patch_statistics=stats,
                  original_zncc=None, alternative_zncc=None, support=None,
                  score_semantics='ZNCC appearance similarity, not probability or correspondence truth',
                  equality_tolerance=1e-10)
    if any(s['status'] == 'NOT_EVALUATED_MASK_BOUNDARY' for s in stats.values()):
        result.update(status='NOT_EVALUATED_MASK_BOUNDARY', reason='AT_LEAST_ONE_PATCH_CROSSES_TARGET_MASK')
        return result
    if any(s['status'] != 'EVALUATED' for s in stats.values()):
        result.update(status='NOT_EVALUATED', reason='PATCH_BOUNDARY_OR_NUMERICALLY_ZERO_TEXTURE')
        return result
    normalized = [(a-a.mean())/a.std() for a, _ in sampled]
    original = float(np.clip(np.mean(normalized[0]*normalized[1]), -1, 1))
    alternative = float(np.clip(np.mean(normalized[0]*normalized[2]), -1, 1))
    result.update(original_zncc=original, alternative_zncc=alternative,
                  support=_support(original, alternative, True))
    return result


def descriptor_evidence(reference, original, alternative):
    """Compare fixed uint8 SIFT vectors after independent L2 normalization."""
    descriptors = (reference, original, alternative)
    if any(not isinstance(a, np.ndarray) or a.dtype != np.uint8 or a.shape != (128,) for a in descriptors):
        raise ValueError('three uint8[128] descriptors required')
    converted = [a.astype(float) for a in descriptors]
    norms = [float(np.linalg.norm(a)) for a in converted]
    result = dict(status='EVALUATED', original_distance=None, alternative_distance=None,
                  support=None, input_norms=norms, equality_tolerance=1e-10,
                  score_semantics='L2-normalized Euclidean descriptor distance, not probability')
    if any(n == 0 for n in norms):
        result.update(status='NOT_EVALUATED', reason='ZERO_DESCRIPTOR_NORM')
        return result
    normalized = [a/n for a, n in zip(converted, norms)]
    original_distance = float(np.linalg.norm(normalized[0]-normalized[1]))
    alternative_distance = float(np.linalg.norm(normalized[0]-normalized[2]))
    result.update(original_distance=original_distance, alternative_distance=alternative_distance,
                  support=_support(original_distance, alternative_distance, False))
    return result
