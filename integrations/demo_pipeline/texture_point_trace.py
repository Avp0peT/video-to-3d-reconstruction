"""Fixed source texture evidence for one fractional CV-coordinate point."""
from copy import deepcopy
import math
import numpy as np
from .area_alignment import align_area, PROFILE as AREA_PROFILE

PROFILE = dict(**deepcopy(AREA_PROFILE), search_size=129, center_cycle_max_px=1.5,
    prediction_cap_px=48, template_policy='FIXED_FIRST_FRAME',
    coordinate_policy='CV_PIXEL_COORDINATES', ambiguity_separation_px=12,
    ambiguity_second_peak_min=.80, ambiguity_margin_min=.04,
    center_mask_support='all bilinear contributors in 3x3 neighborhood',
    permanent_stop=True)
PROFILE['id'] = 'fixed-texture-point-stop-v1'


def _crop(image, center, size):
    x, y = np.rint(center).astype(int)
    h = size // 2
    left, top = max(0, x-h), max(0, y-h)
    right, bottom = min(image.shape[1], x+h+1), min(image.shape[0], y+h+1)
    if right-left < 33 or bottom-top < 33 or (size == 33 and (right-left != 33 or bottom-top != 33)):
        return None, None
    return image[top:bottom, left:right].copy(), np.array([left, top], float)


def _center_supported(mask, point):
    # All bilinear contributors for a 3x3 neighborhood must be foreground.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = np.floor(np.asarray(point)+[dx, dy]).astype(int)
            if x < 0 or y < 0 or x+1 >= mask.shape[1] or y+1 >= mask.shape[0]:
                return False
            if not np.all(mask[y:y+2, x:x+2] == 255):
                return False
    return True


class _Capture:
    def __init__(self, cv):
        self.cv = cv
        self.response = None
    def __getattr__(self, key):
        return getattr(self.cv, key)
    def matchTemplate(self, *args, **kwargs):
        self.response = self.cv.matchTemplate(*args, **kwargs)
        return self.response


def _ambiguous(response, answer, mask, local_seed):
    if answer['status'] != 'ACCEPTED':
        return False, None
    best = answer['initialization']['translation_cv_xy']
    # Native scores with partial masks are not the same objective as the
    # supported NCC. Conservatively stop on a competing finite native peak.
    second = None
    for y, x in np.argwhere(np.isfinite(response) & (response >= .80)):
        if math.hypot(float(x-best[0]), float(y-best[1])) <= 12:
            continue
        candidate = local_seed + [x, y]
        if not _center_supported(mask, candidate):
            continue
        if np.mean(mask[y:y+33, x:x+33] == 255) < .85:
            continue
        value = float(response[y, x])
        second = value if second is None else max(second, value)
    primary = float(answer['initialization']['supported_train_ncc'])
    return second is not None and second >= .80 and primary-second < .04, second


def trace(frames, masks, indices, times, seed_xy, cv, budget=lambda: None):
    """Return a permanent accepted prefix; all coordinates are global CV pixels.

    Native exceptions propagate. A stopped trace never calls native code again.
    """
    n = len(frames)
    if not n or not (len(masks) == len(indices) == len(times) == n):
        raise ValueError('INVALID_SEQUENCE_LENGTH')
    seed = np.asarray(seed_xy, float)
    if seed.shape != (2,) or not np.isfinite(seed).all():
        raise ValueError('INVALID_SEED')
    if any(type(i) is not int for i in indices) or any(b-a != 1 for a, b in zip(indices, indices[1:])):
        raise ValueError('NONCONTIGUOUS_DECODE_INDICES')
    if any(not math.isfinite(float(t)) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('INVALID_TIMES')
    for frame, mask in zip(frames, masks):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 2:
            raise ValueError('INVALID_UINT8_GRAY')
        if mask is not None and (not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != frame.shape or not np.isin(mask, [0, 255]).all()):
            raise ValueError('INVALID_BINARY_MASK')
    result = dict(profile=deepcopy(PROFILE),
        frames=[], physical_identity_verified=False, occlusion_confirmed=False,
        native_calls=dict(matchTemplate=0, findTransformECCWithMask=0), stop_decode_index=None)
    template, origin = _crop(frames[0], seed, 33)
    template_mask = None if origin is None or masks[0] is None else _crop(masks[0], seed, 33)[0]
    accepted = []
    stopped = False
    def stop(row, reason):
        nonlocal stopped
        stopped = True
        row['status'] = 'STOPPED_APPEARANCE_UNVERIFIED'
        row['accepted_xy'] = None
        row['reasons'].append(reason)
        result['stop_decode_index'] = row['decode_index']
    def align(a, b, am, bm, local_seed):
        proxy = _Capture(cv)
        answer = align_area(a, b, am, bm, proxy, budget=budget)
        for key, value in answer['native_calls'].items():
            result['native_calls'][key] += value
        ambiguous, second = _ambiguous(proxy.response, answer, bm, local_seed)
        answer['ambiguity'] = dict(second_peak=second, excluded_radius_px=12, rejected=ambiguous)
        if ambiguous:
            answer['status'] = 'REJECTED'
            answer['reasons'].append('AMBIGUOUS_REPEATED_TEXTURE')
        return answer
    for i, (frame, mask) in enumerate(zip(frames, masks)):
        budget()
        row = dict(decode_index=indices[i], time_seconds=float(times[i]), status='NOT_RUN_AFTER_STOP',
            accepted_xy=None, candidate_xy=None, reasons=[], forward=None, backward=None, cycle_error_px=None)
        result['frames'].append(row)
        if stopped:
            continue
        if mask is None:
            stop(row, 'UNKNOWN_MASK'); continue
        if i == 0:
            if template is None:
                stop(row, 'SOURCE_TEMPLATE_BOUNDS'); continue
            if not _center_supported(mask, seed):
                stop(row, 'SOURCE_CENTER_UNSUPPORTED'); continue
            if np.mean(template_mask == 255) < .85:
                stop(row, 'SOURCE_NEIGHBORHOOD_UNSUPPORTED'); continue
            row.update(status='ACCEPTED', accepted_xy=seed.tolist(), candidate_xy=seed.tolist())
            accepted.append((seed.copy(), float(times[i])))
            continue
        prediction = accepted[-1][0].copy()
        if len(accepted) >= 2:
            displacement = (accepted[-1][0]-accepted[-2][0]) * ((times[i]-accepted[-1][1])/(accepted[-1][1]-accepted[-2][1]))
            norm = float(np.linalg.norm(displacement))
            if norm > 48:
                displacement *= 48/norm
            prediction += displacement
        search, search_origin = _crop(frame, prediction, 129)
        if search is None:
            stop(row, 'SEARCH_BOUNDS'); continue
        search_mask = _crop(mask, prediction, 129)[0]
        forward = align(template, search, template_mask, search_mask, seed-origin)
        forward['template_origin_cv_xy'] = origin.tolist()
        forward['search_origin_cv_xy'] = search_origin.tolist()
        forward['prediction_cv_xy'] = prediction.tolist()
        row['forward'] = forward
        if forward['status'] != 'ACCEPTED':
            stop(row, 'FORWARD_'+forward['status']); row['reasons'].extend(forward['reasons']); continue
        F = np.asarray(forward['matrix_cv2x3'], float)
        point = F[:, :2] @ (seed-origin) + F[:, 2] + search_origin
        row['candidate_xy'] = point.tolist()
        corners = np.array([[0, 0], [32, 0], [32, 32], [0, 32]], float)
        row['mapped_polygon'] = (corners @ F[:, :2].T + F[:, 2] + search_origin).tolist()
        if not _center_supported(mask, point):
            stop(row, 'TARGET_CENTER_UNSUPPORTED'); continue
        reverse_template, reverse_origin = _crop(frame, point, 33)
        source_search, source_origin = _crop(frames[0], seed, 129)
        if reverse_template is None or source_search is None:
            stop(row, 'REVERSE_CROP_BOUNDS'); continue
        reverse_mask = _crop(mask, point, 33)[0]
        source_search_mask = _crop(masks[0], seed, 129)[0]
        backward = align(reverse_template, source_search, reverse_mask, source_search_mask, point-reverse_origin)
        backward['template_origin_cv_xy'] = reverse_origin.tolist()
        backward['search_origin_cv_xy'] = source_origin.tolist()
        row['backward'] = backward
        if backward['status'] != 'ACCEPTED':
            stop(row, 'BACKWARD_'+backward['status']); row['reasons'].extend(backward['reasons']); continue
        B = np.asarray(backward['matrix_cv2x3'], float)
        returned = B[:, :2] @ (point-reverse_origin) + B[:, 2] + source_origin
        cycle = float(np.linalg.norm(returned-seed))
        row['cycle_error_px'] = cycle
        if cycle > 1.5:
            stop(row, 'CENTER_CYCLE_GATE'); continue
        row.update(status='ACCEPTED', accepted_xy=point.tolist())
        accepted.append((point.copy(), float(times[i])))
    result['status'] = 'STOPPED_APPEARANCE_UNVERIFIED' if stopped else 'ROUTE_END'
    result['counts'] = dict(frames=n, seed_frames=int(bool(accepted)),
        accepted_frames=len(accepted), accepted_transitions=max(0, len(accepted)-1),
        stopped_frames=int(stopped), not_run_after_stop=sum(r['status']=='NOT_RUN_AFTER_STOP' for r in result['frames']))
    return result
