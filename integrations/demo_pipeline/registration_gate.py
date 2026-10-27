"""Predict saved-snapshot registration gates; never invoke native estimation."""
from collections import Counter
import math


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('finite plain number required: '+label)
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError('finite plain number required: '+label) from exc
    if not math.isfinite(result):
        raise ValueError('finite plain number required: '+label)
    return result


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('plain integer required: '+label)
    return value


def bogus_camera(width, height, params, minratio, maxratio, maxextra):
    """SIMPLE_RADIAL comparisons in COLMAP 4.2.1 models.h:1071–1126.

    Validation failure is an exception, not a native bogus-camera conclusion.
    The native predicate's inclusive bounds and absolute extra parameter are
    preserved; this does not evaluate radial monotonicity or physical accuracy.
    """
    _integer(width, 'width', 1); _integer(height, 'height', 1)
    if not isinstance(params, (list, tuple)) or len(params) != 4:
        raise ValueError('four SIMPLE_RADIAL parameters required')
    f, cx, cy, k = [_number(v, 'camera parameter') for v in params]
    low, high, extra = [_number(v, 'threshold') for v in (minratio, maxratio, maxextra)]
    if low < 0 or high < low or extra < 0:
        raise ValueError('ordered nonnegative native thresholds required')
    ratio = f * (1.0 / max(width, height))  # Match native floating operation order.
    principal_bad = cx < 0 or cx > width or cy < 0 or cy > height
    focal_bad = ratio < low or ratio > high
    extra_bad = abs(k) > extra
    return dict(bogus=principal_bad or focal_bad or extra_bad,
        principal_point_bogus=principal_bad, focal_length_bogus=focal_bad,
        extra_params_bogus=extra_bad, focal_length_ratio=ratio,
        first_native_rejection=('PRINCIPAL_POINT' if principal_bad else 'FOCAL_LENGTH' if focal_bad else
                                'EXTRA_PARAMS' if extra_bad else None),
        predicate='COLMAP-4.2.1-SIMPLE_RADIAL', physical_camera_verified=False)


def summarize_gate(camera_checks, links, min_inliers, visible_count=None):
    """Replay collection order only, before query-camera reset/refinement.

    camera_checks maps positive camera_id to bogus_camera results. Source row
    means the query image row, target means its corresponding registered image.
    The native ordinary dedup set is updated only after acceptance, per query
    row. Structure-less collection counts all posed valid-camera links and does
    not require point3D or deduplicate by point3D. Its visible 2D–2D precheck is
    not supplied by visible_count (which is the ordinary visible 3D count).
    """
    _integer(min_inliers, 'min_inliers', 1)
    if visible_count is not None:
        _integer(visible_count, 'visible_count')
    if type(camera_checks) is not dict or type(links) is not list or len(links) > 200000:
        raise ValueError('camera mapping and bounded ordered link list required')
    for ident, check in camera_checks.items():
        _integer(ident, 'camera_id', 1)
        if type(check) is not dict or type(check.get('bogus')) is not bool:
            raise ValueError('explicit boolean camera bogus result required')
    accepted = set(); potential = set(); ledger = []; ordinary = Counter(); structureless = Counter()
    for ordinal, link in enumerate(links):
        if type(link) is not dict:
            raise ValueError('link record required')
        query = _integer(link['source_row'], 'source_row')
        _integer(link['target_image_id'], 'target_image_id', 1)
        _integer(link['target_point_row'], 'target_point_row')
        camera = _integer(link['camera_id'], 'camera_id', 1)
        if camera not in camera_checks or type(link.get('has_pose')) is not bool:
            raise ValueError('known camera and boolean has_pose required')
        point = link.get('point3D_id')
        if point is not None:
            _integer(point, 'point3D_id')
        bogus = camera_checks[camera]['bogus']
        if not link['has_pose']:
            reason = 'NO_POSE'
        elif point is None:
            reason = 'NO_POINT3D'
        elif (query, point) in accepted:
            reason = 'DUPLICATE_ACCEPTED_POINT3D_FOR_SOURCE_ROW'
        elif bogus:
            reason = 'BOGUS_SOURCE_CAMERA'
        else:
            reason = 'ACCEPTED'
            accepted.add((query, point))
        if link['has_pose'] and point is not None:
            potential.add((query, point))
        sl_reason = 'NO_POSE' if not link['has_pose'] else 'BOGUS_SOURCE_CAMERA' if bogus else 'ACCEPTED'
        ordinary[reason] += 1; structureless[sl_reason] += 1
        ledger.append(dict(ordinal=ordinal, source_row=query, target_image_id=link['target_image_id'],
            target_point_row=link['target_point_row'], point3D_id=point, camera_id=camera,
            structure_based=reason, structureless=sl_reason))
    visible_pass = None if visible_count is None else visible_count >= min_inliers
    collected_pass = ordinary['ACCEPTED'] >= min_inliers
    return dict(status='PREDICTED_FROM_SAVED_SNAPSHOT', ordered_links=len(links), ledger=ledger,
        structure_based=dict(min_inliers=min_inliers, visible_count=visible_count,
            visible_precheck_pass=visible_pass, potential_unique_2D3D=len(potential),
            retained_count=ordinary['ACCEPTED'], rejected_link_count=len(links)-ordinary['ACCEPTED'],
            potential_pairs_without_valid_source=len(potential)-len(accepted), counts=dict(ordinary),
            collection_gate_pass=collected_pass,
            gate_reachable=False if not collected_pass or visible_pass is False else True if visible_pass is True else None),
        structureless=dict(min_inliers=2*min_inliers, retained_count=structureless['ACCEPTED'],
            rejected_link_count=len(links)-structureless['ACCEPTED'], counts=dict(structureless),
            collection_gate_pass=structureless['ACCEPTED'] >= 2*min_inliers,
            gate_reachable=False if structureless['ACCEPTED'] < 2*min_inliers else None,
            visible_correspondence_precheck='NOT_EVALUATED', registered_image_precheck='NOT_EVALUATED'),
        native_called=False, historical_cause_verified=False, identity_verified=False,
        limitations=['Saved terminal state is not historical registration state.',
            'Passing collection gates does not establish native pose estimation or refinement success.',
            'Structure-less visible-correspondence count and two-registered-image gate require separate evidence.'])
