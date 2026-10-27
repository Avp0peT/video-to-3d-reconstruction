"""Candidate link diagnostics against frozen, original pair models only."""
from collections import Counter
import hashlib
import json

import numpy as np


def _xy(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('two canonical coordinates required')
    if any(isinstance(v, (bool, str)) for v in value):
        raise ValueError('numeric canonical coordinates required')
    point = np.asarray(value, dtype=float)
    if not np.isfinite(point).all():
        raise ValueError('finite canonical coordinates required')
    return point


def _digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def assess_reassociated_links(recovery, original_geometry):
    """Never refit: independent of fitting only when neither endpoint trained F.

    Endpoint exclusion is conservative and does not establish correspondence truth.
    Malformed or ambiguous original evidence cannot grant geometric support.
    """
    pairs = {}
    for pair in original_geometry.get('pairs', []):
        frames = pair.get('frames', [])
        if (isinstance(frames, list) and len(frames) == 2
                and all(type(v) is int for v in frames) and frames[0] < frames[1]):
            pairs.setdefault(tuple(frames), []).append(pair)
    rows = []
    for link in recovery.get('links', []):
        row = {key: link.get(key) for key in ('from_track_id', 'to_track_id', 'direction',
               'from_frame', 'to_frame', 'xy_from', 'xy_to', 'status', 'gap', 'gap_frames') if key in link}
        # Copy JSON-shaped endpoint data; never expose mutable input containers.
        row = json.loads(json.dumps(row))
        row.update(status='NO_INDEPENDENT_GEOMETRY', correspondence_verified=False,
                   geometry_acceptance=False, eligible_for_triangulation=False,
                   planar_or_low_parallax_ambiguity=None)
        rows.append(row)
        try:
            first, second = link.get('from_frame'), link.get('to_frame')
            if type(first) is not int or type(second) is not int or first == second:
                raise ValueError('distinct integer endpoint frames required')
            a, b = _xy(link.get('xy_from')), _xy(link.get('xy_to'))
            if first > second:
                first, second, a, b = second, first, b, a
            row['original_pair_frames'] = [first, second]
            matches = pairs.get((first, second), [])
            if len(matches) != 1:
                raise ValueError('unique original pair model unavailable')
            pair = matches[0]
            row['planar_or_low_parallax_ambiguity'] = pair.get('planar_or_low_parallax_ambiguity')
            row['original_pair_evidence'] = dict(status=pair.get('status'),
                train_ids=list(pair.get('train_ids', [])), holdout_ids=list(pair.get('holdout_ids', [])),
                sha256=_digest(pair))
            if pair.get('status') != 'CANDIDATE_SUPPORT_NOT_VERIFIED':
                raise ValueError('original pair lacks heldout candidate support')
            F = np.asarray(pair.get('F'), dtype=float)
            if (F.shape != (3, 3) or not np.isfinite(F).all()
                    or not np.isfinite(np.linalg.norm(F)) or np.linalg.norm(F) == 0):
                raise ValueError('original F unavailable, nonfinite, malformed or zero')
            # Normalization preserves Sampson distance and avoids arbitrary F scale.
            F = F / np.linalg.norm(F)
            train_ids = pair.get('train_ids')
            if not isinstance(train_ids, list) or not train_ids or len(set(train_ids)) != len(train_ids):
                raise ValueError('original training identity evidence unavailable')
            candidates = {}
            for candidate in pair.get('candidates', []):
                candidates.setdefault(candidate.get('stable_id'), []).append(candidate)
            dependent = []
            for ident in train_ids:
                found = candidates.get(ident, [])
                if len(found) != 1:
                    raise ValueError('original training endpoint evidence unavailable')
                training = found[0]
                if (np.linalg.norm(a - _xy(training.get('xy_first'))) <= 2
                        or np.linalg.norm(b - _xy(training.get('xy_second'))) <= 2):
                    dependent.append(ident)
            if dependent:
                row.update(status='TRAIN_ENDPOINT_DEPENDENT', dependent_train_ids=dependent)
                continue
            x, y = np.r_[a, 1.], np.r_[b, 1.]
            fx, fty = F @ x, F.T @ y
            denominator = float(fx[:2] @ fx[:2] + fty[:2] @ fty[:2])
            if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
                raise ValueError('original F gives zero or invalid Sampson denominator')
            error = float(abs(y @ fx) / np.sqrt(denominator))
            if not np.isfinite(error):
                raise ValueError('nonfinite Sampson residual')
            row.update(status='EXISTING_F_SUPPORT_CANDIDATE' if error <= 1.5 else 'GEOMETRY_DISAGREES',
                       sampson_px=error, training_endpoint_independent=True)
        except (ValueError, TypeError, OverflowError) as exc:
            row['reason'] = str(exc)
    return dict(schema='reassociation-original-geometry-v1', links=rows,
                counts=dict(Counter(row['status'] for row in rows)), model_refit=False,
                correspondence_verified=False, geometry_acceptance=False,
                eligible_for_triangulation=False,
                limitations=['Frozen original F only; endpoint exclusion is not identity verification.',
                    'Residual support remains candidate evidence, including planar or low-parallax cases.'])
